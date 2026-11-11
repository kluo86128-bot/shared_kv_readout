"""Packed C1 recomputation with independent attention and top-k per source.

The original single-request implementation remains the reference and fallback.
Packed GEMMs need not be bitwise identical on BF16 CUDA hardware.
"""
import math

import torch

from .native_blend import BlendResult, native_cacheblend, relocate_keys


@torch.no_grad()
def reference_scores(model,public,relocated,check_layer,execution):
    """Hard selection uses single-request GEMM shapes even when values are packed.

    Before selection the public attention is independent of each writer, so
    replay that short public prefix only once using single-request shapes. This avoids
    changing top-k at tied/near-tied scores just because batch shapes changed.
    """
    length=public.shape[1]
    hidden=public;carry=None
    positions=torch.arange(length,device=public.device)
    for index,layer in enumerate(model.model.layers):
        residual,q,k,v=execution.project(layer,model.model,hidden,positions,carry,blend=True)
        if index==check_layer:
            break
        hidden,carry=execution.finish(layer,residual,q,k,v,paged=False)
    fresh=k[0].transpose(0,1).reshape(length,-1).float()
    return torch.stack([(fresh-writer[check_layer][0][0].transpose(0,1).reshape(length,-1).float())
                        .square().sum(dim=1) for writer in relocated])



def native_cacheblend_many(model, public, writers, *, writer_position_starts,
                           check_layer, ratio, selection_indices=None):
    execution = getattr(model, '_readout_execution', None)
    if execution is None or not execution.fast:
        raise ValueError('Packed C1 requires the fast native execution profile')
    if (not writers or len(writers) != len(writer_position_starts)
            or public.ndim != 3 or public.shape[0] != 1 or public.shape[1] < 1):
        raise ValueError('Require one shared C1 and one position start per writer')
    layers = model.model.layers
    if (not 0 <= check_layer < len(layers) or not math.isfinite(ratio) or not 0 <= ratio <= 1):
        raise ValueError('Invalid C1 check layer or recomputation ratio')
    length = public.shape[1]
    if selection_indices is not None:
        from .c1_selection import check_indices
        if selection_indices.ndim!=2 or selection_indices.shape[0]!=len(writers):
            raise ValueError('Packed random C1 indices must have one row per writer')
        for row in selection_indices:check_indices(row,length,ratio,public.device)
    for writer in writers:
        if len(writer) != len(layers):
            raise ValueError('Writer cache layer count differs from model')
        for key, value in writer:
            if key.shape[0] != 1 or key.shape[2] != length or key.shape != value.shape:
                raise ValueError('Each writer cache must contain exactly its own C1')
    if ratio == 0:
        return [native_cacheblend(model, public, writer, writer_position_start=start,
                                 check_layer=check_layer, ratio=0,
                                 selection_indices=None if selection_indices is None else selection_indices[i])
                for i,(writer, start) in enumerate(zip(writers, writer_position_starts))]
    if len(writers) == 1:
        return [native_cacheblend(model, public, writers[0], check_layer=check_layer,
            ratio=ratio, writer_position_start=writer_position_starts[0],
            selection_indices=None if selection_indices is None else selection_indices[0])]

    batch = len(writers)
    count = max(int(length * ratio), 1)
    relocated = [relocate_keys(model.model, writer, start, 0)
                 for writer, start in zip(writers, writer_position_starts)]
    hidden = public.repeat(1, batch, 1)
    positions = torch.arange(length, device=public.device).repeat(batch)
    selected = scores = None
    carry = None
    mixed = [[] for _ in writers]
    for index, layer in enumerate(layers):
        residual, query, fresh_key, fresh_value = execution.project(
            layer, model.model, hidden, positions, carry, blend=True)
        old_key = torch.cat([writer[index][0] for writer in relocated], dim=2)
        old_value = torch.cat([writer[index][1] for writer in relocated], dim=2)
        if index == check_layer:
            scores = (reference_scores(model,public,relocated,check_layer,execution)
                      if selection_indices is None else public.new_empty((batch,0),dtype=torch.float32))
            with torch.no_grad():
                selected = (scores.topk(count, dim=1).indices.sort(dim=1).values
                            if selection_indices is None else selection_indices.clone())
                packed_indices = (selected + torch.arange(batch, device=public.device)[:, None]
                                  * length).reshape(-1)
            query = query.index_select(2, packed_indices)
            fresh_key = fresh_key.index_select(2, packed_indices)
            fresh_value = fresh_value.index_select(2, packed_indices)
            residual = residual.index_select(1, packed_indices)
            positions = positions.index_select(0, packed_indices)
        if selected is None:
            key, value = fresh_key, fresh_value
            stored = (old_key, old_value)
            query_length = length
        else:
            key = old_key.index_copy(2, packed_indices, fresh_key)
            value = old_value.index_copy(2, packed_indices, fresh_value)
            stored = (key, value)
            query_length = count
        for owner in range(batch):
            begin, end = owner * length, (owner + 1) * length
            mixed[owner].append(tuple(t[:, :, begin:end] for t in stored))
        # Separate boundaries preserve single-request bottom-right causal masking.
        hidden, carry = execution.finish(layer, residual, query, key, value,
            paged=False, qlens=[query_length] * batch, klens=[length] * batch)

    indices = selected.detach().cpu().tolist()
    results = []
    for owner in range(batch):
        audit = dict(algorithm='lmcache_v0.4.2_cacheblend', c1_tokens=length,
            check_layer=check_layer, ratio=ratio, selected_tokens=count,
            selected_indices=indices[owner], layer0_full_attention_tokens=length if check_layer > 0 else 0,
            layer0_query_tokens=length if check_layer > 0 else count, post_check_query_tokens=count,
            position_policy='reader_destination_zero', topk_policy='floor_min_one_fixed_after_check',
            attention_policy='flash_bottom_right_causal', cached_key_policy='relocate_source_to_destination',
            pre_check_cache_policy='retain_relocated_writer', c1_execution='packed_independent',
            c1_batch_records=batch)
        if selection_indices is not None:audit.update(algorithm='native_random_sparse',selection_method='random')
        results.append(BlendResult(tuple(mixed[owner]), selected[owner], scores[owner], audit))
    return results
