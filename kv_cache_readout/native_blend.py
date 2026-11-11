"""Differentiable surrogate for LMCache 0.4.2 CacheBlend.

The implementation deliberately mirrors the pinned native contract:

* reader C1 positions start at zero;
* layers before ``check_layer`` are fully recomputed;
* the check layer scores every fresh post-RoPE key against the writer key;
* ``floor(tokens * ratio)``, with a minimum of one, is selected once;
* the same positions are updated in every later layer; and
* sparse queries use FlashAttention's bottom-right causal alignment.

Hard top-k indices are not differentiated. Tensor values on the selected and
unselected branches remain differentiable.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    rotate_half,
)


@dataclass(frozen=True)
class BlendResult:
    layers: tuple
    selected: torch.Tensor
    scores: torch.Tensor
    audit: dict


def bottom_right_causal_mask(query_tokens: int, key_tokens: int, *, device, dtype):
    """Additive mask equivalent to FlashAttention causal bottom-right alignment."""
    if query_tokens < 1 or key_tokens < query_tokens:
        raise ValueError("Require 1 <= query_tokens <= key_tokens")
    rows = torch.arange(query_tokens, device=device) + (key_tokens - query_tokens)
    columns = torch.arange(key_tokens, device=device)
    allowed = columns[None, :] <= rows[:, None]
    mask = torch.zeros((1, 1, query_tokens, key_tokens), device=device, dtype=dtype)
    return mask.masked_fill(~allowed[None, None], torch.finfo(dtype).min)


def _project(layer, model, hidden, positions):
    residual = hidden
    normed = layer.input_layernorm(hidden)
    attention = layer.self_attn
    shape = (*normed.shape[:-1], -1, attention.head_dim)
    query = attention.q_norm(attention.q_proj(normed).view(shape)).transpose(1, 2)
    key = attention.k_norm(attention.k_proj(normed).view(shape)).transpose(1, 2)
    value = attention.v_proj(normed).view(shape).transpose(1, 2)
    cos, sin = model.rotary_emb(normed, position_ids=positions[None])
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    return residual, query, key, value


def _decoder_layer(layer, residual, query, key, value, mask):
    attention = layer.self_attn
    output, _ = eager_attention_forward(
        attention, query, key, value, mask,
        scaling=attention.scaling, dropout=0.0,
    )
    hidden = residual + attention.o_proj(output.reshape(output.shape[0], output.shape[1], -1).contiguous())
    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden


def native_cacheblend(model_for_causal_lm, public_embeddings, writer_layers, *,
                      check_layer: int, ratio: float, writer_position_start=0, trace=None,
                      selection_indices=None) -> BlendResult:
    """Build the reader-side mixed C1 cache with LMCache 0.4.2 semantics."""
    model = model_for_causal_lm.model
    layers = model.layers
    length = public_embeddings.shape[1]
    if public_embeddings.shape[0] != 1 or length < 1:
        raise ValueError("native_cacheblend currently requires one nonempty C1 sequence")
    if not 0 <= check_layer < len(layers):
        raise ValueError("check_layer is outside the model")
    if not 0 <= ratio <= 1 or not math.isfinite(ratio):
        raise ValueError("ratio must be finite and in [0, 1]")
    if selection_indices is not None:
        from .c1_selection import check_indices
        check_indices(selection_indices,length,ratio,public_embeddings.device)
    if len(writer_layers) != len(layers):
        raise ValueError("Writer cache layer count differs from model")
    for key, value in writer_layers:
        if key.shape[0] != 1 or key.shape[2] != length or value.shape != key.shape:
            raise ValueError("Writer cache must contain exactly the C1 region")

    hidden = public_embeddings
    writer_layers = relocate_keys(model, writer_layers, writer_position_start, 0)
    if ratio == 0:
        selected = torch.empty(0, device=hidden.device, dtype=torch.long)
        return BlendResult(tuple(writer_layers), selected, hidden.new_empty(0),
            dict(algorithm='relocated_reuse_zero_control', ratio=0, check_layer=check_layer,
                 selected_indices=[], selected_tokens=0, c1_tokens=length))
    positions = torch.arange(length, device=hidden.device, dtype=torch.long)
    selected = None
    scores = hidden.new_empty(0, dtype=torch.float32)
    mixed_layers = []
    layer0_query_tokens = None
    execution=getattr(model_for_causal_lm,'_readout_execution',None)
    carry=None
    for layer_id, (layer, (old_key, old_value)) in enumerate(zip(layers, writer_layers)):
        residual, query, fresh_key, fresh_value = (
            execution.project(layer,model,hidden,positions,carry,blend=True) if execution is not None
            else _project(layer, model, hidden, positions))
        if layer_id == check_layer:
            # vLLM keys are [tokens, flattened_kv_heads]. HF keys are
            # [batch, kv_heads, tokens, head_dim]; summing heads and head_dim
            # gives the same per-token ranking as LMCache 0.4.2.
            if selection_indices is not None:
                scores=hidden.new_empty(0,dtype=torch.float32)
            elif execution is not None:
                fresh=fresh_key[0].transpose(0,1).reshape(length,-1)
                old=old_key[0].transpose(0,1).reshape(length,-1)
                scores=(fresh.float()-old.float()).square().sum(dim=1)
            else:
                scores = (fresh_key.float() - old_key.float()).square().sum(dim=(0, 1, 3))
            count = max(int(length * ratio), 1)
            with torch.no_grad():
                selected = (scores.topk(count).indices.sort().values
                            if selection_indices is None else selection_indices.clone())
            query = query.index_select(2, selected)
            fresh_key = fresh_key.index_select(2, selected)
            fresh_value = fresh_value.index_select(2, selected)
            residual = residual.index_select(1, selected)
            positions = positions.index_select(0, selected)

        if selected is None:
            cache_key, cache_value = fresh_key, fresh_value
        else:
            # Functional index_copy is intentional: it preserves gradients from
            # unselected writer KV while routing selected positions to fresh KV.
            cache_key = old_key.index_copy(2, selected, fresh_key)
            cache_value = old_value.index_copy(2, selected, fresh_value)

        # LMCache writes the connector buffer into paged KV. Before the check
        # layer fresh KV only drives internal attention and does not overwrite
        # that buffer (process_qkv's no-imp_indices branch).
        stored_key, stored_value = ((old_key, old_value) if selected is None else (cache_key, cache_value))
        mixed_layers.append((stored_key, stored_value))
        if trace is not None:
            trace.append(dict(layer=layer_id, query=query.detach().cpu(),
                              attention_key=cache_key.detach().cpu(), attention_value=cache_value.detach().cpu(),
                              stored_key=stored_key.detach().cpu(), stored_value=stored_value.detach().cpu(),
                              selected_indices=None if selected is None else selected.detach().cpu().tolist(),
                              scores=scores.detach().cpu() if layer_id==check_layer else None))
        mask = bottom_right_causal_mask(
            query.shape[2], cache_key.shape[2], device=query.device, dtype=query.dtype,
        )
        if layer_id == 0:
            layer0_query_tokens = query.shape[2]
        if execution is not None:
            hidden,carry=execution.finish(layer,residual,query,cache_key,cache_value,paged=False)
        else:
            hidden = _decoder_layer(layer, residual, query, cache_key, cache_value, mask)

    if selected is None:
        raise AssertionError("check_layer was not executed")
    audit = {
        "algorithm": "lmcache_v0.4.2_cacheblend",
        "c1_tokens": length,
        "check_layer": check_layer,
        "ratio": ratio,
        "selected_tokens": len(selected),
        "selected_indices": selected.detach().cpu().tolist(),
        "layer0_full_attention_tokens": length if check_layer > 0 else 0,
        "layer0_query_tokens": layer0_query_tokens,
        "post_check_query_tokens": len(selected),
        "position_policy": "reader_destination_zero",
        "topk_policy": "floor_min_one_fixed_after_check",
        "attention_policy": "flash_bottom_right_causal",
        'cached_key_policy': 'relocate_source_to_destination',
        'pre_check_cache_policy': 'retain_relocated_writer',
    }
    if selection_indices is not None:
        audit.update(algorithm='native_random_sparse',selection_method='random')
    return BlendResult(tuple(mixed_layers), selected, scores, audit)


def relocate_keys(model, layers, source_start, destination_start):
    execution=getattr(model,'_readout_execution',None)
    if execution is not None:
        return execution.relocate(model,layers,source_start,destination_start)
    if source_start == destination_start:
        return layers
    key = layers[0][0]
    length = key.shape[2]
    columns = torch.arange(length, device=key.device)[None]
    old_cos, old_sin = model.rotary_emb(key, position_ids=columns+source_start)
    new_cos, new_sin = model.rotary_emb(key, position_ids=columns+destination_start)
    c = (old_cos.float()*new_cos.float() + old_sin.float()*new_sin.float())[:, None]
    s = (new_sin.float()*old_cos.float() - new_cos.float()*old_sin.float())[:, None]
    return tuple(((k.float()*c + rotate_half(k.float())*s).to(k.dtype), v) for k,v in layers)

