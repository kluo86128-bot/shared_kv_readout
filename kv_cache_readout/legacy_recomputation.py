"""Internal recomputation of C1 or explicitly supplied S, retaining source RoPE positions.

These are controlled reuse simulations, not a serving cache lookup. Sparse
Random/CacheBlend attention reads the remaining old KV of that region; EPIC
substitutes a clean prefix reference of the same region. No P/U/C2/Q/labels are inputs.
"""
import hashlib
import math
import random
import time
from dataclasses import replace

import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward


METHODS = ('vanilla', 'random', 'epic', 'cacheblend')


def validate_settings(method, ratio, seed):
    if method not in METHODS:
        raise ValueError(f'Unknown recomputation method: {method}')
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('Recomputation ratio must be finite and in [0, 1]')
    if type(seed) is not int or seed < 0:
        raise ValueError('Recomputation seed must be a nonnegative integer')
    if method == 'vanilla' and ratio != 0:
        raise ValueError('Vanilla means zero recomputation')
    return {'method': method, 'ratio': ratio, 'seed': seed,
            'context': 'C_internal_only', 'position_policy': 'retain_original_absolute_positions'}


def random_order(length, seed, record_id):
    """Stable per-record stream, independent of rank, global RNG and ratio."""
    digest = hashlib.sha256(f'{seed}|{record_id}|C-recomputation-v1'.encode()).digest()
    order = list(range(length))
    random.Random(int.from_bytes(digest, 'big')).shuffle(order)
    return order


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def recompute_context(backend, context, *, method, ratio, seed=17, record_id='', token_ids=None):
    return _recompute_context(backend, context, method=method, ratio=ratio, seed=seed, record_id=record_id, token_ids=token_ids)


def recompute_training_context(backend, context, *, method, ratio, seed=17, record_id='', token_ids=None):
    """Differentiate the selected branch; hard top-k index changes have no gradient."""
    return _recompute_context(backend, context, method=method, ratio=ratio, seed=seed, record_id=record_id, token_ids=token_ids)


def _recompute_context(backend, context, *, method, ratio, seed, record_id, token_ids=None):
    settings = validate_settings(method, ratio, seed)
    ids = backend.public_ids if token_ids is None else list(token_ids)
    length = len(ids)
    if (context.full_context or context.public_length != length or context.physical_length != length
            or context.next_position != context.public_start + length):
        raise ValueError('Recomputation requires a compact cache matching the region tokens at source positions')
    if backend.model.config.model_type != 'qwen3' or backend.cfg['attention_implementation'] != 'eager':
        raise ValueError('C-internal recomputation is validated for eager full-attention Qwen3 only')
    if len(context.layers) != len(backend.model.model.layers) or length == 0:
        raise ValueError('Missing C cache layers/tokens')
    if any(k.shape[0] != 1 or k.shape[2] != length or v.shape != k.shape for k, v in context.layers):
        raise ValueError('Expected one record with equal-shaped C keys and values in every layer')
    _sync(backend.device)
    started = time.perf_counter()
    layers = tuple((k.clone(), v.clone()) for k, v in context.layers)
    num_layers = len(layers)
    selected = [[] for _ in layers]
    reference_tokens = 0
    if ratio == 1 or (method == 'epic' and int(length * ratio) > 0):
        # EPIC's reference replacement deliberately differs from sparse attention.
        with torch.no_grad():
            _, clean = backend.forward(ids=ids, position_start=context.public_start)
        count = length if ratio == 1 else int(length * ratio)
        layers = (tuple((ck.clone(), cv.clone()) for ck, cv in clean) if ratio == 1 else
                  tuple((torch.cat((ck[:, :, :count], k[:, :, count:]), dim=2),
                         torch.cat((cv[:, :, :count], v[:, :, count:]), dim=2))
                        for (k, v), (ck, cv) in zip(layers, clean)))
        selected = [list(range(count)) for _ in layers]
        reference_tokens = length
    elif ratio > 0 and method in ('random', 'cacheblend'):
        layers, selected = _sparse(backend, context, layers, method, ratio, seed, record_id, ids)
    _sync(backend.device)
    counts = [len(indices) for indices in selected]
    stats = dict(settings, public_tokens=length, layer_counts=counts, selected_by_layer=selected,
                 updated_token_layers=sum(counts), total_token_layers=length * num_layers,
                 actual_ratio=sum(counts) / (length * num_layers),
                 clean_reference_tokens=reference_tokens,
                 budget_definition=('ceil(ratio*C_tokens*layers)' if method == 'cacheblend'
                                    else 'floor(ratio*C_tokens) positions in every layer'),
                 seconds=time.perf_counter() - started)
    if token_ids is not None:
        stats.update(context='S_internal_only', region_tokens=length,
                     budget_definition=('ceil(ratio*S_tokens*layers)' if method=='cacheblend'
                                        else 'floor(ratio*S_tokens) positions in every layer'))
    if any(not torch.isfinite(tensor).all() for layer in layers for tensor in layer):
        raise FloatingPointError('Non-finite recomputed cache')
    return replace(context, layers=layers), stats


def _sparse(backend, context, layers, method, ratio, seed, record_id, token_ids):
    model = backend.model.model
    length, device = len(token_ids), backend.device
    if method == 'random':
        positions = torch.tensor(sorted(random_order(length, seed, record_id)[:int(length * ratio)]),
                                 device=device, dtype=torch.long)
    else:
        positions = torch.arange(length, device=device)
    hidden = backend.embedding(backend.tensor(token_ids)).index_select(1, positions)
    remaining = math.ceil(length * len(layers) * ratio)
    selected = []
    updated = []
    for index, (layer, (keys, values)) in enumerate(zip(model.layers, layers)):
        count = (len(positions) if method == 'random' else
                 min(len(positions), math.ceil(remaining / (len(layers) - index))))
        if count == 0:
            selected.append([])
            updated.append((keys, values))
            continue
        residual = hidden
        normed = layer.input_layernorm(hidden)
        attn = layer.self_attn
        shape = (*normed.shape[:-1], -1, attn.head_dim)
        query = attn.q_norm(attn.q_proj(normed).view(shape)).transpose(1, 2)
        key = attn.k_norm(attn.k_proj(normed).view(shape)).transpose(1, 2)
        value = attn.v_proj(normed).view(shape).transpose(1, 2)
        absolute = (positions + context.public_start)[None]
        cos, sin = model.rotary_emb(normed, position_ids=absolute)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if count < len(positions):
            with torch.no_grad():
                scores = (key.float() - keys.index_select(2, positions).float()).square().mean((0, 1, 3))
                chosen = scores.topk(count, sorted=False).indices.sort().values
            positions = positions.index_select(0, chosen)
            residual = residual.index_select(1, chosen)
            query, key, value = (x.index_select(2, chosen) for x in (query, key, value))
        remaining -= count
        keys = keys.index_copy(2, positions, key)
        values = values.index_copy(2, positions, value)
        selected.append(positions.cpu().tolist())
        updated.append((keys, values))
        if index != len(layers) - 1:
            allowed = torch.arange(length, device=device)[None, :] <= positions[:, None]
            mask = torch.zeros((1, 1, count, length), device=device, dtype=backend.dtype)
            mask.masked_fill_(~allowed[None, None], torch.finfo(backend.dtype).min)
            output, _ = eager_attention_forward(attn, query, keys, values, mask,
                                                 scaling=attn.scaling, dropout=0.)
            hidden = residual + attn.o_proj(output.reshape(1, count, -1).contiguous())
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return tuple(updated), selected
