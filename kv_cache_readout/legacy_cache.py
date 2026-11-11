"""Immutable tensor snapshots; physical cache slots and RoPE positions are distinct."""
from dataclasses import dataclass
import torch
from transformers import DynamicCache


def pairs(cache):
    return tuple((layer.keys, layer.values) for layer in cache.layers)


def fresh_cache(layers):
    result = DynamicCache()
    for index, (key, value) in enumerate(layers):
        result.update(key.clone(), value.clone(), index)
    return result


def slice_layers(layers, start, stop=None):
    return tuple((k[:, :, start:stop].clone(), v[:, :, start:stop].clone()) for k, v in layers)


def causal_mask(count, past, *, device, dtype):
    rows = past + torch.arange(count, device=device)
    cols = torch.arange(past + count, device=device)
    allowed = cols[None, :] <= rows[:, None]
    mask = torch.zeros((1, 1, count, past + count), device=device, dtype=dtype)
    return mask.masked_fill(~allowed[None, None], torch.finfo(dtype).min)


@dataclass(frozen=True)
class Context:
    layers: tuple
    next_position: int
    public_start: int
    public_length: int
    full_context: bool = False
    metadata: dict | None = None

    @property
    def physical_length(self):
        return self.layers[0][0].shape[2]
