"""Preserve source positions at native store/load boundaries.

The native token database's store spans are authoritative. Keep their positions
on each actual CPU cache object independently of LMCache's cached_positions.
Use them at the fused RoPE call, not an inferred offset or a fitted rotation.
"""
from contextvars import ContextVar
from functools import wraps

import torch

CURRENT = ContextVar('readout_rope_transfer', default=None)
POSITION_ATTR = '_readout_native_source_positions'


def remember(memory_objs, starts, ends):
    if len(starts) != len(ends):
        raise RuntimeError('Unequal native store span counts')
    for layer_id, layer in enumerate(memory_objs):
        if len(layer) != len(starts):
            raise RuntimeError('Unequal native store object/span counts')
        for chunk_id, (obj, start, end) in enumerate(zip(layer, starts, ends)):
            tensor = obj.tensor
            # SegmentTokenDatabase emits an empty final segment when a prompt
            # ends in the separator (Qwen3's generation prompt does). Preserve
            # that native object and its empty position vector.
            if (start < 0 or end < start or tensor is None or tensor.ndim != 3
                    or tensor.shape[0] != 2 or tensor.shape[1] != end-start):
                raise RuntimeError('Native store object does not match its token span: '
                    f'layer={layer_id}, chunk={chunk_id}, start={start}, end={end}, '
                    f'shape={None if tensor is None else tuple(tensor.shape)}')
            setattr(obj, POSITION_ATTR, torch.arange(start, end, dtype=torch.long))


def source_positions(obj):
    positions = getattr(obj, POSITION_ATTR, None)
    if (not torch.is_tensor(positions) or positions.dtype != torch.long
            or positions.ndim != 1 or positions.numel() != obj.tensor.shape[1]):
        raise RuntimeError('Cached KV has no verified source positions; use a fresh fixed run')
    return positions


class Transfer:
    """Forward the full generator protocol without losing send() values."""
    def __init__(self, generator, starts, ends, audit):
        self.generator = generator
        self.starts, self.ends, self.audit = list(starts), list(ends), audit
        self.positions = None
        self.layer = -1

    def __iter__(self):
        return self

    def _step(self, operation, *args):
        token = CURRENT.set(self)
        try:
            return operation(*args)
        finally:
            CURRENT.reset(token)

    def __next__(self):
        return self._step(next, self.generator)

    def send(self, objects):
        if objects is not None:
            if len(objects) != len(self.starts):
                raise RuntimeError('Native retrieval object/span counts differ')
            offset = self.starts[0]
            positions = torch.zeros(self.ends[-1]-offset, dtype=torch.long)
            for obj, start, end in zip(objects, self.starts, self.ends):
                saved = source_positions(obj)
                if saved.numel() != end-start:
                    raise RuntimeError('Retrieved source position count differs from destination span')
                positions[start-offset:end-offset] = saved
            self.positions = positions
            self.layer += 1
        return self._step(self.generator.send, objects)

    def throw(self, *args):
        return self._step(self.generator.throw, *args)

    def close(self):
        return self._step(self.generator.close)


def install(connector_class, fused_class, audit=None):
    if getattr(connector_class, '_readout_source_position_fix', False):
        return
    original_store = connector_class.batched_from_gpu
    original_load = connector_class.batched_to_gpu
    original_rope = fused_class.__call__

    @wraps(original_store)
    def store(self, memory_objs, starts, ends, **kwargs):
        remember(memory_objs, starts, ends)
        yield from original_store(self, memory_objs, starts, ends, **kwargs)

    @wraps(original_load)
    def load(self, starts, ends, **kwargs):
        return Transfer(original_load(self, starts, ends, **kwargs), starts, ends, audit)

    @wraps(original_rope)
    def fused(self, old_positions, new_positions, key):
        transfer = CURRENT.get()
        if transfer is None:
            # LMCache also calls this during its own startup correctness test.
            return original_rope(self, old_positions, new_positions, key)
        if transfer.positions is None:
            raise RuntimeError('RoPE relocation occurred before native source positions were loaded')
        expected = transfer.positions.to(device=old_positions.device)
        destination = torch.arange(transfer.starts[0], transfer.ends[-1],
                                   device=new_positions.device, dtype=torch.long)
        if not torch.equal(destination, new_positions) or expected.shape != old_positions.shape:
            raise RuntimeError('Unexpected native RoPE destination layout')
        corrected = not torch.equal(old_positions, expected)
        result = original_rope(self, expected, new_positions, key)
        if transfer.audit is not None:
            transfer.audit(transfer.layer, old_positions, expected, new_positions, corrected)
        return result

    connector_class.batched_from_gpu = store
    connector_class.batched_to_gpu = load
    fused_class.__call__ = fused
    connector_class._readout_source_position_fix = True
