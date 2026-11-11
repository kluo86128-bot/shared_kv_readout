"""Frozen S pairing and the preserved source-position S recomputation objective."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import hashlib
import math
from pathlib import Path

import torch
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    rotate_half,
)

from .config import file_hash, fingerprint, resolve


@lru_cache(maxsize=8)
def _rows(path: str, sha: str):
    if file_hash(path) != sha:
        raise ValueError("Suffix split hash changed")
    rows = tuple(json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines())
    if not rows:
        raise ValueError("Empty suffix split")
    return rows


def suffix_identity(cfg: dict):
    root = resolve(cfg["scoped_cache"]["suffix_directory"])
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "scoped_squad_v1":
        raise ValueError("Expected scoped_squad_v1 suffix data")
    if manifest['max_tokens'] > cfg['scoped_cache']['suffix_max_tokens']:
        raise ValueError('Frozen suffix dataset exceeds the configured token budget')
    for name, expected in manifest["files"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Suffix file changed: {name}")
    return {"manifest_sha256": file_hash(manifest_path), "files": manifest["files"],
            "tokenizer_sha256": manifest["tokenizer_sha256"]}


def pair_for(record: dict, cfg: dict, epoch: int = 0):
    if '_suffix_fixture' in record:
        return record['_suffix_fixture']
    root = resolve(cfg["scoped_cache"]["suffix_directory"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    split = record.get("split", "test")
    rows = _rows(str(root / f"{split}.jsonl"), manifest["files"][f"{split}.jsonl"])
    key = {"seed": cfg["seed"], "record_id": record["record_id"], "split": split,
           "epoch": epoch if split == "train" else 0}
    # Match the original experiment's JSON whitespace exactly: this hash is
    # part of the frozen pairing protocol, independent of config signatures.
    pairing_hash = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return rows[int(pairing_hash[:16], 16) % len(rows)]


def mismatch_for(record, cfg):
    target = pair_for(record, cfg)
    root = resolve(cfg['scoped_cache']['suffix_directory'])
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    split = record.get('split', 'test')
    pool = _rows(str(root / f'{split}.jsonl'), manifest['files'][f'{split}.jsonl'])
    candidates = [s for s in pool if s['suffix_id'] != target['suffix_id'] and s['text'] != target['text']]
    def old_hash(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return min(candidates, key=lambda s: (abs(len(s['ids'])-len(target['ids'])),
               s['article_id'] == target['article_id'],
               old_hash(dict(seed=cfg['seed'], record_id=record['record_id'], suffix_id=s['suffix_id']))))


@dataclass(frozen=True)
class SuffixContext:
    layers: tuple
    start: int
    length: int
    token_ids: tuple


def _source_sparse_recompute(backend, context: SuffixContext, *, ratio: float):
    """Preserve the previous S-internal CacheBlend objective at source positions."""
    if ratio == 0:
        return context.layers
    model = backend.model.model
    device = backend.device
    length = context.length
    positions = torch.arange(length, device=device)
    hidden = backend.embedding(backend.tensor(context.token_ids))
    remaining = math.ceil(length * len(context.layers) * ratio)
    updated = []
    for index, (layer, (old_key, old_value)) in enumerate(zip(model.layers, context.layers)):
        count = min(len(positions), math.ceil(remaining / (len(context.layers) - index)))
        residual = hidden
        normed = layer.input_layernorm(hidden)
        attention = layer.self_attn
        shape = (*normed.shape[:-1], -1, attention.head_dim)
        query = attention.q_norm(attention.q_proj(normed).view(shape)).transpose(1, 2)
        key = attention.k_norm(attention.k_proj(normed).view(shape)).transpose(1, 2)
        value = attention.v_proj(normed).view(shape).transpose(1, 2)
        absolute = (positions + context.start)[None]
        cos, sin = model.rotary_emb(normed, position_ids=absolute)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if count < len(positions):
            with torch.no_grad():
                scores = (key.float() - old_key.index_select(2, positions).float()).square().mean((0, 1, 3))
                chosen = scores.topk(count, sorted=False).indices.sort().values
            positions = positions.index_select(0, chosen)
            residual = residual.index_select(1, chosen)
            query, key, value = (tensor.index_select(2, chosen) for tensor in (query, key, value))
        remaining -= count
        mixed_key = old_key.index_copy(2, positions, key)
        mixed_value = old_value.index_copy(2, positions, value)
        updated.append((mixed_key, mixed_value))
        if index != len(context.layers) - 1:
            allowed = torch.arange(length, device=device)[None, :] <= positions[:, None]
            mask = torch.zeros((1, 1, count, length), device=device, dtype=backend.dtype)
            mask.masked_fill_(~allowed[None, None], torch.finfo(backend.dtype).min)
            output, _ = eager_attention_forward(
                attention, query, mixed_key, mixed_value, mask,
                scaling=attention.scaling, dropout=0.0,
            )
            hidden = residual + attention.o_proj(output.reshape(1, count, -1).contiguous())
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return tuple(updated)


def _keys_without_rope(backend, context: SuffixContext, layers):
    key = layers[0][0]
    positions = torch.arange(context.start, context.start + context.length,
                             device=backend.device)[None]
    cos, sin = backend.model.model.rotary_emb(key, position_ids=positions)
    cos, sin = cos[:, None].float(), sin[:, None].float()
    return tuple(layer_key.float() * cos - rotate_half(layer_key.float()) * sin
                 for layer_key, _ in layers)


def preservation_loss(backend, student: SuffixContext, reference: SuffixContext, *, ratio: float):
    if student.length != reference.length or student.token_ids != reference.token_ids:
        raise ValueError("Student/reference S pairing changed")
    student_layers = _source_sparse_recompute(backend, student, ratio=ratio)
    with torch.no_grad():
        reference_layers = _source_sparse_recompute(backend, reference, ratio=ratio)
    student_keys = _keys_without_rope(backend, student, student_layers)
    reference_keys = _keys_without_rope(backend, reference, reference_layers)
    epsilon = backend.cfg["scoped_cache"]["epsilon"]

    key_loss = torch.stack([relative_per_head(a, b, epsilon) for a, b in zip(student_keys, reference_keys)]).mean()
    value_loss = torch.stack([relative_per_head(a[1], b[1], epsilon)
                              for a, b in zip(student_layers, reference_layers)]).mean()
    return (key_loss + value_loss) / 2, {
        "kv_key_error": key_loss.detach(), "kv_value_error": value_loss.detach(),
    }


def relative_per_head(current, target, epsilon):
    target = target.detach().float()
    error = (current.float() - target).square().mean((-2, -1))
    energy = target.square().mean((-2, -1))
    return (error / energy.clamp_min(epsilon)).mean()

