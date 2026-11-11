"""Portable experiment identity: content is scientific, machine paths are not."""
from copy import deepcopy
import hashlib
from .config import fingerprint


def scientific_config(cfg):
    value = deepcopy(cfg)
    for key in ('model_name_or_path', 'dataset_directory', 'output_directory', 'device'):
        value.pop(key, None)
    value['scoped_cache'].pop('suffix_directory', None)
    for key in ('sample_batch_size', 'questions_per_sample_batch', 'source_batch_records',
                'preservation_batch_records', 'question_batch_size', 'c1_batch_size',
                'question_order', 'checkpoint_questions', 'checkpoint_every_steps', 'max_steps'):
        value['training'].pop(key, None)
    return value


def portable_dataset(value):
    return {k: v for k, v in value.items() if k != 'directory'}


def model_content(value):
    # The complete config JSON is itself in assets; HF's runtime _name_or_path is not.
    return {k: value[k] for k in ('assets', 'tokenizer_sha256')}


def local_batch(batch, rank, world):
    if world < 1 or not 0 <= rank < world:
        raise ValueError('Invalid distributed rank')
    return batch[rank::world]


def remap_seed(seed, epoch, cursor, step, rank, world):
    payload = f'{seed}|remap-v1|{epoch}|{cursor}|{step}|{rank}|{world}'
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], 'big')


def resume_compatibility(state, cfg, dataset, model, code):
    if state.get('schema_version') not in ('native_lmcache_u_v2', 'aligned_readout_v3'):
        raise ValueError('Unsupported checkpoint schema; optimizer migration is not defined')
    import torch
    if not torch.isfinite(state['prompt']).all():
        raise ValueError('Checkpoint prompt contains nonfinite values')
    if portable_dataset(state['dataset']) != portable_dataset(dataset):
        raise ValueError('Dataset/tokenizer/suffix contents changed')
    if model_content(state['model']) != model_content(model):
        raise ValueError('Model/tokenizer asset contents changed')
    old = scientific_config(state['config'])
    new = scientific_config(cfg)
    for key in ('prompt_length', 'prompt_partition', 'public_text', 'delimiter',
                'question_format', 'scene', 'seed', 'loss', 'focal_fields', 'scoped_cache'):
        if old[key] != new[key]:
            raise ValueError(f'Incompatible training protocol: {key}')
    if state.get('code') != code:
        from .config import ROOT
        import json
        migration = json.loads((ROOT / 'migrations/known_code.json').read_text(encoding='utf-8'))
        if state['code'] not in migration:
            raise ValueError('Unknown checkpoint code; refusing an unverified code-identity rewrite')
    return old != new


def condition_id(cfg):
    return fingerprint(scientific_config(cfg))[:16]
