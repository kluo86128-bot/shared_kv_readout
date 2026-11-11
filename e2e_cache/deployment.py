"""Export dedicated input tokens and their trained U/A embedding rows."""
import json
from pathlib import Path
import shutil
import uuid

from .common import atomic_json, path, sha_file


def validate_native_contract(cfg, experiment):
    if experiment.get('c1_selection',{}).get('method','cacheblend')!='cacheblend':
        raise ValueError('Only CacheBlend training is validated by this real-engine deployment contract')
    contract = experiment['native_contract']
    expected = {'method': 'cacheblend', 'ratio': contract['recompute_ratio'],
                'check_layer': contract['check_layer']}
    override = cfg.get('evaluation_override')
    if override is not None:
        if (override != {'check_layer': 0, 'reason': 'layer0_replacement_ablation'}
                or contract['check_layer'] != 1):
            raise ValueError('Invalid layer0 evaluation override for the training contract')
        expected['check_layer'] = 0
    if cfg['recomputation'] != expected:
        raise ValueError('E2E recomputation differs from the U training contract')
    if cfg['cache']['delimiter'] != experiment['delimiter']:
        raise ValueError('E2E delimiter differs from the U training layout')
    if any(material['text'] != experiment['public_text'] for material in cfg['materials']):
        raise ValueError('E2E material differs from the trained C1')
    if cfg['use_a']:
        raise ValueError('The native v1 protocol does not use reader adapter A')


def prepare(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kv_cache_readout.config import load_config, fingerprint
    from kv_cache_readout.data import dataset_identity
    from kv_cache_readout.io import read_checkpoint

    target = path(cfg['deployment'])
    if target.exists():
        raise ValueError('Deployment already exists; choose a new directory')
    experiment = load_config(path(cfg['experiment_config']))
    validate_native_contract(cfg, experiment)
    source = read_checkpoint(path(cfg['source_u']))
    if source.get('schema_version') not in ('native_lmcache_u_v2', 'aligned_readout_v3') or source.get('method') != 'soft':
        raise ValueError('Expected a native_lmcache_u_v2 soft U checkpoint')
    if source['config'] != experiment:
        stored = source['config']
        differences = sorted(k for k in stored.keys() | experiment.keys()
                             if k not in stored or k not in experiment or stored[k] != experiment[k])
        raise ValueError('Experiment configuration differs from U checkpoint. '
            f'Differing keys: {", ".join(differences)}. '
            'Export checkpoint["run_info"]["config"] to a new JSON under configs/ '
            'and point experiment_config to that snapshot.')
    identity = dataset_identity(experiment)
    if source['dataset'] != identity:
        raise ValueError('U dataset identity mismatch')
    if source.get('native_contract') != experiment['native_contract']:
        raise ValueError('U was trained against a different native runtime contract')
    vectors = {'u': source['prompt']} if cfg['use_u'] else {}
    if cfg['use_a']:
        raise ValueError('The refactored native v1 protocol intentionally has no reader adapter A')
    adapter_sha = None
    kwargs = dict(local_files_only=True, trust_remote_code=False)
    model_path = cfg.get('model_name_or_path', experiment['model_name_or_path'])
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    import hashlib
    expected_assets = source['model']['assets']
    model_root = Path(model_path)
    actual_assets = {p.name: {'bytes': p.stat().st_size, 'sha256': sha_file(p)}
                     for p in sorted(model_root.glob('*')) if p.is_file()
                     and p.suffix in ('.safetensors', '.bin', '.json', '.model')}
    if not expected_assets or expected_assets != actual_assets:
        raise ValueError('Base model assets differ from the U training checkpoint')
    if hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode('utf-8')).hexdigest() != source['model']['tokenizer_sha256']:
        raise ValueError('Base tokenizer differs from the U training checkpoint')
    original_vocab = tokenizer.get_vocab().copy()
    names = {kind: [f'<|e2e_{kind}_{i:04d}|>' for i in range(len(value))]
             for kind, value in vectors.items()}
    all_names = [token for tokens in names.values() for token in tokens]
    if any(token in original_vocab for token in all_names):
        raise ValueError('Dedicated U/A token names collide with model vocabulary')
    if tokenizer.add_special_tokens({'additional_special_tokens': all_names},
                                    replace_additional_special_tokens=False) != len(all_names):
        raise ValueError('Unable to allocate all U/A tokens')
    mapping = {}
    for kind, tokens in names.items():
        ids = tokenizer.convert_tokens_to_ids(tokens)
        if tokenizer.encode(''.join(tokens), add_special_tokens=False) != ids:
            raise ValueError('U/A token text does not round-trip')
        mapping[kind] = {'text': ''.join(tokens), 'tokens': tokens, 'ids': ids}
    if any(tokenizer.convert_tokens_to_ids(token) != index for token, index in original_vocab.items()):
        raise ValueError('Original vocabulary IDs changed')
    model = AutoModelForCausalLM.from_pretrained(model_path,
        torch_dtype=getattr(torch, experiment['dtype']), **kwargs)
    original_output_size = model.get_output_embeddings().weight.shape[0]
    model.resize_token_embeddings(max(len(tokenizer), original_output_size), mean_resizing=False)
    with torch.no_grad():
        for kind, value in vectors.items():
            if value.ndim != 2 or value.shape[1] != model.config.hidden_size or not torch.isfinite(value).all():
                raise ValueError(f'Invalid {kind} vector shape/values')
            model.get_input_embeddings().weight[mapping[kind]['ids']] = value.to(model.dtype)
            actual = model.get_input_embeddings().weight[mapping[kind]['ids']].float().cpu()
            if not torch.equal(actual, value.to(model.dtype).float().cpu()):
                raise ValueError('Exported embedding rows differ from checkpoint')
    temporary = target.with_name(target.name + '.preparing-' + uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        torch.save({k: v.to(model.dtype).cpu() for k, v in vectors.items()}, temporary / 'ua_vectors.pt')
        manifest = dict(schema=1, mapping=mapping, experiment=experiment,
            experiment_sha=fingerprint(experiment), dataset=identity,
            source_u_sha256=sha_file(path(cfg['source_u'])), adapter_sha256=adapter_sha,
            enabled_prompts={'u': cfg['use_u'], 'a': cfg['use_a']},
            original_output_size=original_output_size, deployment_assumption='dedicated input token embedding rows',
            files={p.name: sha_file(p) for p in temporary.iterdir() if p.is_file()})
        if cfg.get('evaluation_override') is not None:
            manifest['evaluation_override'] = cfg['evaluation_override']
            manifest['evaluation_recomputation'] = cfg['recomputation']
        if 'model_name_or_path' in cfg:
            # Relocate identical assets without editing the saved training config.
            manifest['deployment_model_name_or_path'] = cfg['model_name_or_path']
        atomic_json(temporary / 'e2e_manifest.json', manifest)
        temporary.rename(target)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return manifest


def verify(cfg):
    from .common import read_json
    target = path(cfg['deployment'])
    manifest_path = target / 'e2e_manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'Deployment manifest is missing: {manifest_path}. '
                         'Complete prepare successfully with this configuration before preflight or run.')
    manifest = read_json(manifest_path)
    validate_native_contract(cfg, manifest['experiment'])
    if manifest.get('evaluation_override') != cfg.get('evaluation_override'):
        raise ValueError('Deployment evaluation override differs; export a separate deployment')
    if manifest.get('deployment_model_name_or_path') != cfg.get('model_name_or_path'):
        raise ValueError('Deployment model path override differs; export a separate deployment')
    if (cfg.get('evaluation_override') is not None
            and manifest.get('evaluation_recomputation') != cfg['recomputation']):
        raise ValueError('Deployment evaluation recomputation differs')
    for name, expected in manifest['files'].items():
        if sha_file(target / name) != expected:
            raise ValueError(f'Deployment changed: {name}')
    requested = {'u': cfg['use_u'], 'a': cfg['use_a']}
    if 'enabled_prompts' in manifest and manifest['enabled_prompts'] != requested:
        raise ValueError('Deployment U/A selection differs from configuration; export a new deployment')
    if any(enabled and kind not in manifest['mapping'] for kind, enabled in requested.items()):
        raise ValueError('Deployment is missing an enabled prompt mapping')
    checkpoints = [('source_u', 'source_u_sha256')]
    if cfg['use_a']:
        checkpoints.append(('adapter', 'adapter_sha256'))
    for key, field in checkpoints:
        if sha_file(path(cfg[key])) != manifest[field]:
            raise ValueError(f'{key} checkpoint changed')
    return manifest
