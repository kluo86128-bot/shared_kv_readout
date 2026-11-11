"""Version-specific instrumentation in real vLLM workers, CPU-only LMCache tier.

No source answers are installed in the worker. Provenance is collected at
LMCache's actual layerwise transfer boundary, including inherited dependencies.
"""
from contextvars import ContextVar
from functools import wraps
import importlib.metadata
import os
import pickle
import sys
import types

from .blend_audit import blend_layer_audit

ACTIVE = ContextVar('e2e_request', default=None)
REQUESTS = []
BACKENDS = []
ORIGINS = {}
EVENTS = []
AUDIT = {}
SETTINGS = None
MAPPING_CHECKS = {'u': 0, 'a': 0}
TRACE_DIRECTORY = None


def install_vllm_attention_compat():
    """Expose the pre-0.18 attention import used by LMCache 0.4.2.

    vLLM 0.18 moved ``Attention`` to
    ``vllm.model_executor.layers.attention`` while LMCache's CacheBlend code
    still imports it from ``vllm.attention``.  Install a process-local module
    alias before importing CacheBlend; no installed package is modified.
    """
    if 'vllm.attention' in sys.modules:
        return
    from vllm.model_executor.layers.attention import Attention
    module = types.ModuleType('vllm.attention')
    module.Attention = Attention
    module.__all__ = ['Attention']
    sys.modules['vllm.attention'] = module


def install_lmcache_dense_attention_compat():
    """Avoid importing LMCache's unused sparse FlashInfer implementation.

    LMCache 0.4.2 imports both attention implementations eagerly.  Its sparse
    implementation targets an older FlashInfer API, although this experiment
    disables sparse blending and uses vLLM's FlashAttention backend.  A narrow
    stub keeps the unused module lazy and fails clearly if it is ever selected.
    """
    name = 'lmcache.v1.compute.attention.flash_infer_sparse'
    if name in sys.modules:
        return
    module = types.ModuleType(name)

    class UnsupportedSparseFlashInfer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('Sparse FlashInfer is disabled for this dense CacheBlend experiment')

    module.LMCFlashInferSparseBackend = UnsupportedSparseFlashInfer
    sys.modules[name] = module


def plain_token_ids(token_ids):
    """Convert vLLM's immutable token container to LMCache's wire format."""
    if type(token_ids) is list:
        return token_ids
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()
    return [int(token_id) for token_id in token_ids]


def install_model_embedding_compat(model):
    """Expose the embedding call expected by LMCache on vLLM 0.18 models."""
    if hasattr(model, 'get_input_embeddings'):
        return
    embedding_call = getattr(model, 'embed_input_ids', None)
    if not callable(embedding_call):
        raise RuntimeError('vLLM model exposes no supported input embedding interface')
    model.get_input_embeddings = embedding_call


def request_for(tokens):
    ids = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
    candidates = [r for r in REQUESTS if ids == r['tokens'][:len(ids)] or r['tokens'] == ids[:len(r['tokens'])]]
    if len(candidates) != 1:
        raise RuntimeError('Cannot uniquely associate LMCache transfer with a registered request')
    return candidates[0]


def event(kind, **values):
    EVENTS.append(dict(kind=kind, **values))


def classify_cache_region(start, end, c1_start, c1_end):
    """Classify a native token chunk against the exact C1 request interval."""
    if end <= c1_start:
        return 'before_C1'
    if start >= c1_end:
        return 'after_C1'
    if start == c1_start and end == c1_end:
        return 'C1_exact'
    if start >= c1_start and end <= c1_end:
        return 'C1_subset'
    return 'C1_overlap'


def backend():
    if len(BACKENDS) != 1:
        raise RuntimeError('Expected exactly one native CPU cache backend')
    return BACKENDS[0]


def track_stores(owner):
    cache = backend()
    with cache.cpu_lock:
        keys = [key.to_string() for key in cache.hot_cache]
    inherited = set(AUDIT[owner['id']]['loaded_sources'])
    if owner['role'] == 'writer':
        inherited.add(owner['id'])
    for key in keys:
        if key not in ORIGINS:
            ORIGINS[key] = sorted(inherited)
            event('store', request=owner['id'], key=key, sources=ORIGINS[key])
    resident = set(keys)
    for key in list(ORIGINS):
        if key not in resident:
            event('evict', key=key, sources=ORIGINS.pop(key))


def wrap_transfer(original, operation):
    @wraps(original)
    def execute(engine, tokens, mask=None, **kwargs):
        import torch
        owner = request_for(tokens)
        if SETTINGS and SETTINGS.get('sharing') == 'isolated':
            kwargs['request_configs'] = dict(kwargs.get('request_configs') or {},
                **{'lmcache.tag.tenant': owner['tenant']})
        audit = AUDIT[owner['id']]
        audit[operation + '_calls'] += 1
        token = ACTIVE.set(owner)
        loaded = []
        loaded_blocks = []
        hit_blocks = []
        try:
            if operation == 'retrieve':
                for start, end, key in engine.token_database.process_tokens(tokens=tokens, mask=mask,
                        request_configs=kwargs.get('request_configs')):
                    layer_keys = key.split_layers(engine.num_layers)
                    cache = backend()
                    with cache.cpu_lock:
                        present = [k for k in layer_keys if k in cache.hot_cache]
                    if len(present) != len(layer_keys):
                        break
                    region = classify_cache_region(
                        start, end, owner['public_start'], owner['public_end'])
                    overlap_start = max(start, owner['public_start'])
                    overlap_end = min(end, owner['public_end'])
                    public_tokens = max(0, overlap_end - overlap_start)
                    block = {
                        'key': key.to_string(),
                        'prompt_start': start,
                        'prompt_end': end,
                        'block_tokens': end - start,
                        'public_start': overlap_start if public_tokens else None,
                        'public_end': overlap_end if public_tokens else None,
                        'public_tokens': public_tokens,
                        'layer_blocks': len(present),
                        'region': region,
                    }
                    hit_blocks.append(block)
                    # Only C1-overlapping blocks support privacy attribution.
                    if public_tokens:
                        loaded.extend(k.to_string() for k in present)
                        loaded_blocks.append(block)
            generator = original(engine, tokens, mask, **kwargs)
            # The caller drives layerwise transfer incrementally. Account after
            # each completed step, since it need not request StopIteration.
            while True:
                try:
                    result = next(generator)
                except StopIteration:
                    break
                if operation == 'store':
                    track_stores(owner)
                if operation == 'retrieve' and torch.is_tensor(result) and result.dtype == torch.bool:
                    audit['retrieved_mask'] = result.tolist()
                    audit['loaded_keys'] = sorted(set(audit['loaded_keys']) | set(loaded))
                    known_blocks = {(item['key'], item['prompt_start'], item['prompt_end'])
                                    for item in audit['loaded_blocks']}
                    for item in loaded_blocks:
                        identity = (item['key'], item['prompt_start'], item['prompt_end'])
                        if identity not in known_blocks:
                            audit['loaded_blocks'].append(item)
                            known_blocks.add(identity)
                    known_hits = {(item['key'], item['prompt_start'], item['prompt_end'])
                                  for item in audit['hit_blocks']}
                    for item in hit_blocks:
                        identity = (item['key'], item['prompt_start'], item['prompt_end'])
                        if identity not in known_hits:
                            audit['hit_blocks'].append(item)
                            known_hits.add(identity)
                    for key in loaded:
                        if key not in ORIGINS:
                            audit['unknown_sources'].append(key)
                        audit['loaded_sources'] = sorted(set(audit['loaded_sources']) | set(ORIGINS.get(key, [])))
                    event('retrieve', request=owner['id'], keys=loaded, blocks=hit_blocks,
                          c1_blocks=loaded_blocks,
                          sources=audit['loaded_sources'], mask=result.tolist())
                ACTIVE.reset(token)
                try:
                    yield result
                finally:
                    token = ACTIVE.set(owner)
        finally:
            ACTIVE.reset(token)
    return execute


def install():
    for package, expected in (('vllm', '0.18.0'), ('lmcache', '0.4.2')):
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f'e2e_cache requires {package}=={expected}')
    install_vllm_attention_compat()
    install_lmcache_dense_attention_compat()
    import torch
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupClient
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    from lmcache.v1.compute.blend.blender import LMCBlender
    from lmcache.v1.token_database import SegmentTokenDatabase
    from vllm.v1.worker.gpu_worker import Worker
    if getattr(Worker, '_e2e_installed', False):
        return
    original_lookup = LMCacheLookupClient.lookup

    @wraps(original_lookup)
    def lookup(self, token_ids, lookup_id, request_configs=None):
        # vLLM 0.18 exposes Request.all_token_ids as ConstantList. LMCache
        # 0.4.2 sends this object through its ZMQ encoder, which only supports
        # a built-in list. Preserve values while normalizing the container.
        return original_lookup(self, plain_token_ids(token_ids), lookup_id,
                               request_configs=request_configs)
    LMCacheLookupClient.lookup = lookup
    original_load = Worker.load_model

    @wraps(original_load)
    def load_model(self, *args, **kwargs):
        original_load(self, *args, **kwargs)
        from lmcache.v1.compute.models.utils import VLLMModelTracker
        from lmcache.integration.vllm.utils import ENGINE_NAME
        model = self.model_runner.model
        install_model_embedding_compat(model)
        VLLMModelTracker.register_model(ENGINE_NAME, model)
    Worker.load_model = load_model
    original_init = LocalCPUBackend.__init__

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        BACKENDS.append(self)
    LocalCPUBackend.__init__ = initialize
    LMCacheEngine.store_layer = wrap_transfer(LMCacheEngine.store_layer, 'store')
    LMCacheEngine.retrieve_layer = wrap_transfer(LMCacheEngine.retrieve_layer, 'retrieve')
    original_segments = SegmentTokenDatabase.__init__

    @wraps(original_segments)
    def segments(self, config, metadata):
        original_segments(self, config, metadata)
        # Upstream 0.4.2 drops token zero assuming an automatic BOS. Qwen3
        # adds no BOS here, so use an explicit no-special-token encoding.
        ids = self.tokenizer.encode(config.blend_special_str, add_special_tokens=False)
        if not ids:
            raise RuntimeError('Empty segment delimiter')
        self.sep_tokens = torch.tensor(ids, device='cpu')
        self.sep_len = len(ids)
    SegmentTokenDatabase.__init__ = segments
    original_outer_blend = LMCBlender.blend

    @wraps(original_outer_blend)
    def blend(self, tokens, mask=None, **kwargs):
        owner = request_for(tokens)
        token = ACTIVE.set(owner)
        if SETTINGS and SETTINGS.get('sharing') == 'isolated':
            kwargs['request_configs'] = {'lmcache.tag.tenant': owner['tenant']}
        try:
            return original_outer_blend(self, tokens, mask, **kwargs)
        finally:
            ACTIVE.reset(token)
    LMCBlender.blend = blend
    original_blend = LMCBlender.process_qkv

    @wraps(original_blend)
    def process_qkv(self, q, k, v, residual, layer_id, attn_output, attn_metadata):
        # CacheBlend executes its native implementation. The optional Random
        # variant changes only top-k selection at its documented selection site.
        owner = ACTIVE.get()
        if owner is None:
            raise RuntimeError('Blending executed outside an audited request')
        original_topk = torch.topk
        trace = None
        if TRACE_DIRECTORY and owner['role'] == 'reader':
            old_k, old_v = self.gpu_connector.get_kv(layer_id)
            trace = dict(layer=layer_id, old_key=old_k.detach().cpu().clone(),
                         old_value=old_v.detach().cpu().clone(), request=owner)
            if layer_id in self.common_metadata.check_layers:
                def capture_topk(input, *args, **kwargs):
                    trace['scores'] = input.detach().cpu()
                    return original_topk(input, *args, **kwargs)
                torch.topk = capture_topk
        if SETTINGS['method'] == 'random' and layer_id in self.common_metadata.check_layers:
            def random_topk(input, k, *args, **kwargs):
                if args or kwargs or input.ndim != 1:
                    raise RuntimeError('Unexpected native blending topk signature')
                from .common import digest
                generator = torch.Generator(device=input.device)
                seed = int(digest([SETTINGS['seed'], owner['id'], layer_id])[:15], 16)
                generator.manual_seed(seed)
                indices = torch.randperm(len(input), device=input.device, generator=generator)[:k]
                return torch.return_types.topk((input[indices], indices))
            torch.topk = random_topk
        try:
            result = original_blend(self, q, k, v, residual, layer_id, attn_output, attn_metadata)
        finally:
            torch.topk = original_topk
        indices = self.metadata.imp_indices
        if trace is not None:
            from pathlib import Path
            from .common import digest
            stored_k, stored_v = self.gpu_connector.get_kv(layer_id)
            trace.update(query=result[0].detach().cpu(), attention_key=result[1].detach().cpu(),
                         attention_value=result[2].detach().cpu(),
                         stored_key=stored_k.detach().cpu().clone(), stored_value=stored_v.detach().cpu().clone(),
                         selected_indices=None if indices is None else indices.tolist())
            torch.save(trace, Path(TRACE_DIRECTORY) / f'{digest(owner["id"])[:16]}_layer_{layer_id:03d}.pt')
        AUDIT[owner['id']]['recomputation'].append(
            blend_layer_audit(result, indices, layer_id, SETTINGS))
        return result
    LMCBlender.process_qkv = process_qkv
    Worker._e2e_installed = True


class Extension:
    """Loaded by vLLM before worker initialization; one GPU per experiment."""

    def e2e_setup(self, manifest, settings, vectors_path):
        import torch
        global SETTINGS
        SETTINGS = settings
        model = self.model_runner.model
        embedding = model.model.embed_tokens
        expected = torch.load(vectors_path, map_location='cpu', weights_only=True)
        for kind, spec in manifest['mapping'].items():
            ids = torch.tensor(spec['ids'], device=embedding.weight.device)
            actual = embedding(ids).detach().float().cpu()
            if not torch.equal(actual, expected[kind].float()):
                raise RuntimeError(f'Worker embedding differs from {kind} checkpoint')

        def verify_mapping(module, inputs, output):
            ids = inputs[0]
            for kind, spec in manifest['mapping'].items():
                for index, token_id in enumerate(spec['ids']):
                    selected = ids == token_id
                    if selected.any():
                        target = expected[kind][index].to(output.device, output.dtype)
                        if not torch.equal(output[selected], target.expand_as(output[selected])):
                            raise RuntimeError('Actual request embedding verification failed')
                        MAPPING_CHECKS[kind] += int(selected.sum())
        embedding.register_forward_hook(verify_mapping)
        original_logits = model.compute_logits
        def compute_logits(*args, **kwargs):
            logits = original_logits(*args, **kwargs)
            if TRACE_DIRECTORY and len(REQUESTS)==1 and REQUESTS[0]['role']=='reader' and logits is not None:
                from pathlib import Path
                target = Path(TRACE_DIRECTORY) / 'first_logits.pt'
                if not target.exists():
                    torch.save(logits.detach().cpu(), target)
            return logits
        model.compute_logits = compute_logits
        backend()
        return {'ready': True, 'pid': os.getpid()}

    def e2e_trace(self, directory):
        from pathlib import Path
        global TRACE_DIRECTORY
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=False)
        TRACE_DIRECTORY = str(target)
        return {'trace_directory': TRACE_DIRECTORY}

    def e2e_register(self, requests):
        global REQUESTS
        REQUESTS = requests
        MAPPING_CHECKS.update(u=0, a=0)
        for r in requests:
            AUDIT[r['id']] = dict(loaded_keys=[], loaded_blocks=[], hit_blocks=[],
                                  loaded_sources=[], unknown_sources=[],
                                  recomputation=[], complete=True, store_calls=0, retrieve_calls=0)

    def e2e_collect(self):
        import torch
        torch.cuda.synchronize()
        result = {'requests': {r['id']: AUDIT[r['id']] for r in REQUESTS},
                  'events': list(EVENTS), 'mapping_checks': dict(MAPPING_CHECKS)}
        EVENTS.clear()
        return result

    def e2e_snapshot(self, filename):
        import torch
        torch.cuda.synchronize()
        cache = backend()
        with cache.cpu_lock:
            if cache.keys_in_request:
                raise RuntimeError('Cannot snapshot a cache with outstanding pinned lookups')
            entries = [(key, obj.tensor.clone(), obj.get_memory_format())
                       for key, obj in cache.hot_cache.items()]
            state = dict(entries=entries, policy=pickle.dumps(cache.cache_policy),
                         origins=dict(ORIGINS), audit=dict(AUDIT))
        torch.save(state, filename)
        return {'entries': len(entries)}

    def e2e_restore(self, filename):
        import torch
        # The coordinator verifies the SHA-256 of this locally generated file.
        state = torch.load(filename, map_location='cpu', weights_only=False)
        cache = backend()
        if cache.hot_cache:
            raise RuntimeError('Restore requires an empty fresh engine')
        for key, tensor, fmt in state['entries']:
            obj = cache.allocate(tensor.shape, tensor.dtype, fmt=fmt, eviction=False)
            if obj is None:
                raise RuntimeError('Snapshot does not fit configured cache capacity')
            obj.tensor.copy_(tensor)
            cache.submit_put_task(key, obj)
            obj.ref_count_down()
        cache.cache_policy = pickle.loads(state['policy'])
        # submit_put_task insertion order restores LRU/FIFO order; LFU bucket
        # frequencies and tie order are restored with the serialized policy.
        ORIGINS.clear()
        ORIGINS.update(state['origins'])
        AUDIT.clear()
        AUDIT.update(state['audit'])
        return {'entries': len(cache.hot_cache)}

    def e2e_clear(self):
        cache = backend()
        for key in list(cache.hot_cache):
            cache.remove(key)
        ORIGINS.clear()
        return True


# Importing the extension precedes Worker.load_model in vLLM's worker wrapper.
install()
