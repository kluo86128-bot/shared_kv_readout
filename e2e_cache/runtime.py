"""Real vLLM batching and LMCache configuration; no offline KV injection."""
import os
import time
from copy import copy

from .common import path


def validate_writer_transfer(request, audit):
    """Require a writer to store new KV or reuse attributable native KV.

    LMCache legitimately skips its store hook when a repeated writer prompt is
    already cached.  Such a request is not a new cache source; provenance must
    remain attached to the earlier request whose blocks were retrieved.
    """
    if request['role'] != 'writer' or audit['store_calls']:
        return
    if audit['retrieve_calls'] and audit['loaded_keys']:
        if audit['unknown_sources']:
            raise RuntimeError('Writer reused native cache with unknown provenance')
        return
    raise RuntimeError('No native store or attributable cache retrieval observed for source request')


class Runtime:
    def __init__(self, cfg, manifest):
        self.cfg, self.manifest = cfg, manifest
        cache, recompute = cfg['cache'], cfg['recomputation']
        # Reject ambient LMCache configuration that could enable an unaudited
        # disk, remote, async or second-tier backend.
        for name in list(os.environ):
            if name.startswith('LMCACHE_'):
                del os.environ[name]
        env = {'LMCACHE_LOCAL_CPU': 'True', 'LMCACHE_MAX_LOCAL_CPU_SIZE': str(cache['cpu_gb']),
            'LMCACHE_CACHE_POLICY': cache['eviction'], 'LMCACHE_ENABLE_BLENDING': 'True',
            # LMCache strips unquoted whitespace before parsing environment
            # values. A Python string literal preserves a newline delimiter.
            'LMCACHE_USE_LAYERWISE': 'True', 'LMCACHE_BLEND_SPECIAL_STR': repr(cache['delimiter']),
            'LMCACHE_BLEND_CHECK_LAYERS': str(recompute['check_layer']),
            'LMCACHE_BLEND_RECOMPUTE_RATIOS': str(recompute['ratio']),
            'LMCACHE_CHUNK_SIZE': str(cache['chunk_size']), 'LMCACHE_ENABLE_ASYNC_LOADING': 'False',
            'LMCACHE_PRE_CACHING_HASH_ALGORITHM': 'sha256_cbor',
            'LMCACHE_SAVE_UNFULL_CHUNK': 'True', 'VLLM_USE_V1': '1',
            'LMCACHE_LOG_LEVEL': 'WARNING', 'VLLM_LOGGING_LEVEL': 'WARNING',
            'CUDA_VISIBLE_DEVICES': str(cfg['engine']['device'])}
        os.environ.update(env)
        from lmcache.v1.config import LMCacheEngineConfig
        native_config = LMCacheEngineConfig.from_env()
        if native_config.blend_special_str != cache['delimiter']:
            raise RuntimeError('LMCache changed the configured segment delimiter during environment parsing')
        from vllm import LLM, SamplingParams
        from vllm.config import KVTransferConfig
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path(cfg['deployment']), local_files_only=True)
        self.llm = LLM(model=str(path(cfg['deployment'])), tokenizer=str(path(cfg['deployment'])),
            tensor_parallel_size=1, enforce_eager=True, enable_prefix_caching=False,
            attention_backend='FLASH_ATTN',
            enable_chunked_prefill=False, max_model_len=cfg['engine']['max_model_len'],
            max_num_seqs=cfg['engine']['max_num_seqs'],
            max_num_batched_tokens=cfg['engine']['max_model_len'],
            gpu_memory_utilization=cfg['engine']['gpu_memory_utilization'],
            worker_extension_cls='e2e_cache.worker.Extension',
            kv_transfer_config=KVTransferConfig(kv_connector='LMCacheConnectorV1', kv_role='kv_both'))
        # Dedicated U/A IDs are input-only.  Preserve the historical evaluator's
        # raw decoding while excluding those newly-added vocabulary entries from
        # generation exactly (logit bias remains defense in depth).
        blocked = {i: -100.0 for spec in manifest['mapping'].values() for i in spec['ids']}
        blocked_words = [token for spec in manifest['mapping'].values()
                         for token in spec['tokens']]
        self.blocked_output_ids = set(blocked)
        self.params = SamplingParams(temperature=0, max_tokens=cfg['engine']['max_new_tokens'],
                                     seed=cfg['seed'], logit_bias=blocked,
                                     bad_words=blocked_words, skip_special_tokens=False)
        ready = self.rpc('e2e_setup', manifest,
            dict(recompute, seed=cfg['seed'], sharing=cache['sharing']), str(path(cfg['deployment']) / 'ua_vectors.pt'))
        if not ready['ready']:
            raise RuntimeError('Worker setup failed')

    def rpc(self, method, *args):
        results = self.llm.collective_rpc(method, args=args)
        if len(results) != 1:
            raise RuntimeError('Each experiment must have exactly one engine worker')
        return results[0]

    def ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def request(self, event, record, material):
        cfg = self.manifest['experiment']
        mapping = self.manifest['mapping']
        separator = cfg['delimiter']
        delimiter = self.cfg['cache']['delimiter']
        if delimiter != separator:
            raise ValueError('Delimiter must match the trained separator to preserve source layout')
        if material['text'] != cfg['public_text']:
            raise ValueError('Runtime material differs from the trained C1 public text')
        public = self.ids(material['text'])
        u = mapping['u']['ids'] if self.cfg['use_u'] and record is not None else []
        a = mapping['a']['ids'] if self.cfg['use_a'] and record is None else []
        if record is not None:
            left = self.ids(record['private_prefix'] + separator)
            keep = cfg['prompt_partition']['keep_length'] if u else 0
            before = u[:-keep] if keep else u
            after = u[-keep:] if keep else []
            tokens = left + before + self.ids(separator)
            before_text = ''.join(mapping['u']['tokens'][:len(before)]) if u else ''
            text = record['private_prefix'] + separator + before_text + separator
            start = len(tokens)
            tokens += public
            text += material['text']
            end = len(tokens)
            from kv_cache_readout.suffix import pair_for
            suffix = pair_for(record, cfg)
            tokens += self.ids(separator) + after + suffix['ids']
            after_text = ''.join(mapping['u']['tokens'][-keep:]) if keep else ''
            text += separator + after_text + suffix['text']
            role = 'writer'
            question = 'Summarize the applicable public guidance briefly.'
        else:
            # Terminate C with the same delimiter as the source. A belongs to
            # the following segment, allowing native non-prefix C lookup.
            start, end = 0, len(public)
            tokens = public + self.ids(separator) + a
            text = material['text'] + separator + (mapping['a']['text'] if a else '')
            role, question = 'reader', event['question']
        if cfg['question_format'] == 'chat':
            qtext = self.tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            qtext = f'Question: {question}\nAnswer:\n'
        tokens += self.ids(qtext)
        text += qtext
        if len(public) != cfg['native_contract']['c1_tokens']:
            raise ValueError('Material must match the trained C1 length')
        if len(tokens) + self.cfg['engine']['max_new_tokens'] > self.cfg['engine']['max_model_len']:
            raise ValueError(f'Request exceeds model budget: {event["id"]}')
        for kind, ids in (('u', u), ('a', a if record is None else [])):
            if ids and self.ids(mapping[kind]['text']) != ids:
                raise ValueError('Actual request token mapping changed')
        return dict(id=event['id'], role=role, tenant=event['tenant'], text=text, tokens=tokens,
                    public_start=start, public_end=end)

    def generate(self, requests):
        # The pinned blender omits native request IDs in its transfer call.
        # Identical prompts are executed separately to keep provenance unique.
        if len({tuple(r['tokens']) for r in requests}) != len(requests):
            batches = [self.generate([request]) for request in requests]
            return dict(rows=[row for batch in batches for row in batch['rows']],
                events=[event for batch in batches for event in batch['events']],
                mapping_checks={k: sum(batch['mapping_checks'][k] for batch in batches) for k in ('u', 'a')},
                seconds=sum(batch['seconds'] for batch in batches))
        self.rpc('e2e_register', requests)
        params = []
        for request in requests:
            parameter = copy(self.params)
            if self.cfg['cache']['sharing'] == 'isolated':
                parameter.extra_args = {'kv_transfer_params': {'lmcache.tag.tenant': request['tenant']}}
            params.append(parameter)
        started = time.monotonic()
        submitted_at = time.time()
        # Submit the exact segment-wise token layout. Retokenizing the readable
        # audit text can merge ordinary tokens across P/U/C/separator boundaries
        # and would no longer represent the layout used during U training.
        prompts = [{'prompt_token_ids': list(r['tokens'])} for r in requests]
        outputs = self.llm.generate(prompts,
                                    sampling_params=params, use_tqdm=False)
        elapsed = time.monotonic() - started
        completed_at = time.time()
        audit = self.rpc('e2e_collect')
        if len(outputs) != len(requests):
            raise RuntimeError('vLLM returned an incomplete batch')
        checks = audit['mapping_checks']
        needed_u = self.cfg['use_u'] and any(r['role'] == 'writer' for r in requests)
        needed_a = self.cfg['use_a'] and any(r['role'] == 'reader' for r in requests)
        if (needed_u and not checks['u']) or (needed_a and not checks['a']):
            raise RuntimeError('Request never passed the U/A embedding verification hook')
        rows = []
        for request, output in zip(requests, outputs):
            if output.prompt_token_ids != request['tokens'] or not output.finished:
                raise RuntimeError('Output/request association mismatch')
            validate_writer_transfer(request, audit['requests'][request['id']])
            completion = output.outputs[0]
            token_ids = [int(token_id) for token_id in completion.token_ids]
            forbidden = sorted(set(token_ids) & self.blocked_output_ids)
            if forbidden:
                raise RuntimeError(f'Dedicated input token generated despite hard exclusion: {forbidden}')
            token_pieces = self.tokenizer.convert_ids_to_tokens(token_ids)
            raw_text = self.tokenizer.decode(
                token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            rows.append(dict(request_id=request['id'], text=completion.text,
                raw_text=raw_text, generated_token_ids=token_ids,
                generated_token_pieces=token_pieces,
                submitted_at=submitted_at, batch_completed_at=completed_at,
                finish_reason=completion.finish_reason,
                generated_tokens=len(token_ids),
                metrics=None if output.metrics is None else {
                    k: getattr(output.metrics, k, None) for k in
                    ('arrival_time', 'first_scheduled_time', 'first_token_time', 'finished_time')},
                audit=audit['requests'][request['id']]))
        return {'rows': rows, 'seconds': elapsed, 'mapping_checks': checks, 'events': audit['events']}

    def clear(self):
        self.rpc('e2e_clear')

    def snapshot(self, filename):
        return self.rpc('e2e_snapshot', str(filename))

    def restore(self, filename):
        return self.rpc('e2e_restore', str(filename))
