"""Worker extension with audited authoritative source-position relocation."""
import json
from pathlib import Path
import pickle

import torch

from e2e_cache import worker as base
from lmcache.v1.compute.positional_encoding import FusedRope
from lmcache.v1.gpu_connector.gpu_connectors import VLLMBufferLayerwiseGPUConnector

from . import POLICY, identity
from .positions import install, source_positions, POSITION_ATTR


def audit_positions(layer, seen, used, destination, corrected):
    owner = base.ACTIVE.get()
    if owner is None:
        raise RuntimeError('RoPE relocation has no audited native request')
    row = dict(layer=layer, corrected=corrected, policy=POLICY,
               observed_source_first=int(seen[0]), observed_source_last=int(seen[-1]),
               used_source_first=int(used[0]), used_source_last=int(used[-1]),
               destination_first=int(destination[0]), destination_last=int(destination[-1]))
    base.AUDIT[owner['id']].setdefault('rope_positions', []).append(row)
    if layer == 0:
        print('[ROPE_FIX] ' + ' | '.join(f'{key}={value}' for key, value in row.items()), flush=True)
    if base.TRACE_DIRECTORY and owner['role'] == 'reader':
        torch.save(dict(**row, observed_positions=seen.detach().cpu(),
                        used_positions=used.detach().cpu(), destination_positions=destination.detach().cpu()),
                   Path(base.TRACE_DIRECTORY)/f'rope_positions_{layer:03d}.pt')


install(VLLMBufferLayerwiseGPUConnector, FusedRope, audit_positions)


class Extension(base.Extension):
    def e2e_trace(self, directory):
        result = base.Extension.e2e_trace(self, directory)
        if not hasattr(self,'_readout_question_trace'):
            from .question_trace import QuestionTrace
            self._readout_question_trace=QuestionTrace(self.model_runner.model,
                lambda:base.TRACE_DIRECTORY,
                lambda:len(base.REQUESTS)==1 and base.REQUESTS[0]['role']=='reader',
                lambda:base.ACTIVE.get() is not None)
        (Path(directory)/'runtime_fix.json').write_text(json.dumps(identity(), indent=2)+'\n')
        return result

    def e2e_snapshot(self, filename):
        torch.cuda.synchronize()
        cache = base.backend()
        with cache.cpu_lock:
            if cache.keys_in_request:
                raise RuntimeError('Cannot snapshot a cache with outstanding pinned lookups')
            entries = [(key, obj.tensor.clone(), obj.get_memory_format()) for key, obj in cache.hot_cache.items()]
            positions = {key.to_string(): source_positions(obj).clone() for key, obj in cache.hot_cache.items()}
            state = dict(entries=entries, source_positions=positions, rope_policy=POLICY,
                         policy=pickle.dumps(cache.cache_policy), origins=dict(base.ORIGINS), audit=dict(base.AUDIT))
        torch.save(state, filename)
        return {'entries': len(entries), 'rope_policy': POLICY}

    def e2e_restore(self, filename):
        state = torch.load(filename, map_location='cpu', weights_only=False)
        if state.get('rope_policy') != POLICY:
            raise RuntimeError('Snapshot predates verified RoPE source positions; start a fresh fixed run')
        cache = base.backend()
        if cache.hot_cache:
            raise RuntimeError('Restore requires an empty fresh engine')
        for key, tensor, fmt in state['entries']:
            positions = state['source_positions'][key.to_string()]
            obj = cache.allocate(tensor.shape, tensor.dtype, fmt=fmt, eviction=False)
            if obj is None:
                raise RuntimeError('Snapshot does not fit configured cache capacity')
            obj.tensor.copy_(tensor)
            setattr(obj, POSITION_ATTR, positions.clone())
            source_positions(obj)
            obj.metadata.cached_positions = positions
            cache.submit_put_task(key, obj)
            obj.ref_count_down()
        cache.cache_policy = pickle.loads(state['policy'])
        base.ORIGINS.clear(); base.ORIGINS.update(state['origins'])
        base.AUDIT.clear(); base.AUDIT.update(state['audit'])
        return {'entries': len(cache.hot_cache), 'rope_policy': POLICY}
