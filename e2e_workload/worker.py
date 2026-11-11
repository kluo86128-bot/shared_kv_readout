"""Native worker plus admission timestamps, direct ownership and hard max-age."""
import os
import time
from functools import wraps

from e2e_cache import worker as base
from .lifecycle import Lifecycle

if os.environ.get('READOUT_WORKLOAD_ROPE_FIX') == '1':
    from runtime_fixes.worker import Extension as Parent
else:
    Parent = base.Extension

STATE = Lifecycle()
ADMISSION_SEQUENCE = 0


def install_admission_clock(backend_class):
    """Stamp successful native insertions, including remove/reinsert of one key."""
    if getattr(backend_class, '_workload_admission_clock', False):
        return
    original = backend_class.submit_put_task
    @wraps(original)
    def put(cache, key, memory_obj, *args, **kwargs):
        global ADMISSION_SEQUENCE
        with cache.cpu_lock:
            previous = cache.hot_cache.get(key)
        result = original(cache, key, memory_obj, *args, **kwargs)
        with cache.cpu_lock:
            admitted = cache.hot_cache.get(key)
            if admitted is not None and admitted is not previous:
                ADMISSION_SEQUENCE += 1
                admitted._workload_admission_sequence = ADMISSION_SEQUENCE
                admitted._workload_admitted_at = time.time()
        return result
    backend_class.submit_put_task = put
    backend_class._workload_admission_clock = True


from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
install_admission_clock(LocalCPUBackend)


def resident_objects():
    cache = base.backend()
    with cache.cpu_lock:
        objects = {k.to_string(): (k, obj) for k, obj in cache.hot_cache.items()}
    resident = {}
    for name, (key, obj) in objects.items():
        if not hasattr(obj, '_workload_admission_sequence'):
            raise RuntimeError('Cache object has no native admission timestamp')
        resident[name] = dict(object_token=obj._workload_admission_sequence,
            admitted_at=obj._workload_admitted_at, group=key.split_layers(1)[0].to_string(),
            bytes=obj.tensor.numel()*obj.tensor.element_size(), can_evict=bool(obj.can_evict))
    return objects, resident


def observe(owner):
    _, resident = resident_objects()
    inherited = base.AUDIT.get(owner['id'], {}).get('loaded_sources', []) if owner else []
    STATE.observe(resident, owner, inherited, time.time())
    base.ORIGINS.clear()
    base.ORIGINS.update({k: e['sources'] for k, e in STATE.entries.items()})


def track_stores(owner):
    observe(owner)


# Keep the original numerical/transfer implementation; replace only provenance
# bookkeeping so a clean Reader or background admission has an explicit owner.
base.track_stores = track_stores


class Extension(Parent):
    def e2e_setup(self, manifest, settings, vectors_path):
        global STATE
        STATE = Lifecycle(settings['max_age_seconds'])
        result = super().e2e_setup(manifest, settings, vectors_path)
        result['lifecycle_schema'] = 1
        return result

    def e2e_expire(self):
        import torch
        torch.cuda.synchronize()
        observe(None)
        objects, resident = resident_objects()
        STATE.expire(resident, lambda name: base.backend().remove(objects[name][0]), time.time())
        base.ORIGINS.clear()
        base.ORIGINS.update({k: e['sources'] for k, e in STATE.entries.items()})

    def e2e_register(self, requests):
        super().e2e_register(requests)
        now = time.time()
        # Retain pre-request metadata even if later allocations evict these blocks.
        self._workload_before = STATE.loaded_metadata(list(STATE.entries), now)

    def e2e_collect(self):
        import torch
        torch.cuda.synchronize()
        owner = base.REQUESTS[0]
        observe(owner)
        result = super().e2e_collect()
        before = {e['key']: e for e in self._workload_before}
        for row in result['requests'].values():
            row['cache_entries'] = [before[k] for k in row['loaded_keys'] if k in before]
            if len(row['cache_entries']) != len(set(row['loaded_keys'])):
                row['complete'] = False
            row['cache_age_clock'] = 'wall_seconds_since_admission'
        result['events'].extend(STATE.events)
        STATE.events.clear()
        result['cache_inventory'] = self.e2e_inventory()
        return result

    def e2e_inventory(self):
        return dict(entries=len(STATE.entries),
                    tensor_bytes=sum(e['bytes'] for e in STATE.entries.values()),
                    age_clock='wall_seconds_since_admission')

    def e2e_snapshot(self, filename):
        import torch
        result = super().e2e_snapshot(filename)
        saved = torch.load(filename, map_location='cpu', weights_only=False)
        saved['workload_lifecycle'] = STATE.state_dict()
        torch.save(saved, filename)
        return dict(result, lifecycle_schema=1)

    def e2e_restore(self, filename):
        import torch
        saved = torch.load(filename, map_location='cpu', weights_only=False)
        if 'workload_lifecycle' not in saved:
            raise RuntimeError('Snapshot predates lifecycle workload; start a new run')
        result = super().e2e_restore(filename)
        _, resident = resident_objects()
        STATE.restore(saved['workload_lifecycle'], resident)
        return result
