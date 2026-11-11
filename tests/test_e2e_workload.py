import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from e2e_workload import common
from e2e_workload.lifecycle import Lifecycle


FIELDS = ['name_full_direct', 'identity_number', 'address', 'email']


def records(n=20):
    return [dict(record_id=f'p{i}', private_prefix='private',
                 qa=[dict(field=f, question=f'What is the {f}?', answer=f'{f}-{i}') for f in FIELDS]) for i in range(n)]


def config():
    return common.load_settings('configs/e2e_workload_validation.json')


def write_config(tmp_path, cfg):
    p = tmp_path/'config.json'
    p.write_text(json.dumps(cfg), encoding='utf-8')
    return p


def test_full_schedule_balanced_random_unique_and_reproducible():
    cfg = config()
    samples = records()
    original = copy.deepcopy(samples)
    plan = common.schedule(cfg, samples)
    assert plan == common.schedule(cfg, samples)
    assert samples == original
    assert Counter(s['gap_bucket'] for s in plan) == {0: 25, 1: 25, 2: 25, 3: 25}
    assert len({s['background_count'] for s in plan}) > 10
    assert len({tuple(w['id'] for w in s['writers']) for s in plan}) == 100
    assert any([w['id'] for w in s['writers']] != sorted(w['id'] for w in s['writers']) for s in plan)
    assert len({tuple(r['field'] for r in s['readers']) for s in plan}) > 5
    for s in plan:
        assert 5 <= len(s['writers']) <= 15
        assert len({w['record_id'] for w in s['writers']}) == len(s['writers'])
        assert s['gap_bucket']*4 <= len(s['backgrounds']) <= s['gap_bucket']*4+3
        assert len({b['document_index'] for b in s['backgrounds']}) == len(s['backgrounds'])
        assert {r['field'] for r in s['readers']} == set(FIELDS)
        assert all('answer' not in r and 'gold' not in r for r in s['readers'])
        ordered = common.ordered_events(s)
        assert ordered[:len(s['writers'])] == s['writers']
        tail = ordered[len(s['writers']):]
        assert tail[0]['role'] == 'reader'
        assert all(e['role'] != 'writer' for e in tail)
        r_positions = [i for i, e in enumerate(tail) if e['role'] == 'reader']
        actual_gaps = [b-a-1 for a, b in zip(r_positions, r_positions[1:])]
        assert actual_gaps == s['reader_gap_background_counts']
        assert all(n in (0, 1) for n in actual_gaps)
        assert len(tail[r_positions[-1]+1:]) == s['cooldown_background_count']
        assert all(e.get('phase') == 'cooldown' for e in tail[r_positions[-1]+1:])
        assert sum(actual_gaps) + s['cooldown_background_count'] == s['background_count']
    cfg['seed'] += 1
    assert common.schedule(cfg, samples) != plan


@pytest.mark.parametrize('mutation', ['overlap', 'rounds', 'pool', 'batch', 'fields', 'ttl', 'renew', 'unknown',
                                   'timing', 'reader_gap', 'reader_gap_budget', 'reader_gap_type'])
def test_invalid_config_rejected(tmp_path, mutation):
    cfg = config()
    if mutation == 'overlap': cfg['reader_schedule']['gaps'][1]['min'] = 3
    elif mutation == 'rounds': cfg['rounds'] = 99
    elif mutation == 'pool': cfg['background']['document_pool_size'] = 10
    elif mutation == 'batch': cfg['writer_batch_size'] = 2
    elif mutation == 'fields': cfg['reader_schedule']['fields'] = ['email', 'email']
    elif mutation == 'ttl': cfg['cache']['max_age_seconds'] = -1
    elif mutation == 'renew': cfg['cache']['renew_age_on_hit'] = True
    elif mutation == 'timing': cfg['reader_schedule']['timing'] = 'invalid'
    elif mutation == 'reader_gap': cfg['reader_schedule']['background_between_readers']['max'] = 2
    elif mutation == 'reader_gap_budget': cfg['reader_schedule']['background_between_readers']['min'] = 1
    elif mutation == 'reader_gap_type': cfg['reader_schedule']['background_between_readers'] = 'invalid'
    else: cfg['unrecognized'] = True
    with pytest.raises(ValueError): common.load_settings(write_config(tmp_path, cfg))


def test_missing_common_field_and_insufficient_writer_population_rejected():
    samples = records()
    samples[0]['qa'][0]['answer'] = ''
    with pytest.raises(ValueError, match='fields missing'): common.schedule(config(), samples)
    with pytest.raises(ValueError, match='distinct writer'): common.schedule(config(), records(3))


def obj(token=1, group='g', can_evict=True):
    return dict(object_token=token, group=group, bytes=100, can_evict=can_evict)


def test_fixed_age_hit_does_not_renew_and_layers_expire_together():
    state = Lifecycle(120)
    owner = dict(id='w0', role='writer')
    state.observe({'k0': obj()}, owner, [], 1000)
    resident = {'k0': obj(), 'k1': obj(2)}
    state.observe(resident, owner, [], 1001)
    state.observe(resident, dict(id='r0', role='reader'), ['w0'], 1100)
    assert state.entries['k1']['created_at'] == 1000
    assert state.expired_groups(1119) == set()
    removed = []
    state.expire(resident, lambda k: removed.append(k) or True, 1120)
    assert removed == ['k0', 'k1'] and not state.entries
    assert all(e['reason'] == 'max_age' for e in state.events if e['kind'] == 'evict')


def test_pinned_expiry_is_atomic_and_capacity_readmission_has_new_owner():
    state = Lifecycle(120)
    state.observe({'k': obj()}, dict(id='w', role='writer'), [], 0)
    with pytest.raises(RuntimeError, match='still in use'):
        state.expire({'k': obj(can_evict=False)}, lambda k: pytest.fail('partial expiration'), 120)
    state.observe({}, None, [], 10)
    state.observe({'k': obj(2)}, dict(id='r', role='reader'), [], 20)
    assert state.entries['k']['sources'] == ['r']
    assert state.entries['k']['direct_role'] == 'reader'
    assert state.entries['k']['created_at'] == 20
    assert any(e.get('reason') == 'capacity_or_native_removal' for e in state.events)


def test_snapshot_keeps_real_age_including_downtime_and_inherited_sources():
    state = Lifecycle(120)
    state.observe({'k': obj()}, dict(id='r', role='reader'), ['w'], 1000)
    snapshot = state.state_dict()
    restored = Lifecycle(120)
    restored.restore(snapshot, {'k': obj(99)})
    assert restored.loaded_metadata(['k'], 1100)[0]['age_seconds'] == 100
    assert restored.entries['k']['sources'] == ['r', 'w']
    assert restored.expired_groups(1120) == {'g'}
    with pytest.raises(RuntimeError, match='mismatch'):
        Lifecycle(None).restore(snapshot, {'k': obj(99)})


def test_background_request_has_no_private_prompt_u_or_reader_a():
    from e2e_workload.runtime import Runtime
    runtime = Runtime.__new__(Runtime)
    runtime.cfg = config()
    runtime.manifest = dict(experiment=dict(question_format='plain', delimiter='\n\n'))
    runtime.ids = lambda text: [ord(c) for c in text]
    request = runtime.request(dict(id='b', role='background', tenant='background'), None,
                              dict(text='operational document', tokens=[10, 20, 30]))
    assert request['role'] == 'background'
    assert request['public_start'] == request['public_end'] == 0
    assert request['tokens'][:3] == [10, 20, 30]
    assert 'private' not in request['text']


def test_background_document_freezing_and_distinctness(tmp_path):
    cfg = config()
    cfg['background'].update(document_path=str(tmp_path/'docs.jsonl'), document_pool_size=2, document_tokens=3)
    tokenizer = SimpleNamespace(encode=lambda s, **kw: list(s.encode()),
                                decode=lambda ids, **kw: bytes(ids).decode())
    Path(cfg['background']['document_path']).write_text('\n'.join(json.dumps(dict(id=str(i), text=t)) for i, t in enumerate(['abcdef', 'xyzabc'])))
    docs = common.prepare_documents(cfg, tokenizer)
    assert docs == common.prepare_documents(cfg, tokenizer)
    assert [d['text'] for d in docs] == ['abc', 'xyz']


def test_actual_dataset_has_all_four_nonempty_fields():
    cfg = config()
    raw = common.path('data/dataset_multiscene_v3_hr_balanced/test.jsonl')
    samples = [json.loads(line) for line in raw.open(encoding='utf-8')]
    samples = [dict(r, qa=r['evaluation_qa']) for r in samples if r['scene'] == 'hr']
    assert len(samples) == 2000
    assert sum(len(s['readers']) for s in common.schedule(cfg, samples)) == 400


@pytest.fixture
def simulated_run(tmp_path, monkeypatch):
    from e2e_workload import __main__ as runner
    from e2e_workload import runtime as native_runtime
    cfg = config()
    cfg.update(rounds=4, output=str(tmp_path/'run'), writers_min=2, writers_max=3)
    cfg['reader_schedule']['gaps'] = [dict(min=i, max=i, rounds=1) for i in range(4)]
    samples = records(4)
    documents = [dict(id=str(i), tokens=[i], text='document') for i in range(16)]
    plan = common.schedule(cfg, samples)
    monkeypatch.setattr(runner.deployment, 'verify', lambda c: {'test_manifest': True})
    monkeypatch.setattr(runner, 'inputs', lambda *a: (samples, documents, plan))
    monkeypatch.setattr(runner, 'code_identity', lambda: {'core': 'unchanged'})
    calls = []
    control = {'pause_round': None, 'fail_on': None, 'restore_calls': 0}

    class FakeRuntime:
        def __init__(self, c, manifest):
            self.source = None
            self.tick = 1000
        def request(self, event, record, material):
            return dict(event, public_start=0, public_end=56)
        def generate(self, requests):
            assert len(requests) == 1
            event = requests[0]
            calls.append({k: v for k, v in event.items() if k not in ('public_start', 'public_end')})
            if event['id'] == control['fail_on']:
                raise RuntimeError('simulated request failure')
            role = event['role']
            if role == 'background': self.source = None
            if role == 'writer' and self.source is None: self.source = event['id']
            audit = dict(loaded_sources=[self.source] if self.source else [],
                         loaded_keys=['C1'] if self.source else [], complete=True,
                         unknown_sources=[], cache_entries=[])
            answer = ''
            if role == 'reader' and self.source:
                source = next(w for s in plan for w in s['writers'] if w['id'] == self.source)
                sample = next(r for r in samples if r['record_id'] == source['record_id'])
                answer = next(q['answer'] for q in sample['qa'] if q['field'] == event['field'])
            if role == 'reader' and control['pause_round'] == event['round']:
                root = Path(cfg['output'])
                (root/'pause.request').write_text('{}')
            self.tick += 1
            return dict(rows=[dict(request_id=event['id'], text=answer, audit=audit,
                        submitted_at=self.tick, batch_completed_at=self.tick+.25)],
                        events=[], seconds=.25, mapping_checks={'u': 1, 'a': 0})
        def inventory(self): return dict(entries=int(self.source is not None))
        def snapshot(self, p):
            Path(p).write_text(json.dumps(dict(source=self.source, tick=self.tick)))
        def restore(self, p):
            control['restore_calls'] += 1
            state = json.loads(Path(p).read_text())
            self.source, self.tick = state['source'], state['tick']
    monkeypatch.setattr(native_runtime, 'Runtime', FakeRuntime)
    return runner, cfg, plan, calls, control


def test_serial_execution_has_four_independent_readers_and_400_style_accounting(simulated_run, capsys):
    import csv
    runner, cfg, plan, calls, _ = simulated_run
    result = runner.execute(cfg)
    expected = [e for s in plan for e in common.ordered_events(s)]
    assert calls == expected
    assert result['completed_rounds'] == 4 and result['completed_queries'] == 16
    assert all(v['queries'] == 4 for v in result['by_field'].values())
    root = Path(cfg['output'])
    with (root/'predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        table = csv.DictReader(stream)
        assert table.fieldnames[:6] == ['record_id', 'round', 'reader_index', 'field', 'answer', 'gold']
        csv_rows = list(table)
        assert len(csv_rows) == 16
        assert 'full_c1_hit' in table.fieldnames and 'direct_sources' in table.fieldnames
    for s in plan:
        rows = common.read_json(root/f"round_{s['index']:06d}"/'result.json')['readers']
        ordered = common.ordered_events(s)
        positions = {e['id']: i for i, e in enumerate(ordered)}
        last_writer = positions[s['writers'][-1]['id']]
        assert [r['intervening_requests'] for r in rows] == [positions[e['id']]-last_writer-1 for e in s['readers']]
        assert rows[0]['intervening_requests'] == 0
        assert [r['background_before_reader'] for r in rows] == [sum(s['reader_gap_background_counts'][:i]) for i in range(4)]
        assert all(r['cache_report']['c1_expected_tokens'] == 56 for r in rows)
    assert len(list(root.glob('cache_*.pt'))) == 1
    text = capsys.readouterr().out
    assert text.count('[E2E] READER_CACHE {') == 16


def test_pause_resume_restores_last_round_without_replaying_requests(simulated_run):
    runner, cfg, plan, calls, control = simulated_run
    control['pause_round'] = 0
    first = runner.execute(cfg)
    assert first['status'] == 'paused' and first['completed_queries'] == 4
    already = len(calls)
    control['pause_round'] = None
    final = runner.execute(cfg, resume=True)
    assert final['status'] == 'complete' and final['completed_queries'] == 16
    assert control['restore_calls'] == 1
    assert all(e['round'] > 0 for e in calls[already:])
    assert all(Counter(e['id'] for e in calls)[e['id']] == 1 for e in calls)


def test_pause_commits_cooldown_before_the_next_writer_group(simulated_run):
    runner, cfg, plan, calls, control = simulated_run
    cfg['reader_schedule']['gaps'] = [dict(min=4+i, max=4+i, rounds=1) for i in range(4)]
    plan[:] = common.schedule(cfg, records(4))
    control['pause_round'] = 0
    result = runner.execute(cfg)
    assert result['status'] == 'paused' and result['completed_groups'] == 1
    assert calls == common.ordered_events(plan[0])
    assert calls[-1]['phase'] == 'cooldown'
    state = common.read_json(Path(cfg['output'])/'state.json')
    cache = common.read_json(Path(cfg['output'])/state['checkpoint']['file'])
    assert cache['source'] is None  # Fake backend evicts C1 on background traffic.
    before = len(calls)
    control['pause_round'] = None
    runner.execute(cfg, resume=True)
    assert calls[before]['id'] == plan[1]['writers'][0]['id']


def test_background_budget_edges_and_legacy_control(tmp_path):
    cfg = config()
    for total in (0, 1, 2, 3, 15):
        cfg['reader_schedule']['gaps'] = [dict(min=total, max=total, rounds=100)]
        cfg = common.load_settings(write_config(tmp_path, cfg))
        for step in common.schedule(cfg, records()):
            assert sum(step['reader_gap_background_counts']) <= total
            assert len(step['backgrounds']) == total
            assert sum(step['reader_gap_background_counts']) + step['cooldown_background_count'] == total
    cfg['reader_schedule']['timing'] = 'background_then_read'
    for step in common.schedule(cfg, records()):
        assert common.ordered_events(step) == step['writers']+step['backgrounds']+step['readers']
        assert step['cooldown_background_count'] == 0
    del cfg['reader_schedule']['timing']
    del cfg['reader_schedule']['background_between_readers']
    cfg = common.load_settings(write_config(tmp_path, cfg))
    assert common.schedule(cfg, records())[0]['timing'] == 'background_then_read'


def test_frozen_event_order_rejects_duplicate_or_missing_requests():
    step = common.schedule(config(), records())[0]
    step['request_order'][0] = step['request_order'][1]
    with pytest.raises(ValueError, match='exactly once'):
        common.ordered_events(step)


def test_partial_round_is_not_committed_and_snapshot_corruption_rejected(simulated_run):
    runner, cfg, plan, calls, control = simulated_run
    control['fail_on'] = plan[1]['readers'][1]['id']
    with pytest.raises(RuntimeError, match='simulated'):
        runner.execute(cfg)
    root = Path(cfg['output'])
    state = common.read_json(root/'state.json')
    assert state['cursor'] == 1
    assert common.read_json(root/'summary.json')['completed_queries'] == 4
    (root/state['checkpoint']['file']).write_text('corrupted')
    with pytest.raises(ValueError, match='corrupted'):
        runner.execute(cfg, resume=True)


def test_changed_config_cannot_resume(simulated_run):
    runner, cfg, _, _, control = simulated_run
    control['pause_round'] = 0
    runner.execute(cfg)
    cfg['cache']['cpu_gb'] = 2
    with pytest.raises(ValueError, match='identity changed'):
        runner.execute(cfg, resume=True)


def test_worker_native_admission_expiry_and_clean_reader_provenance(monkeypatch):
    """Exercise the production worker hooks with the backend's public interfaces."""
    import importlib.util
    import sys
    import threading
    import types
    import torch
    import e2e_cache

    class Key:
        def __init__(self, layer): self.layer = layer
        def __hash__(self): return hash(self.layer)
        def __eq__(self, other): return isinstance(other, Key) and self.layer == other.layer
        def to_string(self): return f'doc@{self.layer}'
        def split_layers(self, n): return [Key(i) for i in range(n)]
    class Backend:
        def __init__(self):
            self.hot_cache, self.cpu_lock = {}, threading.Lock()
        def submit_put_task(self, key, obj):
            with self.cpu_lock:
                if key not in self.hot_cache: self.hot_cache[key] = obj
        def remove(self, key): return self.hot_cache.pop(key, None) is not None
    cache = Backend()
    fake = types.ModuleType('e2e_cache.worker')
    fake.ORIGINS, fake.AUDIT, fake.REQUESTS = {}, {}, []
    fake.backend = lambda: cache
    class Parent:
        def e2e_setup(self, *a): return dict(ready=True)
        def e2e_register(self, requests):
            fake.REQUESTS = requests
            for r in requests:
                fake.AUDIT[r['id']] = dict(loaded_keys=[], loaded_sources=[], complete=True)
        def e2e_collect(self):
            return dict(requests={r['id']: fake.AUDIT[r['id']] for r in fake.REQUESTS}, events=[])
    fake.Extension = Parent
    monkeypatch.setitem(sys.modules, 'e2e_cache.worker', fake)
    monkeypatch.setattr(e2e_cache, 'worker', fake, raising=False)
    for name in ('lmcache', 'lmcache.v1', 'lmcache.v1.storage_backend'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    backend_module = types.ModuleType('lmcache.v1.storage_backend.local_cpu_backend')
    backend_module.LocalCPUBackend = Backend
    monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)
    monkeypatch.setenv('READOUT_WORKLOAD_ROPE_FIX', '0')
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    spec = importlib.util.spec_from_file_location('e2e_workload.worker_test', common.ROOT/'e2e_workload/worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    clock = [100.0]
    monkeypatch.setattr(worker, 'time', SimpleNamespace(time=lambda: clock[0]))
    extension = worker.Extension()
    extension.e2e_setup({}, dict(max_age_seconds=10), '')
    writer = dict(id='w', role='writer')
    extension.e2e_register([writer])
    objs = [SimpleNamespace(tensor=torch.zeros(2), can_evict=True) for _ in range(2)]
    for i, obj_ in enumerate(objs): cache.submit_put_task(Key(i), obj_)
    worker.track_stores(writer)
    assert fake.ORIGINS == {'doc@0': ['w'], 'doc@1': ['w']}
    first_token = objs[0]._workload_admission_sequence
    clock[0] = 104
    reader = dict(id='r', role='reader')
    extension.e2e_expire()
    extension.e2e_register([reader])
    fake.AUDIT['r'].update(loaded_keys=['doc@0', 'doc@1'], loaded_sources=['w'])
    result = extension.e2e_collect()
    assert [e['age_seconds'] for e in result['requests']['r']['cache_entries']] == [4, 4]
    clock[0] = 110
    extension.e2e_expire()
    assert not cache.hot_cache and not fake.ORIGINS
    clean = dict(id='clean', role='reader')
    extension.e2e_register([clean])
    # Even reuse of the same Python object gets a new native admission stamp.
    for i, obj_ in enumerate(objs): cache.submit_put_task(Key(i), obj_)
    worker.track_stores(clean)
    assert objs[0]._workload_admission_sequence > first_token
    assert fake.ORIGINS == {'doc@0': ['clean'], 'doc@1': ['clean']}
    assert all(e['direct_role'] == 'reader' for e in worker.STATE.entries.values())
    result = extension.e2e_collect()
    assert sum(e.get('reason') == 'max_age' for e in result['events']) == 2




