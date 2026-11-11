"""CPU tests verify latest-workload wiring/provenance, not real CUDA parity."""
from copy import deepcopy
import json
from pathlib import Path
import pytest

from kv_cache_readout import workload_parity as parity
from kv_cache_readout.config import ROOT,file_hash,load_config
from kv_cache_readout.io import atomic_json
from e2e_workload.common import load_settings,schedule,ordered_events

FIELDS=['name_full_direct','identity_number','address','email']


def samples():
    return [dict(record_id=f'r{i}',private_prefix=f'private {i}',
                 qa=[dict(field=f,question=f'What is {f}?',answer=f'{f}-{i}') for f in FIELDS])
            for i in range(20)]


def test_latest_serving_sources_match_frozen_upstream_snapshot():
    provenance=json.loads((ROOT/'LATEST_E2E_PROVENANCE.json').read_text(encoding='utf-8'))
    assert len(provenance['files'])>=12
    for path,digest in provenance['files'].items():assert file_hash(ROOT/path)==digest
    from kv_cache_readout.io import code_identity
    code=code_identity()
    assert 'e2e_workload/worker.py' in code and 'runtime_fixes/positions.py' in code


def test_representative_workload_preserves_latest_policy(tmp_path):
    original=load_settings(parity.DEFAULT_WORKLOAD)
    cfg=load_config('configs/train_hr.json')
    selected,meta=parity.validation_settings(cfg,tmp_path,groups=4)
    for key in ('native_rope_fix','background','writer_sampling','writers_min','writers_max'):
        assert selected[key]==original[key]
    assert selected['native_rope_fix'] is True
    assert selected['cache']==original['cache']
    assert [g['rounds'] for g in selected['reader_schedule']['gaps']]==[1,1,1,1]
    assert selected['engine']['max_new_tokens']==128
    assert selected['reader_schedule']['fields']==FIELDS
    assert meta['production_rounds']==100 and meta['validation_groups']==4
    plan=schedule(selected,samples())
    assert {g['gap_bucket'] for g in plan}=={0,1,2,3}
    assert sum(len(g['readers']) for g in plan)==16
    for group in plan:
        events=ordered_events(group)
        assert events[:len(group['writers'])]==group['writers']
        assert all(e['role']!='writer' for e in events[len(group['writers']):])
        assert max(group['reader_gap_background_counts'])<=1


def test_validation_uses_selected_physical_gpu_in_latest_runtime(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3')
    settings,_=parity.validation_settings(load_config('configs/train_hr.json'),tmp_path)
    assert settings['engine']['device']=='3'


@pytest.mark.parametrize('groups',[0,1,3,101])
def test_invalid_coverage_group_count_rejected(tmp_path,groups):
    with pytest.raises(ValueError,match='validation-groups'):
        parity.validation_settings(load_config('configs/train_hr.json'),tmp_path,groups=groups)


def test_cold_source_resolution_never_uses_designated_target():
    records={r['record_id']:r for r in samples()}
    registry={'actual':dict(role='writer',record_id='r0'),
              'target':dict(role='writer',record_id='r1')}
    report=dict(direct_sources=[dict(request_id='actual',role='writer')])
    history={'actual':dict(request=dict(public_start=7,public_end=63),
        response=dict(rows=[dict(audit=dict(retrieved_mask=[False]*80))]))}
    record,qa,reason=parity.resolve_writer(report,registry,records,history,'email','actual question')
    assert record['record_id']=='r0' and qa['answer']=='email-0'
    assert qa['question']=='actual question' and reason=='cold_c1_writer'
    history['actual']['response']['rows'][0]['audit']['retrieved_mask'][9]=True
    assert parity.resolve_writer(report,registry,records,history,'email','q')[2]=='writer_itself_reused_c1'
    mask=history['actual']['response']['rows'][0]['audit']['retrieved_mask']
    mask[9]=False;mask[0]=True
    assert parity.resolve_writer(report,registry,records,history,'email','q')[2]=='writer_itself_reused_other_tokens'
    report['direct_sources']=[dict(request_id='earlier-reader',role='reader')]
    assert parity.resolve_writer(report,registry,records,history,'email','q')[:2]==(None,None)


def test_collect_runs_full_latest_schedule_and_four_reader_trace_directories(tmp_path,monkeypatch):
    import e2e_workload.runtime as runtime
    import e2e_workload.common as common
    import e2e_cache.deployment as deployment
    import e2e_cache.__main__ as engine_main
    cfg=load_config('configs/train_hr.json')
    settings,_=parity.validation_settings(cfg,tmp_path,groups=4)
    config=tmp_path/'oracle.json';atomic_json(config,settings)
    records=samples();calls=[];rpcs=[]
    manifest=dict(experiment=cfg,source_u_sha256='frozen-u')
    class MockRuntime:
        def __init__(self,c,m):
            assert c['native_rope_fix'] and c['cache']['max_age_seconds']==120
            self.tokenizer=object();self.first=None
        def request(self,event,record,document):
            return dict(event,public_start=7 if record else 0,public_end=63 if record else 56,tokens=[1]*80)
        def rpc(self,name,directory):
            rpcs.append(name);Path(directory).mkdir(parents=True)
            atomic_json(Path(directory)/'runtime_fix.json',dict(policy='lmcache042_authoritative_source_positions_v1'))
        def generate(self,requests):
            request=requests[0];calls.append(request['role'])
            if request['role']=='writer' and self.first is None:self.first=request
            audit=dict(complete=True,unknown_sources=[],loaded_keys=[],loaded_blocks=[],hit_blocks=[],
                       retrieved_mask=[False]*80,store_calls=1,retrieve_calls=0)
            if request['role']=='reader':
                audit.update(loaded_keys=['c1-layer0'],loaded_sources=[self.first['id']],
                    retrieved_mask=[True]*56+[False]*24,
                    loaded_blocks=[dict(key='c1',prompt_start=0,prompt_end=56,block_tokens=56)],
                    cache_entries=[dict(key='c1-layer0',direct_request=self.first['id'],direct_role='writer')])
            return dict(rows=[dict(audit=audit,text='mock')],events=[],seconds=0.)
    monkeypatch.setattr(runtime,'Runtime',MockRuntime)
    monkeypatch.setattr(deployment,'verify',lambda c:manifest)
    monkeypatch.setattr(engine_main,'load_inputs',lambda c,m:records)
    monkeypatch.setattr(common,'prepare_documents',lambda c,t:[dict(id=str(i),tokens=[i]) for i in range(16)])
    index=parity.collect(config,tmp_path/'traces')
    assert index['runtime']=='e2e_workload' and index['native_rope_fix']
    assert index['worker']=='e2e_workload.worker.Extension'
    assert index['groups']==4 and len(index['readers'])==16
    assert calls.count('reader')==16 and calls.count('background')>0 and calls.count('writer')>=20
    assert len(rpcs)==16
    inputs=[json.loads((Path(r['directory'])/'inputs.json').read_text()) for r in index['readers']]
    assert all(i['comparison_eligible'] and i['writer_reference_eligible'] for i in inputs)
    assert any(i['record']['record_id']!=i['designated_target_record'] for i in inputs)


@pytest.mark.parametrize('case',['pass','failure','no_hit','no_writer'])
def test_latest_report_does_not_count_missing_evidence_as_pass(tmp_path,monkeypatch,case):
    from kv_cache_readout import parity as core
    readers=[dict(directory=str(tmp_path/f'r{i}'),field=f,group=0,
        comparison_eligible=case!='no_hit',writer_reference_eligible=case!='no_writer')
        for i,f in enumerate(FIELDS)]
    atomic_json(tmp_path/'index.json',dict(readers=readers,fields=FIELDS,groups=1,native_rope_fix=True))
    def compare(*a,**k):return dict(passed=case!='failure',writer_verified=case!='no_writer')
    monkeypatch.setattr(core,'compare',compare)
    result=parity.compare_all(tmp_path,execution={})
    assert result['passed'] is (case=='pass')
    assert (tmp_path/'workload_parity_report.json').exists()


def test_legacy_collect_entry_routes_to_latest_collector(monkeypatch):
    from kv_cache_readout import parity as old_entry
    monkeypatch.setattr(parity,'collect',lambda *a:dict(runtime='e2e_workload'))
    assert old_entry.collect('config','root')['runtime']=='e2e_workload'
