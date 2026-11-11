import copy
import json
import os
import socket
from pathlib import Path
import pytest
import torch

from kv_cache_readout.backend import Backend,per_sample_waves
from kv_cache_readout.config import load_config,validate_config,fingerprint
from kv_cache_readout.launch import parser,effective_config
from kv_cache_readout.identity import (scientific_config,local_batch,remap_seed,
                                       resume_compatibility)
from kv_cache_readout.io import run_lock,atomic_json,read_checkpoint
from kv_cache_readout.native_model import FAST_EXECUTION,EXECUTION
from kv_cache_readout.optim import parameters_for
from kv_cache_readout.training import batch_gradients
from kv_cache_readout.generation import generate_many
from test_native_autograd import native_backend


def fast_backend(fixture,b=2,q=2,ratio=.3):
    original,records,prompt=fixture
    cfg=copy.deepcopy(original.cfg);cfg['execution']=dict(FAST_EXECUTION)
    cfg['training'].update(sample_batch_size=b,questions_per_sample_batch=q)
    cfg['native_contract']['recompute_ratio']=ratio
    validate_config(cfg)
    return Backend(copy.deepcopy(original.model),original.tokenizer,cfg),records,prompt


def test_cli_has_exactly_two_microbatch_knobs_and_four_gpu_allocation():
    args=parser('train').parse_args(['--gpus','0,1,2,3','--sample-batch-size','4',
        '--questions-per-sample-batch','3','--global-batch-size','8','--recompute-ratio','.5',
        '--check-layer','2','--learning-rate','.005','--training-epochs','100'])
    cfg,gpus=effective_config(args)
    assert len(gpus)==4 and cfg['training']['c1_batch_size']==4
    assert cfg['training']['question_batch_size']==12
    assert [len(local_batch(list(range(8)),r,4)) for r in range(4)]==[2]*4
    help_text=parser('train').format_help()
    for forbidden in ('--writer-batch','--source-batch','--c1-batch','--keep-batch','--question-batch-size'):
        assert forbidden not in help_text


def test_reference_mode_rejects_batching():
    args=parser('train').parse_args(['--compute-mode','native-reference'])
    with pytest.raises(ValueError,match='requires'):effective_config(args)
    args=parser('train').parse_args(['--compute-mode','native-reference','--sample-batch-size','1',
                                   '--questions-per-sample-batch','1'])
    cfg,_=effective_config(args)
    assert cfg['execution']==EXECUTION


def test_questions_are_q_per_sample_not_q_per_whole_wave():
    sets=[[{'i':i} for i in range(n)] for n in (7,1,4,0)]
    waves=list(per_sample_waves(sets,3))
    assert [[sum(o==r for o,_,_ in w) for r in range(4)] for w in waves]==[[3,1,3,0],[3,0,1,0],[1,0,0,0]]
    assert [(o,i) for w in waves for o,i,_ in w]==[(o,i) for start in (0,3,6)
        for o,qs in enumerate(sets) for i in range(start,min(start+3,len(qs)))]


@pytest.mark.parametrize('world',[1,2,3,4])
def test_global_records_never_duplicated_or_dropped(world):
    for n in (1,3,8,9):
        assigned=[local_batch(list(range(n)),rank,world) for rank in range(world)]
        assert sorted(i for group in assigned for i in group)==list(range(n))
        assert max(map(len,assigned))-min(map(len,assigned))<=1
    assert remap_seed(17,25,1328,6416,0,world)==remap_seed(17,25,1328,6416,0,world)


def test_runtime_paths_and_batch_limits_are_not_scientific_changes():
    original=load_config('configs/train_hr.json');other=copy.deepcopy(original)
    other.update(device='cuda:3',model_name_or_path='/another/model',dataset_directory='/data/moved',
                 output_directory='/new/run')
    other['scoped_cache']['suffix_directory']='/moved/suffix'
    other['training'].update(sample_batch_size=4,questions_per_sample_batch=1)
    validate_config(other)
    assert scientific_config(original)==scientific_config(other)
    other['native_contract']['recompute_ratio']=.5
    assert scientific_config(original)!=scientific_config(other)


def test_identity_migration_checks_contents_and_known_code():
    cfg=load_config('configs/train_hr.json')
    dataset={'directory':'old','files':{'train':'sha'}}
    model={'assets':{'config.json':'sha'},'tokenizer_sha256':'tok'}
    state=dict(schema_version='aligned_readout_v3',config=copy.deepcopy(cfg),dataset=dataset,
               model=model,code={'current':'sha'},prompt=torch.zeros(64,16))
    assert not resume_compatibility(state,cfg,dict(dataset,directory='new'),model,state['code'])
    cfg['native_contract']['check_layer']=1
    assert resume_compatibility(state,cfg,dataset,model,state['code'])
    with pytest.raises(ValueError,match='asset contents'):
        resume_compatibility(state,cfg,dataset,dict(model,assets={'config.json':'changed'}),state['code'])
    with pytest.raises(ValueError,match='Unknown checkpoint code'):
        resume_compatibility(state,cfg,dataset,model,{'other':'version'})


def test_active_lock_is_not_removed_and_dead_lock_recovers(tmp_path,monkeypatch):
    from kv_cache_readout import io
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError,match='already active'): 
            with run_lock(tmp_path):pass
        assert (tmp_path/'.lock').exists()
    assert not (tmp_path/'.lock').exists()
    atomic_json(tmp_path/'.lock',dict(pid=9999999,host=socket.gethostname()))
    monkeypatch.setattr(io,'pid_alive',lambda pid:False)
    with run_lock(tmp_path):assert json.loads((tmp_path/'.lock').read_text())['pid']==os.getpid()
    assert not (tmp_path/'.lock').exists()


def test_alignment_cannot_be_omitted_or_failed(tmp_path):
    from kv_cache_readout.alignment import verify_attestation
    cfg=load_config('configs/train_hr.json')
    with pytest.raises(ValueError,match='required'):verify_attestation(cfg,None)
    p=tmp_path/'report.json';atomic_json(p,dict(passed=False))
    with pytest.raises(ValueError,match='did not pass'):verify_attestation(cfg,p)


def test_alignment_acceptance_binds_code_hardware_and_evidence(tmp_path,monkeypatch):
    from kv_cache_readout import alignment
    from kv_cache_readout.config import file_hash
    cfg=load_config('configs/train_hr.json')
    bound=dict(world=4,code='verified')
    monkeypatch.setattr(alignment,'binding',lambda *a:bound)
    monkeypatch.delenv('WORLD_SIZE',raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'set_device',lambda d:None)
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda:(8,0))
    evidence=tmp_path/'native_trace.pt';evidence.write_bytes(b'actual-trace-fixture')
    report=tmp_path/'acceptance.json'
    atomic_json(report,dict(passed=True,binding=bound,checkpoint=None,
        tolerance=dict(atol=.03,rtol=.03),compute_capability=[8,0],
        gradient_tolerance=alignment.GRADIENT_TOLERANCE,
        evidence={str(evidence):file_hash(evidence)}))
    assert alignment.verify_attestation(cfg,report)['passed']
    monkeypatch.setattr(alignment,'binding',lambda *a:dict(world=4,code='changed'))
    with pytest.raises(ValueError,match='no longer matches'):alignment.verify_attestation(cfg,report)
    monkeypatch.setattr(alignment,'binding',lambda *a:bound)
    evidence.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='evidence changed'):alignment.verify_attestation(cfg,report)


def test_evaluate_resume_does_not_require_training_latest(tmp_path,monkeypatch,capsys):
    from kv_cache_readout import launch
    import sys
    monkeypatch.setattr(sys,'argv',['evaluate.py','--checkpoint',str(tmp_path/'epoch_030.pt'),
        '--output-dir',str(tmp_path/'new-eval'),'--gpus','0,1','--resume','--dry-run'])
    launch.main('evaluate')
    assert 'epoch_030.pt' in capsys.readouterr().out
    assert not (tmp_path/'new-eval').exists()


def test_changed_condition_automatically_selects_separate_stage(tmp_path,monkeypatch,capsys):
    from kv_cache_readout import launch
    import sys
    source=load_config('configs/train_hr.json')
    monkeypatch.setattr(launch,'read_checkpoint',lambda p:dict(config=source))
    monkeypatch.setattr(sys,'argv',['train.py','--resume-from',str(tmp_path/'parent.pt'),
        '--output-dir',str(tmp_path/'run'),'--recompute-ratio','.5','--dry-run'])
    launch.main('train')
    plan=json.loads(capsys.readouterr().out)
    assert 'stages' in plan['config']['output_directory']
    assert plan['checkpoint']==str(tmp_path/'parent.pt')
    assert not (tmp_path/'run').exists()


def test_pause_entry_works_without_site_packages(tmp_path):
    import subprocess,sys
    from kv_cache_readout.config import ROOT
    result=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/pause.py'),
                           '--run-dir',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert (tmp_path/'pause.request').is_file()


def test_subprocess_watchdog_and_error_propagation():
    import subprocess,sys
    from kv_cache_readout.launch import command
    command([sys.executable,'-c','pass'],dict(os.environ),timeout=5)
    with pytest.raises(subprocess.CalledProcessError) as error:
        command([sys.executable,'-c','raise SystemExit(7)'],dict(os.environ),timeout=5)
    assert error.value.returncode==7
    with pytest.raises(subprocess.TimeoutExpired):
        command([sys.executable,'-c','import time; time.sleep(20)'],dict(os.environ),timeout=.1)


def test_train_and_auto_offline_launch_without_engine_validation(tmp_path,monkeypatch):
    import sys
    from kv_cache_readout import launch
    output=tmp_path/'run';calls=[]
    def forbidden(*a,**k):
        raise AssertionError('Training must never start a real vLLM alignment engine')
    def training_command(arguments,env):
        assert '--attestation' not in arguments
        assert 'alignment' not in arguments
        assert 'torch.distributed.run' in arguments
        calls.append('train')
        atomic_json(output/'summary.json',dict(status='stage_reached',epoch=30,cursor=0,step=1))
    def evaluation(*a):
        assert len(a)==7 and a[-2:] == ('test',None)
        calls.append('offline')
        return dict(status='complete',output=str(output/'offline'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(launch,'ensure_alignment',forbidden)
    monkeypatch.setattr(launch,'command',training_command)
    monkeypatch.setattr(launch,'evaluate_parallel',evaluation)
    monkeypatch.setattr(sys,'argv',['train.py','--output-dir',str(output),
                                  '--gpus','0,1,2,3','--then-offline'])
    launch.main('train')
    assert calls==['train','offline']


def test_standalone_offline_launch_does_not_start_engine(tmp_path,monkeypatch):
    import sys
    from kv_cache_readout import launch
    calls=[]
    def forbidden(*a,**k):raise AssertionError('Offline evaluation started the vLLM engine')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(launch,'ensure_alignment',forbidden)
    monkeypatch.setattr(launch,'evaluate_parallel',lambda *a:calls.append(a))
    monkeypatch.setattr(sys,'argv',['evaluate.py','--output-dir',str(tmp_path/'run'),
                                  '--checkpoint',str(tmp_path/'epoch_030.pt'),'--gpus','1,2'])
    launch.main('evaluate')
    assert len(calls)==1 and calls[0][3]==['1','2']


def test_real_engine_validation_has_its_own_explicit_entry(tmp_path,monkeypatch):
    import sys
    from kv_cache_readout import launch
    calls=[]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(launch,'ensure_alignment',lambda *a,**k:calls.append(a) or 'acceptance.json')
    monkeypatch.setattr(launch,'command',lambda *a:pytest.fail('Diagnostic launched training'))
    monkeypatch.setattr(sys,'argv',['check_alignment.py','--output-dir',str(tmp_path/'diagnostic'),
                                  '--checkpoint',str(tmp_path/'epoch_030.pt'),'--gpus','0,1'])
    launch.main('check-alignment')
    assert len(calls)==1 and calls[0][-1]==str(tmp_path/'epoch_030.pt')


@pytest.mark.parametrize('ratio',[0.,.3,1.])
def test_per_sample_packed_gradient_matches_serial_and_ratio_edges(native_backend,ratio):
    batched,records,prompt=fast_backend(native_backend,ratio=ratio)
    records=copy.deepcopy(records)
    records[0]['qa']*=3
    records[1]['qa']=records[1]['qa'][:1]
    actual,loss,flags=batch_gradients(batched,parameters_for(prompt,batched.cfg),records,len(records),0)
    cfg=copy.deepcopy(batched.cfg);cfg['training'].update(sample_batch_size=1,questions_per_sample_batch=1)
    validate_config(cfg)
    serial=Backend(copy.deepcopy(batched.model),batched.tokenizer,cfg)
    expected,reference,_=batch_gradients(serial,parameters_for(prompt,cfg),records,len(records),0)
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual,expected,atol=4e-5,rtol=5e-4)
    assert loss==pytest.approx(reference,rel=3e-5,abs=3e-5)
    if ratio==1 and cfg['native_contract']['check_layer']==0:
        assert torch.count_nonzero(actual[:48])==0


@pytest.mark.parametrize('world',[3,4])
def test_uneven_global_gradient_is_sum_of_actual_record_weights(native_backend,world):
    backend,records,prompt=fast_backend(native_backend)
    batch=[copy.deepcopy(records[i%2]) for i in range(3)]
    expected,loss,_=batch_gradients(backend,parameters_for(prompt,backend.cfg),batch,len(batch),0)
    gradient=torch.zeros_like(expected);total=0.
    for rank in range(world):
        local=local_batch(batch,rank,world)
        g,value,_=batch_gradients(backend,parameters_for(prompt,backend.cfg),local,len(batch),0)
        gradient+=g;total+=value
    assert torch.allclose(gradient,expected,atol=4e-5,rtol=5e-4)
    assert total==pytest.approx(loss,rel=3e-5)


def test_ragged_greedy_generation_and_returned_cache_match_single(native_backend):
    backend,records,prompt=fast_backend(native_backend)
    with torch.no_grad():
        contexts=backend.blend_many(records,prompt)
        qs=[dict(question='a?'),dict(question='longer variable length question?')]
        batched=generate_many(backend,contexts,qs,max_new_tokens=4)
        single=[backend.generate(c,q,max_new_tokens=4) for c,q in zip(contexts,qs)]
    assert [a[1]['generated_token_ids'] for a in batched]==[a[1]['generated_token_ids'] for a in single]


def test_offline_commits_resume_with_different_gpu_partition(tmp_path,monkeypatch):
    from kv_cache_readout import offline
    cfg=load_config('configs/train_hr.json');cfg['output_directory']=str(tmp_path)
    identity={'checkpoint':'frozen'}
    monkeypatch.setattr(offline,'evaluation_identity',lambda *a:identity)
    records=[dict(record_id=f'r{i}',fields={},qa=[dict(field='title',question='q?',answer='A')])
             for i in range(5)]
    monkeypatch.setattr(offline,'load_split',lambda *a,**k:records)
    output=offline.evaluation_directory(cfg,'unused','test',None);(output/'items').mkdir(parents=True)
    for record in records[:3]:
        qa=record['qa'][0];task=offline.task_id(record,0,qa)
        row=dict(record_id=record['record_id'],field='title',gold='A',prediction='A',em=1.,nll=.1)
        atomic_json(output/'items'/(task+'.json'),dict(identity=fingerprint(identity),task_id=task,
                    row=row,row_sha256=fingerprint(row)))
    first=offline.merge(cfg,'unused','test',None)
    assert first['status']=='paused' and first['missing_questions']==2
    # All remaining immutable task IDs are owned once under a new world size.
    pending=[]
    for rank in range(4):
        for record in records[rank::4]:
            task=offline.task_id(record,0,record['qa'][0]);path=output/'items'/(task+'.json')
            if offline.committed(path,fingerprint(identity),task) is None:pending.append(record)
    assert [r['record_id'] for r in pending]==['r4','r3']
    for record in pending:
        task=offline.task_id(record,0,record['qa'][0])
        row=dict(record_id=record['record_id'],field='title',gold='A',prediction='A',em=1.,nll=.1)
        atomic_json(output/'items'/(task+'.json'),dict(identity=fingerprint(identity),task_id=task,
                    row=row,row_sha256=fingerprint(row)))
    assert offline.merge(cfg,'unused','test',None)['status']=='complete'
    path=next((output/'items').glob('*.json'));item=json.loads(path.read_text());item['row']['gold']='tampered'
    atomic_json(path,item)
    with pytest.raises(ValueError,match='contents changed'):offline.merge(cfg,'unused','test',None)


def test_native_offline_worker_only_computes_remaining_records_after_remap(
        native_backend,tmp_path,monkeypatch):
    from kv_cache_readout import offline
    from kv_cache_readout.io import atomic_torch,code_identity
    backend,records,prompt=fast_backend(native_backend)
    cfg=backend.cfg;cfg['output_directory']=str(tmp_path/'run')
    cfg['generation']['max_new_tokens']=3
    dataset={'fixture':'frozen-offline'}
    monkeypatch.setattr(offline,'dataset_identity',lambda cfg:dataset)
    monkeypatch.setattr(offline,'load_split',lambda *a,**k:records)
    monkeypatch.setattr(offline.Backend,'load',lambda cfg:backend)
    checkpoint=tmp_path/'epoch_030.pt'
    atomic_torch(checkpoint,dict(schema_version='aligned_readout_v3',config=cfg,prompt=prompt,
                               dataset=dataset,model=backend.identity,code=code_identity()))
    seen=[];blend=backend.blend_many
    def observe(rs,*a,**k):
        seen.extend(r['record_id'] for r in rs)
        return blend(rs,*a,**k)
    monkeypatch.setattr(backend,'blend_many',observe)
    offline.evaluate_worker(cfg,str(checkpoint),0,2,'test',None)
    assert offline.merge(cfg,str(checkpoint),'test',None)['missing_questions']==2
    offline.evaluate_worker(cfg,str(checkpoint),0,1,'test',None)
    assert offline.merge(cfg,str(checkpoint),'test',None)['status']=='complete'
    assert seen==[records[0]['record_id'],records[1]['record_id']]


def test_two_knob_training_resume_remaps_rng_without_resetting_adam(
        native_backend,tmp_path,monkeypatch):
    from kv_cache_readout import training
    from kv_cache_readout.io import atomic_torch
    backend,records,prompt=fast_backend(native_backend)
    cfg=backend.cfg;cfg['training'].update(epochs=2,batch_records=2)
    template=copy.deepcopy(backend.model)
    monkeypatch.setattr(Backend,'initial_prompt',lambda self:prompt.clone())
    monkeypatch.setattr(training.Backend,'load',lambda c:Backend(copy.deepcopy(template),backend.tokenizer,c))
    monkeypatch.setattr(training,'load_split',lambda *a,**k:records)
    monkeypatch.setattr(training,'dataset_identity',lambda cfg:{'fixture':'new-entry'})
    cfg['output_directory']=str(tmp_path/'continuous');training.train(cfg)
    continuous=read_checkpoint(tmp_path/'continuous/latest.pt')
    cfg['output_directory']=str(tmp_path/'resumed');training.train(cfg,max_updates=1)
    path=tmp_path/'resumed/latest.pt';saved=read_checkpoint(path)
    # Emulate the saved RNG envelope of a two-rank checkpoint. Global U,
    # optimizer, scheduler and cursor are unchanged; resume now has one rank.
    saved['rng_by_rank']*=2;atomic_torch(path,saved)
    training.train(cfg,resume=True)
    resumed=read_checkpoint(path)
    assert torch.equal(continuous['prompt'],resumed['prompt'])
    assert continuous['scheduler']==resumed['scheduler']
    assert resumed['resume_policy']['previous_world_size']==2
    assert resumed['resume_policy']['rng']=='seeded_or_remapped'
    for a,b in zip(continuous['optimizer']['state'].values(),resumed['optimizer']['state'].values()):
        for key in a:assert torch.equal(a[key],b[key])
