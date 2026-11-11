import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from kv_cache_readout.config import ROOT,load_config,validate_config
from kv_cache_readout.data import load_split
from kv_cache_readout.joint import question,layout,date_component_owners,reduce_losses,score_rows
from kv_cache_readout.readout import reduce_fieldwise
from kv_cache_readout.post_training import run_followups,run_joint_e2e
from kv_cache_readout.io import atomic_json
from test_joint_readout import tiny_joint


def settings_backend():
    b,records,initial=tiny_joint()
    b.cfg['joint_loss']['fields']={'birth_date':dict(type='ce',weight=1.,u_group='focal',reduction='ymd_mean')}
    b.cfg['joint_loss']['format']['weight']=b.cfg['joint_loss']['missing']['weight']=0
    return b,records,initial


@pytest.mark.parametrize('kind',['ce','focal'])
def test_date_component_mean_and_exact_gradient_weights(kind):
    b,_,_=settings_backend();b.cfg['joint_loss']['fields']['birth_date']['type']=kind
    qa=question(dict(birth_date='1995-03-08'))
    labels=date_component_owners(b,qa)
    ce=torch.tensor([1. if i==0 else 2. if i==1 else 3. if i==2 else 100. for i in labels],requires_grad=True)
    result=reduce_losses(b,qa,ce)
    reference=torch.tensor([1.,2.,3.])
    if kind=='focal':reference=(-torch.expm1(-reference)).pow(2)*reference
    torch.testing.assert_close(result[1],reference.mean())
    assert result[0]==0
    result.sum().backward()
    assert all(ce.grad[i]==0 for i,label in enumerate(labels) if label<0)
    if kind=='ce':
        for i,label in enumerate(labels):
            if label>=0:assert float(ce.grad[i])==pytest.approx(1/(3*labels.count(label)))


def test_fieldwise_date_also_honors_component_reduction_and_missing_date_still_works():
    b,_,_=settings_backend()
    qa=dict(field='birth_date',question='birthday?',answer='1995-03-08',prediction_kind='fieldwise_readout')
    owners=date_component_owners(b,qa)
    ce=torch.tensor([float(i+1) if i>=0 else 100. for i in owners])
    torch.testing.assert_close(reduce_fieldwise(b,qa,ce),torch.tensor([0.,2.]))
    qa=question(dict(name='A'));owners,_=layout(b,qa)
    assert torch.isfinite(reduce_losses(b,qa,torch.ones(len(owners)))).all()


def test_frozen_tokenizer_all_real_date_components():
    tok=AutoTokenizer.from_pretrained(ROOT/'data/tokenizer/Qwen3-4B',local_files_only=True)
    class Adapter:
        def question_text(self,text):
            return tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        def question_and_target(self,qa):
            prefix=self.question_text(qa['question']);q=tok.encode(prefix,add_special_tokens=False)
            joined=tok.encode(prefix+qa['answer'],add_special_tokens=False)
            return q,joined[len(q):]+[tok.eos_token_id]
    b=Adapter();b.tokenizer=tok
    count=0
    for split in ('train','validation','test'):
        for row in load_split(load_config('configs/train_hr_joint_ymd_c128_server2.json'),split):
            if 'birth_date' not in row['fields']:continue
            owners=date_component_owners(b,row['qa'][0]);field_owners,spans=layout(b,row['qa'][0])
            assert [owners.count(i) for i in range(3)]==[4,2,2]
            index=next(i for i,s in enumerate(spans) if s['field']=='birth_date')
            assert all(field_owners[i]==index for i,x in enumerate(owners) if x>=0)
            assert 'birth_year' not in json.loads(row['qa'][0]['answer'])
            count+=1
    assert count==2864


def test_date_scoring_still_outputs_full_date_and_reports_parts():
    qa=question(dict(birth_date='1995-03-08'))
    obj=json.loads(qa['answer']);obj['birth_date']='1995-04-08'
    result,_=score_rows([dict(record_id='r',gold=qa['answer'],prediction=json.dumps(obj))])
    assert result['per_field']['birth_date']['em']==0
    assert [result['birth_date_component_em'][k]['em'] for k in ('year','month','day')]==[1,0,1]


def test_bad_date_reduction_rejected():
    cfg=load_config('configs/train_hr_joint_ymd_c128_server2.json')
    cfg['joint_loss']['fields']['birth_date']['reduction']='unknown'
    with pytest.raises(ValueError,match='reduction'):validate_config(cfg)


def test_native_date_reduction_has_finite_partitioned_gradients():
    from kv_cache_readout.training import batch_gradients
    from kv_cache_readout.optim import parameters_for
    b,records,initial=tiny_joint()
    b.cfg['joint_loss']['fields']['birth_date']['reduction']='ymd_mean'
    for record in records:
        record['fields']['birth_date']='1995-03-08'
        record['qa']=[question(record['fields'])]
    grad,loss,active=batch_gradients(b,parameters_for(initial,b.cfg),records,2,0)
    assert active==[True,True,False]
    assert torch.isfinite(grad).all() and grad[:48].abs().sum()>0 and grad[48:].count_nonzero()==0


def pipeline_fixture(tmp_path):
    cfg=load_config('configs/train_hr_joint_ymd_c128_server2.json');cfg['output_directory']=str(tmp_path/'run')
    source=tmp_path/'checkpoint.pt';torch.save(dict(config=cfg),source)
    args=SimpleNamespace(offline_sample_batch_size=16,then_e2e=True,e2e_gpu='1',e2e_template='configs/e2e_workload_joint.json')
    return cfg,source,args


def test_pipeline_order_frozen_config_and_resume(tmp_path):
    cfg,source,args=pipeline_fixture(tmp_path);calls=[]
    def evaluate(evaluated,*rest):
        assert evaluated['training']['sample_batch_size']==16
        assert cfg['training']['sample_batch_size']==4
        calls.append('offline');return dict(status='complete',output='offline')
    def execute(command,env):
        action=command[3];calls.append(action)
        config=json.loads(Path(command[-1]).read_text(encoding='utf-8'))
        assert env['CUDA_VISIBLE_DEVICES']=='1'
        assert config['reader_schedule']['fields']==['all_fields']
        if action=='prepare':atomic_json(Path(config['deployment'])/'e2e_manifest.json',{})
        if action in ('run','resume'):atomic_json(Path(config['output'])/'state.json',dict(status='complete',cursor=100))
    run_followups(cfg,tmp_path/'launch.json',str(source),['1','2'],{},args,evaluate,execute)
    assert calls==['offline','prepare','preflight','run']
    marker=json.loads((Path(cfg['output_directory'])/'post_training_state.json').read_text())
    assert marker['status']=='complete'
    calls.clear()
    run_joint_e2e(cfg,source,'1',args.e2e_template,{},execute)
    assert calls==['preflight','resume']


@pytest.mark.parametrize('status',['paused','error'])
def test_failed_or_paused_offline_never_starts_e2e(tmp_path,status):
    cfg,source,args=pipeline_fixture(tmp_path)
    def evaluate(*a):
        if status=='error':raise RuntimeError('offline failed')
        return dict(status='paused')
    def forbidden(*a):raise AssertionError('E2E must not start')
    if status=='error':
        with pytest.raises(RuntimeError,match='offline failed'):
            run_followups(cfg,tmp_path/'launch.json',source,['1','2'],{},args,evaluate,forbidden)
    else:run_followups(cfg,tmp_path/'launch.json',source,['1','2'],{},args,evaluate,forbidden)
    marker=json.loads((Path(cfg['output_directory'])/'post_training_state.json').read_text())
    assert marker['status']==('failed' if status=='error' else 'paused')


def test_joint_workload_has_four_independent_json_readers_per_group():
    from e2e_workload.common import load_settings,schedule,ordered_events
    from kv_cache_readout.joint import PROMPT
    cfg=load_settings('configs/e2e_workload_joint.json')
    records=load_split(load_config('configs/train_hr_joint_ymd_c128_server2.json'),'test')
    plan=schedule(cfg,records)
    assert len(plan)==100 and sum(len(s['readers']) for s in plan)==400
    for group in plan:
        assert len({r['id'] for r in group['readers']})==4
        assert all(r['field']=='all_fields' and r['question']==PROMPT and 'answer' not in r for r in group['readers'])
        assert all(n in (0,1) for n in group['reader_gap_background_counts'])
        assert len(ordered_events(group))==len(group['writers'])+len(group['backgrounds'])+4


def test_joint_e2e_report_uses_actual_source_not_designated_target(tmp_path):
    from test_workload_source_csv import result,REGISTRY
    from e2e_workload.joint_report import write_joint_report
    row=result();row['reader']['field']='all_fields'
    gold=question(dict(name='Actual'))['answer']
    row['score']['expected_answer']=question(dict(name='Target'))['answer']
    row['score']['source_answers'][0]['answer']=gold
    row['response']['text']=json.dumps(json.loads(gold),indent=2)
    excluded=copy.deepcopy(row);excluded['reader']['id']='reader_unknown'
    excluded['cache_report']['c1_hit']=False
    summary=write_joint_report(tmp_path,[row,excluded],REGISTRY)
    assert summary['total_joint_queries']==2 and summary['verified_unique_writer_queries']==1
    assert summary['record_em']==1 and summary['excluded_queries']==1
    import csv
    with (tmp_path/'joint_field_predictions.csv').open(encoding='utf-8-sig') as f:
        fields=list(csv.DictReader(f))
    assert len(fields)==10 and all(r['record_id']=='actual_sample' for r in fields)
