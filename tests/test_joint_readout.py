import copy
import hashlib
import json
import re

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from kv_cache_readout.backend import Backend
from kv_cache_readout.config import ROOT, load_config, validate_config
from kv_cache_readout.data import load_split
from kv_cache_readout.joint import (FIELDS, PROMPT, question, layout, reduce_losses,
                                   default_settings, score_rows)
from kv_cache_readout.optim import parameters_for
from kv_cache_readout.training import batch_gradients
from test_training_v2 import Tokenizer


class CharTokenizer(Tokenizer):
    def __call__(self, text, **kwargs):
        return dict(input_ids=self.encode(text), offset_mapping=[(i,i+1) for i in range(len(text))])


def tiny_joint(settings=None):
    torch.set_num_threads(1);torch.manual_seed(17)
    cfg=load_config('configs/train_hr_joint_c128.json')
    cfg.update(device='cpu',dtype='float32',public_text='x'*128,
               model_name_or_path='missing-tiny-assets',question_format='plain')
    cfg['scoped_cache']['preservation_weight']=0
    if settings is not None:cfg['joint_loss']=settings
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=128,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=2048,eos_token_id=0))
    backend=Backend(model,CharTokenizer(),cfg)
    records=[]
    for i in range(2):
        fs=dict(name='A'+str(i),identity_number='123-45-678'+str(i),address='Road',email='a@b.c')
        records.append(dict(record_id='r'+str(i),private_prefix='private'+' more'*i,
            fields=fs,qa=[question(fs)],_suffix_fixture=dict(ids=[65,66],text='AB',suffix_id='s',article_id='a')))
    return backend,records,torch.randn(64,16)*.02


def test_entire_joint_dataset_and_sources():
    cfg=load_config('configs/train_hr_joint_c128.json')
    root=ROOT/cfg['dataset_directory']
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    donors={};numbers={}
    for split,n in [('train',2000),('validation',400),('test',2000)]:
        path=root/f'{split}.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['files'][path.name]
        rows=[json.loads(line) for line in path.open(encoding='utf-8')]
        parent=[json.loads(line) for line in (ROOT/'data/dataset_hr_v4_reduced'/path.name).open(encoding='utf-8')]
        assert len(rows)==len(load_split(cfg,split))==n
        donors[split]=set();numbers[split]=set()
        for row,old in zip(rows,parent):
            assert row['record_id']==old['record_id'] and row['group_id']==old['group_id']
            assert {k:v for k,v in row['fields'].items() if k!='identity_number'}=={k:v for k,v in old['fields'].items() if k!='identity_number'}
            assert re.fullmatch(r'\d{3}-\d{2}-\d{4}',row['fields']['identity_number'])
            for name in ('qa','training_qa','evaluation_qa'):
                assert row[name]==[question(row['fields'])]
                assert row[name][0]['question']==PROMPT
            for field,a in row['field_annotations'].items():
                s,e=a['evidence_span'];assert row['private_prefix'][s:e]==row['fields'][field]
            if row['joint_derivation']['identity_action']=='source_replacement':
                a=row['field_annotations']['identity_number']
                assert a['source_ref'].startswith('Gretel-PII:')
                assert a['normalization']['source_split']==split
                assert a['source_evidence']==row['fields']['identity_number']
                donors[split].add(a['source_ref']);numbers[split].add(row['fields']['identity_number'])
    for left,right in [('train','test'),('train','validation'),('test','validation')]:
        assert not donors[left]&donors[right]
        assert not numbers[left]&numbers[right]


def test_frozen_tokenizer_layout_all_records():
    cfg=load_config('configs/train_hr_joint_c128.json')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'data/tokenizer/Qwen3-4B',local_files_only=True)
    class Adapter:
        def question_text(self,text):
            return self.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        def question_and_target(self,qa):
            prefix=self.question_text(qa['question'])
            q=self.tokenizer.encode(prefix,add_special_tokens=False)
            both=self.tokenizer.encode(prefix+qa['answer'],add_special_tokens=False)
            assert both[:len(q)]==q
            return q,both[len(q):]+[self.tokenizer.eos_token_id]
    adapter=Adapter();adapter.tokenizer=tokenizer
    for split in ('train','validation','test'):
        for row in load_split(cfg,split):
            owners,spans=layout(adapter,row['qa'][0])
            assert len(owners)==len(adapter.question_and_target(row['qa'][0])[1])
            assert set(owners)==set(range(10))|{-1}
            assert all(qa['field']=='all_fields' for qa in row['qa'])


def test_weighted_field_mean_and_routing_independent_of_length_and_presence():
    settings=default_settings();settings['fields']={}
    settings['format']['weight']=settings['missing']['weight']=0
    b,_,_=tiny_joint(settings)
    for fields in (dict(name='A'),{k:'long content' for k in FIELDS}):
        qa=question(fields);owners,_=layout(b,qa)
        ce=torch.full((len(owners),),2.,requires_grad=True)
        result=reduce_losses(b,qa,ce)
        torch.testing.assert_close(result,torch.tensor([2.,0.]))
    qa=question(dict(name='A',email='long@address.example'))
    settings['fields']['email']=dict(type='focal',gamma=0.,alpha=1.,weight=3.,u_group='focal')
    owners,spans=layout(b,qa)
    ce=torch.ones(len(owners),requires_grad=True)
    result=reduce_losses(b,qa,ce)
    torch.testing.assert_close(result,torch.tensor([.25,.75]))
    settings['fields']['email']['type']='ce'
    torch.testing.assert_close(reduce_losses(b,qa,ce),result)
    result.sum().backward();assert torch.isfinite(ce.grad).all()


def test_joint_native_gradient_and_inactive_u_partition():
    settings=default_settings();settings['fields']={}
    b,records,initial=tiny_joint(settings)
    grad,loss,active=batch_gradients(b,parameters_for(initial,b.cfg),records,2,0)
    assert active==[True,False,False]
    assert torch.isfinite(grad).all() and grad[:32].abs().sum()>0
    assert grad[32:].count_nonzero()==0
    settings['fields']['identity_number']=dict(type='focal',u_group='focal')
    b.cfg['joint_loss']=settings
    grad,loss,active=batch_gradients(b,parameters_for(initial,b.cfg),records,2,0)
    assert active==[True,True,False] and grad[32:48].abs().sum()>0
    assert grad[48:].count_nonzero()==0


def test_joint_json_scoring_rejects_malformed_duplicate_and_missing_keys():
    qa=question(dict(name='A',identity_number='123-45-6789'))
    perfect=dict(record_id='r',gold=qa['answer'],prediction=qa['answer'])
    stats,fields=score_rows([perfect]);assert stats['record_em']==stats['macro_field_em']==1
    assert stats['missing_error_rate']==0 and len(fields)==10
    for text in ('not JSON',qa['answer'][:-1]+',"name":"B"}', '{}', '[]', '{"name":NaN}'):
        stats,_=score_rows([dict(perfect,prediction=text)])
        assert stats['record_em']==0 and stats['schema_valid_rate']==0
    obj=json.loads(qa['answer']);obj['salary']='100'
    stats,_=score_rows([dict(perfect,prediction=json.dumps(obj))])
    assert stats['macro_field_em']==1 and stats['missing_false_fill_rate']==1/8


def test_bad_loss_config_and_missing_joint_config_rejected():
    cfg=load_config('configs/train_hr_joint_c128.json')
    for bad in ('typo',):
        c=copy.deepcopy(cfg);c['joint_loss']['fields'][bad]={}
        with pytest.raises(ValueError):validate_config(c)
    for v in (-1,float('nan')):
        c=copy.deepcopy(cfg);c['joint_loss']['format']['weight']=v
        with pytest.raises(ValueError):validate_config(c)
    del cfg['joint_loss']
    with pytest.raises(ValueError,match='requires joint_loss'):load_split(cfg,'test')


def test_offline_merge_exports_joint_and_field_results_and_resumes(tmp_path,monkeypatch):
    from kv_cache_readout import offline
    from kv_cache_readout.io import atomic_json
    cfg=load_config('configs/train_hr_joint_c128.json')
    records=[dict(record_id='r',fields=dict(name='A'),qa=[question(dict(name='A'))])]
    output=tmp_path/'evaluation';output.mkdir()
    monkeypatch.setattr(offline,'load_split',lambda *a,**k:records)
    monkeypatch.setattr(offline,'evaluation_identity',lambda *a,**k:{'fixed':'identity'})
    monkeypatch.setattr(offline,'evaluation_directory',lambda *a,**k:output)
    qa=records[0]['qa'][0];task=offline.task_id(records[0],0,qa)
    row=dict(record_id='r',field='all_fields',gold=qa['answer'],prediction=qa['answer'],
             prediction_kind='joint_json',nll=1.,em=1.,truncated=False)
    from kv_cache_readout.config import fingerprint
    atomic_json(output/'items'/(task+'.json'),dict(identity=fingerprint({'fixed':'identity'}),
        task_id=task,row=row,row_sha256=fingerprint(row)))
    result=offline.merge(cfg,'unused','test',None)
    assert result['status']=='complete' and result['metrics']['record_em']==1
    import csv
    with (output/'predictions.csv').open(encoding='utf-8-sig') as f:
        assert len(list(csv.DictReader(f)))==1
    with (output/'field_predictions.csv').open(encoding='utf-8-sig') as f:
        assert len(list(csv.DictReader(f)))==10
    assert offline.merge(cfg,'unused','test',None)==result


def test_four_gpu_one_record_allocation():
    from kv_cache_readout.launch import parser,effective_config
    from kv_cache_readout.identity import local_batch
    args=parser('train').parse_args(['--config','configs/train_hr_joint_c128.json','--gpus','0,1,2,3'])
    cfg,gpus=effective_config(args)
    assert cfg['training']['batch_records']==4
    assert cfg['training']['sample_batch_size']==cfg['training']['questions_per_sample_batch']==1
    assert [len(local_batch(list(range(4)),i,4)) for i in range(4)]==[1]*4
