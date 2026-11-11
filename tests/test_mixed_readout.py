import copy
import hashlib
import json

import pytest
import torch
from transformers import AutoTokenizer

from kv_cache_readout.config import ROOT, load_config, validate_config
from kv_cache_readout.data import load_split
from kv_cache_readout.joint import FIELDS, PROMPT, question, options
from kv_cache_readout.readout import (field_questions, combine_tasks, task_weights,
                                    select_questions, joint_evaluation)
from kv_cache_readout.training import batch_gradients
from kv_cache_readout.optim import parameters_for
from kv_cache_readout.launch import parser,effective_config
from test_joint_readout import tiny_joint


def explicit_cfg():
    return load_config('configs/train_hr_mixed_c128.json')


def test_full_mixed_dataset_preserves_all_prefixes_and_answers_and_modes():
    cfg=explicit_cfg();root=ROOT/cfg['dataset_directory']
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for split,n in [('train',2000),('validation',400),('test',2000)]:
        path=root/f'{split}.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['files'][path.name]
        records=[json.loads(line) for line in path.open(encoding='utf-8')]
        parent=[json.loads(line) for line in (ROOT/'data/dataset_hr_v5_joint'/path.name).open(encoding='utf-8')]
        assert len(records)==n
        for row,old in zip(records,parent):
            for key in ('record_id','group_id','private_prefix','fields','field_annotations'):
                assert row[key]==old[key]
            assert row['joint_qa']==[question(row['fields'])]
            assert row['fieldwise_qa']==field_questions(row)
            assert row['training_qa']==row['fieldwise_qa']+row['joint_qa']
            assert row['evaluation_qa']==row['joint_qa']
            for qa in row['fieldwise_qa']:
                a,b=qa['evidence_span'];assert row['private_prefix'][a:b]==qa['answer']
        mixed=load_split(cfg,split,training=True)
        assert all(len(r['qa'])==len(r['fields'])+1 for r in mixed)
        evaluated=load_split(cfg,split,training=False)
        assert all(len(r['qa'])==1 and r['qa'][0]['question']==PROMPT for r in evaluated)
        for mode in ('fieldwise','joint'):
            other=copy.deepcopy(cfg);other['readout']['train_mode']=mode
            selected=load_split(other,split,training=True)
            assert all(len(r['qa'])==(1 if mode=='joint' else len(r['fields'])) for r in selected)
            assert all(len(r['qa'])==1 for r in load_split(other,split,training=False))


def test_all_fieldwise_frozen_token_boundaries():
    tok=AutoTokenizer.from_pretrained(ROOT/'data/tokenizer/Qwen3-4B',local_files_only=True)
    cfg=explicit_cfg();cfg['readout']['train_mode']='fieldwise'
    for split in ('train','validation','test'):
        for row in load_split(cfg,split,training=True):
            for qa in row['qa']:
                prefix=tok.apply_chat_template([dict(role='user',content=qa['question'])],
                    tokenize=False,add_generation_prompt=True,enable_thinking=False)
                qids=tok.encode(prefix,add_special_tokens=False)
                ids=tok.encode(prefix+qa['answer'],add_special_tokens=False)
                assert ids[:len(qids)]==qids


@pytest.mark.parametrize('count',[1,4,10])
def test_family_balance_independent_of_field_count(count):
    cfg=explicit_cfg();cfg['joint_loss']['fields']={}
    fields={k:'A' for k in FIELDS[:count]}
    field_q=[dict(field=k,prediction_kind='fieldwise_readout') for k in fields]
    qa=question(fields)
    values=[(torch.tensor(2.,requires_grad=True),torch.tensor(0.,requires_grad=True)) for _ in fields]
    values.append((torch.tensor(6.,requires_grad=True),torch.tensor(0.,requires_grad=True)))
    result,routes=combine_tasks(cfg,field_q+[qa],values)
    torch.testing.assert_close(result[0],torch.tensor(4.))
    assert routes=={'other'}
    cfg['readout'].update(fieldwise_weight=1.,joint_weight=3.)
    result,_=combine_tasks(cfg,field_q+[qa],values)
    torch.testing.assert_close(result[0],torch.tensor(5.))
    result[0].backward()
    assert values[-1][0].grad==.75


def test_mixed_native_gradient_is_weighted_sum_of_pure_task_gradients():
    b,records,initial=tiny_joint()
    cfg=b.cfg;cfg['readout']=dict(train_mode='mixed',eval_mode='joint',fieldwise_weight=.25,joint_weight=.75)
    cfg['training'].update(sample_batch_size=2,questions_per_sample_batch=1)
    cfg['scoped_cache']['preservation_weight']=.1
    validate_config(cfg)
    cfg['scoped_cache']['preservation_weight']=0  # isolate the two readout task families
    for record in records:
        record['field_annotations']={k:dict(locator='the employee record',evidence_span=[0,1]) for k in record['fields']}
        record['fieldwise_qa']=field_questions(record)
        record['joint_qa']=[question(record['fields'])]
    gradients={};losses={}
    for mode in ('fieldwise','joint','mixed'):
        cfg['readout']['train_mode']=mode
        selected=[dict(r,qa=select_questions(r,cfg,True)) for r in records]
        grad,loss,active=batch_gradients(b,parameters_for(initial,cfg),selected,2,0)
        assert active==[True,True,False]
        assert torch.isfinite(grad).all() and grad[48:].count_nonzero()==0
        gradients[mode]=grad;losses[mode]=loss
    torch.testing.assert_close(gradients['mixed'],.25*gradients['fieldwise']+.75*gradients['joint'],atol=3e-6,rtol=3e-4)
    assert losses['mixed']==pytest.approx(.25*losses['fieldwise']+.75*losses['joint'],rel=3e-5)


def test_cli_mixed_modes_and_four_card_b2():
    args=parser('train').parse_args(['--config','configs/train_hr_mixed_c128.json',
        '--gpus','0,1,2,3','--global-batch-size','8','--sample-batch-size','2',
        '--questions-per-sample-batch','1','--train-mode','mixed','--eval-mode','joint',
        '--fieldwise-weight','1','--joint-weight','3'])
    cfg,gpus=effective_config(args)
    assert len(gpus)==4 and cfg['training']['c1_batch_size']==2
    assert cfg['training']['batch_records']==8 and cfg['training']['question_batch_size']==2
    assert task_weights(cfg)=={'fieldwise':.25,'joint':.75}
    assert joint_evaluation(cfg)
    for mode in ('fieldwise','joint'):
        args.train_mode=mode;cfg,_=effective_config(args)
        assert task_weights(cfg)=={mode:1.}


def test_zero_family_and_invalid_mode_or_weights():
    cfg=explicit_cfg();row=load_split(cfg,'train',training=True)[0]
    # Original task collections remain available in the loaded row.
    for zero,expected in [('fieldwise_weight','joint_json'),('joint_weight','fieldwise_readout')]:
        other=copy.deepcopy(cfg);other['readout'][zero]=0
        assert all(q['prediction_kind']==expected for q in select_questions(row,other,True))
    for values in (dict(fieldwise_weight=0,joint_weight=0),dict(joint_weight=-1),
                   dict(fieldwise_weight=float('nan')),dict(train_mode='invalid')):
        other=copy.deepcopy(cfg);other['readout'].update(values)
        with pytest.raises(ValueError):validate_config(other)
