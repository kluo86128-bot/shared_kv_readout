"""Explicit training task families, shared field losses, and independent evaluation mode."""
import math
from .joint import FIELDS, options

LABELS = dict(name='complete name', identity_number='identity number', address='residential address',
              email='email address', birth_date='complete birth date', position='job position',
              department='department', salary='salary', leave_duration='leave duration',
              leave_reason='leave reason')


def field_questions(record):
    result = []
    for field in FIELDS:
        if field not in record['fields']:
            continue
        annotation = record['field_annotations'][field]
        locator = annotation.get('locator', 'the employee record')
        result.append(dict(field=field, group=field,
            question=f"In {locator}, what is the target employee's recorded {LABELS[field]}? Return only the complete value exactly as recorded.",
            answer=record['fields'][field], evidence_span=annotation['evidence_span'],
            template_id=field+':mixed_complete_v1', prediction_kind='fieldwise_readout'))
    return result


def validate_readout(value):
    if set(value) != {'train_mode','eval_mode','fieldwise_weight','joint_weight'}:
        raise ValueError('readout requires train_mode, eval_mode and both task weights')
    if value['train_mode'] not in ('fieldwise','joint','mixed') or value['eval_mode'] not in ('fieldwise','joint'):
        raise ValueError('Invalid readout mode')
    for key in ('fieldwise_weight','joint_weight'):
        number=value[key]
        if type(number) not in (int,float) or not math.isfinite(number) or number < 0:
            raise ValueError('Task weights must be finite and nonnegative')
    if value['train_mode']=='mixed' and value['fieldwise_weight']+value['joint_weight']<=0:
        raise ValueError('Mixed task weights cannot both be zero')


def task_weights(cfg):
    value=cfg['readout'];mode=value['train_mode']
    if mode!='mixed':return {mode:1.}
    total=value['fieldwise_weight']+value['joint_weight']
    return {key:weight/total for key,weight in
            [('fieldwise',value['fieldwise_weight']),('joint',value['joint_weight'])] if weight>0}


def joint_evaluation(cfg):
    return cfg['readout']['eval_mode']=='joint' if 'readout' in cfg else 'joint_loss' in cfg


def select_questions(record,cfg,training):
    fields=record.get('fieldwise_qa');joint=record.get('joint_qa')
    if not fields or not joint or len(joint)!=1 or joint[0].get('prediction_kind')!='joint_json':
        raise ValueError('Explicit readout modes require fieldwise_qa and one joint_qa')
    if len(fields)!=len(record['fields']) or {q['field'] for q in fields}!=set(record['fields']):
        raise ValueError('Fieldwise tasks differ from the record fields')
    if any(q.get('prediction_kind')!='fieldwise_readout' or q['answer']!=record['fields'][q['field']] for q in fields):
        raise ValueError('Fieldwise target mismatch')
    if training:
        weights=task_weights(cfg)
        return (fields if 'fieldwise' in weights else [])+(joint if 'joint' in weights else [])
    return joint if joint_evaluation(cfg) else fields


def reduce_fieldwise(backend,qa,ce):
    import torch
    opt=options(backend.cfg['joint_loss'],qa['field'])
    loss=ce
    if opt['type']=='focal':
        loss=opt['alpha']*(-torch.expm1(-ce)).clamp_min(torch.finfo(ce.dtype).tiny).pow(opt['gamma'])*ce
    zero=ce.sum()*0
    pair=[zero,zero]
    if qa['field']=='birth_date' and opt.get('reduction')=='ymd_mean':
        from .joint import date_component_owners
        groups=torch.tensor(date_component_owners(backend,qa),device=ce.device)
        pair[int(opt['u_group']=='focal')]=torch.stack([loss[groups==i].mean() for i in range(3)]).mean()*opt['weight']
        fmt=options(backend.cfg['joint_loss'],'format');extra=ce[groups<0]
        if extra.numel():
            if fmt['type']=='focal':
                extra=fmt['alpha']*(-torch.expm1(-extra)).clamp_min(torch.finfo(extra.dtype).tiny).pow(fmt['gamma'])*extra
            route=int(fmt['u_group']=='focal')
            pair[route]=pair[route]+extra.mean()*fmt['weight']*opt['weight']
    else:
        pair[int(opt['u_group']=='focal')]=loss.mean()*opt['weight']
    return torch.stack(pair)


def combine_tasks(cfg,questions,values):
    """Normalize fields within a family, then task families within each sample."""
    from .joint import active_groups
    weights=task_weights(cfg)
    output=[values[0][0]*0,values[0][1]*0];routes=set()
    field_rows=[];joint_rows=[]
    for qa,value in zip(questions,values):
        if qa.get('prediction_kind')=='joint_json':joint_rows.append((qa,value))
        elif qa.get('prediction_kind')=='fieldwise_readout':field_rows.append((qa,value))
        else:raise ValueError('Unknown explicit readout task')
    if bool(field_rows)!=('fieldwise' in weights) or bool(joint_rows)!=('joint' in weights) or len(joint_rows)>1:
        raise ValueError('Training tasks do not match selected mode/weights')
    if field_rows:
        if len({qa['field'] for qa,_ in field_rows})!=len(field_rows):
            raise ValueError('Duplicate fieldwise training task')
        denominator=sum(options(cfg['joint_loss'],qa['field'])['weight'] for qa,_ in field_rows)
        for qa,value in field_rows:
            opt=options(cfg['joint_loss'],qa['field'])
            for i in range(2):output[i]=output[i]+value[i]*weights['fieldwise']/(denominator or 1.)
            if opt['weight']>0:
                routes.add(opt['u_group'])
                if qa['field']=='birth_date' and opt.get('reduction')=='ymd_mean':
                    fmt=options(cfg['joint_loss'],'format')
                    if fmt['weight']>0:routes.add(fmt['u_group'])
    for qa,value in joint_rows:
        for i in range(2):output[i]=output[i]+value[i]*weights['joint']
        routes.update(active_groups(qa,cfg['joint_loss']))
    return output,routes
