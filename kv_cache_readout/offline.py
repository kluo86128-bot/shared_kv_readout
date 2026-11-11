"""Question-atomic native offline evaluation, portable GPU repartition on resume."""
import csv
import json
import os
from pathlib import Path
import time
import uuid
import torch

from .backend import Backend,per_sample_waves
from .config import resolve,fingerprint,file_hash
from .data import load_split,dataset_identity
from .identity import scientific_config,portable_dataset,resume_compatibility
from .io import read_checkpoint,atomic_json,code_identity
from .generation import generate_many
from .evaluation import metrics,reconstruction_rows
from .readout import joint_evaluation


def evaluation_identity(cfg,checkpoint,split,limit):
    if limit is not None and limit < 1:raise ValueError('--limit must be positive')
    return dict(schema='aligned_offline_v1',checkpoint=file_hash(resolve(checkpoint)),
                science=scientific_config(cfg),dataset=portable_dataset(dataset_identity(cfg)),
                code=code_identity(),split=split,limit=limit)


def evaluation_directory(cfg,checkpoint,split,limit):
    key=fingerprint(evaluation_identity(cfg,checkpoint,split,limit))[:20]
    return resolve(cfg['output_directory'])/('offline_'+split+'_'+key)


def task_id(record, index, qa):
    return fingerprint(dict(record_id=record['record_id'],index=index,qa=qa))


def committed(path,identity,task):
    if not path.exists():return None
    item=json.loads(path.read_text(encoding='utf-8'))
    if item.get('identity')!=identity or item.get('task_id')!=task:
        raise ValueError(f'Committed answer identity mismatch: {path}')
    row=item['row']
    if item.get('row_sha256')!=fingerprint(row):
        raise ValueError(f'Committed answer contents changed: {path}')
    return row


@torch.no_grad()
def evaluate_worker(cfg,checkpoint,rank,world,split,limit):
    ident=evaluation_identity(cfg,checkpoint,split,limit);key=fingerprint(ident)
    output=evaluation_directory(cfg,checkpoint,split,limit)
    items=output/'items';items.mkdir(parents=True,exist_ok=True)
    state=read_checkpoint(resolve(checkpoint))
    records=load_split(cfg,split,training=False)[:limit]
    # Partition the immutable full record list before checking completion files.
    # Startup skew between workers can never duplicate or drop pending tasks.
    assigned=records[rank::world]
    work=[]
    for record in assigned:
        pending=[(i,qa) for i,qa in enumerate(record['qa'])
                 if committed(items/(task_id(record,i,qa)+'.json'),key,task_id(record,i,qa)) is None]
        if pending:work.append((record,pending))
    if not work:return
    backend=Backend.load(dict(cfg,device='cuda:0'))
    # Evaluation can change compute/rate/layer via its own explicitly identified condition.
    resume_compatibility(state,cfg,dataset_identity(cfg),backend.identity,code_identity())
    prompt=state['prompt'].to(backend.device,torch.float32)
    cap=cfg['training']['sample_batch_size'];qcap=cfg['training']['questions_per_sample_batch']
    atomic_json(output/f'worker_{rank}_runtime.json',dict(rank=rank,world=world,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),sample_batch_size=cap,
                questions_per_sample_batch=qcap,
                execution_policy='aligned_native_operators_without_inference_engine'))
    for start in range(0,len(work),cap):
        if (output/'pause.request').exists():return
        group=work[start:start+cap]
        contexts=backend.blend_many([r for r,_ in group],prompt)
        questions=[[q for _,q in pending] for _,pending in group]
        for wave in per_sample_waves(questions,qcap):
            if (output/'pause.request').exists():return
            started=time.monotonic()
            predictions=generate_many(backend,[contexts[o] for o,_,_ in wave],[q for _,_,q in wave])
            # Joint training returns U-routed objectives; reported NLL must remain
            # the ordinary complete-answer CE, not just the U_other objective.
            losses=([backend.answer_loss_components(contexts[o],qa) for o,_,qa in wave]
                    if 'joint_loss' in cfg else backend._question_wave(contexts,wave))
            for (owner,local_index,qa),(prediction,audit),loss in zip(wave,predictions,losses):
                record,pending=group[owner];original_index,_=pending[local_index]
                task=task_id(record,original_index,qa)
                row=dict(record_id=record['record_id'],question_index=original_index,field=qa['field'],
                    group=qa.get('group',qa['field']),question=qa['question'],gold=qa['answer'],
                    prediction=prediction,em=float(prediction.strip()==qa['answer'].strip()),
                    nll=float(loss[0]),prediction_kind=qa.get('prediction_kind','generated'),
                    condition='aligned_native',recompute_ratio=cfg['native_contract']['recompute_ratio'],
                    check_layer=cfg['native_contract']['check_layer'],**audit)
                if 'c1_selection' in cfg:
                    row.update(condition='aligned_native_'+cfg['c1_selection']['method'],selection_method=cfg['c1_selection']['method'],
                               selection_seed=cfg['c1_selection']['seed'],c1_selection_audit=contexts[owner].audit)
                atomic_json(items/(task+'.json'),dict(identity=key,task_id=task,row=row,
                                                        row_sha256=fingerprint(row)))
            print(f'[EVAL] rank={rank} | records={len(group)} | questions={len(wave)} | seconds={time.monotonic()-started:.2f}',flush=True)


def merge(cfg,checkpoint,split,limit):
    ident=evaluation_identity(cfg,checkpoint,split,limit);key=fingerprint(ident)
    output=evaluation_directory(cfg,checkpoint,split,limit);output.mkdir(parents=True,exist_ok=True)
    rows=[];missing=0
    for record in load_split(cfg,split,training=False)[:limit]:
        record_rows=[]
        for index,qa in enumerate(record['qa']):
            task=task_id(record,index,qa)
            row=committed(output/'items'/(task+'.json'),key,task)
            if row is None:missing+=1
            else:record_rows.append(row)
        rows.extend(record_rows)
        # Derived fields are only emitted after every component has been committed.
        if len(record_rows)==len(record['qa']):rows.extend(reconstruction_rows(record,record_rows))
    columns=list(dict.fromkeys(k for row in rows for k in row))
    target=output/'predictions.csv';temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v
                             for k,v in row.items()})
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,target)
    if joint_evaluation(cfg):
        from .joint import score_rows
        summary, fields = score_rows(rows)
        field_path=output/'field_predictions.csv'
        tmp=field_path.with_name(field_path.name+'.'+uuid.uuid4().hex+'.tmp')
        with tmp.open('w',encoding='utf-8-sig',newline='') as stream:
            columns=['record_id','field','gold','prediction','present','prediction_key_present','em','json_valid']
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(fields)
            stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,field_path)
    else:
        summary=metrics(rows) if rows else None
    result=dict(status='paused' if missing else 'complete',missing_questions=missing,
                identity=ident,output=str(output),metrics=summary)
    atomic_json(output/'summary.json',result)
    return result
