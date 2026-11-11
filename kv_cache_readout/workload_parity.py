"""Trace the actual latest workload, including four Readers and cache provenance."""
from copy import deepcopy
import json
import os
from pathlib import Path
import time

from .config import ROOT,resolve,file_hash,fingerprint
from .io import atomic_json

DEFAULT_WORKLOAD='configs/e2e_workload_validation.json'


def validation_settings(cfg,root,template=None,groups=4):
    """Preserve numerical/cache/request policies; reduce only group counts."""
    from e2e_workload.common import load_settings
    original=load_settings(resolve(template or DEFAULT_WORKLOAD))
    settings=deepcopy(original)
    gaps=settings['reader_schedule']['gaps']
    active=[i for i,g in enumerate(gaps) if g['rounds']]
    if type(groups) is not int or not len(active)<=groups<=original['rounds']:
        raise ValueError(f'validation-groups must be between {len(active)} and {original["rounds"]}')
    counts=[int(i in active) for i in range(len(gaps))]
    while sum(counts)<groups:
        for i in active:
            if counts[i]<gaps[i]['rounds']:
                counts[i]+=1
                if sum(counts)==groups:break
    for gap,count in zip(gaps,counts):gap['rounds']=count
    root=resolve(root)
    settings.update(experiment_config=str(root/'experiment.json'),source_u=str(root/'source.pt'),
        deployment=str(ROOT/'deployments'/('alignment_'+root.name)),output=str(root/'workload'),
        model_name_or_path=cfg['model_name_or_path'],rounds=groups,
        seed=cfg['seed'],materials=[dict(id='c1',text=cfg['public_text'],weight=1)])
    settings['recomputation']=dict(method='cacheblend',ratio=cfg['native_contract']['recompute_ratio'],
                                   check_layer=cfg['native_contract']['check_layer'])
    settings['cache']['delimiter']=cfg['delimiter']
    # The latest Runtime writes this value back to CUDA_VISIBLE_DEVICES, so
    # preserve the physical ID supplied to the isolated diagnostic child.
    settings['engine']['device']=(os.environ.get('CUDA_VISIBLE_DEVICES') or '0').split(',')[0]
    if len(settings['reader_schedule']['fields'])!=4 or settings['reader_schedule']['timing']!='read_then_cooldown':
        raise ValueError('Latest parity requires four Readers with read_then_cooldown timing')
    return settings,dict(template_sha256=file_hash(resolve(template or DEFAULT_WORKLOAD)),
                         production_rounds=original['rounds'],validation_groups=groups)


def resolve_writer(report,registry,records,history,qa_field,question):
    """Never substitute the designated Reader target for direct cache ownership."""
    origins=report['direct_sources']
    ids={s['request_id'] for s in origins}
    if len(ids)!=1 or any(s['role']!='writer' for s in origins):
        return None,None,'direct_source_not_one_writer'
    source_id=next(iter(ids));event=registry[source_id]
    record=records[event['record_id']]
    source=history.get(source_id)
    if source is None:return None,None,'writer_request_evidence_missing'
    mask=source['response']['rows'][0]['audit'].get('retrieved_mask')
    start,end=source['request']['public_start'],source['request']['public_end']
    if not isinstance(mask,list) or len(mask)<end:
        audit=source['response']['rows'][0]['audit']
        if not (audit.get('complete') and audit.get('store_calls',0)>0
                and audit.get('retrieve_calls')==0 and not audit.get('hit_blocks')
                and not audit.get('loaded_keys')):
            return None,None,'writer_c1_retrieval_mask_missing'
    elif any(mask):
        return None,None,('writer_itself_reused_c1' if any(mask[start:end])
                          else 'writer_itself_reused_other_tokens')
    qa=next((q for q in record['qa'] if q['field']==qa_field),None)
    if qa is None:return None,None,'direct_writer_field_missing'
    return record,dict(qa,question=question),'cold_c1_writer'


def collect(config,output):
    from e2e_workload.common import load_settings,schedule,ordered_events,prepare_documents
    from e2e_workload.runtime import Runtime
    from e2e_workload.cache_report import reader_cache_report,log_reader_cache
    from e2e_cache.deployment import verify
    from e2e_cache.__main__ import load_inputs
    cfg=load_settings(resolve(config));manifest=verify(cfg)
    records=load_inputs(cfg,manifest);by_id={r['record_id']:r for r in records}
    runtime=Runtime(cfg,manifest)
    documents=prepare_documents(cfg,runtime.tokenizer)
    plan=schedule(cfg,records)
    registry={e['id']:e for group in plan for e in ordered_events(group)}
    root=resolve(output);root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'schedule.json',plan)
    atomic_json(root/'workload_config.json',cfg)
    material=cfg['materials'][0];history={};readers=[];last=None
    common=dict(source_u=cfg['source_u'],source_u_sha256=manifest['source_u_sha256'],
                experiment=manifest['experiment'],recomputation=cfg['recomputation'],
                model_name_or_path=cfg['model_name_or_path'],runtime='e2e_workload',
                native_rope_fix=cfg['native_rope_fix'])
    for group in plan:
        events=[]
        for event in ordered_events(group):
            delay=cfg['reader_schedule']['submit_delay_seconds']
            if last is not None and delay:time.sleep(max(0,last+delay-time.monotonic()))
            record=by_id[event['record_id']] if event['role']=='writer' else None
            document=documents[event['document_index']] if event['role']=='background' else material
            request=runtime.request(event,record,document)
            folder=None
            if event['role']=='reader':
                folder=root/f'group_{group["index"]:04d}'/event['field']
                folder.mkdir(parents=True)
                runtime.rpc('e2e_trace',str(folder/'native'))
            response=runtime.generate([request]);last=time.monotonic()
            events.append(dict(event=event,request=request,response=response))
            print(f'[WORKLOAD_PARITY] group={group["index"]} | role={event["role"]} | id={event["id"]}',flush=True)
            if event['role']=='writer':history[event['id']]=events[-1]
            if folder is None:continue
            audit=response['rows'][0]['audit']
            cache=reader_cache_report(request,audit,registry);log_reader_cache(event,cache)
            actual,qa,reason=resolve_writer(cache,registry,by_id,history,event['field'],event['question'])
            usable=cache['only_c1_hit'] and cache['audit_complete']
            inputs=dict(common,reader=request,record=actual,qa=qa,
                cache_report=cache,source_reference_status=reason,
                writer_reference_eligible=bool(actual is not None),
                designated_target_record=event['source_record'],response=response,
                comparison_eligible=bool(usable),field=event['field'],question=event['question'],
                generation_limit=cfg['engine']['max_new_tokens'])
            atomic_json(folder/'inputs.json',inputs)
            readers.append(dict(directory=str(folder),field=event['field'],group=group['index'],
                comparison_eligible=bool(usable),writer_reference_eligible=bool(actual is not None),
                cache_report=cache,source_reference_status=reason))
        atomic_json(root/f'group_{group["index"]:04d}'/'events.json',events)
    result=dict(schema='latest_workload_trace_v1',runtime='e2e_workload',
                worker='e2e_workload.worker.Extension',native_rope_fix=cfg['native_rope_fix'],
                groups=len(plan),readers=readers,fields=cfg['reader_schedule']['fields'])
    atomic_json(root/'index.json',result)
    return result


def compare_all(directory,execution,atol=.03,rtol=.03):
    from .parity import compare
    root=resolve(directory);index=json.loads((root/'index.json').read_text(encoding='utf-8'))
    rows=[]
    for reader in index['readers']:
        row=dict(reader)
        if not reader['comparison_eligible']:
            row.update(passed=None,status='not_verified',reason='requires_audited_C1_only_hit')
        else:
            try:
                report=compare(reader['directory'],execution=execution,atol=atol,rtol=rtol)
                row.update(passed=report['passed'],status='passed' if report['passed'] else 'failed',
                           writer_verified=report['writer_verified'])
            except Exception as exc:
                row.update(passed=False,status='failed',error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    covered={r['field'] for r in rows if r.get('passed')}
    # Skipped hits never count as a pass. Require each question field and at
    # least one verified cold Writer -> Reader chain, not just native KV replay.
    passed=(set(index['fields'])<=covered and not any(r.get('passed') is False for r in rows)
            and any(r.get('writer_verified') and r.get('passed') for r in rows))
    result=dict(passed=passed,runtime='e2e_workload',native_rope_fix=index['native_rope_fix'],
                groups=index['groups'],covered_fields=sorted(covered),readers=rows,
                atol=atol,rtol=rtol,scope='Latest workload Runtime/Worker, four Readers, direct-source-aware KV/logit parity')
    atomic_json(root/'workload_parity_report.json',result)
    return result
