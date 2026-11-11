"""One machine-independent CLI; only this process owns launch/evaluation locks."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

from .config import ROOT, load_config, resolve, validate_config, fingerprint, file_hash
from .io import atomic_json, read_checkpoint, run_lock
from .identity import scientific_config, condition_id


def parser(action):
    p=argparse.ArgumentParser(description='Aligned native Qwen3 '+action)
    p.add_argument('--config',default='configs/train_hr.json')
    p.add_argument('--gpus',help='Physical CUDA IDs, e.g. 0,1,2,3')
    p.add_argument('--num-gpus',type=int)
    p.add_argument('--model-path');p.add_argument('--dataset-path');p.add_argument('--suffix-path')
    p.add_argument('--output-dir');p.add_argument('--run-name')
    p.add_argument('--compute-mode',choices=('native-reference','native-batched'))
    p.add_argument('--recompute-ratio',type=float);p.add_argument('--check-layer',type=int)
    p.add_argument('--selection-method',choices=('cacheblend','random','epic','none'))
    p.add_argument('--selection-seed',type=int)
    p.add_argument('--sample-batch-size',type=int)
    p.add_argument('--questions-per-sample-batch',type=int)
    p.add_argument('--global-batch-size',type=int)
    p.add_argument('--learning-rate',type=float);p.add_argument('--training-epochs',type=int)
    p.add_argument('--save-every',type=int)
    p.add_argument('--train-mode',choices=('fieldwise','joint','mixed'))
    p.add_argument('--eval-mode',choices=('fieldwise','joint'))
    p.add_argument('--fieldwise-weight',type=float)
    p.add_argument('--joint-weight',type=float)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--resume-from',help='Existing compatible checkpoint; migrated automatically')
    p.add_argument('--dry-run',action='store_true')
    if action=='train':
        p.add_argument('--stop-after-epochs',type=int,default=30)
        p.add_argument('--max-updates',type=int)
        p.add_argument('--then-offline',action='store_true')
        p.add_argument('--then-e2e',action='store_true',help='After successful offline evaluation, export and run real joint E2E')
        p.add_argument('--offline-sample-batch-size',type=int)
        p.add_argument('--e2e-template',default='configs/e2e_workload_joint.json')
        p.add_argument('--e2e-gpu',help='Physical GPU for the single-worker E2E engine; default first training GPU')
    elif action=='evaluate':
        p.add_argument('--checkpoint',required=True)
        p.add_argument('--split',choices=('test','validation'),default='test')
        p.add_argument('--limit',type=int)
        p.add_argument('--max-new-tokens',type=int)
    elif action=='check-alignment':
        p.add_argument('--checkpoint',help='Validate this U; omitted means seed initialization')
        p.add_argument('--workload-config',default='configs/e2e_workload_validation.json')
        p.add_argument('--validation-groups',type=int,default=4,
                       help='Representative groups covering all background buckets; default 4, full workload 100')
    else:
        raise ValueError(f'Unknown action: {action}')
    return p


def effective_config(args):
    cfg=load_config(args.config)
    if getattr(args,'selection_method',None) is not None or getattr(args,'selection_seed',None) is not None:
        selection=dict(cfg.get('c1_selection',dict(method='cacheblend',seed=cfg['seed'])))
        if args.selection_method is not None:selection['method']=args.selection_method
        if args.selection_seed is not None:selection['seed']=args.selection_seed
        cfg['c1_selection']=selection
    readout_overrides={key:getattr(args,arg,None) for arg,key in
                      [('train_mode','train_mode'),('eval_mode','eval_mode'),
                       ('fieldwise_weight','fieldwise_weight'),('joint_weight','joint_weight')]}
    if any(v is not None for v in readout_overrides.values()):
        if 'readout' not in cfg:
            raise ValueError('Readout CLI overrides require an explicit-mode config and dataset')
        cfg['readout'].update({k:v for k,v in readout_overrides.items() if v is not None})
    # New public interface has exactly two batching dimensions. Discard derived fields.
    for key in ('source_batch_records','preservation_batch_records','question_batch_size',
                'c1_batch_size','question_order'):
        cfg['training'].pop(key,None)
    for arg,key in (('model_path','model_name_or_path'),('dataset_path','dataset_directory'),
                    ('output_dir','output_directory')):
        value=getattr(args,arg,None)
        if value is not None:cfg[key]=value
    if args.run_name:
        if not re.fullmatch('[A-Za-z0-9_-]+',args.run_name):raise ValueError('Invalid run name')
        cfg['output_directory']='runs/'+args.run_name
    if args.suffix_path:cfg['scoped_cache']['suffix_directory']=args.suffix_path
    for arg,key in (('sample_batch_size','sample_batch_size'),
                    ('questions_per_sample_batch','questions_per_sample_batch'),
                    ('global_batch_size','batch_records'),('learning_rate','learning_rate'),
                    ('training_epochs','epochs'),('save_every','checkpoint_every_steps')):
        value=getattr(args,arg,None)
        if value is not None:cfg['training'][key]=value
    for arg,key in (('recompute_ratio','recompute_ratio'),('check_layer','check_layer')):
        value=getattr(args,arg,None)
        if value is not None:cfg['native_contract'][key]=value
    if cfg.get('c1_selection',{}).get('method')=='none':
        if args.recompute_ratio not in (None,0):
            raise ValueError('none means zero recomputation; choose a method before setting a positive ratio')
        cfg['native_contract']['recompute_ratio']=0.
    if args.compute_mode:
        from .native_model import EXECUTION,FAST_EXECUTION
        cfg['execution']=dict(EXECUTION if args.compute_mode=='native-reference' else FAST_EXECUTION)
    if getattr(args,'max_new_tokens',None) is not None:
        cfg['generation']['max_new_tokens']=args.max_new_tokens
    cfg['device']='cuda:0'
    validate_config(cfg)
    gpu_text=args.gpus or ','.join(map(str,range(args.num_gpus or 1)))
    ids=gpu_text.split(',')
    if any(not x.isdigit() for x in ids) or len(ids)!=len(set(ids)):
        raise ValueError('--gpus requires distinct physical integer GPU IDs')
    if args.num_gpus is not None and args.num_gpus!=len(ids):
        raise ValueError('--num-gpus and --gpus disagree')
    if not ids:raise ValueError('At least one GPU is required')
    return cfg,ids


def command(arguments,env,timeout=None):
    print('[LAUNCH] '+ ' '.join(map(str,arguments)),flush=True)
    process=subprocess.Popen(list(map(str,arguments)),cwd=ROOT,env=env,
                             start_new_session=os.name!='nt')
    def terminate_group(sig):
        try:
            if os.name!='nt':os.killpg(process.pid,sig)
            elif process.poll() is None:process.kill()
        except ProcessLookupError:pass
    try:
        code=process.wait(timeout=timeout)
        if code:raise subprocess.CalledProcessError(code,arguments)
        # A completed collector must not leave an orphan inference engine
        # holding GPU memory when the next isolated validation phase starts.
        if os.name!='nt':
            terminate_group(signal.SIGTERM)
            deadline=time.monotonic()+1
            while time.monotonic()<deadline:
                try:os.killpg(process.pid,0)
                except ProcessLookupError:break
                time.sleep(.1)
            else:terminate_group(signal.SIGKILL)
    except BaseException:
        terminate_group(signal.SIGTERM)
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:pass
        terminate_group(signal.SIGKILL if os.name!='nt' else signal.SIGTERM)
        process.wait()
        raise


def ensure_alignment(cfg,world,env,checkpoint=None,*,workload_config=None,validation_groups=4):
    from .alignment import binding,verify_attestation,ATOL,RTOL,GRADIENT_TOLERANCE
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Native aligned training requires CUDA')
    capabilities=[torch.cuda.get_device_capability(i) for i in range(world)]
    if len(set(capabilities))!=1:
        raise ValueError('This alignment gate requires GPUs with the same compute capability')
    bound=binding(cfg,world,checkpoint,workload_config,validation_groups)
    key=fingerprint(bound)
    base=ROOT/'alignments'/key
    report_path=base/'acceptance.json'
    if report_path.exists():
        verify_attestation(cfg,report_path)
        return str(report_path)
    attempt=base/('attempt_'+uuid.uuid4().hex[:10]);attempt.mkdir(parents=True)
    config_path=attempt/'effective.json';atomic_json(config_path,cfg)
    one_gpu=dict(env,CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'].split(',')[0])
    for name in ('bootstrap','export','collect','compare','batch'):
        timeout=max(1800,validation_groups*300) if name in ('collect','compare') else 1800
        print(f'[ALIGN] phase={name} | timeout={timeout}s',flush=True)
        call=[sys.executable,str(ROOT/'scripts/_worker.py'),'alignment','--config',str(config_path),
              '--phase',name,'--root',str(attempt),'--world',str(world)]
        call.extend(('--validation-groups',str(validation_groups)))
        if workload_config:call.extend(('--workload-config',str(resolve(workload_config))))
        if checkpoint:call.extend(('--checkpoint',str(resolve(checkpoint))))
        command(call,one_gpu,timeout=timeout)
    evidence={str(p):file_hash(p) for p in sorted(attempt.rglob('*')) if p.is_file()}
    # Pin the hardware class as an additional acceptance fact (GPU IDs may change).
    result=dict(passed=True,binding=bound,checkpoint=str(resolve(checkpoint)) if checkpoint else None,
                tolerance=dict(atol=ATOL,rtol=RTOL),compute_capability=list(capabilities[0]),evidence=evidence,
                gradient_tolerance=GRADIENT_TOLERANCE,
                workload_config=str(resolve(workload_config)) if workload_config else None,
                validation_groups=validation_groups,
                scope='Latest e2e_workload Runtime/Worker and RoPE policy, four Readers, background lifecycle; packed QA/gradient checks',
                zero_ratio_control=cfg['native_contract']['recompute_ratio']==0)
    atomic_json(report_path,result)
    print(f'[ALIGN] passed | acceptance={report_path}',flush=True)
    return str(report_path)


def main(action):
    args=parser(action).parse_args()
    cfg,ids=effective_config(args)
    if cfg.get('c1_selection',{}).get('method','cacheblend')!='cacheblend' and (action=='check-alignment' or getattr(args,'then_e2e',False)):
        raise ValueError('This selector supports native training/offline only; real-engine parity/E2E is currently limited to CacheBlend')
    if action=='train':
        if not 1 <= args.stop_after_epochs <= cfg['training']['epochs']:
            raise ValueError('--stop-after-epochs must be within --training-epochs')
        if args.max_updates is not None and args.max_updates < 1:
            raise ValueError('--max-updates must be positive')
        if args.offline_sample_batch_size is not None and args.offline_sample_batch_size<1:
            raise ValueError('--offline-sample-batch-size must be positive')
        if args.e2e_gpu is not None and args.e2e_gpu not in ids:
            raise ValueError('--e2e-gpu must be one of the selected training GPUs')
        if args.then_e2e:
            from e2e_workload.common import load_settings
            settings=load_settings(args.e2e_template)
            if 'joint_loss' not in cfg or cfg.get('readout',{}).get('eval_mode','joint')!='joint' or settings['reader_schedule']['fields']!=['all_fields']:
                raise ValueError('Automatic joint E2E requires joint evaluation and all_fields template')
    elif action=='evaluate' and args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    checkpoint=args.resume_from if action=='train' else None
    base=resolve(cfg['output_directory'])
    if action=='train' and args.resume and not checkpoint:checkpoint=str(base/'latest.pt')
    if checkpoint:
        checkpoint=str(resolve(checkpoint))
        state=read_checkpoint(checkpoint)
        if scientific_config(state['config'])!=scientific_config(cfg):
            cfg['output_directory']=str(base/'stages'/condition_id(cfg))
            # Resuming an already-created new stage must not restart its parent.
            stage_latest=resolve(cfg['output_directory'])/'latest.pt'
            if stage_latest.exists():checkpoint=str(stage_latest)
    if action in ('evaluate','check-alignment') and args.checkpoint:
        checkpoint=str(resolve(args.checkpoint))
    output=resolve(cfg['output_directory'])
    plan=dict(action=action,config=cfg,gpus=ids,world_size=len(ids),checkpoint=checkpoint,
              per_gpu_records=[len(list(range(cfg['training']['batch_records']))[r::len(ids)])
                               for r in range(len(ids))])
    if action=='train':
        plan['post_training']=dict(offline=args.then_offline or args.then_e2e,e2e=args.then_e2e,
            offline_sample_batch_size=args.offline_sample_batch_size or cfg['training']['sample_batch_size'],
            e2e_gpu=args.e2e_gpu or ids[0],e2e_template=args.e2e_template if args.then_e2e else None)
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    if args.dry_run:return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(ids),PYTHONHASHSEED='0',PYTHONUNBUFFERED='1',
             TORCHELASTIC_SIGNALS_TO_HANDLE='SIGTERM,SIGINT,SIGQUIT')
    os.environ['CUDA_VISIBLE_DEVICES']=env['CUDA_VISIBLE_DEVICES']
    with run_lock(output/'.launcher'):
        launches=output/'launches';launches.mkdir(parents=True,exist_ok=True)
        stamp=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        config_path=launches/(stamp+'.json');atomic_json(config_path,cfg)
        atomic_json(launches/(stamp+'_launch.json'),plan)
        if action=='train' and not checkpoint and (output/'latest.pt').exists():
            raise ValueError('Existing run found; use --resume')
        if action=='check-alignment':
            report=ensure_alignment(cfg,len(ids),env,checkpoint,
                                    workload_config=args.workload_config,validation_groups=args.validation_groups)
            print(f'[ALIGNMENT_REPORT] {report}',flush=True)
        elif action=='train':
            call=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(len(ids)),
                  str(ROOT/'scripts/_worker.py'),'train','--config',str(config_path),
                  '--stop-after-epochs',str(args.stop_after_epochs)]
            if checkpoint:call.extend(('--checkpoint',checkpoint))
            if args.max_updates is not None:call.extend(('--max-updates',str(args.max_updates)))
            command(call,env)
            result=json.loads((output/'summary.json').read_text(encoding='utf-8'))
            if (args.then_offline or args.then_e2e) and result['status'] in ('stage_reached','complete'):
                checkpoint=str(output/'latest.pt')
                from .post_training import run_followups
                run_followups(cfg,config_path,checkpoint,ids,env,args,evaluate_parallel,command)
        else:
            evaluate_parallel(cfg,config_path,checkpoint,ids,env,args.split,args.limit)


def evaluate_parallel(cfg,config_path,checkpoint,ids,env,split,limit):
    from .offline import evaluation_directory,merge
    output=evaluation_directory(cfg,checkpoint,split,limit)
    with run_lock(output):
        (output/'pause.request').unlink(missing_ok=True)
        processes=[]
        try:
            for rank,gpu in enumerate(ids):
                call=[sys.executable,str(ROOT/'scripts/_worker.py'),'evaluate','--config',str(config_path),
                      '--checkpoint',checkpoint,'--rank',str(rank),
                      '--world',str(len(ids)),'--split',split]
                if limit is not None:call.extend(('--limit',str(limit)))
                processes.append(subprocess.Popen(call,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=gpu)))
            # Check every worker, including failures while another GPU is still busy.
            while any(p.poll() is None for p in processes):
                failed=[p.returncode for p in processes if p.poll() not in (None,0)]
                if failed:raise RuntimeError(f'Offline evaluation worker failed: {failed}')
                time.sleep(.2)
            if any(p.returncode for p in processes):raise RuntimeError('Offline evaluation failed')
        finally:
            for process in processes:
                if process.poll() is None:process.terminate()
            for process in processes:process.wait()
        result=merge(cfg,checkpoint,split,limit)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        return result
