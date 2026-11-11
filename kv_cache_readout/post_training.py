"""Checked sequential post-training pipeline using an immutable U snapshot."""
import copy
import json
import shutil
import sys

from .config import ROOT,resolve,file_hash,validate_config
from .io import atomic_json,read_checkpoint


def offline_configuration(cfg,batch_size):
    result=copy.deepcopy(cfg)
    if batch_size is not None:result['training']['sample_batch_size']=batch_size
    validate_config(result)
    return result


def run_joint_e2e(cfg,checkpoint,gpu,template,env,execute):
    if cfg.get('c1_selection',{}).get('method','cacheblend')!='cacheblend':
        raise ValueError('This real-engine E2E pipeline is currently limited to CacheBlend')
    from e2e_workload.common import load_settings
    from e2e_cache.deployment import validate_native_contract
    source=resolve(checkpoint)
    checksum=file_hash(source)
    root=resolve(cfg['output_directory'])/('e2e_joint_'+checksum[:20])
    root.mkdir(parents=True,exist_ok=True)
    snapshot=root/'source.pt'
    if snapshot.exists():
        if file_hash(snapshot)!=checksum:raise ValueError('Existing E2E U snapshot changed')
    else:shutil.copy2(source,snapshot)
    state=read_checkpoint(snapshot)
    experiment=state['config']
    from .identity import scientific_config
    if scientific_config(experiment)!=scientific_config(cfg):
        raise ValueError('E2E source differs from the completed training condition')
    if 'joint_loss' not in experiment:
        raise ValueError('Joint E2E pipeline requires a joint-readout checkpoint')
    if experiment.get('readout',{}).get('eval_mode','joint')!='joint':
        raise ValueError('Joint E2E requires joint evaluation')
    experiment_path=root/'training_snapshot.json'
    atomic_json(experiment_path,experiment)
    settings=load_settings(template)
    settings.update(experiment_config=str(experiment_path),source_u=str(snapshot),
        model_name_or_path=experiment['model_name_or_path'],deployment=str(root/'deployment'),
        output=str(root/'workload'),use_u=True,use_a=False,adapter=None)
    settings['materials'][0]['text']=experiment['public_text']
    settings['cache']['delimiter']=experiment['delimiter']
    settings['recomputation']=dict(method='cacheblend',ratio=experiment['native_contract']['recompute_ratio'],
                                    check_layer=experiment['native_contract']['check_layer'])
    settings['engine'].update(device=str(gpu),max_new_tokens=experiment['generation']['max_new_tokens'])
    if settings['reader_schedule']['fields']!=['all_fields']:
        raise ValueError('Joint E2E template must use all_fields Readers')
    validate_native_contract(settings,experiment)
    config_path=root/'workload_config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8'))!=settings:
        raise ValueError('Existing E2E workload configuration changed')
    atomic_json(config_path,settings)
    load_settings(config_path)
    marker=root/'pipeline_state.json'
    worker_env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONHASHSEED='0',PYTHONUNBUFFERED='1')
    def call(action):
        atomic_json(marker,dict(status='running',stage=action,config=str(config_path),gpu=str(gpu)))
        execute([sys.executable,'-m','e2e_workload',action,'--config',str(config_path)],worker_env)
    try:
        if not (root/'deployment/e2e_manifest.json').exists():call('prepare')
        call('preflight')
        workload_state=root/'workload/state.json'
        status=json.loads(workload_state.read_text(encoding='utf-8')) if workload_state.exists() else None
        # Completed runs also verify commits; execute() returns before engine
        # creation when the complete immutable plan has already been committed.
        call('resume' if status else 'run')
        status=json.loads(workload_state.read_text(encoding='utf-8'))
        if status.get('status')!='complete':
            atomic_json(marker,dict(status='paused',stage='workload',config=str(config_path)))
            return dict(status='paused',output=str(root/'workload'))
        atomic_json(marker,dict(status='complete',config=str(config_path),checkpoint_sha256=checksum))
        return dict(status='complete',output=str(root/'workload'))
    except BaseException as error:
        atomic_json(marker,dict(status='failed',error=str(error),config=str(config_path)))
        raise


def run_followups(cfg,config_path,checkpoint,ids,env,args,evaluate,execute):
    marker=resolve(cfg['output_directory'])/'post_training_state.json'
    try:
        atomic_json(marker,dict(status='running',stage='offline'))
        evaluated=offline_configuration(cfg,getattr(args,'offline_sample_batch_size',None))
        offline_path=config_path.with_name(config_path.stem+'_offline.json')
        atomic_json(offline_path,evaluated)
        result=evaluate(evaluated,offline_path,checkpoint,ids,env,'test',None)
        if result['status']!='complete':
            atomic_json(marker,dict(status='paused',stage='offline',output=result.get('output')))
            return
        if getattr(args,'then_e2e',False):
            atomic_json(marker,dict(status='running',stage='e2e'))
            result=run_joint_e2e(cfg,checkpoint,args.e2e_gpu or ids[0],args.e2e_template,env,execute)
            if result['status']!='complete':
                atomic_json(marker,dict(status='paused',stage='e2e',output=result['output']))
                return
        atomic_json(marker,dict(status='complete',last_stage='e2e' if getattr(args,'then_e2e',False) else 'offline',
                                output=result.get('output')))
    except BaseException as error:
        atomic_json(marker,dict(status='failed',error=str(error)))
        raise
