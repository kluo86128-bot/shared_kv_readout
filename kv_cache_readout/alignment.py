"""Optional real-engine diagnostic, separate from training/offline execution."""
import importlib.metadata
import json
import os
from pathlib import Path
from .config import ROOT, resolve, file_hash, fingerprint
from .data import dataset_identity
from .identity import scientific_config, portable_dataset
from .io import code_identity, atomic_json

ATOL = .03
RTOL = .03
GRADIENT_TOLERANCE = dict(atol=3e-4,rtol=.03,relative_l2=.05)


def binding(cfg, world, checkpoint=None, workload_config=None, validation_groups=4):
    from .workload_parity import DEFAULT_WORKLOAD
    template=resolve(workload_config or DEFAULT_WORKLOAD)
    model = resolve(cfg['model_name_or_path'])
    assets = {p.name: file_hash(p) for p in sorted(model.glob('*'))
              if p.is_file() and p.suffix in ('.safetensors','.bin','.json','.model')}
    if not assets:
        raise ValueError(f'Model assets are missing: {model}')
    versions = {name: importlib.metadata.version(name)
                for name in ('torch','transformers','vllm','lmcache')}
    for name,field in (('vllm','vllm_version'),('lmcache','lmcache_version')):
        if versions[name]!=cfg['native_contract'][field]:
            raise ValueError(f'Native contract requires {name}=={cfg["native_contract"][field]}, got {versions[name]}')
    return dict(protocol='aligned-latest-workload-v2', science=scientific_config(cfg),
                sample_batch=cfg['training']['sample_batch_size'],
                question_batch=cfg['training']['questions_per_sample_batch'], world=world,
                model=assets, dataset=portable_dataset(dataset_identity(cfg)),
                code=code_identity(), versions=versions,
                source_checkpoint=file_hash(resolve(checkpoint)) if checkpoint else 'seed-initialization',
                workload=json.loads(template.read_text(encoding='utf-8')),
                workload_sha256=file_hash(template),validation_groups=validation_groups,
                background_sha256=file_hash(resolve(json.loads(template.read_text(encoding='utf-8'))['background']['document_path'])))


def verify_attestation(cfg, path):
    if not path:
        raise ValueError('A diagnostic report is required; run scripts/check_alignment.py explicitly')
    report = json.loads(resolve(path).read_text(encoding='utf-8'))
    if not report.get('passed') or report.get('tolerance') != dict(atol=ATOL,rtol=RTOL):
        raise ValueError('Native alignment gate did not pass with the fixed tolerances')
    if report.get('gradient_tolerance') != GRADIENT_TOLERANCE:
        raise ValueError('Gradient acceptance tolerances differ from the fixed contract')
    expected = binding(cfg,report['binding']['world'],report.get('checkpoint'),
                       report.get('workload_config'),report.get('validation_groups',4))
    if expected != report['binding']:
        raise ValueError('Alignment acceptance no longer matches config, code, model, data or environment')
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) != report['binding']['world']:
        raise ValueError('Alignment acceptance was collected for a different training world size')
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    if not torch.cuda.is_available() or list(torch.cuda.get_device_capability()) != report['compute_capability']:
        raise ValueError('Alignment acceptance does not match the current CUDA architecture')
    for name, sha in report['evidence'].items():
        if file_hash(resolve(name)) != sha:
            raise ValueError(f'Alignment evidence changed: {name}')
    return report


def batch_checks(cfg, source, world,report_path=None):
    """Fixed representative records, exact top-k, numerical KV/logit/gradient checks."""
    import torch
    from .backend import Backend, per_sample_waves
    from .native_model import NativeExecution, EXECUTION
    from .data import load_split
    from .io import read_checkpoint
    from .training import batch_gradients
    from .optim import parameters_for
    from .generation import generate_many
    backend=Backend.load(dict(cfg,device='cuda:0'))
    prompt=read_checkpoint(source)['prompt'].to(backend.device,torch.float32)
    count=min(cfg['training']['sample_batch_size'],
              (cfg['training']['batch_records']+world-1)//world)
    records=load_split(cfg,'train',training=True)[:count]
    questions=[r['qa'][:cfg['training']['questions_per_sample_batch']+i]
               for i,r in enumerate(records)]
    checks=[]
    def close(a,b,label,atol=ATOL,rtol=RTOL):
        ok=torch.allclose(a.float(),b.float(),atol=atol,rtol=rtol)
        delta=torch.where(a==b,torch.zeros_like(a),a.float()-b.float())
        item=dict(label=label,passed=bool(ok),max_abs=float(delta.abs().max()),
                  rmse=float(delta.float().square().mean().sqrt()),
                  failed_elements=int((~torch.isclose(a.float(),b.float(),atol=atol,rtol=rtol)).sum()),
                  elements=a.numel(),atol=atol,rtol=rtol)
        checks.append(item)
        if not ok:
            if report_path:atomic_json(report_path,dict(passed=False,checks=checks,first_failure=item))
            raise ValueError(f'Native batching alignment failed: {label}; max_abs={item["max_abs"]}; '
                             f'RMSE={item["rmse"]}; failed_elements={item["failed_elements"]}/{item["elements"]}')
    fast=backend.native
    def activate(execution):
        backend.native=execution
        backend.model._readout_execution=execution
        backend.model.model._readout_execution=execution
    with torch.no_grad():
        packed=backend.blend_many(records,prompt)
        activate(NativeExecution(backend.model,EXECUTION))
        singles=[backend.blended_context(r,prompt) for r in records]
        for owner,(a,b) in enumerate(zip(packed,singles)):
            if a.audit['selected_indices'] != b.audit['selected_indices']:
                raise ValueError(f'Exact C1 selection differs for representative source {owner}')
            for layer,(akv,bkv) in enumerate(zip(a.layers,b.layers)):
                for part,(x,y) in enumerate(zip(akv,bkv)):
                    close(x,y,f'C1:{owner}:{layer}:{part}')
        for wave in per_sample_waves(questions,cfg['training']['questions_per_sample_batch']):
            expected=[]
            for owner,_,qa in wave:
                q,t=backend.question_and_target(qa)
                expected.append(backend._native_teacher_logits(singles[owner],backend.delimiter_ids+q,t))
            activate(fast)
            if fast.fast:
                pairs=[backend.question_and_target(qa) for _,_,qa in wave]
                actual,sizes=backend._parallel_teacher_batch([packed[o] for o,_,_ in wave],
                    [(backend.delimiter_ids+q,t) for q,t in pairs])
                offset=0
                for index,(target,size) in enumerate(zip(expected,sizes)):
                    close(actual[:,offset:offset+size],target,f'QA-wave-logits:{index}')
                    offset+=size
                generated=generate_many(backend,[packed[o] for o,_,_ in wave],
                                         [qa for _,_,qa in wave],max_new_tokens=4)
                for i,((text,audit),(owner,_,qa)) in enumerate(zip(generated,wave)):
                    single=backend.generate(packed[owner],qa,max_new_tokens=4)
                    if audit['generated_token_ids'] != single[1]['generated_token_ids']:
                        raise ValueError(f'Packed greedy decoding differs at representative question {i}')
            activate(NativeExecution(backend.model,EXECUTION))
    activate(fast)
    # Native inference has no gradients. Compare analytic backward to the FP32
    # mathematical backward using the same frozen weights and loss normalization.
    short=[dict(r,qa=q[:1]) for r,q in zip(records,questions)]
    g1,loss1,_=batch_gradients(backend,parameters_for(prompt,cfg),short,len(short),0)
    activate(NativeExecution(backend.model,EXECUTION))
    old=backend.cfg['training'].copy()
    backend.cfg['training'].update(source_batch_records=1,c1_batch_size=1,question_batch_size=1)
    g2,loss2,_=batch_gradients(backend,parameters_for(prompt,cfg),short,len(short),0)
    backend.cfg['training']=old
    close(g1,g2,'analytic-vs-mathematical-U-gradient',
          atol=GRADIENT_TOLERANCE['atol'],rtol=GRADIENT_TOLERANCE['rtol'])
    reference_norm=float(g2.norm())
    relative_error=float((g1-g2).norm())/max(reference_norm,1e-6)
    gradient_ok=relative_error<=GRADIENT_TOLERANCE['relative_l2']
    checks.append(dict(label='gradient-relative-l2',passed=gradient_ok,value=relative_error))
    if not gradient_ok:raise ValueError('Analytic gradient relative L2 error exceeded 5%')
    close(torch.tensor(loss1),torch.tensor(loss2),'objective')
    return dict(passed=True,records=[r['record_id'] for r in records],checks=checks,
                actual_sample_batch=len(records),questions_per_sample=cfg['training']['questions_per_sample_batch'],
                scope='Representative fixed records only; no assertion of BF16 bitwise equivalence')


def phase(name,cfg,root,world,checkpoint=None,workload_config=None,validation_groups=4):
    """Each phase runs in a fresh process so CUDA engines cannot overlap."""
    root=resolve(root)
    source=root/'source.pt'; oracle_config=root/'oracle.json'
    if name=='bootstrap':
        import torch
        from .backend import Backend
        from .optim import seed_all
        from .io import read_checkpoint,atomic_torch
        seed_all(cfg['seed'])
        backend=Backend.load(dict(cfg,device='cpu'))
        if checkpoint:
            from .identity import resume_compatibility
            state=read_checkpoint(resolve(checkpoint))
            resume_compatibility(state,cfg,dataset_identity(cfg),backend.identity,code_identity())
            initial=state['prompt']
        else:
            initial=backend.initial_prompt().cpu()
        if tuple(initial.shape)!=(cfg['prompt_length'],backend.model.config.hidden_size):
            raise ValueError('Checkpoint prompt shape is incompatible with this model')
        oracle=dict(cfg)
        oracle['native_contract']=dict(cfg['native_contract'])
        if oracle['native_contract']['recompute_ratio']==0:
            oracle['native_contract']['recompute_ratio']=.3
        atomic_json(root/'experiment.json',oracle)
        atomic_torch(source,dict(schema_version='aligned_readout_v3',method='soft',prompt=initial,
                     config=oracle,dataset=dataset_identity(oracle),model=backend.identity,
                     native_contract=oracle['native_contract']))
        from .workload_parity import validation_settings
        settings,metadata=validation_settings(oracle,root,workload_config,validation_groups)
        atomic_json(oracle_config,settings)
        atomic_json(root/'workload_origin.json',metadata)
    elif name=='export':
        from e2e_cache.deployment import prepare
        from e2e_workload.common import load_settings
        prepare(load_settings(oracle_config))
    elif name=='collect':
        from .workload_parity import collect
        collect(oracle_config,root/'trace')
    elif name=='compare':
        from .workload_parity import compare_all
        report=compare_all(root/'trace',atol=ATOL,rtol=RTOL,execution=cfg['execution'])
        if not report['passed']:
            raise ValueError('Latest e2e_workload parity failed or coverage incomplete; see trace/workload_parity_report.json')
    elif name=='batch':
        target=root/'batch_report.json'
        try:
            atomic_json(target,batch_checks(cfg,source,world,report_path=target))
        except Exception as exc:
            if not target.exists():atomic_json(target,dict(passed=False,error=f'{type(exc).__name__}: {exc}'))
            raise
    else:
        raise ValueError(name)
