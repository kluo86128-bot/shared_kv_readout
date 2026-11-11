"""Collect actual LMCache tensors and compare surrogate KV, indices and logits."""
import json
from pathlib import Path
import torch

from .config import resolve
from .io import read_checkpoint, atomic_json
from .backend import Backend, Context
from .native_blend import native_cacheblend, relocate_keys


def collect(e2e_config, output):
    from .workload_parity import collect as latest_collect
    return latest_collect(e2e_config,output)


def compare(trace_directory, *, atol=.03, rtol=.03, execution=None, gradient_check=False):
    root = resolve(trace_directory)
    inputs = json.loads((root / 'inputs.json').read_text(encoding='utf-8'))
    if inputs.get('runtime')=='e2e_workload' and inputs.get('native_rope_fix'):
        from runtime_fixes import identity
        policy=json.loads((root/'native/runtime_fix.json').read_text(encoding='utf-8'))
        if policy!=identity():
            raise ValueError('Trace was not captured with the latest authoritative RoPE source-position policy')
    cfg = dict(inputs['experiment'])
    if execution is not None:
        cfg['execution']=dict(execution)
    cfg['model_name_or_path'] = inputs.get('model_name_or_path', cfg['model_name_or_path'])
    backend = Backend.load(cfg)
    export_alignment=None
    if backend.native is not None:
        export_alignment=align_exported_u(backend,inputs)
    records = [read_checkpoint(path) for path in sorted((root / 'native').glob('*_layer_*.pt'))]
    if len(records) != len(backend.model.model.layers):
        raise ValueError('Incomplete native layer trace')
    expected = backend.public_ids
    actual = inputs['reader']['tokens'][:len(expected)]
    if actual != expected or records[0]['old_key'].shape[0] != len(expected):
        raise ValueError('Native blend span differs from the expected C1-only span')
    head_dim = backend.model.config.head_dim
    def hf(tensor):
        return tensor.to(backend.device).reshape(tensor.shape[0], -1, head_dim).transpose(0,1)[None]
    old_layers = tuple((hf(r['old_key']), hf(r['old_value'])) for r in records)
    trace = []
    with torch.no_grad():
        from .config import file_hash
        if file_hash(resolve(inputs['source_u'])) != inputs['source_u_sha256']:
            raise ValueError('Parity source U checkpoint changed')
        source = read_checkpoint(resolve(inputs['source_u']))
        prompt = source['prompt'].to(backend.device)
        writer_eligible=inputs.get('writer_reference_eligible',True)
        expected_writer=None
        if writer_eligible:
            writer_layers, writer_audit = backend.writer_cache(inputs['record'], prompt)
            expected_writer = relocate_keys(backend.model.model, writer_layers, writer_audit['writer_c1_start'], 0)
        recomputation = inputs.get('recomputation', {
            'check_layer': cfg['native_contract']['check_layer'],
            'ratio': cfg['native_contract']['recompute_ratio']})
        result = native_cacheblend(backend.model, backend.embedding(backend.tensor(expected)), old_layers,
                                  check_layer=recomputation['check_layer'],
                                  ratio=recomputation['ratio'], trace=trace)
        context = Context(result.layers, len(expected), {})
        hidden, _ = backend._append(context, inputs['reader']['tokens'][len(expected):])
        logits = backend.model.lm_head(hidden[:, -1]).float().cpu()
        own_logits=None
        if expected_writer is not None:
            own=native_cacheblend(backend.model,backend.embedding(backend.tensor(expected)),expected_writer,
                                 check_layer=recomputation['check_layer'],ratio=recomputation['ratio'])
            own_hidden,_=backend._append(Context(own.layers,len(expected),{}),inputs['reader']['tokens'][len(expected):])
            own_logits=backend.model.lm_head(own_hidden[:,-1]).float().cpu()
    report = []
    def flat(tensor):
        return tensor[0].transpose(0,1).reshape(tensor.shape[2], -1).float()
    def error(a,b):
        a,b = a.float().cpu(), b.float().cpu()
        if a.shape != b.shape:
            return dict(passed=False, shape_a=list(a.shape), shape_b=list(b.shape))
        return dict(passed=bool(torch.allclose(a,b,atol=atol,rtol=rtol)),
                    max_abs=float((a-b).abs().max()), rmse=float((a-b).square().mean().sqrt()))
    for surrogate, native in zip(trace, records):
        row = dict(layer=native['layer'], indices_match=surrogate['selected_indices']==native['selected_indices'])
        for key in ('query', 'attention_key', 'attention_value', 'stored_key', 'stored_value'):
            row[key] = error(flat(surrogate[key]), native[key])
        if surrogate['scores'] is not None:
            row['scores'] = error(surrogate['scores'], native['scores'])
        report.append(row)
    writer_report = ([{'layer': i, 'key': error(flat(pair[0]), records[i]['old_key']),
                      'value': error(flat(pair[1]), records[i]['old_value'])} for i,pair in enumerate(expected_writer)]
                     if expected_writer is not None else [])
    native_logits_path = root / 'native' / 'first_logits.pt'
    if not native_logits_path.exists():
        raise ValueError('Native first-step logits trace is missing')
    native_logits = torch.load(native_logits_path, map_location='cpu', weights_only=True)
    native_logits = native_logits.reshape(-1, native_logits.shape[-1])[:1, :logits.shape[-1]]
    logits_report = error(logits, native_logits)
    end_to_end_logits=error(own_logits,native_logits) if own_logits is not None else None
    generation=None
    own_generation=None
    if inputs.get('runtime')=='e2e_workload':
        native_ids=inputs['response']['rows'][0]['generated_token_ids']
        def normalized(ids):
            end=next((i for i,t in enumerate(ids) if t in backend.stops),len(ids))
            return ids[:end]
        native_ids=normalized(native_ids)
        def generated(context):
            text,audit=backend.generate(context,dict(question=inputs['question']),
                                         max_new_tokens=inputs['generation_limit'])
            actual=normalized(audit['generated_token_ids'])
            mismatch=next((i for i,(a,b) in enumerate(zip(actual,native_ids)) if a!=b),None)
            if mismatch is None and len(actual)!=len(native_ids):mismatch=min(len(actual),len(native_ids))
            return dict(passed=actual==native_ids,first_mismatch=mismatch,
                        training_tokens=actual,native_tokens=native_ids,training_text=text)
        generation=generated(context)
        if expected_writer is not None:own_generation=generated(Context(own.layers,len(expected),{}))
    passed = logits_report['passed'] and all(row['indices_match'] and all(v['passed'] for v in row.values() if isinstance(v,dict)) for row in report)
    passed = passed and all(r['key']['passed'] and r['value']['passed'] for r in writer_report)
    passed=passed and (end_to_end_logits is None or end_to_end_logits['passed'])
    passed=passed and (generation is None or generation['passed']) and (own_generation is None or own_generation['passed'])
    result = dict(passed=passed, atol=atol, rtol=rtol, layers=report, first_logits=logits_report,
                  writer=writer_report,end_to_end_logits=end_to_end_logits,
                  generation=generation,end_to_end_generation=own_generation,
                  writer_verified=bool(writer_report) and all(r['key']['passed'] and r['value']['passed'] for r in writer_report),
                  source_reference_status=inputs.get('source_reference_status','legacy_single_writer'),
                  runtime=inputs.get('runtime','legacy_e2e_cache_trace'),
                  scope='Actual loaded C1 replay, native blend indices/KV, question first logits; Writer checked only against verified cold direct source')
    if backend.native is not None:
        result['execution']=cfg['execution']
        result['export_alignment']=export_alignment
        result['trace_execution_override']=inputs['experiment'].get('execution')!=cfg['execution']
    if gradient_check:
        if not writer_eligible:
            raise ValueError('Writer-to-U gradient check requires a verified cold direct Writer')
        if backend.native is None:
            raise ValueError('--gradient-check requires the native_autograd_v1 execution profile')
        with torch.enable_grad():
            prompt=source['prompt'].to(backend.device,torch.float32).requires_grad_(True)
            context=backend.blended_context(inputs['record'],prompt)
            question=inputs['reader']['tokens'][len(expected):]
            hidden,_=backend._append(context,question)
            target=backend.question_and_target(inputs['qa'])[1][0]
            loss=torch.nn.functional.cross_entropy(backend.output_logits(hidden[:,-1]),
                torch.tensor([target],device=backend.device))
            gradient,=torch.autograd.grad(loss,prompt)
        split=cfg['prompt_partition']
        segments=gradient.split((split['other_length'],split['focal_length'],split['keep_length']))
        norms=[float(part.norm()) for part in segments]
        ok=bool(torch.isfinite(gradient).all()) and all(value>0 for value in norms[:2]) and norms[2]==0
        result['gradient_smoke']=dict(passed=ok,loss=float(loss.detach()),segment_norms=norms,
            keep_expected_zero=True,scope='Reader CE to writer U; S preservation gradients covered by training tests',
            model_weights_have_grad=any(p.grad is not None for p in backend.model.parameters()))
        result['passed']=result['passed'] and ok and not result['gradient_smoke']['model_weights_have_grad']
    atomic_json(root / ('parity_report_native_autograd.json' if backend.native is not None else 'parity_report.json'), result)
    return result


def align_exported_u(backend,inputs):
    """Mirror actual export, including tied output rows, for raw-logit parity."""
    from .config import ROOT,file_hash,fingerprint
    candidates=[]
    for path in (ROOT/'deployments').rglob('e2e_manifest.json'):
        value=json.loads(path.read_text(encoding='utf-8'))
        if (value.get('source_u_sha256')==inputs['source_u_sha256']
                and value.get('experiment_sha')==fingerprint(inputs['experiment'])):
            candidates.append((path,value))
    if not candidates:
        raise ValueError('No matching exported deployment for native raw-logit comparison')
    if len({json.dumps(value['mapping'],sort_keys=True) for _,value in candidates})!=1:
        raise ValueError('Matching deployments disagree on their dedicated U mapping')
    path,manifest=candidates[0]
    if not manifest.get('enabled_prompts',{}).get('u') or manifest['enabled_prompts'].get('a'):
        raise ValueError('Native parity requires deployed U and no reader adapter A')
    source_path=resolve(inputs['source_u'])
    if file_hash(source_path)!=inputs['source_u_sha256']:
        raise ValueError('Parity source U checkpoint changed')
    source=read_checkpoint(source_path)
    ids=manifest['mapping']['u']['ids']
    if set(ids)&set(inputs['reader']['tokens']):
        raise ValueError('Reader contains an input-only U token')
    required=max(manifest['original_output_size'],max(ids)+1)
    if backend.embedding.weight.shape[0]<required:
        backend.model.resize_token_embeddings(required,mean_resizing=False)
        backend.embedding=backend.model.get_input_embeddings()
    with torch.no_grad():
        backend.embedding.weight[ids]=source['prompt'].to(backend.device,backend.dtype)
    return dict(manifest=str(path),input_rows=len(ids),
        tied_input_output_weights=backend.embedding.weight.data_ptr()==backend.model.lm_head.weight.data_ptr())
