import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.autograd import gradcheck

from kv_cache_readout import native_ops as ops
from kv_cache_readout.native_model import EXECUTION,FAST_EXECUTION,NativeExecution,WRITER_QUESTION
from kv_cache_readout.backend import Backend,Context
from kv_cache_readout.config import load_config
from kv_cache_readout.optim import parameters_for
from kv_cache_readout.training import batch_gradients
from test_training_v2 import Tokenizer
from transformers import Qwen3Config,Qwen3ForCausalLM


def test_native_bridge_gradchecks():
    torch.set_num_threads(1)
    torch.manual_seed(43)
    x=torch.randn(2,8,dtype=torch.double,requires_grad=True)
    w=torch.randn(8,dtype=torch.double,requires_grad=True)
    r=torch.randn_like(x,requires_grad=True)
    assert gradcheck(lambda x,w:ops.RMS.apply(x,w,1e-6),(x,w),fast_mode=True)
    assert gradcheck(lambda x,r,w:ops.AddRMS.apply(x,r,w,1e-6),(x,r,w),fast_mode=True)
    assert gradcheck(ops.SiLU.apply,(x,),fast_mode=True)
    phase=torch.randn(12,4,dtype=torch.double)
    table=torch.cat((phase.cos(),phase.sin()),dim=-1)
    positions=torch.tensor([0,3,7])
    q=torch.randn(1,3,2,8,dtype=torch.double,requires_grad=True)
    assert gradcheck(lambda q:ops.Rotary.apply(q,positions,table),(q,),fast_mode=True)
    key=torch.randn(3,2,8,dtype=torch.double,requires_grad=True)
    assert gradcheck(lambda k:ops.Relocate.apply(k,positions,positions+2,table),(key,),fast_mode=True)


@pytest.mark.parametrize('page_size',[0,16])
def test_attention_gqa_sparse_causal_gradcheck(page_size):
    torch.manual_seed(23)
    q=torch.randn(1,4,3,4,dtype=torch.double,requires_grad=True)
    k=torch.randn(1,2,7,4,dtype=torch.double,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    assert gradcheck(lambda q,k,v:ops.Attention.apply(q,k,v,.5,page_size,2),(q,k,v),fast_mode=True)


def test_attention_backward_tiles_match_dense_reference():
    torch.manual_seed(45)
    xs=[torch.randn(1,4,65,4,requires_grad=True),torch.randn(1,2,70,4,requires_grad=True),
        torch.randn(1,2,70,4,requires_grad=True)]
    output=ops.Attention.apply(*xs,.5,16,2)
    grad=torch.randn_like(output)
    actual=torch.autograd.grad(output,xs,grad)
    reference=torch.autograd.grad(ops.attention_reference(*xs,.5),xs,grad)
    assert all(torch.allclose(a,b,atol=2e-5,rtol=2e-4) for a,b in zip(actual,reference))


@pytest.fixture(params=[0,1])
def native_backend(request):
    torch.set_num_threads(1); torch.manual_seed(17)
    cfg=load_config(Path(__file__).resolve().parent/'fixtures/base.json')
    cfg.update(device='cpu',dtype='float32',public_text='x'*56,question_format='plain',
        model_name_or_path='missing-tiny-assets',execution=dict(EXECUTION))
    cfg['native_contract']['check_layer']=request.param
    cfg['training'].update(source_batch_records=2,question_batch_size=2,preservation_batch_records=2)
    model_cfg=Qwen3Config(vocab_size=128,hidden_size=16,intermediate_size=24,
        num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=2048,eos_token_id=0)
    backend=Backend(Qwen3ForCausalLM(model_cfg),Tokenizer(),cfg)
    records=[dict(record_id=f'n{i}',private_prefix='private'+(' more'*i),split='train',
        qa=[dict(field='title',group='title',question='title?',answer='M'),
            dict(field='email',group='email',question='email?',answer='A')],
        _suffix_fixture=dict(ids=[65+i,66,67,68],text='ABCD',suffix_id=f's{i}',article_id='a'))
        for i in range(2)]
    return backend,records,torch.randn(64,16)*.02


def test_native_writer_full_request_and_single_source_paths_agree(native_backend,monkeypatch):
    backend,records,prompt=native_backend
    counts=[]
    original=backend.native.forward
    def observe(model,embeddings,**kwargs):
        counts.append(embeddings.shape)
        return original(model,embeddings,**kwargs)
    monkeypatch.setattr(backend.native,'forward',observe)
    with torch.no_grad():
        many,suffixes=backend.source_many(records,prompt)
        single,audit=backend.writer_cache(records[0],prompt)
    assert all(shape[0]==1 for shape in counts)
    expected=suffixes[0].start+suffixes[0].length+len(backend.ids(backend.question_text(WRITER_QUESTION)))
    assert counts[0][1]==expected and audit['writer_c1_end']<expected
    assert all(torch.equal(a,b) for pair,other in zip(many[0],single) for a,b in zip(pair,other))
    changed=prompt.clone(); changed[-16:]+=1
    with torch.no_grad():
        other,_=backend.writer_cache(records[0],changed)
    assert all(torch.equal(a,b) for pair,changed in zip(single,other) for a,b in zip(pair,changed))


def test_native_training_all_prompt_segments_get_gradients_and_checkpoint_matches(native_backend):
    backend,records,prompt=native_backend
    parameters=parameters_for(prompt,backend.cfg)
    actual,loss,flags=batch_gradients(backend,parameters,records,2,0)
    backend.cfg['training'].update(source_batch_records=1,preservation_batch_records=1,
                                    question_batch_size=1,checkpoint_questions=False)
    reference,ref_loss,ref_flags=batch_gradients(backend,parameters,records,2,0)
    assert flags==ref_flags==[True,True,True]
    assert abs(loss-ref_loss)<2e-5
    assert torch.allclose(actual,reference,atol=3e-5,rtol=3e-4)
    assert torch.isfinite(actual).all() and all(part.norm()>0 for part in actual.split([32,16,16]))
    assert all(p.grad is None for p in backend.model.parameters())


def test_native_teacher_forcing_preserves_prefill_and_single_token_shapes(native_backend,monkeypatch):
    backend,records,prompt=native_backend
    with torch.no_grad():
        context=backend.blended_context(records[0],prompt)
    counts=[]
    original=backend._append
    def observe(context,ids):
        counts.append(len(ids))
        return original(context,ids)
    monkeypatch.setattr(backend,'_append',observe)
    question,target=backend.question_and_target(records[0]['qa'][0])
    prefix=backend.delimiter_ids+question
    with torch.no_grad():
        logits=backend._native_teacher_logits(context,prefix,target)
    assert counts==[len(prefix)]+[1]*(len(target)-1)
    assert logits.shape==(1,len(target),128)
    backend.input_only_ids=(126,127)
    hidden=torch.randn(1,2,16,requires_grad=True)
    masked=backend.output_logits(hidden)
    assert torch.isneginf(masked[...,126:]).all()
    gradient=torch.autograd.grad(masked[...,:126].sum(),hidden)[0]
    assert torch.isfinite(gradient).all() and gradient.norm()>0




def test_native_training_resume_matches_continuous_optimizer_updates(native_backend,tmp_path,monkeypatch):
    from kv_cache_readout import training
    from kv_cache_readout.io import read_checkpoint
    backend,records,initial=native_backend
    cfg=copy.deepcopy(backend.cfg)
    cfg['training'].update(epochs=2,batch_records=2)
    template=copy.deepcopy(backend.model)
    monkeypatch.setattr(Backend,'initial_prompt',lambda self:initial.clone())
    monkeypatch.setattr(training.Backend,'load',lambda config:Backend(copy.deepcopy(template),Tokenizer(),config))
    monkeypatch.setattr(training,'load_split',lambda *args,**kwargs:records)
    monkeypatch.setattr(training,'dataset_identity',lambda cfg:{'fixture':'native-frozen'})
    cfg['output_directory']=str(tmp_path/'continuous')
    training.train(cfg)
    continuous=read_checkpoint(tmp_path/'continuous/latest.pt')
    cfg['output_directory']=str(tmp_path/'resumed')
    training.train(cfg,max_updates=1)
    training.train(cfg,resume=True)
    resumed=read_checkpoint(tmp_path/'resumed/latest.pt')
    assert torch.equal(continuous['prompt'],resumed['prompt'])
    assert continuous['scheduler']==resumed['scheduler']
    for a,b in zip(continuous['optimizer']['state'].values(),resumed['optimizer']['state'].values()):
        assert all(torch.equal(a[key],b[key]) for key in a)


def test_production_parity_includes_writer_raw_logits_and_u_gradient(native_backend,tmp_path,monkeypatch):
    import json
    from kv_cache_readout import parity,config
    from kv_cache_readout.native_blend import native_cacheblend,relocate_keys
    backend,records,prompt=native_backend
    monkeypatch.setattr(config,'ROOT',tmp_path)
    monkeypatch.setattr(parity.Backend,'load',lambda cfg:backend)
    source=tmp_path/'u.pt'
    torch.save(dict(prompt=prompt,config=backend.cfg),source)
    qa=records[0]['qa'][0]
    tokens=backend.public_ids+backend.delimiter_ids+backend.ids(backend.question_text(qa['question']))
    inputs=dict(experiment=backend.cfg,record=records[0],qa=qa,reader=dict(tokens=tokens),
        source_u=str(source),source_u_sha256=config.file_hash(source))
    deployment=tmp_path/'deployments/tiny'
    deployment.mkdir(parents=True)
    manifest=dict(source_u_sha256=inputs['source_u_sha256'],experiment_sha=config.fingerprint(backend.cfg),
        enabled_prompts=dict(u=True,a=False),mapping={'u':{'ids':list(range(128,192))}},original_output_size=128)
    (deployment/'e2e_manifest.json').write_text(json.dumps(manifest))
    parity.align_exported_u(backend,inputs)
    root=tmp_path/'trace'; (root/'native').mkdir(parents=True)
    (root/'inputs.json').write_text(json.dumps(inputs))
    with torch.no_grad():
        writer,audit=backend.writer_cache(records[0],prompt)
        old=relocate_keys(backend.model.model,writer,audit['writer_c1_start'],0)
        trace=[]
        blended=native_cacheblend(backend.model,backend.embedding(backend.tensor(backend.public_ids)),
            old,check_layer=backend.cfg['native_contract']['check_layer'],ratio=.3,trace=trace)
        hidden,_=backend._append(Context(blended.layers,56,{}),tokens[56:])
        torch.save(backend.model.lm_head(hidden[:,-1]),root/'native/first_logits.pt')
    def flat(tensor):
        return tensor[0].transpose(0,1).reshape(tensor.shape[2],-1)
    for row,pair in zip(trace,old):
        value=dict(row,old_key=flat(pair[0]),old_value=flat(pair[1]))
        for name in ('query','attention_key','attention_value','stored_key','stored_value'):
            value[name]=flat(row[name])
        torch.save(value,root/f'native/test_layer_{row["layer"]:03d}.pt')
    result=parity.compare(str(root),execution=EXECUTION,gradient_check=True)
    assert result['passed'] and result['gradient_smoke']['passed']
    assert result['first_logits']['max_abs']==0
    assert all(row['key']['max_abs']==row['value']['max_abs']==0 for row in result['writer'])
    assert result['gradient_smoke']['segment_norms'][2]==0
    assert (root/'parity_report_native_autograd.json').is_file()
    # A latest-workload trace must prove its worker/RoPE policy, and must not
    # invent a Writer when direct cache ownership belongs to a previous Reader.
    from runtime_fixes import identity
    latest=dict(inputs,runtime='e2e_workload',native_rope_fix=True,
                writer_reference_eligible=True,source_reference_status='cold_c1_writer',
                question=qa['question'],generation_limit=3,
                response=dict(rows=[dict(generated_token_ids=backend.generate(
                    Context(blended.layers,56,{}),qa,max_new_tokens=3)[1]['generated_token_ids'])]))
    (root/'inputs.json').write_text(json.dumps(latest))
    (root/'native/runtime_fix.json').write_text(json.dumps(identity()))
    checked=parity.compare(str(root),execution=EXECUTION)
    assert checked['passed'] and checked['writer_verified']
    assert checked['end_to_end_logits']['max_abs']==0 and checked['runtime']=='e2e_workload'
    assert checked['generation']['passed'] and checked['end_to_end_generation']['passed']
    latest.update(record=None,qa=None,writer_reference_eligible=False,
                  source_reference_status='direct_source_not_one_writer')
    (root/'inputs.json').write_text(json.dumps(latest))
    conditional=parity.compare(str(root),execution=EXECUTION)
    assert conditional['passed'] and not conditional['writer_verified']
    assert conditional['writer']==[] and conditional['end_to_end_logits'] is None
    (root/'inputs.json').write_text(json.dumps(inputs))
    bad=torch.load(root/'native/test_layer_001.pt',weights_only=True)
    bad['old_key'][0,0]+=10
    torch.save(bad,root/'native/test_layer_001.pt')
    failed=parity.compare(str(root),execution=EXECUTION)
    assert not failed['passed'] and not failed['writer'][1]['key']['passed']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires the pinned server CUDA environment')
def test_cuda_production_forward_matches_verified_paged_diagnostic_and_has_gradients():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('native_diagnostic',root/'tests/native_decoder_reference.py')
    diagnostic=importlib.util.module_from_spec(spec); spec.loader.exec_module(diagnostic)
    torch.manual_seed(17)
    cfg=Qwen3Config(vocab_size=128,hidden_size=64,intermediate_size=128,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=32)
    model=Qwen3ForCausalLM(cfg).to(device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    execution=NativeExecution(model,EXECUTION)
    embedding=model.get_input_embeddings()
    backend=SimpleNamespace(model=model,embedding=embedding,device=torch.device('cuda'),
        tensor=lambda ids:torch.tensor([ids],device='cuda'))
    layers=tuple((torch.randn(1,1,56,32,device='cuda',dtype=torch.bfloat16).requires_grad_(),
                  torch.randn(1,1,56,32,device='cuda',dtype=torch.bfloat16).requires_grad_()) for _ in range(2))
    with torch.no_grad():
        expected=diagnostic.NativeDecoderReplay().question_hidden(backend,Context(layers,56,{}),
                                                                  [11,12,13,14],page_size=16)
    inputs=embedding(backend.tensor([11,12,13,14])).detach().requires_grad_()
    actual,_=execution.forward(model,inputs,layers=layers,position_start=56)
    assert torch.equal(actual,expected)
    parameters=(inputs,*(t for pair in layers for t in pair))
    grads=torch.autograd.grad(actual.float().square().mean(),parameters)
    assert all(torch.isfinite(g).all() and g.float().norm()>0 for g in grads)

