import copy

import pytest
import torch
from torch.autograd import gradcheck

from kv_cache_readout.backend import Backend,Context
from kv_cache_readout.native_model import EXECUTION,FAST_EXECUTION
from kv_cache_readout.fast_ops import AnalyticRMS,AnalyticAddRMS,PackedAttention,PageWorkspace
from kv_cache_readout.native_ops import norm_reference,add_norm_reference,attention_reference
from kv_cache_readout.native_blend import native_cacheblend
from kv_cache_readout.optim import parameters_for
from kv_cache_readout.training import batch_gradients
from test_native_autograd import native_backend


def test_analytic_norm_gradchecks():
    torch.manual_seed(311)
    x=torch.randn(2,3,8,dtype=torch.double,requires_grad=True)
    r=torch.randn_like(x,requires_grad=True)
    w=torch.randn(8,dtype=torch.double,requires_grad=True)
    assert gradcheck(lambda x,w:AnalyticRMS.apply(x,w,1e-6),(x,w),fast_mode=True)
    assert gradcheck(lambda x,r,w:AnalyticAddRMS.apply(x,r,w,1e-6),(x,r,w),fast_mode=True)


def test_ragged_attention_gradcheck_and_sequence_isolation():
    torch.manual_seed(312)
    q=torch.randn(1,4,5,4,dtype=torch.double,requires_grad=True)
    k=torch.randn(1,2,12,4,dtype=torch.double,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    workspace=PageWorkspace()
    operation=lambda q,k,v:PackedAttention.apply(q,k,v,.5,(2,3),(5,7),16,2,workspace)
    assert gradcheck(operation,(q,k,v),fast_mode=True)
    output=operation(q,k,v)
    dk,dv=torch.autograd.grad(output[:,:,:2].sum(),(k,v))
    assert dk[:,:,5:].count_nonzero()==0 and dv[:,:,5:].count_nonzero()==0
    changed=v.detach().clone(); changed[:,:,5:]+=100
    assert torch.equal(operation(q,k,changed)[:,:,:2],output[:,:,:2])


def test_ragged_attention_analytic_backward_matches_full_reference_across_tiles():
    torch.manual_seed(313)
    q=torch.randn(1,4,69,4,requires_grad=True)
    k=torch.randn(1,2,82,4,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    reference=torch.cat((attention_reference(q[:,:,:65],k[:,:,:72],v[:,:,:72],.5),
        attention_reference(q[:,:,65:],k[:,:,72:],v[:,:,72:],.5)),dim=2)
    actual=PackedAttention.apply(q,k,v,.5,(65,4),(72,10),16,2,PageWorkspace())
    grad=torch.randn_like(actual)
    a=torch.autograd.grad(actual,(q,k,v),grad)
    b=torch.autograd.grad(reference,(q,k,v),grad)
    assert torch.equal(actual,reference)
    assert all(torch.allclose(x,y,atol=3e-5,rtol=3e-4) for x,y in zip(a,b))


def test_page_scratch_reuses_memory_and_metadata_without_saving_a_graph():
    pool=PageWorkspace()
    key=torch.randn(12,2,4,requires_grad=True)
    value=torch.randn_like(key,requires_grad=True)
    k,v=pool.pack(key,value,(5,7),16)
    pointer=k.data_ptr()
    assert not k.requires_grad and not v.requires_grad
    assert torch.equal(k[0,:5],key[:5]) and torch.equal(k[1,:7],key[5:])
    changed=key.detach()+1
    k2,_=pool.pack(changed,value,(5,7),16)
    assert k2.data_ptr()==pointer and pool.allocations==1
    assert torch.equal(k2[0,:5],changed[:5])
    a=pool.layout((2,3),(5,7),16,key.device)
    b=pool.layout((2,3),(5,7),16,key.device)
    assert a is b and a['cu_seqlens_q'].tolist()==[0,2,5]
    assert a['block_table'].tolist()==[[0],[1]] and a['seqused_k'].tolist()==[5,7]


@pytest.fixture
def fast_pair(native_backend):
    strict,records,prompt=native_backend
    cfg=copy.deepcopy(strict.cfg); cfg['execution']=dict(FAST_EXECUTION)
    fast=Backend(copy.deepcopy(strict.model),strict.tokenizer,cfg)
    # Unequal question and answer lengths exercise ragged masking and CE offsets.
    records=copy.deepcopy(records)
    records[0]['qa'][0]['answer']='Manager'
    records[1]['qa'][1]['answer']='AB'
    return strict,fast,records,prompt


def test_writer_and_c1_values_selection_are_unchanged(fast_pair):
    strict,fast,records,prompt=fast_pair
    with torch.no_grad():
        wa,_=strict.writer_cache(records[0],prompt)
        wb,_=fast.writer_cache(records[0],prompt)
        assert all(torch.equal(a,b) for pair,other in zip(wa,wb) for a,b in zip(pair,other))
        a=strict.blend_many(records,prompt)
        b=fast.blend_many(records,prompt)
    for ca,cb in zip(a,b):
        assert ca.audit['selected_indices']==cb.audit['selected_indices']
        assert all(torch.equal(x,y) for pair,other in zip(ca.layers,cb.layers) for x,y in zip(pair,other))


def test_question_batch_is_one_packed_forward_and_matches_serial_loss(fast_pair,monkeypatch):
    strict,fast,records,prompt=fast_pair
    with torch.no_grad():
        contexts=fast.blend_many(records,prompt)
    calls=[]; original=fast.native.forward_batch
    def observe(model,embeddings,past,lengths,starts):
        calls.append((embeddings.shape,tuple(lengths),tuple(starts)))
        return original(model,embeddings,past,lengths,starts)
    monkeypatch.setattr(fast.native,'forward_batch',observe)
    fast.cfg['training']['checkpoint_questions']=False
    group=[(0,0,records[0]['qa'][0]),(1,1,records[1]['qa'][1])]
    with torch.no_grad():
        actual=fast._question_wave(contexts,group)
        expected=torch.stack([torch.stack(strict.answer_loss_components(contexts[owner],qa))
                              for owner,_,qa in group])
    assert len(calls)==1 and calls[0][0][1]==sum(calls[0][1])
    assert calls[0][1][0]!=calls[0][1][1] and calls[0][2]==(56,56)
    assert torch.allclose(actual,expected,atol=2e-5,rtol=2e-5)


def test_fast_loss_gradients_batching_and_checkpoint_agree(fast_pair):
    strict,fast,records,prompt=fast_pair
    parameters=parameters_for(prompt,fast.cfg)
    gb,lb,flags=batch_gradients(fast,parameters,records,2,0)
    fast.cfg['training'].update(question_batch_size=1,checkpoint_questions=False)
    gs,ls,_=batch_gradients(fast,parameters,records,2,0)
    gr,lr,_=batch_gradients(strict,parameters,records,2,0)
    assert flags==[True,True,True] and torch.isfinite(gb).all()
    assert all(part.norm()>0 for part in gb.split([32,16,16]))
    assert abs(lb-ls)<2e-5 and abs(lb-lr)<2e-5
    assert torch.allclose(gb,gs,atol=3e-5,rtol=3e-4)
    assert torch.allclose(gb,gr,atol=3e-5,rtol=3e-4)
    assert all(p.grad is None for p in fast.model.parameters())


def test_parallel_teacher_forcing_does_not_expose_future_answer_tokens(fast_pair):
    _,fast,records,prompt=fast_pair
    with torch.no_grad():
        context=fast.blend_many(records,prompt)[0]
        prefix=fast.delimiter_ids+fast.question_and_target(records[0]['qa'][0])[0]
        target=fast.question_and_target(records[0]['qa'][0])[1]
        changed=list(target); changed[-2]=ord('Z')
        a,_=fast._parallel_teacher_batch([context],[(prefix,target)])
        b,_=fast._parallel_teacher_batch([context],[(prefix,changed)])
    assert torch.equal(a[:,0],b[:,0])


def test_checkpoint_is_valid_after_rope_cache_grows(fast_pair):
    _,fast,records,prompt=fast_pair
    prompt=prompt.clone().requires_grad_()
    contexts=fast.blend_many(records,prompt)
    group=[(0,0,records[0]['qa'][0]),(1,1,records[1]['qa'][1])]
    values=fast._question_wave(contexts,group)
    fast.native.table(fast.model.model,fast.embedding.weight,4096)
    grad,=torch.autograd.grad(values.sum(),prompt)
    assert torch.isfinite(grad).all() and grad.norm()>0


def test_fast_training_resume_matches_continuous_updates(fast_pair,tmp_path,monkeypatch):
    from kv_cache_readout import training
    from kv_cache_readout.io import read_checkpoint
    _,fast,records,prompt=fast_pair
    cfg=copy.deepcopy(fast.cfg); cfg['training'].update(epochs=2,batch_records=2)
    template=copy.deepcopy(fast.model)
    monkeypatch.setattr(Backend,'initial_prompt',lambda self:prompt.clone())
    monkeypatch.setattr(training.Backend,'load',lambda config:Backend(copy.deepcopy(template),fast.tokenizer,config))
    monkeypatch.setattr(training,'load_split',lambda *args,**kwargs:records)
    monkeypatch.setattr(training,'dataset_identity',lambda cfg:{'fixture':'fast-frozen'})
    cfg['output_directory']=str(tmp_path/'continuous')
    training.train(cfg)
    a=read_checkpoint(tmp_path/'continuous/latest.pt')
    cfg['output_directory']=str(tmp_path/'resumed')
    training.train(cfg,max_updates=1)
    training.train(cfg,resume=True)
    b=read_checkpoint(tmp_path/'resumed/latest.pt')
    assert torch.equal(a['prompt'],b['prompt']) and a['scheduler']==b['scheduler']
    for first,second in zip(a['optimizer']['state'].values(),b['optimizer']['state'].values()):
        assert all(torch.equal(first[key],second[key]) for key in first)


def test_fast_backward_does_not_call_nested_autograd(monkeypatch):
    x=torch.randn(2,8,requires_grad=True); r=torch.randn_like(x,requires_grad=True)
    w=torch.ones(8)
    norm,carry=AnalyticAddRMS.apply(x,r,w,1e-6)
    q=torch.randn(1,4,3,4,requires_grad=True)
    k=torch.randn(1,2,7,4,requires_grad=True); v=torch.randn_like(k,requires_grad=True)
    output=PackedAttention.apply(q,k,v,.5,(3,),(7,),16,2,PageWorkspace())
    def forbidden(*args,**kwargs):
        raise AssertionError('Nested autograd.grad in the optimized backward')
    monkeypatch.setattr(torch.autograd,'grad',forbidden)
    (norm.sum()+carry.sum()+output.sum()).backward()
    assert x.grad is not None and q.grad is not None and k.grad is not None




@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires the pinned server CUDA environment')
def test_cuda_fast_preserves_writer_and_c1_forward():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from kv_cache_readout.native_model import NativeExecution
    torch.manual_seed(315)
    cfg=Qwen3Config(vocab_size=128,hidden_size=64,intermediate_size=128,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=32)
    model=Qwen3ForCausalLM(cfg).to(device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    strict=NativeExecution(model,EXECUTION); fast=NativeExecution(model,FAST_EXECUTION)
    inputs=torch.randn(1,128,64,device='cuda',dtype=torch.bfloat16)
    public=inputs[:,32:88].clone()
    outputs=[]
    with torch.no_grad():
        for execution in (strict,fast):
            model._readout_execution=execution
            model.model._readout_execution=execution
            _,layers=execution.forward(model,inputs)
            writer=tuple(tuple(t[:,:,32:88].clone() for t in pair) for pair in layers)
            result=native_cacheblend(model,public,writer,check_layer=0,ratio=.3,writer_position_start=32)
            outputs.append((writer,result))
    a,b=outputs
    assert all(torch.equal(x,y) for pair,other in zip(a[0],b[0]) for x,y in zip(pair,other))
    assert torch.equal(a[1].selected,b[1].selected)
    assert all(torch.equal(x,y) for pair,other in zip(a[1].layers,b[1].layers) for x,y in zip(pair,other))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires the pinned server CUDA environment')
def test_cuda_ragged_gradients_survive_workspace_overwrite():
    torch.manual_seed(316)
    q=torch.randn(1,4,8,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    k=torch.randn(1,2,19,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    pool=PageWorkspace()
    result=PackedAttention.apply(q,k,v,32**-.5,(3,5),(8,11),16,2,pool)
    before=result.detach().clone()
    PackedAttention.apply(q.detach()+1,k.detach()+1,v.detach()+1,32**-.5,(3,5),(8,11),16,2,pool)
    assert torch.equal(result,before) and pool.allocations==1
    grad=torch.randn_like(result)
    actual=torch.autograd.grad(result,(q,k,v),grad)
    refs=[t.detach().float().cpu().requires_grad_() for t in (q,k,v)]
    qr,kr,vr=refs
    reference=torch.cat((attention_reference(qr[:,:,:3],kr[:,:,:8],vr[:,:,:8],32**-.5),
        attention_reference(qr[:,:,3:],kr[:,:,8:],vr[:,:,8:],32**-.5)),dim=2)
    expected=torch.autograd.grad(reference,refs,grad.float().cpu())
    for a,b in zip(actual,expected):
        assert torch.allclose(a.float().cpu(),b,atol=2e-3,rtol=.02)

