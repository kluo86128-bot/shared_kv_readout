"""Native inference forwards with explicit mathematical training backwards.

vLLM/LMCache do not expose training gradients. These bridges keep their CUDA
forward values untouched and differentiate the corresponding smooth equations
in FP32 (FP64 for CPU gradcheck). Rounding and hard selection are not themselves
differentiated. No installed inference package is patched.
"""
from functools import lru_cache
import importlib.metadata

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


def work(x):
    return x if x.dtype==torch.float64 else x.float()


def norm_reference(x, weight, eps):
    y=work(x)
    return (y*torch.rsqrt(y.square().mean(-1,keepdim=True)+eps)).to(x.dtype)*weight


def add_norm_reference(x, residual, weight, eps):
    total=work(x)+work(residual)
    normalized=total*torch.rsqrt(total.square().mean(-1,keepdim=True)+eps)
    return normalized.to(x.dtype)*weight,total.to(x.dtype)


def rotate(x, cos, sin):
    a,b=work(x).chunk(2,dim=-1)
    return torch.cat((a*work(cos)-b*work(sin),b*work(cos)+a*work(sin)),dim=-1).to(x.dtype)


def attention_reference(q,k,v,scale,start=0,end=None):
    """Grouped-query, bottom-right causal attention, optionally a query tile."""
    end=q.shape[2] if end is None else end
    groups=q.shape[1]//k.shape[1]
    key=work(k).repeat_interleave(groups,dim=1)
    value=work(v).repeat_interleave(groups,dim=1)
    scores=(work(q[:,:,start:end])@key.transpose(-1,-2))*scale
    rows=torch.arange(start,end,device=q.device)+k.shape[2]-q.shape[2]
    allowed=torch.arange(k.shape[2],device=q.device)[None]<=rows[:,None]
    scores=scores.masked_fill(~allowed[None,None],float('-inf'))
    return (scores.softmax(-1)@value).to(q.dtype)


def vjp(function, tensors, gradients):
    with torch.enable_grad():
        inputs=[x.detach().requires_grad_(True) for x in tensors]
        output=function(*inputs)
        outputs=output if isinstance(output,tuple) else (output,)
        grads=[torch.zeros_like(y) if g is None else g for y,g in zip(outputs,gradients)]
        return torch.autograd.grad(outputs,inputs,grads,create_graph=False)


@lru_cache(maxsize=1)
def cuda_kernels():
    for name,version in (('vllm','0.18.0'),('lmcache','0.4.2')):
        if importlib.metadata.version(name)!=version:
            raise ValueError(f'Native training requires {name}=={version}')
    from vllm import _custom_ops
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    import lmcache.c_ops
    return _custom_ops,flash_attn_varlen_func,lmcache.c_ops


class RMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,eps):
        ctx.save_for_backward(x,weight); ctx.eps=eps
        if not x.is_cuda:
            return norm_reference(x,weight,eps)
        out=torch.empty_like(x,memory_format=torch.contiguous_format)
        cuda_kernels()[0].rms_norm(out,x,weight,eps)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        return (*vjp(lambda x,w:norm_reference(x,w,ctx.eps),ctx.saved_tensors,(grad,)),None)


class AddRMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,residual,weight,eps):
        ctx.save_for_backward(x,residual,weight); ctx.eps=eps
        if not x.is_cuda:
            return add_norm_reference(x,residual,weight,eps)
        shape=x.shape
        out=x.reshape(-1,shape[-1]).contiguous().clone()
        carried=residual.reshape_as(out).contiguous().clone()
        cuda_kernels()[0].fused_add_rms_norm(out,carried,weight,eps)
        return out.reshape(shape),carried.reshape(shape)

    @staticmethod
    @once_differentiable
    def backward(ctx,grad,grad_residual):
        return (*vjp(lambda x,r,w:add_norm_reference(x,r,w,ctx.eps),ctx.saved_tensors,
                     (grad,grad_residual)),None)


class SiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        if not x.is_cuda:
            a,b=work(x).chunk(2,dim=-1)
            return (F.silu(a)*b).to(x.dtype)
        cuda_kernels()
        out=x.new_empty((*x.shape[:-1],x.shape[-1]//2))
        torch.ops._C.silu_and_mul(out,x)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        x,=ctx.saved_tensors
        a,b=work(x).chunk(2,dim=-1)
        sig=a.sigmoid(); g=work(grad)
        return torch.cat((g*b*sig*(1+a*(1-sig)),g*a*sig),dim=-1).to(x.dtype)


class Rotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,positions,cache):
        ctx.save_for_backward(positions,cache); ctx.shape=x.shape
        if not x.is_cuda:
            cos,sin=cache[positions].chunk(2,dim=-1)
            return rotate(x,cos[None,:,None],sin[None,:,None])
        dim=x.shape[-1]
        out=x.reshape(len(positions),-1).contiguous().clone()
        cuda_kernels()[0].rotary_embedding(positions,out,None,dim,cache,True)
        return out.reshape(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        positions,cache=ctx.saved_tensors
        cos,sin=cache[positions].chunk(2,dim=-1)
        return rotate(grad,cos[None,:,None],-sin[None,:,None]),None,None


class Relocate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,key,old,new,cache):
        ctx.save_for_backward(old,new,cache)
        ctx.shape=key.shape
        if not key.is_cuda:
            return Relocate.reference(key,old,new,cache)
        out=key.contiguous().clone()
        cuda_kernels()[2].rotary_embedding_k_fused(old,new,out,key.shape[-1],cache,True)
        return out

    @staticmethod
    def reference(key,old,new,cache):
        oc,os=work(cache[old]).chunk(2,dim=-1)
        nc,ns=work(cache[new]).chunk(2,dim=-1)
        return rotate(key,(oc*nc+os*ns)[:,None],(ns*oc-nc*os)[:,None])

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        old,new,cache=ctx.saved_tensors
        return Relocate.reference(grad,new,old,cache),None,None,None


class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,scale,page_size,fa_version):
        ctx.save_for_backward(q,k,v); ctx.scale=scale
        if not q.is_cuda:
            return attention_reference(q,k,v,scale)
        query,key,value=(t[0].transpose(0,1) for t in (q,k,v))
        count=len(key)
        output=torch.empty_like(query,memory_format=torch.contiguous_format)
        if page_size:
            pages=(count+page_size-1)//page_size
            cache=key.new_zeros((2,pages,page_size,*key.shape[1:]))
            cache[0].view(-1,*key.shape[1:])[:count].copy_(key)
            cache[1].view(-1,*value.shape[1:])[:count].copy_(value)
            key,value=cache.unbind(0)
            layout=dict(block_table=torch.arange(pages,device=q.device,dtype=torch.int32)[None],
                        seqused_k=torch.tensor([count],device=q.device,dtype=torch.int32))
        else:
            layout=dict(cu_seqlens_k=torch.tensor([0,count],device=q.device,dtype=torch.int32))
        cuda_kernels()[1](q=query,k=key,v=value,out=output,
            cu_seqlens_q=torch.tensor([0,len(query)],device=q.device,dtype=torch.int32),
            max_seqlen_q=len(query),max_seqlen_k=count,softmax_scale=scale,
            causal=True,window_size=(-1,-1),softcap=0.0,alibi_slopes=None,
            fa_version=fa_version,**layout)
        return output.transpose(0,1)[None]

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        q,k,v=ctx.saved_tensors
        # Recompute only one probability tile at a time, not an entire writer
        # attention matrix for every layer. Accumulate shared GQA KV gradients
        # in FP32 before casting once to the corresponding input dtype.
        dq,dk,dv=(torch.zeros_like(work(t)) for t in (q,k,v))
        for start in range(0,q.shape[2],64):
            end=min(start+64,q.shape[2])
            with torch.enable_grad():
                a,b,c=(work(t).detach().requires_grad_(True) for t in (q,k,v))
                result=attention_reference(a,b,c,ctx.scale,start,end)
                grads=torch.autograd.grad(result,(a,b,c),work(grad[:,:,start:end]))
            dq+=grads[0]; dk+=grads[1]; dv+=grads[2]
        return dq.to(q.dtype),dk.to(k.dtype),dv.to(v.dtype),None,None,None
