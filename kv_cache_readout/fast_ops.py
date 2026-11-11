"""Ragged native attention, reusable page scratch, and analytic backwards.

Only immutable logical Q/K/V are saved for backward. Paged buffers are scratch
on the current CUDA stream, and are never autograd inputs or saved tensors.
"""
from collections import OrderedDict

import torch
from torch.autograd.function import once_differentiable

from .native_ops import RMS,AddRMS,work,attention_reference,cuda_kernels


def norm_grad(x,weight,eps,grad,need_weight=True):
    value=work(x)
    inv=torch.rsqrt(value.square().mean(-1,keepdim=True)+eps)
    if grad is None:
        return torch.zeros_like(value),None
    g=work(grad); scaled=g*work(weight)
    dx=inv*(scaled-value*(scaled*value).mean(-1,keepdim=True)*inv.square())
    dw=(g*value*inv).sum(tuple(range(value.ndim-1))) if need_weight else None
    return dx,dw


class AnalyticRMS(RMS):
    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        x,weight=ctx.saved_tensors
        dx,dw=norm_grad(x,weight,ctx.eps,grad,ctx.needs_input_grad[1])
        return dx.to(x.dtype),dw.to(weight.dtype) if ctx.needs_input_grad[1] and dw is not None else None,None


class AnalyticAddRMS(AddRMS):
    @staticmethod
    @once_differentiable
    def backward(ctx,grad,grad_residual):
        x,residual,weight=ctx.saved_tensors
        dx,dw=norm_grad(work(x)+work(residual),weight,ctx.eps,grad,ctx.needs_input_grad[2])
        if grad_residual is not None:
            dx=dx+work(grad_residual)
        return (dx.to(x.dtype),dx.to(residual.dtype),
                dw.to(weight.dtype) if ctx.needs_input_grad[2] and dw is not None else None,None)


def validate_lengths(q,k,v,qlens,klens):
    if (q.ndim!=4 or q.shape[0]!=1 or k.ndim!=4 or k.shape!=v.shape or k.shape[0]!=1
            or not qlens or len(qlens)!=len(klens) or sum(qlens)!=q.shape[2]
            or sum(klens)!=k.shape[2] or q.shape[1]%k.shape[1]
            or any(a<1 or a>b for a,b in zip(qlens,klens))):
        raise ValueError('Invalid packed attention boundaries or GQA shape')


class PageWorkspace:
    def __init__(self):
        self.buffers={}
        self.layouts=OrderedDict()
        self.allocations=0

    def layout(self,qlens,klens,page_size,device):
        stream=torch.cuda.current_stream(device).cuda_stream if device.type=='cuda' else None
        key=(qlens,klens,page_size,device,stream)
        if key in self.layouts:
            self.layouts.move_to_end(key)
            return self.layouts[key]
        cuq=[0]; cuk=[0]
        for q,k in zip(qlens,klens):
            cuq.append(cuq[-1]+q); cuk.append(cuk[-1]+k)
        result=dict(cu_seqlens_q=torch.tensor(cuq,device=device,dtype=torch.int32),
                    max_seqlen_q=max(qlens),max_seqlen_k=max(klens))
        if page_size:
            counts=[(n+page_size-1)//page_size for n in klens]
            table=torch.zeros((len(klens),max(counts)),dtype=torch.int32)
            offset=0
            for row,count in enumerate(counts):
                table[row,:count]=torch.arange(offset,offset+count,dtype=torch.int32)
                offset+=count
            result.update(block_table=table.to(device),
                          seqused_k=torch.tensor(klens,device=device,dtype=torch.int32))
        else:
            result['cu_seqlens_k']=torch.tensor(cuk,device=device,dtype=torch.int32)
        self.layouts[key]=result
        if len(self.layouts)>64:
            self.layouts.popitem(last=False)
        return result

    @torch.no_grad()
    def pack(self,key,value,lengths,page_size):
        if type(page_size) is not int or page_size<16 or page_size%16:
            raise ValueError('Page size must be a positive multiple of 16')
        if key.ndim!=3 or key.shape!=value.shape or sum(lengths)!=len(key):
            raise ValueError('Packed KV lengths differ from their contents')
        pages=sum((n+page_size-1)//page_size for n in lengths)
        stream=torch.cuda.current_stream(key.device).cuda_stream if key.is_cuda else None
        signature=(key.device,key.dtype,key.shape[1:],page_size,stream)
        cache=self.buffers.get(signature)
        if cache is None or cache.shape[1]<pages:
            capacity=max(pages,2*cache.shape[1] if cache is not None else pages)
            cache=key.new_empty((2,capacity,page_size,*key.shape[1:]))
            self.buffers[signature]=cache
            self.allocations+=1
        token=page=0
        for length in lengths:
            count=(length+page_size-1)//page_size
            cache[0,page:page+count].reshape(-1,*key.shape[1:])[:length].copy_(key[token:token+length])
            cache[1,page:page+count].reshape(-1,*value.shape[1:])[:length].copy_(value[token:token+length])
            token+=length; page+=count
        # Unused tails are never read: each sequence has an explicit seqused_k.
        return cache[0,:pages],cache[1,:pages]


class PackedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,scale,qlens,klens,page_size,fa_version,workspace):
        qlens,klens=tuple(qlens),tuple(klens)
        validate_lengths(q,k,v,qlens,klens)
        ctx.save_for_backward(q,k,v)
        ctx.scale,ctx.qlens,ctx.klens=scale,qlens,klens
        if not q.is_cuda:
            result=[]; qs=ks=0
            for nq,nk in zip(qlens,klens):
                result.append(attention_reference(q[:,:,qs:qs+nq],k[:,:,ks:ks+nk],v[:,:,ks:ks+nk],scale))
                qs+=nq; ks+=nk
            return torch.cat(result,dim=2)
        query,key,value=(t[0].transpose(0,1) for t in (q,k,v))
        layout=workspace.layout(qlens,klens,page_size,q.device)
        if page_size:
            key,value=workspace.pack(key,value,klens,page_size)
        output=torch.empty_like(query,memory_format=torch.contiguous_format)
        cuda_kernels()[1](q=query,k=key,v=value,out=output,softmax_scale=scale,
            causal=True,window_size=(-1,-1),softcap=0.0,alibi_slopes=None,
            fa_version=fa_version,**layout)
        return output.transpose(0,1)[None]

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        original=ctx.saved_tensors
        q,k,v=original
        q,k,v,g=(work(t) for t in (q,k,v,grad))
        dq=torch.empty_like(q); dk=torch.zeros_like(k); dv=torch.zeros_like(v)
        heads=k.shape[1]; groups=q.shape[1]//heads; dim=q.shape[-1]
        qs=ks=0
        for nq,nk in zip(ctx.qlens,ctx.klens):
            key=k[0,:,ks:ks+nk]
            value=v[0,:,ks:ks+nk]
            columns=torch.arange(nk,device=q.device)
            for start in range(0,nq,64):
                end=min(start+64,nq)
                query=q[0,:,qs+start:qs+end].reshape(heads,groups,end-start,dim)
                go=g[0,:,qs+start:qs+end].reshape_as(query)
                query_flat=query.reshape(heads,-1,dim)
                go_flat=go.reshape(heads,-1,dim)
                scores=(query_flat@key.transpose(-1,-2)).reshape(heads,groups,end-start,nk)*ctx.scale
                rows=torch.arange(start,end,device=q.device)+nk-nq
                probabilities=scores.masked_fill(~(columns[None]<=rows[:,None]),float('-inf')).softmax(-1)
                dp=(go_flat@value.transpose(-1,-2)).reshape_as(scores)
                ds=probabilities*(dp-(dp*probabilities).sum(-1,keepdim=True))
                dq[0,:,qs+start:qs+end]=(ds.reshape(heads,-1,nk)@key*ctx.scale).reshape(q.shape[1],end-start,dim)
                dk[0,:,ks:ks+nk]+=(ds.permute(0,3,1,2).reshape(heads,nk,-1)@query_flat)*ctx.scale
                dv[0,:,ks:ks+nk]+=(probabilities.permute(0,3,1,2).reshape(heads,nk,-1)@go_flat)
            qs+=nq; ks+=nk
        return (dq.to(original[0].dtype),dk.to(original[1].dtype),dv.to(original[2].dtype),
                None,None,None,None,None,None)
