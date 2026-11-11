"""Functional native Qwen3 execution shared by training and parity checks."""
import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .native_ops import RMS,AddRMS,SiLU,Rotary,Relocate,Attention

EXECUTION = dict(profile='native_autograd_v1',page_size=16,fa_version=2,
    backward='mathematical_fp32',writer='single_full_request',
    teacher_forcing='prefill_then_decode')
FAST_EXECUTION = dict(profile='native_autograd_fast_v1',page_size=16,fa_version=2,
    backward='analytic_fp32',writer='single_full_request',
    teacher_forcing='parallel_ragged_batch')
WRITER_QUESTION='Summarize the applicable public guidance briefly.'


class NativeExecution:
    def __init__(self,model,settings):
        if settings not in (EXECUTION,FAST_EXECUTION):
            raise ValueError('Unsupported native execution settings')
        if (getattr(model.config,'rope_scaling',None) is not None
                or getattr(model.config,'partial_rotary_factor',1.0)!=1.0):
            raise ValueError('Native execution requires unscaled full-head Qwen3 RoPE')
        self.settings=dict(settings)
        self.fast=settings==FAST_EXECUTION
        if self.fast:
            from .fast_ops import PageWorkspace
            self.workspace=PageWorkspace()
        self.weights={}
        self.tables={}

    def packed(self,layer):
        key=id(layer)
        if key not in self.weights:
            projections=(layer.self_attn.q_proj,layer.self_attn.k_proj,layer.self_attn.v_proj)
            mlp=(layer.mlp.gate_proj,layer.mlp.up_proj)
            if any(p.bias is not None or p.weight.requires_grad for p in (*projections,*mlp)):
                raise ValueError('Native U training requires frozen, bias-free model projections')
            self.weights[key]=(torch.cat([p.weight for p in projections]).detach(),
                               torch.cat([p.weight for p in mlp]).detach())
        return self.weights[key]

    def table(self,model,x,length):
        dim=model.config.head_dim; theta=model.config.rope_theta
        key=(dim,theta,x.device,x.dtype)
        cache=self.tables.get(key)
        if cache is None or len(cache)<length:
            inv=1.0/(theta**(torch.arange(0,dim,2,device=x.device,dtype=torch.float32)/dim))
            phase=torch.einsum('i,j->ij',torch.arange(length,device=x.device,dtype=torch.float32),inv)
            cache=torch.cat((phase.cos(),phase.sin()),dim=-1).to(x.dtype)
            self.tables[key]=cache
        # Checkpoint replay must see the same saved-tensor shape even if a
        # later request has grown the shared immutable table in the meantime.
        return cache[:length]

    def norm(self,module,hidden,residual=None):
        rms,add=RMS,AddRMS
        if self.fast:
            from .fast_ops import AnalyticRMS,AnalyticAddRMS
            rms,add=AnalyticRMS,AnalyticAddRMS
        if residual is None:
            return rms.apply(hidden,module.weight,module.variance_epsilon),hidden
        return add.apply(hidden,residual,module.weight,module.variance_epsilon)

    def project(self,layer,model,hidden,positions,carry=None,*,blend=False,rope_length=None):
        normalized,residual=self.norm(layer.input_layernorm,hidden,carry)
        attention=layer.self_attn
        packed,_=self.packed(layer)
        sizes=[p.weight.shape[0] for p in (attention.q_proj,attention.k_proj,attention.v_proj)]
        q,k,v=F.linear(normalized,packed).split(sizes,dim=-1)
        shape=(*normalized.shape[:-1],-1,attention.head_dim)
        q=self.norm(attention.q_norm,q.reshape(shape))[0]
        k=self.norm(attention.k_norm,k.reshape(shape))[0]
        v=v.reshape(shape).transpose(1,2)
        if blend:
            # Native LMCache blend uses its tensor RoPE path; the ordinary
            # vLLM prefill/decode uses the native cached-table CUDA operator.
            cos,sin=model.rotary_emb(normalized,position_ids=positions[None])
            q,k=apply_rotary_pos_emb(q.transpose(1,2),k.transpose(1,2),cos,sin)
        else:
            cache=self.table(model,hidden,rope_length if rope_length is not None else int(positions.max())+1)
            q=Rotary.apply(q,positions,cache).transpose(1,2)
            k=Rotary.apply(k,positions,cache).transpose(1,2)
        return residual,q,k,v

    def finish(self,layer,residual,q,k,v,*,paged,qlens=None,klens=None):
        if self.fast:
            from .fast_ops import PackedAttention
            output=PackedAttention.apply(q,k,v,layer.self_attn.scaling,
                tuple(qlens) if qlens is not None else (q.shape[2],),
                tuple(klens) if klens is not None else (k.shape[2],),
                self.settings['page_size'] if paged else 0,self.settings['fa_version'],self.workspace)
        else:
            output=Attention.apply(q,k,v,layer.self_attn.scaling,
                                   self.settings['page_size'] if paged else 0,
                                   self.settings['fa_version'])
        output=output.transpose(1,2).reshape(1,q.shape[2],-1).contiguous()
        hidden=layer.self_attn.o_proj(output)
        normalized,carry=self.norm(layer.post_attention_layernorm,hidden,residual)
        _,gate_up=self.packed(layer)
        hidden=layer.mlp.down_proj(SiLU.apply(F.linear(normalized,gate_up)))
        return hidden,carry

    def forward(self,model,embeddings,*,layers=(),position_start=0,positions=None):
        if embeddings.ndim!=3 or embeddings.shape[0]!=1:
            raise ValueError('Native execution requires one unpadded request per forward')
        length=embeddings.shape[1]
        if positions is None:
            positions=torch.arange(position_start,position_start+length,device=embeddings.device)
            rope_length=position_start+length
        else:
            rope_length=int(positions.max())+1
        positions=positions.reshape(-1)
        if len(positions)!=length:
            raise ValueError('Native position count differs from the request')
        carry=None; hidden=embeddings; updated=[]
        if layers and len(layers)!=len(model.model.layers):
            raise ValueError('Native past KV layer count differs from model')
        for index,layer in enumerate(model.model.layers):
            residual,q,k,v=self.project(layer,model.model,hidden,positions,carry,rope_length=rope_length)
            if layers:
                k=torch.cat((layers[index][0],k),dim=2)
                v=torch.cat((layers[index][1],v),dim=2)
            updated.append((k,v))
            hidden,carry=self.finish(layer,residual,q,k,v,paged=True)
        hidden,_=self.norm(model.model.norm,hidden,carry)
        return hidden,tuple(updated)

    def forward_batch(self,model,embeddings,past,lengths,starts,*,return_cache=False):
        """One packed question GEMM; independent, unpadded causal sequences."""
        if not self.fast or embeddings.shape[0]!=1 or sum(lengths)!=embeddings.shape[1]:
            raise ValueError('Packed questions require the fast execution profile and valid lengths')
        if len(past)!=len(lengths) or len(starts)!=len(lengths):
            raise ValueError('Packed question context count differs from sequence count')
        layer_count=len(model.model.layers)
        if any(len(cache)!=layer_count for cache in past):
            raise ValueError('Incomplete packed question context')
        positions=torch.cat([torch.arange(start,start+length,device=embeddings.device)
                             for start,length in zip(starts,lengths)])
        rope_length=max(start+length for start,length in zip(starts,lengths))
        klens=[cache[0][0].shape[2]+length for cache,length in zip(past,lengths)]
        hidden=embeddings; carry=None
        updated=[[] for _ in past] if return_cache else None
        for index,layer in enumerate(model.model.layers):
            residual,q,k,v=self.project(layer,model.model,hidden,positions,carry,rope_length=rope_length)
            keys=[]; values=[]; offset=0
            for cache,length,total in zip(past,lengths,klens):
                pk,pv=cache[index]
                if pk.shape!=pv.shape or pk.shape[0]!=1 or pk.shape[2]!=total-length:
                    raise ValueError('Packed past KV lengths changed between layers')
                keys.append(torch.cat((pk,k[:,:,offset:offset+length]),dim=2))
                values.append(torch.cat((pv,v[:,:,offset:offset+length]),dim=2))
                if return_cache:
                    updated[len(keys)-1].append((keys[-1],values[-1]))
                offset+=length
            hidden,carry=self.finish(layer,residual,q,torch.cat(keys,dim=2),
                torch.cat(values,dim=2),paged=True,qlens=lengths,klens=klens)
        result=self.norm(model.model.norm,hidden,carry)[0]
        return (result,[tuple(cache) for cache in updated]) if return_cache else result

    def relocate(self,model,layers,source_start,destination_start):
        if source_start==destination_start:
            return layers
        key=layers[0][0]; length=key.shape[2]
        old=torch.arange(source_start,source_start+length,device=key.device,dtype=torch.long)
        new=torch.arange(destination_start,destination_start+length,device=key.device,dtype=torch.long)
        cache=self.table(model,key,max(source_start,destination_start)+length)
        result=[]
        for k,v in layers:
            flat=k[0].transpose(0,1)
            relocated=Relocate.apply(flat,old,new,cache).transpose(0,1)[None]
            result.append((relocated,v))
        return tuple(result)
