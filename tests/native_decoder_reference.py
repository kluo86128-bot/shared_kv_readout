"""Test-only forward oracle with independent page packing; no training entry."""
import torch
import torch.nn.functional as F

def snapshot(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:snapshot(v) for k,v in value.items()}
    return value

def paged_kv(key, value, page_size):
    """Pack identical [tokens, kv_heads, dim] values into vLLM NHD pages."""
    if type(page_size) is not int or page_size<16 or page_size%16:
        raise ValueError('FA2 page size must be a positive multiple of 16')
    if key.shape!=value.shape or key.ndim!=3 or not len(key):
        raise ValueError('Invalid diagnostic KV shape')
    pages=(len(key)+page_size-1)//page_size
    cache=key.new_zeros((2,pages,page_size,*key.shape[1:]))
    cache[0].view(-1,*key.shape[1:])[:len(key)].copy_(key)
    cache[1].view(-1,*value.shape[1:])[:len(value)].copy_(value)
    table=torch.arange(pages,device=key.device,dtype=torch.int32)[None]
    return cache[0],cache[1],table


class NativeDecoderReplay:
    """Forward-only kernel control for C1; never use this as a trainer.

    LMCache 0.4.2 carries MLP output and residual separately between layers.
    Use the installed vLLM kernels rather than guessing their BF16 rounding.
    Question prefill uses HF unless explicitly requested as a separate control.
    """
    def __init__(self, ops=None, flash=None, fa_version=2):
        if ops is None or flash is None:
            import importlib.metadata
            if importlib.metadata.version('vllm') != '0.18.0':
                raise ValueError('Native decoder replay requires vLLM 0.18.0')
            from vllm import _custom_ops
            from vllm.vllm_flash_attn import flash_attn_varlen_func
            from types import SimpleNamespace
            ops=SimpleNamespace(rms_norm=_custom_ops.rms_norm,
                fused_add_rms_norm=_custom_ops.fused_add_rms_norm,
                silu_and_mul=torch.ops._C.silu_and_mul,
                rotary_embedding=_custom_ops.rotary_embedding)
            flash=flash_attn_varlen_func
        self.ops,self.flash,self.fa_version=ops,flash,fa_version
        self.carry=None
        self.pending=None
        self.rope_cache={}
        self.trace=None
        self.trace_layer=None

    def record(self, **values):
        if self.trace is not None and self.trace_layer is not None and self.trace_layer<2:
            self.trace.setdefault('layers',{}).setdefault(str(self.trace_layer),{}).update(snapshot(values))

    def norm(self, layernorm, hidden, residual=None):
        if hidden.requires_grad or (residual is not None and residual.requires_grad):
            raise ValueError('Native decoder replay is forward-only; run under torch.no_grad()')
        weight=layernorm.weight
        eps=layernorm.variance_epsilon
        if residual is None:
            out=torch.empty_like(hidden, memory_format=torch.contiguous_format)
            self.ops.rms_norm(out,hidden,weight,eps)
            return out,hidden
        # The native ops mutate both buffers. Never mutate the source trace or
        # a residual still owned by the original model.
        shape=hidden.shape
        out=hidden.reshape(-1,shape[-1]).contiguous().clone()
        carried=residual.reshape_as(out).contiguous().clone()
        self.ops.fused_add_rms_norm(out,carried,weight,eps)
        return out.reshape(shape),carried.reshape(shape)

    def project(self, layer, model, hidden, positions, *, rotary='hf'):
        from kv_cache_readout.native_blend import apply_rotary_pos_emb
        if layer is model.layers[0]:
            self.carry=None
            self.pending=None
        if self.pending is not None:
            raise RuntimeError('Native decoder projection ran before its previous decoder')
        self.trace_layer=next(i for i,item in enumerate(model.layers) if item is layer)
        self.record(positions=positions,hidden_in=hidden[0],
                    residual_in=None if self.carry is None else self.carry[0])
        normed,residual=self.norm(layer.input_layernorm,hidden,self.carry)
        attention=layer.self_attn
        projections=(attention.q_proj,attention.k_proj,attention.v_proj)
        if any(p.bias is not None for p in projections):
            raise ValueError('Native decoder replay is scoped to bias-free Qwen3')
        qkv=F.linear(normed,torch.cat([p.weight for p in projections]))
        self.record(qkv_input=normed[0],qkv=qkv[0])
        q,k,v=qkv.split(
            [p.weight.shape[0] for p in projections],dim=-1)
        shape=(*normed.shape[:-1],-1,attention.head_dim)
        q=self.norm(attention.q_norm,q.reshape(shape))[0]
        k=self.norm(attention.k_norm,k.reshape(shape))[0]
        self.record(query_norm=q[0],key_norm=k[0])
        v=v.reshape(shape).transpose(1,2)
        if rotary=='hf':
            cos,sin=model.rotary_emb(normed,position_ids=positions[None])
            q,k=apply_rotary_pos_emb(q.transpose(1,2),k.transpose(1,2),cos,sin)
        elif rotary=='vllm_table':
            cfg=model.config
            if (getattr(cfg,'rope_scaling',None) is not None
                    or getattr(cfg,'partial_rotary_factor',1.0)!=1.0):
                raise ValueError('Question RoPE control requires unscaled, full-head Qwen3 RoPE')
            dim=attention.head_dim
            theta=cfg.rope_theta
            length=int(positions.max())+1
            cache_key=(dim,theta,q.dtype,q.device)
            cache=self.rope_cache.get(cache_key)
            if cache is None or len(cache)<length:
                # Same standard-RoPE table construction as vLLM 0.18.0
                # rotary_embedding/base.py, computed on the active GPU.
                inv=1.0/(theta**(torch.arange(0,dim,2,device=q.device,dtype=torch.float32)/dim))
                phase=torch.einsum('i,j->ij',torch.arange(length,device=q.device,dtype=torch.float32),inv)
                cache=torch.cat((phase.cos(),phase.sin()),dim=-1).to(q.dtype)
                self.rope_cache[cache_key]=cache
            q=q.reshape(len(positions),-1).contiguous().clone()
            k=k.reshape(len(positions),-1).contiguous().clone()
            self.ops.rotary_embedding(positions,q,k,dim,cache,True)
            q=q.reshape(1,len(positions),-1,dim).transpose(1,2)
            k=k.reshape(1,len(positions),-1,dim).transpose(1,2)
        else:
            raise ValueError(f'Unknown diagnostic RoPE: {rotary}')
        self.record(rotary=[q[0].transpose(0,1).reshape(len(positions),-1),
                            k[0].transpose(0,1).reshape(len(positions),-1)])
        self.record(attention_input=[q[0].transpose(0,1).reshape(len(positions),-1),
            k[0].transpose(0,1).reshape(len(positions),-1),
            v[0].transpose(0,1).reshape(len(positions),-1)])
        self.pending=layer
        return residual,q,k,v

    def attention(self, query, key, value, scaling, *, page_size=None):
        # C1 blending uses contiguous KV; ordinary vLLM prefill uses paged KV.
        # Keep these calls separate: identical mathematics can select different
        # FlashAttention kernels and BF16 accumulation schedules.
        q,k,v=(t[0].transpose(0,1) for t in (query,key,value))
        output=torch.empty_like(q,memory_format=torch.contiguous_format)
        length=len(k)
        if page_size is None:
            layout=dict(cu_seqlens_k=torch.tensor([0,length],device=q.device,dtype=torch.int32))
        else:
            k,v,table=paged_kv(k,v,page_size)
            layout=dict(seqused_k=torch.tensor([length],device=q.device,dtype=torch.int32),block_table=table)
        self.flash(q=q,k=k,v=v,out=output,
            cu_seqlens_q=torch.tensor([0,len(q)],device=q.device,dtype=torch.int32),
            max_seqlen_q=len(q),max_seqlen_k=length,
            softmax_scale=scaling,causal=True,
            window_size=(-1,-1),softcap=0.0,alibi_slopes=None,
            fa_version=self.fa_version,**layout)
        return output

    def decoder(self, layer, residual, query, key, value, mask, *, page_size=None):
        if self.pending is not layer:
            raise RuntimeError('Native decoder layer does not match its projection')
        output=self.attention(query,key,value,layer.self_attn.scaling,page_size=page_size)
        self.record(attention_output=output)
        hidden=layer.self_attn.o_proj(output.reshape(1,output.shape[0],-1))
        self.record(o_proj=hidden[0])
        normed,self.carry=self.norm(layer.post_attention_layernorm,hidden,residual)
        self.record(post_norm=[normed[0],self.carry[0]])
        mlp=layer.mlp
        if mlp.gate_proj.bias is not None or mlp.up_proj.bias is not None:
            raise ValueError('Native decoder replay requires bias-free gate/up projections')
        gate_up=F.linear(normed,torch.cat((mlp.gate_proj.weight,mlp.up_proj.weight)))
        activated=gate_up.new_empty((*gate_up.shape[:-1],gate_up.shape[-1]//2))
        self.ops.silu_and_mul(activated,gate_up)
        self.record(gate_up=gate_up[0],activation=activated[0])
        self.pending=None
        output=mlp.down_proj(activated)
        self.record(mlp_output=output[0],layer_output=[output[0],self.carry[0]])
        return output

    def question_hidden(self, backend, context, ids, *, rotary='vllm_table',page_size=None):
        """Native-kernel prefill over an unchanged captured C1 KV.

        The scheduler and physical allocation are not replayed. With page_size
        set, identical logical KV is packed into synthetic NHD pages and the
        paged FlashAttention entry is used. C1 construction is kept separate.
        """
        if not ids or len(context.layers)!=len(backend.model.model.layers):
            raise ValueError('Incomplete question context')
        past=context.layers[0][0].shape[2]
        if context.next_position!=past:
            raise ValueError('Question control requires contiguous positions after C1')
        hidden=backend.embedding(backend.tensor(ids))
        positions=torch.arange(past,past+len(ids),device=hidden.device,dtype=torch.long)
        model=backend.model.model
        for layer,pair in zip(model.layers,context.layers):
            if any(t.shape[2]!=past for t in pair):
                raise ValueError('Question cache layers have different lengths')
            residual,q,k,v=self.project(layer,model,hidden,positions,rotary=rotary)
            k=torch.cat((pair[0],k),dim=2)
            v=torch.cat((pair[1],v),dim=2)
            hidden=self.decoder(layer,residual,q,k,v,None,page_size=page_size)
        output=self.norm(model.norm,hidden,self.carry)[0]
        if self.trace is not None:
            self.trace['final_norm']=output[0].detach().cpu()
            self.trace['logits_hidden']=output[:,-1].detach().cpu()
        return output


