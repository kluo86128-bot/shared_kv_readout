"""Read-only hooks for the first ordinary vLLM reader prefill, not blending.

Only the first two decoder layers and final outputs are captured. The hooks
are installed by the explicit parity-trace RPC, never by a normal E2E run.
"""
from pathlib import Path
from functools import wraps

import torch


def snapshot(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,(tuple,list)):
        return [snapshot(item) for item in value]
    if isinstance(value,dict):
        return {key:snapshot(item) for key,item in value.items()}
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    return {'type':type(value).__name__}


class QuestionTrace:
    def __init__(self, model, directory, is_reader, is_blending, layers=2):
        self.directory,self.is_reader,self.is_blending=directory,is_reader,is_blending
        self.path=None
        self.data=None
        self.recording=False
        self.done=False
        self.current_layer=None
        self.layer_limit=layers
        self.handles=[]
        self.attention_backends={}
        core=model.model
        if not getattr(core,'do_not_compile',True):
            raise ValueError('Question tracing requires vLLM enforce_eager=True')
        # vLLM support_torch_compile overrides __call__: even in eager mode
        # it invokes forward directly and bypasses nn.Module's outer hooks.
        # Wrap that actual instance entry; invoke its original forward once.
        self.original_forward=core.forward
        @wraps(self.original_forward)
        def forward(*args,**kwargs):
            opened=self.begin(core,args,kwargs)
            try:
                output=self.original_forward(*args,**kwargs)
            except BaseException:
                if opened:
                    self.end(core,args,None)
                raise
            if opened:
                self.end(core,args,output)
            return output
        core.forward=forward
        for index,layer in enumerate(model.model.layers):
            self.handles.append(layer.register_forward_pre_hook(self.layer_input(index),with_kwargs=True))
            if index>=layers:
                continue
            impl=getattr(layer.self_attn.attn,'impl',None)
            self.attention_backends[str(index)]=dict(implementation=type(impl).__name__,
                **{name:snapshot(getattr(impl,name,None)) for name in
                   ('vllm_flash_attn_version','scale','sliding_window','logits_soft_cap')})
            self.handles.append(layer.register_forward_hook(self.capture(index,'layer_output')))
            for module,name in ((layer.input_layernorm,'input_norm'),
                    (layer.self_attn.qkv_proj,'qkv'),(layer.self_attn.q_norm,'query_norm'),
                    (layer.self_attn.k_norm,'key_norm'),(layer.self_attn.rotary_emb,'rotary'),
                    (layer.self_attn.attn,'attention_output'),(layer.self_attn.o_proj,'o_proj'),
                    (layer.post_attention_layernorm,'post_norm'),(layer.mlp.gate_up_proj,'gate_up'),
                    (layer.mlp.act_fn,'activation'),(layer.mlp.down_proj,'mlp_output')):
                self.handles.append(module.register_forward_hook(self.capture(index,name)))
            self.handles.append(layer.self_attn.qkv_proj.register_forward_pre_hook(self.capture_input(index,'qkv_input')))
            self.handles.append(layer.self_attn.attn.register_forward_pre_hook(self.capture_input(index,'attention_input')))
        self.handles.append(model.model.norm.register_forward_hook(self.capture_global('final_norm')))
        self.original_logits=model.compute_logits
        def logits(*args,**kwargs):
            # The original wrapper records first_logits.pt at this same call.
            result=self.original_logits(*args,**kwargs)
            if self.is_reader() and self.directory():
                expected_path=Path(self.directory())/'question_forward.pt'
                if self.path!=expected_path or not self.done or self.data is None:
                    raise RuntimeError('Actual vLLM reader forward was not captured before compute_logits')
                if 'logits_hidden' in self.data:
                    return result
                missing={str(i):[name for name in ('positions','qkv_input','qkv','attention_output')
                    if name not in self.data['layers'].get(str(i),{})] for i in range(self.layer_limit)}
                missing={index:names for index,names in missing.items() if names}
                if missing:
                    raise RuntimeError(f'Actual question trace is missing native layer stages: {missing}')
                hidden=args[0] if args else kwargs.get('hidden_states')
                self.data['logits_hidden']=snapshot(hidden)
                self.data['capture_complete']=result is not None and hidden is not None
                torch.save(self.data,self.path)
                positions=self.data['layers']['0']['positions'].flatten()
                print(f'[QUESTION_TRACE] saved={self.path} | layers={self.layer_limit} | '
                      f'tokens={len(positions)} | first_position={int(positions[0])} | '
                      f'last_position={int(positions[-1])} | complete={self.data["capture_complete"]}',flush=True)
            return result
        self.logits_wrapper=logits
        model.compute_logits=logits

    def active(self):
        return self.recording and not self.is_blending()

    def begin(self,module,args,kwargs):
        directory=self.directory()
        if not directory or not self.is_reader() or self.is_blending():
            return False
        path=Path(directory)/'question_forward.pt'
        if self.path==path and self.done:
            return False
        if self.recording:
            raise RuntimeError('Nested ordinary reader forward during question tracing')
        self.path=path
        self.data=dict(schema=1,scope='First ordinary vLLM reader prefill, first two layers',
            capture_complete=False,model_args=snapshot(args),model_kwargs=snapshot(kwargs),layers={},
            attention_backends=self.attention_backends)
        self.recording=True
        self.done=False
        self.current_layer=None
        return True

    def end(self,module,args,output):
        if self.recording:
            self.data['model_hidden']=snapshot(output)
            self.recording=False
            self.done=True
            torch.save(self.data,self.path)

    def layer_input(self,index):
        def hook(module,args,kwargs):
            if self.active():
                self.current_layer=index
                if index>=self.layer_limit:
                    return
                names=('positions','hidden_in','residual_in')
                row=self.data['layers'].setdefault(str(index),{})
                for position,name in enumerate(names):
                    keyword='hidden_states' if name=='hidden_in' else 'residual' if name=='residual_in' else name
                    row[name]=snapshot(args[position] if len(args)>position else kwargs.get(keyword))
        return hook

    def capture(self,index,name):
        def hook(module,args,output):
            if self.active() and self.current_layer==index:
                self.data['layers'].setdefault(str(index),{})[name]=snapshot(output)
        return hook

    def capture_input(self,index,name):
        def hook(module,args):
            if self.active() and self.current_layer==index:
                self.data['layers'].setdefault(str(index),{})[name]=snapshot(args)
        return hook

    def capture_global(self,name):
        def hook(module,args,output):
            if self.active():
                self.data[name]=snapshot(output)
        return hook

