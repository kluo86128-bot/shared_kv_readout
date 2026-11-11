from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .config import fingerprint, file_hash, ROOT, resolve
import json
from pathlib import Path
from .native_blend import native_cacheblend


def _pairs(cache):
    return tuple((layer.keys, layer.values) for layer in cache.layers)


def _fresh_cache(layers):
    result = DynamicCache()
    for index, (key, value) in enumerate(layers):
        result.update(key.clone(), value.clone(), index)
    return result


def _causal_mask(count, past, *, device, dtype):
    rows = past + torch.arange(count, device=device)
    columns = torch.arange(past + count, device=device)
    allowed = columns[None, :] <= rows[:, None]
    mask = torch.zeros((1, 1, count, past + count), device=device, dtype=dtype)
    return mask.masked_fill(~allowed[None, None], torch.finfo(dtype).min)


@dataclass(frozen=True)
class Context:
    layers: tuple
    next_position: int
    audit: dict


def question_waves(question_sets, size, order='record_major'):
    """Yield real questions only, retaining owner/index for loss reconstruction.

    Round-robin waves never fill a short round from the next round: at most one
    question from each active source appears in a wave. Sequence packing is
    handled separately by the existing variable-length attention path.
    """
    if type(size) is not int or size < 1:
        raise ValueError('Question batch size must be positive')
    if order == 'record_major':
        flat = [(owner, i, qa) for owner, qs in enumerate(question_sets) for i, qa in enumerate(qs)]
        for offset in range(0, len(flat), size):
            yield flat[offset:offset+size]
    elif order == 'round_robin':
        for i in range(max(map(len, question_sets), default=0)):
            active = [(owner, i, qs[i]) for owner, qs in enumerate(question_sets) if i < len(qs)]
            for offset in range(0, len(active), size):
                yield active[offset:offset+size]
    else:
        raise ValueError('Unknown question ordering')


class Backend:
    def __init__(self, model, tokenizer, cfg):
        if model.config.model_type != "qwen3":
            raise ValueError("The native surrogate is validated for Qwen3 only")
        if any(kind != "full_attention" for kind in getattr(model.config, "layer_types", [])):
            raise ValueError("Sliding-attention models are not supported")
        self.model, self.tokenizer, self.cfg = model, tokenizer, cfg
        self.model.eval().requires_grad_(False)
        self.embedding = model.get_input_embeddings()
        self.device = self.embedding.weight.device
        self.dtype = self.embedding.weight.dtype
        self.native=None
        self.input_only_ids=()
        if 'execution' in cfg:
            from .native_model import NativeExecution
            self.native=NativeExecution(model,cfg['execution'])
            model._readout_execution=self.native
            model.model._readout_execution=self.native
            if self.device.type=='cuda':
                from .native_ops import cuda_kernels
                if self.dtype!=torch.bfloat16:
                    raise ValueError('The pinned native CUDA profile requires BF16')
                cuda_kernels()
            if hasattr(tokenizer,'add_special_tokens'):
                from copy import deepcopy
                export_tokenizer=deepcopy(tokenizer)
                names=[f'<|e2e_u_{i:04d}|>' for i in range(cfg['prompt_length'])]
                original_vocab=export_tokenizer.get_vocab()
                if any(name in original_vocab for name in names):
                    raise ValueError('Dedicated U names collide with base tokenizer')
                export_tokenizer.add_special_tokens({'additional_special_tokens':names},
                                                    replace_additional_special_tokens=False)
                self.input_only_ids=tuple(i for i in export_tokenizer.convert_tokens_to_ids(names)
                                          if i<model.get_output_embeddings().weight.shape[0])
            elif self.device.type=='cuda':
                raise ValueError('Native execution requires an export-compatible tokenizer')
        self.public_ids = self.ids(cfg["public_text"])
        expected = cfg["native_contract"]["c1_tokens"]
        if len(self.public_ids) != expected:
            raise ValueError(f"public_text must tokenize to exactly {expected}, got {len(self.public_ids)}")
        self.delimiter_ids = self.ids(cfg["delimiter"])
        if not self.delimiter_ids:
            raise ValueError("delimiter must produce at least one token")
        self.eos = int(tokenizer.eos_token_id)
        stops = model.generation_config.eos_token_id
        self.stops = set(stops if isinstance(stops, (list, tuple)) else [stops]) | {self.eos}
        self.stops.discard(None)
        tokenizer_payload = tokenizer.backend_tokenizer.to_str().encode("utf-8")
        self.identity = {
            "model_config_sha256": fingerprint(model.config.to_dict()),
            "tokenizer_sha256": hashlib.sha256(tokenizer_payload).hexdigest(),
        }
        root = resolve(cfg['model_name_or_path'])
        self.identity['assets'] = {p.name: {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
                                   for p in sorted(root.glob('*')) if p.is_file()
                                   and (p.suffix in ('.safetensors', '.bin', '.json', '.model'))}
        self.identity['torch_version'] = str(torch.__version__)
        self.identity['transformers_version'] = __import__('transformers').__version__
        if self.native is not None:
            self.identity['execution']=dict(cfg['execution'])
            self.identity['execution_device']='native_cuda' if self.device.type=='cuda' else 'cpu_reference_only'

    @classmethod
    def load(cls, cfg):
        cfg = dict(cfg, model_name_or_path=str(resolve(cfg["model_name_or_path"])))
        device = torch.device(cfg["device"])
        if device.type == "cuda":
            torch.cuda.set_device(device)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_name_or_path"], local_files_only=True, trust_remote_code=False,
        )
        snapshot = ROOT / 'data/tokenizer/Qwen3-4B/tokenizer.json'
        if not snapshot.is_file() or json.loads(tokenizer.backend_tokenizer.to_str()) != json.loads(snapshot.read_text(encoding='utf-8')):
            raise ValueError('Model tokenizer differs from the frozen Qwen3 dataset tokenizer')
        suffix_manifest = json.loads((resolve(cfg['scoped_cache']['suffix_directory']) / 'manifest.json').read_text(encoding='utf-8'))
        if file_hash(snapshot) != suffix_manifest['tokenizer_sha256']:
            raise ValueError('Frozen tokenizer differs from the S dataset tokenizer identity')
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name_or_path"], local_files_only=True, trust_remote_code=False,
            torch_dtype=getattr(torch, cfg["dtype"]), attn_implementation="eager",
        ).to(device)
        return cls(model, tokenizer, cfg)

    def ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def tensor(self, ids) -> torch.Tensor:
        return torch.tensor([list(ids)], device=self.device, dtype=torch.long)

    def question_text(self, question: str) -> str:
        if self.cfg["question_format"] == "plain":
            return f"Question: {question}\nAnswer:\n"
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

    def question_and_target(self, qa: dict):
        text = self.question_text(qa["question"])
        question = self.ids(text)
        joined = self.ids(text + qa["answer"])
        if joined[:len(question)] != question:
            raise ValueError("Question/answer boundary changes tokenization")
        return question, joined[len(question):] + [self.eos]

    def forward(self, *, ids, position_start=0):
        hidden, layers = self._model_forward(self.embedding(self.tensor(ids)), position_start=position_start)
        return None, layers

    def initial_prompt(self) -> torch.Tensor:
        choices = []
        for word in self.cfg["initialization_words"]:
            ids = self.ids(" " + word)
            if len(ids) == 1 and ids[0] not in self.tokenizer.all_special_ids:
                choices.append(ids[0])
        if not choices:
            raise ValueError("No stable one-token initialization words")
        rng = random.Random(self.cfg["seed"])
        ids = rng.choices(choices, k=self.cfg["prompt_length"])
        return self.embedding(self.tensor(ids))[0].detach().float().clone()

    def _model_forward(self, embeddings, *, layers=(), position_start=0, positions=None, mask=None):
        if getattr(self,'native',None) is not None:
            if mask is not None:
                raise ValueError('Native execution takes unpadded single requests, not arbitrary batched masks')
            return self.native.forward(self.model,embeddings,layers=layers,
                                       position_start=position_start,positions=positions)
        count = embeddings.shape[1]
        past = layers[0][0].shape[2] if layers else 0
        if positions is None:
            positions = torch.arange(position_start, position_start + count,
                                     device=self.device, dtype=torch.long)[None]
        cache_positions = torch.arange(past, past + count, device=self.device)
        result = self.model.model(
            inputs_embeds=embeddings,
            past_key_values=_fresh_cache(layers) if layers else None,
            position_ids=positions,
            cache_position=cache_positions,
            attention_mask=(_causal_mask(count, past, device=self.device, dtype=self.dtype) if mask is None else mask),
            use_cache=True,
            return_dict=True,
        )
        return result.last_hidden_state, _pairs(result.past_key_values)

    def source_many(self, records, prompt, *, epoch=0):
        """Batch complete writers with left padding and true per-record RoPE."""
        from .suffix import SuffixContext, pair_for
        if self.native is not None:
            pairs=[self._native_source(record,prompt,epoch=epoch) for record in records]
            return [pair[0] for pair in pairs],[pair[1] for pair in pairs]
        sequences, boundaries = [], []
        keep = self.cfg['prompt_partition']['keep_length']
        for record in records:
            suffix = pair_for(record, self.cfg, epoch)
            if not 1 <= len(suffix['ids']) <= self.cfg['scoped_cache']['suffix_max_tokens']:
                raise ValueError('S exceeds the configured token budget')
            prefix = self.ids(record['private_prefix'] + self.cfg['delimiter'])
            chunks = [self.embedding(self.tensor(prefix))]
            before = 0 if prompt is None else len(prompt) - keep
            if prompt is not None:
                chunks.append(prompt[:before].to(self.dtype)[None])
            chunks.append(self.embedding(self.tensor(self.delimiter_ids + self.public_ids + self.delimiter_ids)))
            if prompt is not None:
                chunks.append(prompt[before:].to(self.dtype)[None])
            chunks.append(self.embedding(self.tensor(suffix['ids'])))
            sequence = torch.cat(chunks, 1)
            start = len(prefix) + before + len(self.delimiter_ids)
            sstart = sequence.shape[1] - len(suffix['ids'])
            sequences.append(sequence)
            boundaries.append((start, start + len(self.public_ids), sstart, suffix))
        width = max(x.shape[1] for x in sequences)
        pads = [width - x.shape[1] for x in sequences]
        embeddings = torch.cat([F.pad(x, (0, 0, pad, 0)) for x, pad in zip(sequences, pads)])
        columns = torch.arange(width, device=self.device)[None]
        pad_tensor = torch.tensor(pads, device=self.device)[:, None]
        valid = columns >= pad_tensor
        positions = (columns - pad_tensor).clamp_min(0)
        mask = _causal_mask(width, 0, device=self.device, dtype=self.dtype).expand(len(records), -1, -1, -1).clone()
        mask.masked_fill_(~valid[:, None, None, :], torch.finfo(self.dtype).min)
        _, layers = self._model_forward(embeddings, positions=positions, mask=mask)
        c1s, suffixes = [], []
        for owner, ((start, end, sstart, suffix), pad) in enumerate(zip(boundaries, pads)):
            def region(a, b):
                return tuple(tuple(t[owner:owner+1, :, a+pad:b+pad].clone() for t in pair) for pair in layers)
            c1s.append(region(start, end))
            suffixes.append(SuffixContext(region(sstart, width-pad), sstart, len(suffix['ids']), tuple(suffix['ids'])))
        return c1s, suffixes

    def blend_many(self, records, prompt, *, epoch=0):
        writers, suffixes = self.source_many(records, prompt, epoch=epoch)
        contract = self.cfg['native_contract']
        public = self.embedding(self.tensor(self.public_ids))
        method=self.cfg.get('c1_selection',{}).get('method','cacheblend')
        if method in ('epic','none'):
            return self._replacement_contexts(records,writers,suffixes,prompt,public,method)
        cap = self.cfg['training'].get('c1_batch_size', 1)
        if self.native is not None and self.native.fast and cap > 1:
            from .c1_batch import native_cacheblend_many
            from .progress import emit
            starts = [suffix.start - len(self.public_ids) - len(self.delimiter_ids)
                      - (self.cfg['prompt_partition']['keep_length'] if prompt is not None else 0)
                      for suffix in suffixes]
            contexts = []
            for offset in range(0, len(writers), cap):
                group = writers[offset:offset+cap]
                emit(self.cfg, 'c1_batch', records=len(group), batch_limit=cap, state='begin')
                results = native_cacheblend_many(self.model, public, group,
                    writer_position_starts=starts[offset:offset+cap],
                    check_layer=contract['check_layer'], ratio=contract['recompute_ratio'],
                    **self._c1_selection(records[offset:offset+cap],packed=True))
                for record,result in zip(records[offset:offset+cap],results):self._selection_audit(record,result)
                contexts.extend(Context(r.layers, len(self.public_ids), r.audit) for r in results)
                emit(self.cfg, 'c1_batch', records=len(group), batch_limit=cap, state='done')
            return contexts
        contexts = []
        for record,writer, suffix in zip(records,writers, suffixes):
            source_start = suffix.start - len(self.public_ids) - len(self.delimiter_ids)
            if prompt is not None:
                source_start -= self.cfg['prompt_partition']['keep_length']
            result = native_cacheblend(self.model, public, writer,
                                       check_layer=contract['check_layer'], ratio=contract['recompute_ratio'],
                                       writer_position_start=source_start,**self._c1_selection([record]))
            self._selection_audit(record,result)
            contexts.append(Context(result.layers, len(self.public_ids), result.audit))
        return contexts

    def _c1_selection(self,records,packed=False):
        selection=self.cfg.get('c1_selection',{})
        if selection.get('method')!='random':return {}
        from .c1_selection import random_indices
        rows=[random_indices(len(self.public_ids),self.cfg['native_contract']['recompute_ratio'],
            selection['seed'],r['record_id'],self.device) for r in records]
        return dict(selection_indices=torch.stack(rows) if packed else rows[0])

    def _replacement_contexts(self,records,writers,suffixes,prompt,public,method):
        from .native_blend import relocate_keys
        length=len(self.public_ids);ratio=self.cfg['native_contract']['recompute_ratio']
        if method=='none' and ratio!=0:raise ValueError('none requires ratio=0')
        count=int(length*ratio) if method=='epic' else 0
        clean=None
        if count:
            with torch.no_grad():_,clean=self._model_forward(public,position_start=0)
        contexts=[]
        for record,writer,suffix in zip(records,writers,suffixes):
            start=suffix.start-length-len(self.delimiter_ids)
            if prompt is not None:start-=self.cfg['prompt_partition']['keep_length']
            relocated=relocate_keys(self.model.model,writer,start,0)
            mixed=(tuple(tuple(torch.cat((fresh[:,:,:count],old[:,:,count:]),dim=2)
                                for fresh,old in zip(cached,source))
                         for cached,source in zip(clean,relocated)) if count else relocated)
            audit=dict(algorithm='native_epic_clean_prefix' if method=='epic' else 'relocated_reuse_zero_control',
                selection_method=method,selection_record_id=record['record_id'],ratio=ratio,
                c1_tokens=length,selected_tokens=count,selected_indices=list(range(count)),
                selection_policy='clean_prefix_replacement' if method=='epic' else 'no_recomputation',
                clean_reference_tokens=length if count else 0,check_layer_used=False,
                position_policy='reader_destination_zero',cached_key_policy='relocate_source_to_destination')
            contexts.append(Context(mixed,length,audit))
        return contexts

    def _selection_audit(self,record,result):
        selection=self.cfg.get('c1_selection',{})
        if selection.get('method')=='random':
            result.audit.update(selection_method='random',selection_seed=selection['seed'],
                selection_record_id=record['record_id'],selection_policy='stable_per_record_seed_v1')

    def answer_components_many(self, contexts, question_sets):
        """Batch questions across records; checkpoint each wave of decoder activations."""
        result = [[None] * len(qs) for qs in question_sets]
        size = self.cfg['training']['question_batch_size']
        order = self.cfg['training'].get('question_order', 'record_major')
        total = sum(map(len, question_sets))
        completed = 0
        for group in (per_sample_waves(question_sets, self.cfg['training']['questions_per_sample_batch'])
                      if order == 'per_sample' else question_waves(question_sets, size, order)):
            from .progress import emit
            emit(self.cfg,'question_batch',questions=len(group), owners=[owner for owner, _, _ in group],
                 question_order=order, completed=f'{completed}/{total}',state='begin')
            values = self._question_wave(contexts, group)
            for (owner, i, _), value in zip(group, values):
                result[owner][i] = (value[0], value[1])
            completed += len(group)
            emit(self.cfg,'question_batch',questions=len(group),completed=f'{completed}/{total}',state='done')
        return result

    def _question_wave(self, contexts, group):
        if self.native is not None:
            return self._native_question_wave(contexts,group)
        pairs = [self.question_and_target(qa) for _, _, qa in group]
        sequences = [self.delimiter_ids + q + target[:-1] for q, target in pairs]
        width = max(map(len, sequences))
        lengths = torch.tensor([len(s) for s in sequences], device=self.device)
        columns = torch.arange(width, device=self.device)[None]
        pads = width - lengths[:, None]
        valid = columns >= pads
        ids = torch.full((len(group), width), self.eos, device=self.device, dtype=torch.long)
        for i, seq in enumerate(sequences):
            ids[i, width-len(seq):] = torch.tensor(seq, device=self.device)
        layers = tuple(tuple(torch.cat([contexts[owner].layers[layer][kv] for owner, _, _ in group])
                             for kv in (0, 1)) for layer in range(len(contexts[0].layers)))
        past = layers[0][0].shape[2]
        starts = torch.tensor([contexts[owner].next_position for owner, _, _ in group], device=self.device)[:, None]
        positions = starts + (columns - pads).clamp_min(0)
        mask = _causal_mask(width, past, device=self.device, dtype=self.dtype).expand(len(group), -1, -1, -1).clone()
        key_valid = torch.cat((torch.ones(len(group), past, dtype=torch.bool, device=self.device), valid), 1)
        mask.masked_fill_(~key_valid[:, None, None, :], torch.finfo(self.dtype).min)
        def calculate(*flat):
            cache = tuple(zip(flat[::2], flat[1::2]))
            hidden, _ = self._model_forward(self.embedding(ids), layers=cache, positions=positions, mask=mask)
            values = []
            for row, (_, target) in enumerate(pairs):
                logits = self.model.lm_head(hidden[row:row+1, width-len(target):]).float()
                ce = F.cross_entropy(logits.transpose(1, 2), self.tensor(target), reduction='none')
                options = self.cfg['loss']
                focal = options['alpha'] * (-torch.expm1(-ce)).clamp_min(torch.finfo(ce.dtype).tiny).pow(options['gamma']) * ce
                values.append(self._reduce_question_tokens(group[row][2], ce.reshape(-1)))
            return torch.stack(values)
        flat = tuple(t for pair in layers for t in pair)
        if torch.is_grad_enabled() and self.cfg['training']['checkpoint_questions']:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(calculate, *flat, use_reentrant=False, preserve_rng_state=False)
        return calculate(*flat)

    def _reduce_question_tokens(self, qa, ce):
        if qa.get('prediction_kind') == 'fieldwise_readout' and 'readout' in self.cfg:
            from .readout import reduce_fieldwise
            return reduce_fieldwise(self,qa,ce.reshape(-1))
        if qa.get('prediction_kind') == 'joint_json':
            from .joint import reduce_losses
            return reduce_losses(self, qa, ce.reshape(-1))
        options = self.cfg['loss']
        focal = options['alpha']*(-torch.expm1(-ce)).clamp_min(torch.finfo(ce.dtype).tiny).pow(options['gamma'])*ce
        return torch.stack((ce.mean(), focal.mean()))

    def writer_cache(self, record: dict, prompt: torch.Tensor):
        if prompt.shape != (self.cfg["prompt_length"], self.model.config.hidden_size):
            raise ValueError("U has the wrong shape")
        if self.native is not None:
            layers,suffix=self._native_source(record,prompt)
            start=suffix.start-len(self.public_ids)-len(self.delimiter_ids)-self.cfg['prompt_partition']['keep_length']
            return layers,dict(writer_c1_start=start,writer_c1_end=start+len(self.public_ids),
                writer_layout=self.cfg['native_contract']['writer_layout'],execution=self.cfg['execution']['profile'])
        left_ids = self.ids(record["private_prefix"] + self.cfg["delimiter"])
        keep = self.cfg["prompt_partition"]["keep_length"]
        before = len(prompt) - keep
        chunks = [
            self.embedding(self.tensor(left_ids)),
            prompt[:before].to(device=self.device, dtype=self.dtype)[None],
            self.embedding(self.tensor(self.delimiter_ids)),
            self.embedding(self.tensor(self.public_ids)),
        ]
        embeddings = torch.cat(chunks, dim=1)
        c1_start = embeddings.shape[1] - len(self.public_ids)
        _, layers = self._model_forward(embeddings)
        c1 = tuple((key[:, :, c1_start:].clone(), value[:, :, c1_start:].clone())
                   for key, value in layers)
        return c1, {"writer_c1_start": c1_start, "writer_c1_end": embeddings.shape[1],
                    "writer_layout": self.cfg["native_contract"]["writer_layout"]}

    def suffix_context(self, record: dict, prompt: torch.Tensor | None, *, epoch: int):
        if self.native is not None:
            return self._native_source(record,prompt,epoch=epoch)[1]
        from .suffix import SuffixContext, pair_for
        suffix = pair_for(record, self.cfg, epoch)
        if not 1 <= len(suffix["ids"]) <= self.cfg["scoped_cache"]["suffix_max_tokens"]:
            raise ValueError("Suffix length exceeds the frozen budget")
        left_ids = self.ids(record["private_prefix"] + self.cfg["delimiter"])
        chunks = [self.embedding(self.tensor(left_ids))]
        before = 0
        keep = 0
        if prompt is not None:
            keep = self.cfg["prompt_partition"]["keep_length"]
            before = len(prompt) - keep
            chunks.append(prompt[:before].to(self.device, self.dtype)[None])
        chunks.append(self.embedding(self.tensor(self.delimiter_ids + self.public_ids + self.delimiter_ids)))
        if prompt is not None:
            chunks.append(prompt[before:].to(self.device, self.dtype)[None])
        chunks.append(self.embedding(self.tensor(suffix["ids"])))
        embeddings = torch.cat(chunks, dim=1)
        suffix_start = embeddings.shape[1] - len(suffix["ids"])
        _, layers = self._model_forward(embeddings)
        suffix_layers = tuple((key[:, :, suffix_start:].clone(), value[:, :, suffix_start:].clone())
                              for key, value in layers)
        return SuffixContext(suffix_layers, suffix_start, len(suffix["ids"]), tuple(suffix["ids"]))

    def blended_context(self, record: dict, prompt: torch.Tensor) -> Context:
        if self.cfg.get('c1_selection',{}).get('method') in ('epic','none'):
            return self.blend_many([record],prompt)[0]
        writer, source_audit = self.writer_cache(record, prompt)
        public = self.embedding(self.tensor(self.public_ids))
        contract = self.cfg["native_contract"]
        result = native_cacheblend(
            self.model, public, writer,
            check_layer=contract["check_layer"], ratio=contract["recompute_ratio"],
            writer_position_start=source_audit['writer_c1_start'],
            **self._c1_selection([record]),
        )
        self._selection_audit(record,result)
        return Context(result.layers, len(self.public_ids), dict(source_audit, **result.audit))

    def _append(self, context: Context, ids: list[int]):
        embeddings = self.embedding(self.tensor(ids))
        return self._model_forward(embeddings, layers=context.layers,
                                   position_start=context.next_position)

    def output_logits(self, hidden):
        logits=self.model.lm_head(hidden).float()
        if getattr(self,'native',None) is not None and self.input_only_ids:
            ids=torch.tensor(self.input_only_ids,device=logits.device,dtype=torch.long)
            logits=logits.index_fill(-1,ids,float('-inf'))
        return logits

    def _native_source(self,record,prompt,*,epoch=0):
        from .suffix import SuffixContext,pair_for
        from .native_model import WRITER_QUESTION
        suffix=pair_for(record,self.cfg,epoch)
        if not 1<=len(suffix['ids'])<=self.cfg['scoped_cache']['suffix_max_tokens']:
            raise ValueError('S exceeds the configured token budget')
        if prompt is not None and prompt.shape!=(self.cfg['prompt_length'],self.model.config.hidden_size):
            raise ValueError('U has the wrong shape')
        prefix=self.ids(record['private_prefix']+self.cfg['delimiter'])
        chunks=[self.embedding(self.tensor(prefix))]
        keep=self.cfg['prompt_partition']['keep_length'] if prompt is not None else 0
        before=len(prompt)-keep if prompt is not None else 0
        if prompt is not None:
            chunks.append(prompt[:before].to(self.dtype)[None])
        chunks.append(self.embedding(self.tensor(self.delimiter_ids+self.public_ids+self.delimiter_ids)))
        if prompt is not None:
            chunks.append(prompt[before:].to(self.dtype)[None])
        start=len(prefix)+before+len(self.delimiter_ids)
        sstart=sum(chunk.shape[1] for chunk in chunks)
        chunks.append(self.embedding(self.tensor(suffix['ids'])))
        # Match the real writer's complete prefill GEMM shape, including the
        # instruction following S. Causality alone does not guarantee BF16
        # bitwise equality for truncated versus full-request GEMMs.
        chunks.append(self.embedding(self.tensor(self.ids(self.question_text(WRITER_QUESTION)))))
        _,layers=self._model_forward(torch.cat(chunks,dim=1))
        def region(a,b):
            return tuple(tuple(t[:,:,a:b].clone() for t in pair) for pair in layers)
        return region(start,start+len(self.public_ids)),SuffixContext(
            region(sstart,sstart+len(suffix['ids'])),sstart,len(suffix['ids']),tuple(suffix['ids']))

    def _native_teacher_logits(self,context,prefix,target):
        hidden,layers=self._append(context,prefix)
        outputs=[self.output_logits(hidden[:,-1])]
        position=context.next_position+len(prefix)
        for token in target[:-1]:
            hidden,layers=self._append(Context(layers,position,{}),[token])
            outputs.append(self.output_logits(hidden[:,-1]))
            position+=1
        return torch.stack(outputs,dim=1)

    def _native_question_wave(self,contexts,group):
        pairs=[self.question_and_target(qa) for _,_,qa in group]
        count=len(contexts[0].layers)
        flat=tuple(t for owner,_,_ in group for pair in contexts[owner].layers for t in pair)
        def calculate(*values):
            if self.native.fast:
                caches=[]
                for row,(owner,_,_) in enumerate(group):
                    cache=tuple(zip(values[row*count*2:(row+1)*count*2:2],
                                    values[row*count*2+1:(row+1)*count*2:2]))
                    caches.append(Context(cache,contexts[owner].next_position,{}))
                logits,sizes=self._parallel_teacher_batch(caches,
                    [(self.delimiter_ids+question,target) for question,target in pairs])
                labels=self.tensor([token for _,target in pairs for token in target])
                ce=F.cross_entropy(logits.transpose(1,2),labels,reduction='none')[0]
                options=self.cfg['loss']
                focal=options['alpha']*(-torch.expm1(-ce)).clamp_min(torch.finfo(ce.dtype).tiny).pow(options['gamma'])*ce
                rows=[]; offset=0
                for size, (_, _, qa) in zip(sizes, group):
                    rows.append(self._reduce_question_tokens(qa, ce[offset:offset+size]))
                    offset+=size
                return torch.stack(rows)
            losses=[]
            for row,((owner,_,_), (question,target)) in enumerate(zip(group,pairs)):
                cache=tuple(zip(values[row*count*2:(row+1)*count*2:2],
                                values[row*count*2+1:(row+1)*count*2:2]))
                context=Context(cache,contexts[owner].next_position,{})
                logits=self._native_teacher_logits(context,self.delimiter_ids+question,target)
                ce=F.cross_entropy(logits.transpose(1,2),self.tensor(target),reduction='none')
                options=self.cfg['loss']
                focal=options['alpha']*(-torch.expm1(-ce)).clamp_min(torch.finfo(ce.dtype).tiny).pow(options['gamma'])*ce
                losses.append(self._reduce_question_tokens(group[row][2], ce.reshape(-1)))
            return torch.stack(losses)
        if torch.is_grad_enabled() and self.cfg['training']['checkpoint_questions']:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(calculate,*flat,use_reentrant=False,preserve_rng_state=False)
        return calculate(*flat)

    def _parallel_teacher_batch(self,contexts,pairs):
        sequences=[prefix+target[:-1] for prefix,target in pairs]
        lengths=[len(sequence) for sequence in sequences]
        ids=[token for sequence in sequences for token in sequence]
        hidden=self.native.forward_batch(self.model,self.embedding(self.tensor(ids)),
            [context.layers for context in contexts],lengths,[context.next_position for context in contexts])
        selected=[]; offset=0
        for sequence,(prefix,target) in zip(sequences,pairs):
            selected.extend(range(offset+len(prefix)-1,offset+len(sequence)))
            offset+=len(sequence)
        positions=torch.tensor(selected,device=self.device,dtype=torch.long)
        return self.output_logits(hidden.index_select(1,positions)),[len(target) for _,target in pairs]

    def answer_loss(self, context: Context, qa: dict) -> torch.Tensor:
        return self.answer_loss_components(context, qa)[0]

    def answer_loss_components(self, context: Context, qa: dict):
        question, target = self.question_and_target(qa)
        prefix = ([] if context.audit.get('legacy_question_boundary') else self.delimiter_ids) + question
        inputs = prefix + target[:-1]
        start = len(prefix) - 1
        if self.native is not None and self.native.fast:
            logits,_=self._parallel_teacher_batch([context],[(prefix,target)])
        elif self.native is not None:
            logits=self._native_teacher_logits(context,prefix,target)
        else:
            hidden, _ = self._append(context, inputs)
            logits = self.output_logits(hidden[:, start:start + len(target)])
        labels = self.tensor(target)
        token_ce = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        ce = token_ce.mean()
        options = self.cfg["loss"]
        modulation = (-torch.expm1(-token_ce)).clamp_min(torch.finfo(token_ce.dtype).tiny)
        focal = (options["alpha"] * modulation.pow(options["gamma"]) * token_ce).mean()
        return ce, focal

    @torch.no_grad()
    def conditional_scores(self, context, qa, *, content_only=False):
        question, target = self.question_and_target(qa)
        prefix = ([] if context.audit.get('legacy_question_boundary') else self.delimiter_ids) + question
        if self.native is not None and self.native.fast:
            logits=self._parallel_teacher_batch([context],[(prefix,target)])[0][0]
        elif self.native is not None:
            logits=self._native_teacher_logits(context,prefix,target)[0]
        else:
            hidden, _ = self._append(context, prefix + target[:-1])
            logits = self.output_logits(hidden[:, len(prefix)-1:len(prefix)-1+len(target)])[0]
        logp = logits.log_softmax(-1)
        labels = torch.tensor(target, device=self.device)
        losses = -logp.gather(1, labels[:, None]).squeeze(1)
        correct = logits.argmax(-1).eq(labels)
        count = len(target)-1 if content_only else len(target)
        first = logp[0]
        top = first.topk(min(10,first.numel()))
        return dict(nll=float(losses[:count].mean()) if count else None,
                    token_count=count, token_correct=correct[:count].int().tolist(),
                    token_nll=losses[:count].tolist(),
                    token_accuracy=float(correct[:count].float().mean()) if count else None,
                    first8_accuracy=float(correct[:min(8,count)].float().mean()) if count else None,
                    first_stop_probability=float(first[list(self.stops)].exp().sum().clamp(0,1)),
                    first_gold_probability=float(first[target[0]].exp()),
                    first_gold_rank=int((first > first[target[0]]).sum())+1,
                    first_accuracy=int(correct[0]), first_gold_id=target[0], first_predicted_id=int(first.argmax()),
                    first_top10=[dict(id=int(i), probability=float(p.exp()),
                                      text=self.tokenizer.decode([int(i)], skip_special_tokens=False))
                                 for i,p in zip(top.indices,top.values)],
                    terminal_eos_nll=float(losses[-1]),
                    first_logits=logits[0].detach().cpu())

    def preservation_loss(self, record: dict, prompt: torch.Tensor, *, epoch: int):
        from .suffix import preservation_loss
        student = self.suffix_context(record, prompt, epoch=epoch)
        with torch.no_grad():
            reference = self.suffix_context(record, None, epoch=epoch)
        return preservation_loss(
            self, student, reference, ratio=self.cfg["native_contract"]["recompute_ratio"],
        )

    @torch.no_grad()
    def generate(self, context: Context, qa: dict, *, max_new_tokens=None):
        question = ([] if context.audit.get('legacy_question_boundary') else self.delimiter_ids) + self.ids(self.question_text(qa["question"]))
        hidden, layers = self._append(context, question)
        logits = self.output_logits(hidden[:, -1:])
        output, stop = [], "max_new_tokens"
        position = context.next_position + len(question)
        for _ in range(max_new_tokens or self.cfg["generation"]["max_new_tokens"]):
            token = int(logits[0, -1].argmax())
            if token in self.stops:
                stop = "eos"
                break
            output.append(token)
            embeddings = self.embedding(self.tensor([token]))
            hidden, layers = self._model_forward(embeddings, layers=layers, position_start=position)
            logits = self.output_logits(hidden[:, -1:])
            position += 1
        return self.tokenizer.decode(output, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False), {
            "generated_token_ids": output, "stop_reason": stop,
            "truncated": stop == "max_new_tokens",
        }



def per_sample_waves(question_sets, count):
    """Up to Q actual questions per active record, in stable original order."""
    if count < 1:
        raise ValueError('Q must be positive')
    for offset in range(0, max(map(len, question_sets), default=0), count):
        yield [(owner, index, questions[index]) for owner, questions in enumerate(question_sets)
               for index in range(offset, min(offset+count, len(questions)))]
