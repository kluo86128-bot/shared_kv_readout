"""Independent ragged prefill and shrinking active-set greedy decoding."""
import torch
from .backend import Context


@torch.no_grad()
def generate_many(backend, contexts, questions, max_new_tokens=None):
    if len(contexts) != len(questions):
        raise ValueError('Context/question count mismatch')
    if not questions:
        return []
    if backend.native is None or not backend.native.fast:
        return [backend.generate(c, q, max_new_tokens=max_new_tokens)
                for c, q in zip(contexts, questions)]
    cap = max_new_tokens or backend.cfg['generation']['max_new_tokens']
    sequences = [backend.delimiter_ids + backend.ids(backend.question_text(q['question']))
                 for q in questions]
    outputs=[[] for _ in questions]
    reasons=['max_new_tokens']*len(questions)
    active=list(range(len(questions)))
    caches=[c.layers for c in contexts]
    starts=[c.next_position for c in contexts]
    for step in range(cap):
        lengths=list(map(len,sequences))
        tokens=[token for seq in sequences for token in seq]
        hidden,updated=backend.native.forward_batch(backend.model,
            backend.embedding(backend.tensor(tokens)),caches,lengths,starts,return_cache=True)
        ends=torch.tensor(lengths,device=backend.device).cumsum(0)-1
        predicted=backend.output_logits(hidden.index_select(1,ends))[0].argmax(-1).tolist()
        next_active=[]; next_caches=[]; next_starts=[]; next_sequences=[]
        for row,(owner,token) in enumerate(zip(active,predicted)):
            if token in backend.stops:
                reasons[owner]='eos'
                continue
            outputs[owner].append(token)
            next_active.append(owner);next_caches.append(updated[row])
            next_starts.append(starts[row]+lengths[row]);next_sequences.append([token])
        active,caches,starts,sequences=next_active,next_caches,next_starts,next_sequences
        if not active:
            break
    return [(backend.tokenizer.decode(ids,skip_special_tokens=False,
             clean_up_tokenization_spaces=False),dict(generated_token_ids=ids,
             stop_reason=reason,truncated=reason=='max_new_tokens'))
            for ids,reason in zip(outputs,reasons)]
