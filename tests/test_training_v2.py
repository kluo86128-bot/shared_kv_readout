from pathlib import Path
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from kv_cache_readout.backend import Backend
from kv_cache_readout.config import load_config
from kv_cache_readout.optim import parameters_for, assign_gradients, schedule_factor
from kv_cache_readout.suffix import pair_for, relative_per_head
from kv_cache_readout.training import batch_gradients
from e2e_cache.common import score


class Tokenizer:
    eos_token_id = 0
    all_special_ids = [0]
    backend_tokenizer = SimpleNamespace(to_str=lambda: '{}')

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)


@pytest.fixture(params=[0, 1], ids=['layer0_aligned', 'layer1_baseline'])
def tiny(request):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    cfg = load_config(Path(__file__).resolve().parent/'fixtures/base.json')
    cfg['native_contract']['check_layer'] = request.param
    cfg.update(device='cpu', dtype='float32', public_text='x'*56, question_format='plain',
               model_name_or_path='missing-tiny-assets')
    cfg['training'].update(source_batch_records=2, question_batch_size=2, preservation_batch_records=2)
    model_cfg = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=24,
                           num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=8, max_position_embeddings=2048, eos_token_id=0)
    model_cfg._attn_implementation = 'eager'
    backend = Backend(Qwen3ForCausalLM(model_cfg), Tokenizer(), cfg)
    records = []
    for i in range(2):
        suffix = dict(ids=[65+i,66,67,68], text='ABCD', suffix_id=f's{i}', article_id='a')
        qs = [dict(field='title', group='title', question='title?', answer='M'),
              dict(field='email', group='email', question='email?', answer='A')]
        records.append(dict(record_id=f'r{i}', private_prefix='private'+(' more'*i), split='train',
                            qa=qs, fields={'title':'M','email':'A'}, _suffix_fixture=suffix))
    initial = torch.randn(64,16)*.02
    return backend, records, initial


def test_three_parameter_adam_skips_inactive_momentum():
    cfg = load_config(Path(__file__).resolve().parent/'fixtures/base.json')
    ps = parameters_for(torch.ones(64,4), cfg)
    optimizer = torch.optim.AdamW(ps, lr=.01, weight_decay=0)
    assign_gradients(ps, torch.ones(64,4), [True,True,True]); optimizer.step()
    before = ps[1].detach().clone()
    old_step = optimizer.state[ps[1]]['step'].clone()
    assign_gradients(ps, torch.ones(64,4), [True,False,True]); optimizer.step()
    assert torch.equal(before, ps[1])
    assert torch.equal(old_step, optimizer.state[ps[1]]['step'])


def test_original_schedule_first_and_warmup_boundary():
    assert schedule_factor(0,25000,.05) == 1/1250
    assert schedule_factor(1249,25000,.05) == 1
    assert schedule_factor(1250,25000,.05) == 1
    assert schedule_factor(25000,25000,.05) == 0


def test_suffix_pairing_matches_original_serialization():
    from kv_cache_readout.config import resolve
    cfg = load_config(Path(__file__).resolve().parent/'fixtures/base.json')
    record = dict(record_id='hr_narrative_train_000001', split='train')
    path = resolve(cfg['scoped_cache']['suffix_directory']) / 'train.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    key = dict(seed=17,record_id=record['record_id'],split='train',epoch=3)
    old_hash = hashlib.sha256(json.dumps(key,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    assert pair_for(record,cfg,3)['suffix_id'] == rows[int(old_hash[:16],16)%len(rows)]['suffix_id']


def test_keep_error_weights_each_head_equally():
    target = torch.tensor([1.,100.]).reshape(1,2,1,1)
    current = torch.tensor([2.,100.]).reshape(1,2,1,1)
    assert relative_per_head(current,target,1e-6).item() == .5


def test_batched_sources_and_question_checkpoint_match_individual_gradients(tiny, monkeypatch):
    backend, records, initial = tiny
    # Layer0 K is context independent. RoPE/GEMM roundoff can break its
    # mathematical score ties differently in batched versus single prefill.
    # Hold that nondifferentiable branch fixed to compare gradient batching;
    # the separate sparse tests exercise the actual top-k selection.
    choices, replay, replay_index = [], False, 0
    original_topk = torch.Tensor.topk
    if backend.cfg['native_contract']['check_layer'] == 0:
        def topk(tensor, k, *args, **kwargs):
            nonlocal replay_index
            if tensor.ndim != 1 or tensor.numel() != 56 or k != 16:
                return original_topk(tensor, k, *args, **kwargs)
            if not replay:
                result = original_topk(tensor, k, *args, **kwargs)
                choices.append(result.indices.clone())
                return result
            indices = choices[replay_index]
            replay_index += 1
            return torch.return_types.topk((tensor[indices], indices))
        monkeypatch.setattr(torch.Tensor, 'topk', topk)
    ps = parameters_for(initial, backend.cfg)
    g_batch, loss_batch, flags = batch_gradients(backend,ps,records,2,0)
    replay = True
    backend.cfg['training'].update(source_batch_records=1,preservation_batch_records=1,
                                    question_batch_size=1,checkpoint_questions=False)
    g_single, loss_single, _ = batch_gradients(backend,ps,records,2,0)
    if choices:
        assert replay_index == len(choices) == len(records)
    assert flags == [True,True,True]
    assert abs(loss_batch-loss_single) < 1e-5
    assert torch.allclose(g_batch,g_single,atol=3e-5,rtol=3e-4)
    assert torch.isfinite(g_batch).all() and all(g.norm()>0 for g in g_batch.split([32,16,16]))
    assert all(p.grad is None for p in backend.model.parameters())


def test_keep_vectors_cannot_change_c1(tiny):
    backend,records,initial=tiny
    changed=initial.clone(); changed[-16:]+=1
    a,_=backend.source_many(records,initial)
    b,_=backend.source_many(records,changed)
    assert all(torch.equal(x,y) for ar,br in zip(a,b) for ap,bp in zip(ar,br) for x,y in zip(ap,bp))


def test_training_and_offline_readout_share_the_configured_layer0_path(tiny, monkeypatch):
    import kv_cache_readout.native_blend as blend
    backend, records, initial = tiny
    original = blend._decoder_layer
    counts = []

    def capture(layer, residual, query, key, value, mask):
        counts.append(query.shape[2])
        return original(layer, residual, query, key, value, mask)

    monkeypatch.setattr(blend, '_decoder_layer', capture)
    batched = backend.blend_many(records[:1], initial)[0]
    single = backend.blended_context(records[0], initial)
    first = 16 if backend.cfg['native_contract']['check_layer'] == 0 else 56
    assert counts == [first, 16, 16] * 2
    assert batched.audit['check_layer'] == single.audit['check_layer'] == backend.cfg['native_contract']['check_layer']
    assert batched.audit['selected_indices'] == single.audit['selected_indices']
    for a, b in zip(batched.layers, single.layers):
        assert all(torch.allclose(x, y, atol=1e-6, rtol=1e-5) for x, y in zip(a, b))


def test_checkpoint_rejects_a_different_recomputation_start(tiny):
    from kv_cache_readout.config import fingerprint
    from kv_cache_readout.training import validate_checkpoint, CHECKPOINT_SCHEMA
    backend, _, _ = tiny
    cfg = backend.cfg
    old_contract = dict(cfg['native_contract'], check_layer=1-cfg['native_contract']['check_layer'])
    state = dict(schema_version=CHECKPOINT_SCHEMA, config_sha256=fingerprint(cfg),
                 dataset={'fixture': 'frozen'}, native_contract=old_contract)
    with pytest.raises(ValueError, match='native_contract'):
        validate_checkpoint(state, cfg, backend, {'fixture': 'frozen'})


def test_e2e_actual_source_success_is_separate_from_target():
    records={'a':dict(qa=[dict(field='title',question='title?',answer='A')]),
             'b':dict(qa=[dict(field='title',question='title?',answer='B')])}
    history=[dict(id='w0',record_id='a'),dict(id='w1',record_id='b')]
    reader=dict(field='title',question='title?',source_request='w1',source_record='b')
    audit=dict(loaded_sources=['w0'],loaded_keys=['k'],complete=True,unknown_sources=[])
    result=score(reader,'A',audit,history,records)
    assert result['success'] and result['source_has_field']
    assert not result['target_success'] and not result['target_source_loaded']


def test_training_resume_matches_continuous_three_adam_states(tiny,tmp_path,monkeypatch):
    import kv_cache_readout.training as training
    from kv_cache_readout.io import read_checkpoint
    backend,records,_=tiny
    cfg=copy.deepcopy(backend.cfg)
    cfg['training'].update(epochs=2,batch_records=2)
    # Initialization words need at least one single character in the tiny tokenizer.
    monkeypatch.setattr(Backend,'initial_prompt',lambda self: torch.ones(64,16)*.01)
    monkeypatch.setattr(training.Backend,'load',lambda config: Backend(copy.deepcopy(backend.model),Tokenizer(),config))
    monkeypatch.setattr(training,'load_split',lambda *a,**kw: records)
    monkeypatch.setattr(training,'dataset_identity',lambda cfg: {'fixture':'frozen'})
    cfg['output_directory']=str(tmp_path/'continuous')
    training.train(cfg)
    continuous=read_checkpoint(tmp_path/'continuous'/'latest.pt')
    cfg['output_directory']=str(tmp_path/'resumed')
    training.train(cfg,stop_after_epochs=1)
    training.train(cfg,resume=True)
    resumed=read_checkpoint(tmp_path/'resumed'/'latest.pt')
    assert torch.equal(continuous['prompt'],resumed['prompt'])
    for a,b in zip(continuous['optimizer']['state'].values(),resumed['optimizer']['state'].values()):
        assert all(torch.equal(a[k],b[k]) for k in a)
    assert continuous['scheduler']==resumed['scheduler']


def test_all_19_conditions_and_evaluation_resume(tiny,tmp_path,monkeypatch):
    import kv_cache_readout.evaluation as evaluation
    from kv_cache_readout.training import CHECKPOINT_SCHEMA
    from kv_cache_readout.config import fingerprint
    from kv_cache_readout.io import code_identity, atomic_torch
    backend,records,initial=tiny
    donor=dict(ids=[70,71,72,73],text='FGHI',suffix_id='wrong',article_id='b')
    monkeypatch.setattr(evaluation,'mismatch_for',lambda *a:donor)
    monkeypatch.setattr(backend,'generate',lambda ctx,qa,**kw:(qa['answer'],dict(stop_reason='eos',truncated=False)))
    monkeypatch.setattr(evaluation,'load_split',lambda *a,**kw:records)
    monkeypatch.setattr(evaluation,'dataset_identity',lambda *a:{'fixture':'frozen'})
    monkeypatch.setattr(evaluation.Backend,'load',lambda cfg:backend)
    checkpoint=tmp_path/'u.pt'
    atomic_torch(checkpoint,dict(schema_version=CHECKPOINT_SCHEMA,prompt=initial,
        config_sha256=fingerprint(backend.cfg),dataset={'fixture':'frozen'},
        native_contract=backend.cfg['native_contract'],model=backend.identity,code=code_identity()))
    summaries=evaluation.evaluate_suite(backend.cfg,str(checkpoint),str(tmp_path/'suite'))
    assert len(summaries)==19 and all(row['status']=='complete' for row in summaries)
    def no_generation(*a,**kw):
        raise AssertionError('Committed rows were regenerated')
    monkeypatch.setattr(backend,'generate',no_generation)
    resumed=evaluation.evaluate_suite(backend.cfg,str(checkpoint),str(tmp_path/'suite'),resume=True)
    assert resumed==summaries
    target=tmp_path/'suite'/'c1_native_r30'/'record_000000.json'
    value=json.loads(target.read_text()); value['rows'][0]['prediction']='tampered'
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Committed prediction changed'):
        evaluation.evaluate_suite(backend.cfg,str(checkpoint),str(tmp_path/'suite'),resume=True)


def test_layer_zero_stored_cache_retains_writer_and_relocation_matches_clean_rope(tiny):
    from kv_cache_readout.native_blend import relocate_keys, native_cacheblend
    backend,records,initial=tiny
    writer,audit=backend.writer_cache(records[0],initial)
    relocated=relocate_keys(backend.model.model,writer,audit['writer_c1_start'],0)
    trace=[]
    result=native_cacheblend(backend.model,backend.embedding(backend.tensor(backend.public_ids)),writer,
                             writer_position_start=audit['writer_c1_start'],check_layer=1,ratio=.3,trace=trace)
    assert torch.equal(result.layers[0][0],relocated[0][0])
    assert torch.equal(result.layers[0][1],writer[0][1])
    assert not torch.equal(trace[0]['attention_key'],trace[0]['stored_key'])
    reverse=relocate_keys(backend.model.model,relocated,0,audit['writer_c1_start'])
    assert torch.allclose(writer[1][0],reverse[1][0],atol=1e-6,rtol=1e-5)


def test_failed_keep_phase_does_not_commit_partial_qa_update(tiny,tmp_path,monkeypatch):
    import kv_cache_readout.training as training
    from kv_cache_readout.io import read_checkpoint
    backend,records,_=tiny
    cfg=copy.deepcopy(backend.cfg)
    cfg['training'].update(epochs=2,batch_records=2)
    cfg['output_directory']=str(tmp_path/'failure')
    monkeypatch.setattr(Backend,'initial_prompt',lambda self: torch.ones(64,16)*.01)
    monkeypatch.setattr(training.Backend,'load',lambda config:Backend(copy.deepcopy(backend.model),Tokenizer(),config))
    monkeypatch.setattr(training,'load_split',lambda *a,**kw:records)
    monkeypatch.setattr(training,'dataset_identity',lambda cfg:{'fixture':'frozen'})
    training.train(cfg,max_updates=1)
    path=tmp_path/'failure'/'latest.pt'
    before=read_checkpoint(path)
    def fail(*a,**kw):
        raise RuntimeError('injected S failure')
    monkeypatch.setattr(training,'keep_phase',fail)
    with pytest.raises(RuntimeError,match='No optimizer update committed'):
        training.train(cfg,resume=True)
    after=read_checkpoint(path)
    # Finishing the saved epoch boundary may normalize cursor/epoch and save
    # an equivalent checkpoint, but must not commit the failed QA gradients.
    assert torch.equal(after['prompt'],before['prompt'])
    assert after['progress']['step']==before['progress']['step']
    assert after['scheduler']==before['scheduler']
    for a,b in zip(after['optimizer']['state'].values(),before['optimizer']['state'].values()):
        assert all(torch.equal(a[k],b[k]) for k in a)


def test_reconstruction_preserves_predicted_components_without_gold_correction():
    from kv_cache_readout.evaluation import reconstruction_rows
    record=dict(record_id='r',fields={'name':'A B','birth_date':'1980-01-02'})
    predictions={'name_first':'A','name_middle':'wrong','name_last':'B',
                 'birth_year':'1980','birth_month':'1','birth_day':'02'}
    rows=[dict(field=k,prediction=v,nll=1.) for k,v in predictions.items()]
    reconstructed=reconstruction_rows(record,rows)
    assert reconstructed[0]['prediction']=='A wrong B' and reconstructed[0]['em']==0
    assert reconstructed[1]['prediction']=='1980-1-02' and reconstructed[1]['em']==0

