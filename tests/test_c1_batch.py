import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from kv_cache_readout.backend import Backend
from kv_cache_readout.c1_batch import native_cacheblend_many
from kv_cache_readout.config import validate_config, fingerprint
from kv_cache_readout.native_blend import native_cacheblend
from kv_cache_readout.native_model import FAST_EXECUTION
from kv_cache_readout.optim import parameters_for
from kv_cache_readout.training import batch_gradients
from test_native_autograd import native_backend


@pytest.fixture
def pair(native_backend):
    original, records, prompt = native_backend
    cfg = copy.deepcopy(original.cfg)
    cfg['execution'] = dict(FAST_EXECUTION)
    return Backend(copy.deepcopy(original.model), original.tokenizer, cfg), records, prompt


def inputs(backend, records, prompt):
    writers, suffixes = backend.source_many(records, prompt)
    starts = [s.start - 56 - len(backend.delimiter_ids) - 16 for s in suffixes]
    return backend.embedding(backend.tensor(backend.public_ids)), writers, starts


@pytest.mark.parametrize('batch_size', [2, 4])
def test_packed_c1_matches_serial_values_selection_and_gradients(pair, monkeypatch, batch_size):
    backend, records, initial = pair
    records = [dict(copy.deepcopy(records[i % 2]),
                    private_prefix=records[i % 2]['private_prefix'] + ' extra' * (i // 2))
               for i in range(batch_size)]
    prompt = initial.clone().requires_grad_()
    public, writers, starts = inputs(backend, records, prompt)
    serial = [native_cacheblend(backend.model, public, w, check_layer=backend.cfg['native_contract']['check_layer'],
              ratio=.3, writer_position_start=s) for w, s in zip(writers, starts)]
    calls = []
    original = backend.native.project
    def observe(*args, **kwargs):
        calls.append(args[2].shape[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(backend.native, 'project', observe)
    packed = native_cacheblend_many(backend.model, public, writers, writer_position_starts=starts,
        check_layer=backend.cfg['native_contract']['check_layer'], ratio=.3)
    assert len(calls) == len(writers[0]) + backend.cfg['native_contract']['check_layer'] + 1 and calls[0] == batch_size * 56
    weights = [[torch.randn_like(t) for layer in r.layers for t in layer] for r in serial]
    def loss(results):
        return sum((t * weight).sum() for r, ws in zip(results, weights)
                   for t, weight in zip((t for layer in r.layers for t in layer), ws))
    for a, b in zip(serial, packed):
        assert torch.equal(a.selected, b.selected) and len(b.selected) == 16
        assert torch.allclose(a.scores, b.scores, atol=2e-5, rtol=2e-5)
        assert all(torch.allclose(x, y, atol=2e-5, rtol=2e-5)
                   for la, lb in zip(a.layers, b.layers) for x, y in zip(la, lb))
    a, = torch.autograd.grad(loss(serial), prompt, retain_graph=True)
    b, = torch.autograd.grad(loss(packed), prompt)
    assert torch.allclose(a, b, atol=5e-5, rtol=5e-4)


def test_c1_packed_samples_cannot_read_or_receive_gradients_from_other_writer(pair):
    backend, records, prompt = pair
    with torch.no_grad():
        public, sources, starts = inputs(backend, records, prompt)
    writers = [tuple(tuple(t.detach().requires_grad_() for t in layer) for layer in source)
               for source in sources]
    options = dict(writer_position_starts=starts,
                   check_layer=backend.cfg['native_contract']['check_layer'], ratio=.3)
    results = native_cacheblend_many(backend.model, public, writers, **options)
    second = [t for layer in writers[1] for t in layer]
    gradients = torch.autograd.grad(sum(t.sum() for layer in results[0].layers for t in layer),
                                    second, allow_unused=True)
    assert all(g is None or g.count_nonzero() == 0 for g in gradients)
    changed = [writers[0], tuple(tuple(t.detach() + 30 for t in layer) for layer in writers[1])]
    other = native_cacheblend_many(backend.model, public, changed, **options)
    assert torch.equal(results[0].selected, other[0].selected)
    assert all(torch.equal(x, y) for a, b in zip(results[0].layers, other[0].layers) for x, y in zip(a, b))


def test_full_training_gradient_with_c1_batch_and_checkpoint_matches_serial(pair):
    backend, records, prompt = pair
    parameters = parameters_for(prompt, backend.cfg)
    a, la, fa = batch_gradients(backend, parameters, records, 2, 0)
    backend.cfg['training']['c1_batch_size'] = 2
    b, lb, fb = batch_gradients(backend, parameters, records, 2, 0)
    assert fa == fb == [True, True, True]
    assert abs(la - lb) < 2e-5
    assert torch.allclose(a, b, atol=3e-5, rtol=3e-4)
    # A final partial group uses the unchanged single-C1 reference.
    with torch.no_grad():
        one = backend.blend_many(records[:1], prompt)
    assert 'c1_batch_records' not in one[0].audit


def test_c1_batch_config_is_opt_in_and_restricted_to_fast(pair):
    backend, _, _ = pair
    original = copy.deepcopy(backend.cfg)
    validate_config(backend.cfg)
    assert backend.cfg == original and 'c1_batch_size' not in backend.cfg['training']
    backend.cfg['training']['c1_batch_size'] = 2
    validate_config(backend.cfg)
    for bad in (0, True, 1.5):
        backend.cfg['training']['c1_batch_size'] = bad
        with pytest.raises(ValueError):
            validate_config(backend.cfg)




@pytest.mark.parametrize('order', ['record_major', 'round_robin'])
def test_packed_c1_resume_matches_continuous_training(pair, tmp_path, monkeypatch, order):
    from kv_cache_readout import training
    from kv_cache_readout.io import read_checkpoint
    backend, records, prompt = pair
    cfg = copy.deepcopy(backend.cfg)
    cfg['training'].update(c1_batch_size=2, epochs=2, batch_records=2, question_order=order)
    template = copy.deepcopy(backend.model)
    monkeypatch.setattr(Backend, 'initial_prompt', lambda self: prompt.clone())
    monkeypatch.setattr(training.Backend, 'load', lambda c: Backend(copy.deepcopy(template), backend.tokenizer, c))
    monkeypatch.setattr(training, 'load_split', lambda *a, **kw: records)
    monkeypatch.setattr(training, 'dataset_identity', lambda c: {'fixture': 'packed-c1'})
    cfg['output_directory'] = str(tmp_path/'continuous')
    training.train(cfg)
    a = read_checkpoint(tmp_path/'continuous/latest.pt')
    cfg['output_directory'] = str(tmp_path/'resumed')
    training.train(cfg, max_updates=1)
    training.train(cfg, resume=True)
    b = read_checkpoint(tmp_path/'resumed/latest.pt')
    assert torch.equal(a['prompt'], b['prompt'])
    assert a['progress'] == b['progress'] and a['scheduler'] == b['scheduler']
    for first, second in zip(a['optimizer']['state'].values(), b['optimizer']['state'].values()):
        assert all(torch.equal(first[k], second[k]) for k in first)




@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires server CUDA/vLLM/LMCache')
def test_cuda_packed_c1_matches_independent_reference():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from kv_cache_readout.native_model import NativeExecution
    torch.manual_seed(133)
    cfg = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=32)
    model = Qwen3ForCausalLM(cfg).to(device='cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
    execution = NativeExecution(model, FAST_EXECUTION)
    model._readout_execution = model.model._readout_execution = execution
    public = torch.randn(1, 56, 64, device='cuda', dtype=torch.bfloat16)
    writers = [tuple((torch.randn(1, 1, 56, 32, device='cuda', dtype=torch.bfloat16),
                      torch.randn(1, 1, 56, 32, device='cuda', dtype=torch.bfloat16)) for _ in range(3))
               for _ in range(2)]
    packed = native_cacheblend_many(model, public, writers, writer_position_starts=[0, 0], check_layer=0, ratio=.3)
    serial = [native_cacheblend(model, public, w, check_layer=0, ratio=.3) for w in writers]
    for a, b in zip(packed, serial):
        assert torch.equal(a.selected, b.selected)
        for first, second in zip(a.layers, b.layers):
            for x, y in zip(first, second):
                torch.testing.assert_close(x, y, atol=.03, rtol=.03)

