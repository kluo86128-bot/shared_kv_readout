import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer as FrozenTokenizer
from transformers import Qwen3Config, Qwen3ForCausalLM

from kv_cache_readout.backend import Backend
from kv_cache_readout.config import ROOT, load_config, validate_config
from kv_cache_readout.data import load_split, dataset_identity
from kv_cache_readout.native_model import FAST_EXECUTION
from kv_cache_readout.optim import parameters_for
from kv_cache_readout.training import batch_gradients
from test_training_v2 import Tokenizer

REMOVED = {'hire_date', 'leave_start', 'complaint_target', 'complaint_content'}


def test_all_v4_records_preserve_splits_answers_and_valid_evidence():
    cfg = load_config('configs/train_hr_v4_c128.json')
    dest = ROOT/cfg['dataset_directory']
    manifest = json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((dest/name).read_bytes()).hexdigest() == expected
    for split, count in [('train', 2000), ('validation', 400), ('test', 2000)]:
        original = [r for line in (ROOT/'data/dataset_multiscene_v3_hr_balanced'/f'{split}.jsonl').open(encoding='utf-8')
                    if (r := json.loads(line))['scene'] == 'hr']
        records = [json.loads(line) for line in (dest/f'{split}.jsonl').open(encoding='utf-8')]
        assert len(records) == count
        assert [r['record_id'] for r in records] == [r['record_id'] for r in original]
        for old, new in zip(original, records):
            assert new['group_id'] == old['group_id']
            assert new['fields'] == {('position' if k == 'title' else k): v
                                     for k, v in old['fields'].items() if k not in REMOVED}
            text = new['private_prefix']
            assert not any(label in text for label in
                           ('hire_date:', '\ntitle:', 'Leave start: ', 'Person complained about: ', 'Conduct complained about: '))
            for k, annotation in new['field_annotations'].items():
                assert k not in REMOVED | {'title'}
                a, b = annotation['evidence_span']
                assert text[a:b] == annotation['value'] == new['fields'][k]
            for name in ('qa', 'training_qa', 'evaluation_qa'):
                expected = {('position' if q['field'] == 'title' else q['field']): q['answer']
                            for q in old[name] if q['field'] not in REMOVED}
                assert {q['field']: q['answer'] for q in new[name]} == expected
                for qa in new[name]:
                    assert qa['group'] not in REMOVED | {'title'}
                    if 'evidence_span' in qa:
                        a, b = qa['evidence_span']
                        assert text[a:b] == qa['answer']
        assert len(load_split(cfg, split, training=split == 'train')) == count
    assert dataset_identity(cfg)['files']['train.jsonl']['sha256'] == manifest['files']['train.jsonl']


def test_c128_configs_match_frozen_tokenizer_and_keep_u64():
    tok = FrozenTokenizer.from_file(str(ROOT/'data/tokenizer/Qwen3-4B/tokenizer.json'))
    for filename in ('train_hr_v4_c128.json', 'train_hr_v4_c128_server2.json'):
        cfg = load_config('configs/'+filename)
        assert len(tok.encode(cfg['public_text'], add_special_tokens=False).ids) == cfg['native_contract']['c1_tokens'] == 128
        assert cfg['prompt_length'] == 64
        assert not REMOVED & set(cfg['focal_fields'])
    for invalid in (0, -1, 55, True, 128.0):
        cfg = load_config('configs/train_hr_v4_c128.json')
        cfg['native_contract']['c1_tokens'] = invalid
        with pytest.raises(ValueError, match='C1'):
            validate_config(cfg)


def test_c128_native_training_forward_and_gradient():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    cfg = load_config('configs/train_hr_v4_c128.json')
    cfg.update(device='cpu', dtype='float32', public_text='x'*128,
               model_name_or_path='missing-tiny-assets', question_format='plain', execution=dict(FAST_EXECUTION))
    # This character tokenizer lets a tiny Qwen model exercise exactly 128 positions.
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=128, hidden_size=16,
        intermediate_size=24, num_hidden_layers=3, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, max_position_embeddings=2048, eos_token_id=0))
    backend = Backend(model, Tokenizer(), cfg)
    records = [dict(record_id=f'c128_{i}', private_prefix='private'+' extra'*i,
        qa=[dict(field='position', group='position', question='position?', answer='M'),
            dict(field='email', group='email', question='email?', answer='A')],
        _suffix_fixture=dict(ids=[65,66,67], text='ABC', suffix_id=f's{i}', article_id='a')) for i in range(2)]
    initial = torch.randn(64, 16)*.02
    with torch.no_grad():
        contexts = backend.blend_many(records, initial)
        assert all(c.layers[0][0].shape[2] == c.next_position == 128 for c in contexts)
    grad, loss, active = batch_gradients(backend, parameters_for(initial, cfg), records, 2, 0)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert torch.isfinite(torch.as_tensor(loss))
