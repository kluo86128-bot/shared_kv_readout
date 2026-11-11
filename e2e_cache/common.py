"""Configuration, deterministic workloads, attribution and durable run files."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import uuid

ROOT = Path(__file__).resolve().parents[1]


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha_file(value):
    result = hashlib.sha256()
    with Path(value).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(target, value):
    target = Path(target)
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def read_json(value):
    return json.loads(Path(value).read_text(encoding='utf-8'))


def load_settings(value):
    cfg = read_json(path(value))
    required = {'experiment_config', 'source_u', 'deployment', 'output',
                'rounds', 'seed', 'writers_min', 'writers_max', 'writer_batch_size',
                'workload', 'materials', 'recomputation', 'cache', 'engine', 'use_u', 'use_a'}
    if not required <= cfg.keys() or cfg.keys() - required - {
            'adapter', 'evaluation_override', 'model_name_or_path'}:
        raise ValueError('Missing required configuration keys or unexpected keys')
    if 'model_name_or_path' in cfg and (
            not isinstance(cfg['model_name_or_path'], str) or not cfg['model_name_or_path']):
        raise ValueError('model_name_or_path must be a nonempty local model path')
    cfg.setdefault('adapter', None)
    for key in ('rounds', 'writers_min', 'writers_max', 'writer_batch_size'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(cfg['seed']) is not int or cfg['seed'] < 0 or cfg['writers_min'] > cfg['writers_max']:
        raise ValueError('Invalid seed or writer range')
    if any(type(cfg[k]) is not bool for k in ('use_u', 'use_a')):
        raise ValueError('use_u/use_a must be booleans')
    if cfg['use_a'] and (not isinstance(cfg['adapter'], str) or not cfg['adapter']):
        raise ValueError('use_a=true requires an adapter checkpoint path')
    if cfg['workload'] not in ('hotspot', 'competition'):
        raise ValueError('workload must be hotspot or competition')
    if not cfg['materials'] or any(set(m) != {'id', 'text', 'weight'} for m in cfg['materials']):
        raise ValueError('Each material requires id, text, weight')
    if len({m['id'] for m in cfg['materials']}) != len(cfg['materials']):
        raise ValueError('Duplicate material IDs')
    if len({m['text'] for m in cfg['materials']}) != len(cfg['materials']):
        raise ValueError('Materials must have distinct text')
    for m in cfg['materials']:
        if not m['id'] or not m['text'] or not math.isfinite(m['weight']) or m['weight'] <= 0:
            raise ValueError('Invalid material')
    if cfg['workload'] == 'hotspot' and len(cfg['materials']) != 1:
        raise ValueError('hotspot requires exactly one material')
    r = cfg['recomputation']
    if set(r) != {'method', 'ratio', 'check_layer'} or r['method'] not in ('cacheblend', 'random'):
        raise ValueError('Native runner supports cacheblend and experimental random selector; EPIC requires a separate port')
    if not math.isfinite(r['ratio']) or not 0 < r['ratio'] <= 1 or type(r['check_layer']) is not int or r['check_layer'] < 0:
        raise ValueError('Invalid blending ratio/check layer')
    override = cfg.get('evaluation_override')
    if override is not None and override != {
            'check_layer': 0, 'reason': 'layer0_replacement_ablation'}:
        raise ValueError('Only the explicit layer0 replacement ablation override is supported')
    if override is not None and (r['method'] != 'cacheblend' or r['check_layer'] != 0):
        raise ValueError('Layer0 ablation requires cacheblend with check_layer=0')
    c = cfg['cache']
    if set(c) != {'cpu_gb', 'eviction', 'sharing', 'chunk_size', 'delimiter'}:
        raise ValueError('Invalid cache configuration')
    if c['eviction'] not in ('LRU', 'LFU', 'FIFO', 'MRU') or c['sharing'] not in ('shared', 'isolated', 'cold'):
        raise ValueError('Invalid eviction/sharing setting')
    if not math.isfinite(c['cpu_gb']) or c['cpu_gb'] <= 0 or type(c['chunk_size']) is not int or c['chunk_size'] < 1 or not c['delimiter']:
        raise ValueError('Invalid cache size/delimiter')
    allowed = {'max_model_len', 'gpu_memory_utilization', 'max_num_seqs', 'max_new_tokens', 'device'}
    if set(cfg['engine']) != allowed:
        raise ValueError(f'engine requires {sorted(allowed)}')
    for k in ('max_model_len', 'max_num_seqs', 'max_new_tokens'):
        if type(cfg['engine'][k]) is not int or cfg['engine'][k] < 1:
            raise ValueError(f'Invalid engine {k}')
    if not 0 < cfg['engine']['gpu_memory_utilization'] < 1:
        raise ValueError('Invalid GPU memory utilization')
    return cfg


def schedule(cfg, records):
    rng = random.Random(cfg['seed'])
    result = []
    by_id = {record['record_id']: record for record in records}
    for index in range(cfg['rounds']):
        writers = []
        for j in range(rng.randint(cfg['writers_min'], cfg['writers_max'])):
            record = rng.choice(records)
            material = rng.choices(cfg['materials'], weights=[m['weight'] for m in cfg['materials']])[0]
            writers.append({'id': f'r{index}:w{j}', 'round': index, 'record_id': record['record_id'],
                            'material': material['id'], 'tenant': f'writer:{record["record_id"]}'})
        material = rng.choices(cfg['materials'], weights=[m['weight'] for m in cfg['materials']])[0]
        # Bind every read to a concrete writer and one of that source record's
        # actual fields. This prevents impossible absent-field rounds.
        source = rng.choice(writers)
        available = [qa for qa in by_id[source['record_id']]['qa'] if qa['answer'] != '']
        if not available:
            raise ValueError(f'Writer record has no nonempty readable fields: {source["record_id"]}')
        qa = rng.choice(available)
        result.append({'index': index, 'writers': writers, 'reader': {
            'id': f'r{index}:reader', 'round': index, 'tenant': 'reader', 'material': material['id'],
            'field': qa['field'], 'question': qa['question'],
            'source_request': source['id'], 'source_record': source['record_id']}})
    return result


def score(reader, text, audit, history, records):
    """Strict EM against prior writers, with actual block-source support."""
    prior = {event['id']: event for event in history}
    matches = []
    for event in history:
        record = records[event['record_id']]
        if any(q['field'] == reader['field'] and q['answer'] == text and text != '' for q in record['qa']):
            matches.append(event['id'])
    loaded = set(audit['loaded_sources'])
    attributable = sorted(loaded & set(matches) & prior.keys())
    distinct_records = {prior[k]['record_id'] for k in matches}
    complete = audit['complete'] and not audit.get('unknown_sources')
    expected_request = reader['source_request']
    expected_record = records[reader['source_record']]
    expected_answers = [q['answer'] for q in expected_record['qa']
                        if q['field'] == reader['field'] and q['question'] == reader['question']]
    if len(expected_answers) != 1:
        raise ValueError('Reader question no longer identifies exactly one source answer')
    exact = text == expected_answers[0] and text != ''
    distances = []
    for i, event in enumerate(history):
        if event['id'] in attributable:
            writer_round = event.get('round', 0)
            reader_round = reader.get('round', writer_round)
            distances.append(dict(request_id=event['id'], round_gap=reader_round - writer_round,
                scheduled_intervening_requests=len(history) - i - 1 + reader_round - writer_round))
    source_answers = [{'request_id': event['id'], 'record_id': event['record_id'],
                       'answer': qa['answer']}
                      for event in history if event['id'] in loaded
                      for qa in records[event['record_id']]['qa']
                      if qa['field'] == reader['field'] and qa['answer'] != '']
    return {'success': bool(attributable) and complete,
            'matched_requests': matches, 'attributed_requests': attributable,
            'expected_request': expected_request, 'expected_record': reader['source_record'],
            'expected_answer': expected_answers[0], 'exact_match': exact,
            'target_success': exact and expected_request in loaded and complete,
            'source_has_field': bool(source_answers), 'source_answers': source_answers,
            'target_source_loaded': expected_request in loaded,
            'ambiguous_value': len(distinct_records) > 1,
            'audit_complete': complete, 'loaded': bool(audit['loaded_keys']),
            'unknown_sources': audit.get('unknown_sources', []), 'source_distances': distances}
