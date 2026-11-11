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


def validate_base(cfg):
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


def load_settings(value):
    cfg = read_json(path(value))
    extras = {'background', 'reader_schedule', 'writer_sampling', 'native_rope_fix'}
    if not extras <= cfg.keys():
        raise ValueError('Lifecycle workload requires background, reader_schedule, writer_sampling and native_rope_fix')
    legacy = {k: v for k, v in cfg.items() if k not in extras}
    cache = dict(cfg['cache'])
    for k in ('max_age_seconds', 'renew_age_on_hit'):
        if k not in cache:
            raise ValueError(f'Missing cache {k}')
    legacy['cache'] = {k: v for k, v in cache.items() if k not in ('max_age_seconds', 'renew_age_on_hit')}
    validate_base(legacy)
    cfg.setdefault('adapter', None)
    def positive(v):
        return type(v) is int and v > 0
    if type(cfg['native_rope_fix']) is not bool:
        raise ValueError('native_rope_fix must be boolean')
    if cfg['writer_batch_size'] != 1 or cfg['engine']['max_num_seqs'] != 1:
        raise ValueError('Lifecycle requests must be serial: writer_batch_size=max_num_seqs=1')
    if cfg['cache']['sharing'] != 'shared' or len(cfg['materials']) != 1:
        raise ValueError('Lifecycle workload requires one C1 and a shared cache')
    age = cache['max_age_seconds']
    if age is not None and (type(age) not in (int, float) or not math.isfinite(age) or age <= 0):
        raise ValueError('max_age_seconds must be positive or null')
    if cache['renew_age_on_hit'] is not False:
        raise ValueError('Only fixed maximum age (renew_age_on_hit=false) is supported')
    sampling = cfg['writer_sampling']
    if set(sampling) != {'replacement_within_round', 'shuffle_within_round'} or any(type(v) is not bool for v in sampling.values()):
        raise ValueError('Invalid writer_sampling')
    bg = cfg['background']
    if set(bg) != {'document_path', 'document_pool_size', 'document_tokens', 'max_new_tokens', 'unique_documents_within_round'}:
        raise ValueError('Invalid background configuration')
    if not isinstance(bg['document_path'], str) or not bg['document_path'] or any(not positive(bg[k]) for k in ('document_pool_size', 'document_tokens', 'max_new_tokens')) or type(bg['unique_documents_within_round']) is not bool:
        raise ValueError('Invalid background documents/length')
    rs = cfg['reader_schedule']
    required_rs = {'gaps', 'shuffle', 'submit_delay_seconds', 'fields'}
    if not required_rs <= rs.keys() or rs.keys() - required_rs - {'timing', 'background_between_readers','repeats_per_field'} or type(rs['shuffle']) is not bool:
        raise ValueError('Invalid reader_schedule')
    if type(rs.get('repeats_per_field',1)) is not int or rs.get('repeats_per_field',1)<1:
        raise ValueError('repeats_per_field must be a positive integer')
    timing = rs.get('timing', 'background_then_read')
    if timing not in ('read_then_cooldown', 'background_then_read'):
        raise ValueError('Invalid reader timing')
    between = rs.get('background_between_readers')
    if timing == 'read_then_cooldown' and between is None:
        raise ValueError('read_then_cooldown requires background_between_readers')
    if between is not None and (not isinstance(between, dict) or set(between) != {'min', 'max'} or
            any(type(v) is not int for v in between.values()) or
            not 0 <= between['min'] <= between['max'] <= 1):
        raise ValueError('Reader gaps support only 0 or 1 background request')
    delay = rs['submit_delay_seconds']
    if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
        raise ValueError('submit_delay_seconds must be finite and nonnegative')
    if not isinstance(rs['fields'], list) or not rs['fields'] or any(not isinstance(f, str) or not f for f in rs['fields']) or len(set(rs['fields'])) != len(rs['fields']):
        raise ValueError('Reader fields must be unique nonempty strings')
    if not rs['gaps']:
        raise ValueError('At least one gap range is required')
    covered = set()
    for gap in rs['gaps']:
        if set(gap) != {'min', 'max', 'rounds'} or any(type(gap[k]) is not int for k in gap) or not 0 <= gap['min'] <= gap['max'] or gap['rounds'] < 0:
            raise ValueError('Invalid inclusive gap range')
        values = set(range(gap['min'], gap['max'] + 1))
        if values & covered:
            raise ValueError('Gap ranges must not overlap')
        covered |= values
        if bg['unique_documents_within_round'] and gap['max'] > bg['document_pool_size']:
            raise ValueError('Background pool is smaller than the maximum unique gap')
        if timing == 'read_then_cooldown' and gap['rounds'] and gap['min'] < (len(rs['fields'])-1)*between['min']:
            raise ValueError('Background budget cannot meet the minimum Reader gaps')
    if sum(g['rounds'] for g in rs['gaps']) != cfg['rounds']:
        raise ValueError('Gap round counts must sum to rounds')
    return cfg


def schedule(cfg, records):
    # Independent streams prevent one sampling option from silently changing others.
    def rng_for(round_index, purpose):
        return random.Random(digest([cfg['seed'], round_index, purpose]))
    if len({r['record_id'] for r in records}) != len(records) or not records:
        raise ValueError('Writer records require unique IDs')
    if not cfg['writer_sampling']['replacement_within_round'] and cfg['writers_max'] > len(records):
        raise ValueError('Not enough distinct writer samples')
    fields = cfg['reader_schedule']['fields']
    for record in records:
        if not set(fields) <= {q['field'] for q in record['qa'] if isinstance(q['answer'], str) and q['answer'].strip()}:
            raise ValueError(f"Reader fields missing from sample {record['record_id']}")
    slots = [i for i, g in enumerate(cfg['reader_schedule']['gaps']) for _ in range(g['rounds'])]
    if cfg['reader_schedule']['shuffle']:
        rng_for(-1, 'gap-order').shuffle(slots)
    result = []
    material = cfg['materials'][0]['id']
    for index, bucket in enumerate(slots):
        rng = rng_for(index, 'writer-sample')
        count = rng.randint(cfg['writers_min'], cfg['writers_max'])
        selected = rng.choices(records, k=count) if cfg['writer_sampling']['replacement_within_round'] else rng.sample(records, count)
        writers = [dict(id=f'r{index}:w{j}', role='writer', round=index, record_id=r['record_id'],
                        material=material, tenant=f"writer:{r['record_id']}") for j, r in enumerate(selected)]
        if cfg['writer_sampling']['shuffle_within_round']:
            rng_for(index, 'writer-order').shuffle(writers)
        gap = cfg['reader_schedule']['gaps'][bucket]
        rng = rng_for(index, 'background')
        count = rng.randint(gap['min'], gap['max'])
        pool = list(range(cfg['background']['document_pool_size']))
        chosen = rng.sample(pool, count) if cfg['background']['unique_documents_within_round'] else rng.choices(pool, k=count)
        backgrounds = [dict(id=f'r{index}:b{j}', role='background', round=index, document_index=d,
                            tenant='background') for j, d in enumerate(chosen)]
        source = rng_for(index, 'designated-target').choice(writers)
        record = next(r for r in selected if r['record_id'] == source['record_id'])
        readers = []
        for field in fields:
            qa = next(q for q in record['qa'] if q['field'] == field and q['answer'].strip())
            repeats=cfg['reader_schedule'].get('repeats_per_field',1)
            for repeat in range(repeats):
                suffix=f':{repeat}' if repeats>1 else ''
                readers.append(dict(id=f'r{index}:reader:{field}{suffix}', role='reader', round=index, tenant='reader',
                    material=material, field=field, question=qa['question'],
                    source_request=source['id'], source_record=source['record_id']))
        rng_for(index, 'reader-order').shuffle(readers)
        timing = cfg['reader_schedule'].get('timing', 'background_then_read')
        between_counts = [0]*(len(readers)-1)
        if timing == 'read_then_cooldown':
            limits = cfg['reader_schedule']['background_between_readers']
            slots = len(between_counts)
            rng = rng_for(index, 'reader-background-placement')
            # Uniform over all feasible 0/1 patterns, i.e. independent fair bits
            # conditioned on the group's total budget. No early-slot preference.
            possible = list(range(slots*limits['min'], min(count, slots*limits['max'])+1))
            inside = rng.choices(possible, weights=[math.comb(slots, k) for k in possible])[0]
            between_counts = [1]*inside + [0]*(slots-inside)
            rng.shuffle(between_counts)
            ordered = list(writers)
            offset = 0
            for j, reader in enumerate(readers):
                ordered.append(reader)
                if j < slots:
                    for event in backgrounds[offset:offset+between_counts[j]]:
                        event.update(phase='between_readers', reader_gap_index=j)
                        ordered.append(event)
                    offset += between_counts[j]
            cooldown = backgrounds[offset:]
            for event in cooldown:
                event['phase'] = 'cooldown'
            ordered.extend(cooldown)
        else:
            for event in backgrounds:
                event['phase'] = 'before_readers'
            cooldown = []
            ordered = writers + backgrounds + readers
        result.append(dict(index=index, gap_bucket=bucket, background_count=count,
            timing=timing, background_count_scope='per_group_total',
            reader_gap_background_counts=between_counts, cooldown_background_count=len(cooldown),
            request_order=[e['id'] for e in ordered],
            writers=writers, backgrounds=backgrounds, readers=readers))
    return result


def ordered_events(step):
    """The frozen request order is authoritative for execution and distances."""
    events = [e for kind in ('writers', 'backgrounds', 'readers') for e in step[kind]]
    if 'request_order' not in step:
        return events  # Old saved plans remain readable by diagnostics.
    registry = {e['id']: e for e in events}
    order = step['request_order']
    if len(registry) != len(events) or len(order) != len(events) or set(order) != set(registry):
        raise ValueError('Request order must contain each event exactly once')
    return [registry[i] for i in order]


def prepare_documents(cfg, tokenizer):
    bg = cfg['background']
    records = [json.loads(line) for line in path(bg['document_path']).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(records) < bg['document_pool_size']:
        raise ValueError('Background corpus has too few documents')
    result = []
    for document in records[:bg['document_pool_size']]:
        text = document['text']
        if any(m['text'] in text for m in cfg['materials']) or cfg['cache']['delimiter'] in text:
            raise ValueError('Background must not contain C1 or internal blend delimiters')
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < bg['document_tokens']:
            raise ValueError(f"Background document too short: {document['id']}")
        ids = ids[:bg['document_tokens']]
        result.append(dict(id=document['id'], text=tokenizer.decode(ids, skip_special_tokens=False,
                          clean_up_tokenization_spaces=False), tokens=ids))
    if len({d['id'] for d in result}) != len(result) or len({tuple(d['tokens']) for d in result}) != len(result):
        raise ValueError('Background IDs and tokenized documents must be distinct')
    return result

