"""Source-aligned CSVs from committed Reader evidence, without model execution."""
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import uuid


FIRST = ['record_id', 'round', 'reader_index', 'field', 'answer', 'gold']


def _json(value):
    return json.dumps(value, ensure_ascii=False)


def _atomic_csv(path, rows):
    columns = FIRST + sorted({k for row in rows for k in row} - set(FIRST))
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path, value):
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_rows(row, registry):
    score, reader, response = row['score'], row['reader'], row['response']
    target_id = score.get('expected_record', reader['source_record'])
    if target_id != reader['source_record'] or not isinstance(score['expected_answer'], str) or not isinstance(response['text'], str):
        raise ValueError('Invalid designated target identity/answer')
    report = row.get('cache_report', {})
    hit = report.get('c1_hit')
    verified = bool(report.get('audit_complete') and score.get('audit_complete'))
    origins = report.get('direct_sources', []) if hit else []
    candidates = []
    for origin in origins:
        event = registry.get(origin['request_id'])
        role = origin['role']
        record_id = origin.get('record_id')
        if event is None or event.get('role') != role:
            raise ValueError('Direct cache source is absent from the frozen schedule')
        if role == 'writer' and (not record_id or record_id != event.get('record_id')):
            raise ValueError('Direct Writer sample disagrees with the frozen schedule')
        if role != 'writer' and record_id is not None:
            raise ValueError('Non-Writer source must not be assigned a private sample')
        answers = sorted({v['answer'] for v in score.get('source_answers', [])
                          if role == 'writer' and v['request_id'] == origin['request_id']
                          and v['record_id'] == record_id and isinstance(v['answer'], str) and v['answer'] != ''})
        status = ('unverified_source' if not verified else
                  'reader_source' if role == 'reader' else
                  'non_writer_source' if role != 'writer' else
                  'source_field_missing' if not answers else
                  'ambiguous_source_gold' if len(answers) != 1 else 'writer_source')
        gold = answers[0] if status == 'writer_source' else ''
        candidates.append(dict(source_request_id=origin['request_id'], source_role=role,
            record_id=record_id or '', gold=gold, gold_candidates=answers, source_status=status,
            source_em=(response['text'] == gold) if status == 'writer_source' else None))
    status = ('source_unknown' if hit is None else 'cache_miss' if not hit else 'unverified_source' if not verified else
              'source_unknown' if not candidates else 'reader_source' if all(c['source_role'] == 'reader' for c in candidates)
              else 'mixed_direct_sources' if any(c['source_role'] != 'writer' for c in candidates)
              else 'multiple_direct_writers' if len({c['record_id'] for c in candidates}) != 1
              else 'source_field_missing' if any(c['source_status'] == 'source_field_missing' for c in candidates)
              else 'ambiguous_source_gold' if any(c['source_status'] != 'writer_source' for c in candidates)
                       or len({c['gold'] for c in candidates}) != 1
              else 'writer_source')
    single_writer_sample = bool(candidates) and all(c['source_role'] == 'writer' for c in candidates) and len({c['record_id'] for c in candidates}) == 1
    record_id = candidates[0]['record_id'] if verified and single_writer_sample else ''
    gold = candidates[0]['gold'] if status == 'writer_source' else ''
    source_em = response['text'] == gold if status == 'writer_source' else None
    entries = response['audit'].get('cache_entries', [])
    ages = [e['age_seconds'] for e in entries]
    base = {k: row.get(k) for k in ('round', 'reader_index', 'gap_bucket', 'background_count',
            'writer_to_reader_seconds', 'intervening_requests', 'background_before_reader',
            'cooldown_background_count', 'timing')}
    base.update(query_id=reader['id'], record_id=record_id, field=reader['field'],
        answer=response['text'], gold=gold, source_status=status, source_em=source_em,
        success=source_em, success_metric='direct_source_exact_match', provenance_complete=verified,
        lineage_readout_success=score.get('success'), lineage_source_has_field=score.get('source_has_field'),
        source_has_field=status == 'writer_source', loaded=score.get('loaded'),
        audit_complete=score.get('audit_complete'), ambiguous_value=score.get('ambiguous_value'),
        loaded_writer_records=_json(row.get('loaded_writer_records', [])),
        attributed_writer_records=_json(row.get('attributed_writer_records', [])),
        loaded_sources=_json(response['audit'].get('loaded_sources', [])),
        direct_cache_roles=_json(sorted({o['role'] for o in origins})),
        direct_source_request_ids=_json([c['source_request_id'] for c in candidates]),
        direct_source_record_ids=_json(sorted({c['record_id'] for c in candidates if c['record_id']})),
        direct_sources=_json(origins), lineage_sources=_json(report.get('lineage_sources', [])),
        cache_age_min_seconds=min(ages) if ages else '', cache_age_max_seconds=max(ages) if ages else '')
    base.update({k: report.get(k) for k in ('c1_hit', 'c1_cached_tokens', 'c1_expected_tokens',
                'full_c1_hit', 'exact_c1_block_hit', 'only_c1_hit', 'total_cached_tokens')})
    detailed = []
    for candidate in candidates:
        detail = dict(base)
        detail.update({k: v for k, v in candidate.items() if k != 'gold_candidates'})
        detail['gold_candidates'] = _json(candidate['gold_candidates'])
        detail['success'] = candidate['source_em']
        detail['source_has_field'] = candidate['source_status'] == 'writer_source'
        detailed.append(detail)
    return base, detailed


def write_exports(root, rows, registry, filename='predictions.csv'):
    root = Path(root)
    if Path(filename).name != filename or not filename.endswith('.csv'):
        raise ValueError('CSV output must be a filename ending in .csv')
    primary, detailed = [], []
    for row in rows:
        main, sources = source_rows(row, registry)
        primary.append(main)
        detailed.extend(sources)
    eligible = [r for r in primary if r['source_status'] == 'writer_source']
    stats = dict(schema=1, comparison='actual_direct_writer_sample', queries=len(primary),
        comparable_queries=len(eligible), correct_queries=sum(r['source_em'] for r in eligible),
        exact_match_rate=sum(r['source_em'] for r in eligible)/len(eligible) if eligible else None,
        source_status_counts=dict(Counter(r['source_status'] for r in primary)),
        source_comparison_rows=len(detailed), multi_source_policy='primary left blank; compare each source in companion CSV')
    stem = Path(filename).stem
    companion = 'source_comparisons.csv' if filename == 'predictions.csv' else stem+'_sources.csv'
    summary = 'source_comparison_summary.json' if filename == 'predictions.csv' else stem+'_summary.json'
    # Build/validate every row before replacing any existing export.
    _atomic_csv(root/filename, primary)
    _atomic_csv(root/companion, detailed)
    _atomic_json(root/summary, stats)
    return stats


def export_existing(root, filename='predictions_by_source.csv'):
    root = Path(root)
    state = json.loads((root/'state.json').read_text(encoding='utf-8'))
    plan = json.loads((root/'schedule.json').read_text(encoding='utf-8'))
    if type(state['cursor']) is not int or not 0 <= state['cursor'] <= len(plan):
        raise ValueError('Invalid committed group count')
    registry = {e['id']: e for step in plan for kind in ('writers', 'backgrounds', 'readers') for e in step[kind]}
    rows = []
    for index in range(state['cursor']):
        folder = root/f'round_{index:06d}'
        payload = (folder/'result.json').read_bytes()
        commit = json.loads((folder/'commit.json').read_text(encoding='utf-8'))
        if hashlib.sha256(payload).hexdigest() != commit['files']['result.json']:
            raise ValueError('Committed Reader result changed')
        rows.extend(json.loads(payload)['readers'])
    return write_exports(root, rows, registry, filename)
