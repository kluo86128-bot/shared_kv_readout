"""C1 token coverage and direct/ancestral cache ownership for each Reader."""
import json


def reader_cache_report(request, audit, registry):
    start, end = request['public_start'], request['public_end']
    expected = end-start
    blocks = audit.get('loaded_blocks', [])
    hits = audit.get('hit_blocks', blocks)
    keys = set(audit.get('loaded_keys', []))
    covered = set()
    total_covered = set()
    for block in blocks:
        left, right = max(start, block['prompt_start']), min(end, block['prompt_end'])
        covered.update(range(left, max(left, right)))
    for block in hits:
        total_covered.update(range(block['prompt_start'], block['prompt_end']))
    mask = audit.get('retrieved_mask')
    valid_mask = (isinstance(mask, list) and len(mask) >= end
                  and all(type(v) is bool for v in mask))
    if valid_mask:
        cached = sum(mask[start:end])
        total = sum(mask)
        evidence = 'native_retrieved_mask'
    elif blocks:
        cached, total = len(covered), len(total_covered)
        evidence = 'native_loaded_blocks'
    elif not keys:
        cached, total = 0, len(total_covered)
        evidence = 'no_c1_loaded_keys'
    else:
        cached, total = None, None
        evidence = 'insufficient_token_evidence'
    evidence_consistent = not (valid_mask and bool(keys) != bool(cached))
    if valid_mask and blocks and cached != len(covered):
        evidence_consistent = False
    # Pre-lookup candidates are not source evidence when the transfer mask says miss.
    effective_keys = keys if cached is None or cached > 0 else set()
    full = bool(expected > 0 and cached == expected and keys)
    exact = full and any(b['prompt_start'] == start and b['prompt_end'] == end
                         and b.get('block_tokens') == expected for b in blocks)

    def source(source_id, role=None):
        event = registry.get(source_id, {})
        actual_role = role or event.get('role', 'unknown')
        # Reader.source_record is a designated scoring target, not its own sample.
        return dict(request_id=source_id, role=actual_role,
                    record_id=event.get('record_id') if actual_role == 'writer' else None,
                    round=event.get('round'), field=event.get('field'))
    direct = {}
    missing_metadata = []
    entries = {entry['key']: entry for entry in audit.get('cache_entries', [])}
    for key in sorted(effective_keys):
        entry = entries.get(key)
        if entry is None:
            missing_metadata.append(key)
            continue
        identity = (entry['direct_request'], entry['direct_role'])
        if identity not in direct:
            direct[identity] = dict(source(*identity), layer_entries=0)
        direct[identity]['layer_entries'] += 1
    lineage = [source(s) for s in sorted(set(audit.get('loaded_sources', [])))] if effective_keys else []
    any_hit = bool(total) if valid_mask else bool(hits or keys)
    return dict(any_cache_hit=any_hit,
                c1_hit=bool(keys) and (cached is None or cached > 0),
                c1_cached_tokens=cached, c1_expected_tokens=expected,
                full_c1_hit=full, exact_c1_block_hit=exact,
                only_c1_hit=bool(exact and total == expected), total_cached_tokens=total,
                non_c1_hit_blocks=sum(b['prompt_end'] <= start or b['prompt_start'] >= end for b in hits),
                token_evidence=evidence, evidence_consistent=evidence_consistent,
                direct_sources=list(direct.values()),
                lineage_sources=lineage, missing_source_metadata=missing_metadata,
                audit_complete=bool(audit.get('complete') and not audit.get('unknown_sources')
                                    and not missing_metadata and evidence_consistent))


def log_reader_cache(reader, report):
    summary = dict(request_id=reader['id'], field=reader['field'],
                   **{k: report[k] for k in ('any_cache_hit', 'c1_hit', 'c1_cached_tokens',
                      'c1_expected_tokens', 'full_c1_hit', 'exact_c1_block_hit', 'only_c1_hit',
                      'total_cached_tokens', 'non_c1_hit_blocks', 'token_evidence',
                      'evidence_consistent', 'audit_complete')})
    print('[E2E] READER_CACHE ' + json.dumps(summary, ensure_ascii=False), flush=True)
    for kind, sources in (('direct', report['direct_sources']), ('lineage', report['lineage_sources'])):
        for origin in sources:
            print('[E2E] READER_CACHE_SOURCE ' + json.dumps(dict(reader_request=reader['id'],
                  relationship=kind, **origin), ensure_ascii=False), flush=True)
    if report['missing_source_metadata']:
        print('[E2E] READER_CACHE_SOURCE ' + json.dumps(dict(reader_request=reader['id'],
              relationship='unknown', keys=report['missing_source_metadata']), ensure_ascii=False), flush=True)
