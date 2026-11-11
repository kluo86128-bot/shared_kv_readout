import copy
import csv
import hashlib
import json

import pytest

from e2e_workload.source_csv import source_rows, write_exports, export_existing


REGISTRY = {
    'writer_actual': dict(role='writer', record_id='actual_sample'),
    'writer_target': dict(role='writer', record_id='target_sample'),
    'reader_previous': dict(role='reader', source_record='target_sample'),
}


def result():
    return dict(round=0, reader_index=0, reader=dict(id='reader_now', field='email', source_record='target_sample'),
        response=dict(text='actual@example.com', audit=dict(loaded_sources=['writer_actual'], cache_entries=[])),
        score=dict(expected_record='target_sample', expected_answer='target@example.com',
                   source_answers=[dict(request_id='writer_actual', record_id='actual_sample', answer='actual@example.com')],
                   audit_complete=True, success=True, loaded=True, source_has_field=True,
                   target_success=False, target_source_loaded=False, exact_match=False),
        cache_report=dict(c1_hit=True, audit_complete=True,
                          direct_sources=[dict(request_id='writer_actual', role='writer', record_id='actual_sample')],
                          lineage_sources=[]))


def test_actual_source_record_gold_and_em_match_without_target_columns():
    row = result()
    before = copy.deepcopy(row)
    main, detailed = source_rows(row, REGISTRY)
    assert main['record_id'] == 'actual_sample'
    assert main['gold'] == main['answer'] == 'actual@example.com'
    assert main['success'] is True and main['source_em'] is True
    assert not any(key.startswith('target_') for key in main)
    assert all(not any(key.startswith('target_') for key in source) for source in detailed)
    assert detailed[0]['record_id'] == main['record_id']
    assert row == before


def test_incorrect_answer_does_not_change_the_selected_source_gold():
    row = result()
    row['response']['text'] = 'target@example.com'
    main, _ = source_rows(row, REGISTRY)
    assert main['record_id'] == 'actual_sample' and main['gold'] == 'actual@example.com'
    assert main['success'] is False


def test_reader_origin_with_writer_lineage_never_uses_target_or_lineage_as_direct_gold():
    row = result()
    row['cache_report']['direct_sources'] = [dict(request_id='reader_previous', role='reader', record_id=None)]
    row['cache_report']['lineage_sources'] = [dict(request_id='writer_actual', role='writer', record_id='actual_sample')]
    main, detailed = source_rows(row, REGISTRY)
    assert main['source_status'] == 'reader_source'
    assert main['record_id'] == main['gold'] == '' and main['success'] is None
    assert not any(key.startswith('target_') for key in main)
    assert detailed[0]['source_request_id'] == 'reader_previous' and detailed[0]['gold'] == ''


@pytest.mark.parametrize('scenario', ['cache_miss', 'unverified_source', 'source_unknown', 'missing_report'])
def test_ungradable_queries_are_explicit_not_assigned_target_gold(scenario):
    row = result()
    if scenario == 'cache_miss': row['cache_report']['c1_hit'] = False
    elif scenario == 'unverified_source': row['cache_report']['audit_complete'] = False
    elif scenario == 'source_unknown': row['cache_report']['direct_sources'] = []
    else: row.pop('cache_report')
    main, _ = source_rows(row, REGISTRY)
    assert main['record_id'] == main['gold'] == ''
    assert main['success'] is None
    assert main['source_status'] == ('source_unknown' if scenario == 'missing_report' else scenario)


def test_multiple_writers_get_individual_comparisons_without_answer_based_selection():
    row = result()
    row['cache_report']['direct_sources'].append(dict(request_id='writer_target', role='writer', record_id='target_sample'))
    row['score']['source_answers'].append(dict(request_id='writer_target', record_id='target_sample', answer='target@example.com'))
    main, detailed = source_rows(row, REGISTRY)
    assert main['source_status'] == 'multiple_direct_writers' and main['record_id'] == main['gold'] == ''
    assert main['success'] is None
    assert [(r['record_id'], r['gold'], r['success']) for r in detailed] == [
        ('actual_sample', 'actual@example.com', True), ('target_sample', 'target@example.com', False)]


@pytest.mark.parametrize('answers,status', [([], 'source_field_missing'),
                                         (['actual@example.com', 'other@example.com'], 'ambiguous_source_gold')])
def test_missing_or_ambiguous_gold_is_not_replaced_with_target(answers, status):
    row = result()
    row['score']['source_answers'] = [dict(request_id='writer_actual', record_id='actual_sample', answer=a) for a in answers]
    main, detailed = source_rows(row, REGISTRY)
    assert main['record_id'] == 'actual_sample'
    assert main['gold'] == '' and main['success'] is None and main['source_status'] == status
    assert detailed[0]['gold'] == ''


def test_bad_source_identity_does_not_replace_existing_csv(tmp_path):
    path = tmp_path/'predictions.csv'
    path.write_bytes(b'existing')
    row = result()
    row['cache_report']['direct_sources'][0]['record_id'] = 'target_sample'
    with pytest.raises(ValueError, match='disagrees'):
        write_exports(tmp_path, [row], REGISTRY)
    assert path.read_bytes() == b'existing'


def test_no_gpu_reexport_only_reads_committed_groups_and_leaves_originals(tmp_path):
    row = result()
    folder = tmp_path/'round_000000'
    folder.mkdir()
    payload = json.dumps(dict(readers=[row])).encode()
    (folder/'result.json').write_bytes(payload)
    (folder/'commit.json').write_text(json.dumps(dict(files={'result.json': hashlib.sha256(payload).hexdigest()})))
    (tmp_path/'state.json').write_text(json.dumps(dict(cursor=1)))
    plan = [dict(writers=[dict(id=k, **v) for k, v in REGISTRY.items() if v['role'] == 'writer'],
                 backgrounds=[], readers=[dict(id='reader_previous', **REGISTRY['reader_previous'])])]
    (tmp_path/'schedule.json').write_text(json.dumps(plan))
    original = tmp_path/'predictions.csv'
    original.write_text('old target-based CSV')
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    stats = export_existing(tmp_path)
    assert stats['queries'] == stats['comparable_queries'] == stats['correct_queries'] == 1
    assert stats['exact_match_rate'] == 1
    with (tmp_path/'predictions_by_source.csv').open(encoding='utf-8-sig', newline='') as stream:
        table = csv.DictReader(stream)
        assert table.fieldnames[:6] == ['record_id', 'round', 'reader_index', 'field', 'answer', 'gold']
        assert not any(key.startswith('target_') for key in table.fieldnames)
        values = next(table)
        assert values['record_id'] == 'actual_sample' and values['gold'] == values['answer']
    with (tmp_path/'predictions_by_source_sources.csv').open(encoding='utf-8-sig', newline='') as stream:
        assert not any(key.startswith('target_') for key in csv.DictReader(stream).fieldnames)
    assert all(p.read_bytes() == value for p, value in before.items())
    (folder/'result.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        export_existing(tmp_path)
