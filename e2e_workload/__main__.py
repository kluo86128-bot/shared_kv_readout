"""Continuous Writer/background/four-Reader workload with durable cache state."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid

from e2e_cache.__main__ import progress, verified_checkpoint, remove_stale_checkpoints
from e2e_cache.common import score
from e2e_cache import deployment
from .common import (ROOT, atomic_json, digest, load_settings, path, prepare_documents,
                     read_json, schedule, sha_file, ordered_events)
from .cache_report import reader_cache_report, log_reader_cache


def code_identity():
    from kv_cache_readout.io import code_identity as core_identity
    result = core_identity()
    for folder in ('e2e_workload', 'runtime_fixes'):
        result.update({f'{folder}/{p.name}': sha_file(p) for p in sorted((ROOT/folder).glob('*.py'))})
    return result


def inputs(cfg, manifest):
    from transformers import AutoTokenizer
    from e2e_cache.__main__ import load_inputs
    records = load_inputs(cfg, manifest)
    tokenizer = AutoTokenizer.from_pretrained(path(cfg['deployment']), local_files_only=True)
    documents = prepare_documents(cfg, tokenizer)
    return records, documents, schedule(cfg, records)


def summarize(root, cursor, cfg, status):
    rows, events = [], []
    for index in range(cursor):
        result = read_json(root / f'round_{index:06d}' / 'result.json')
        rows.extend(result['readers'])
        events.extend(read_json(root / f'round_{index:06d}' / 'events.json'))
    def metrics(selected):
        count = len(selected)
        eligible = sum(r['score']['loaded'] and r['score']['source_has_field'] and r['score']['audit_complete'] for r in selected)
        successes = sum(r['score']['success'] for r in selected)
        return dict(queries=count, successes=successes,
            readout_rate=successes/count if count else None,
            cache_load_rate=sum(r['score']['loaded'] for r in selected)/count if count else None,
            source_field_readout_rate=successes/eligible if eligible else None,
            source_field_eligible_queries=eligible,
            target_successes=sum(r['score']['target_success'] for r in selected))
    source_rows = defaultdict(list)
    for row in rows:
        for record_id in row['loaded_writer_records']:
            source_rows[record_id].append(row)
    per_source = {k: dict(reads=len(v), successes=sum(k in r['attributed_writer_records'] for r in v),
                         readout_rate=sum(k in r['attributed_writer_records'] for r in v)/len(v))
                  for k, v in sorted(source_rows.items())}
    result = dict(schema=2, status=status, completed_rounds=cursor, planned_rounds=cfg['rounds'],
        completed_groups=cursor, planned_groups=cfg['rounds'],
        timing=cfg['reader_schedule'].get('timing', 'background_then_read'),
        background_bucket_scope='per_group_total',
        planned_queries=cfg['rounds']*len(cfg['reader_schedule']['fields']), completed_queries=len(rows),
        **metrics(rows), by_field={f: metrics([r for r in rows if r['reader']['field'] == f])
                                  for f in cfg['reader_schedule']['fields']},
        by_gap_bucket={str(i): metrics([r for r in rows if r['gap_bucket'] == i])
                       for i in range(len(cfg['reader_schedule']['gaps']))},
        unique_loaded_writer_records=len(per_source), per_source=per_source,
        source_macro_readout_rate=sum(v['readout_rate'] for v in per_source.values())/len(per_source) if per_source else None,
        eviction_layer_entries_by_reason=dict(Counter(e.get('reason') for e in events if e['kind'] == 'evict')),
        audit_complete=all(r['score']['audit_complete'] for r in rows),
        age_clock='wall_seconds_since_admission; downtime included',
        eviction_count_unit='per-layer entry, not logical all-layer block')
    from .source_csv import write_exports
    plan = read_json(root/'schedule.json')
    registry = {e['id']: e for step in plan for kind in ('writers', 'backgrounds', 'readers') for e in step[kind]}
    result['direct_source_comparison'] = write_exports(root, rows, registry)
    from .joint_report import write_joint_report
    joint=write_joint_report(root,rows,registry)
    if joint is not None:result['joint_readout']=joint
    atomic_json(root/'summary.json', result)
    return result


def check_commits(root, state):
    for index in range(state['cursor']):
        folder = root/f'round_{index:06d}'
        commit = read_json(folder/'commit.json')
        for name, sha in commit['files'].items():
            if Path(name).name != name or sha_file(folder/name) != sha:
                raise ValueError('Committed workload artifact changed')
    if state['cursor']:
        snapshot = verified_checkpoint(root, state)
        if state['checkpoint']['round'] != state['cursor']-1:
            raise ValueError('Lifecycle resume requires a cache snapshot for the latest committed round')
        if read_json(root/f"round_{state['cursor']-1:06d}"/'commit.json')['checkpoint'] != state['checkpoint']:
            raise ValueError('Snapshot commit disagrees with run state')
        return snapshot


def execute(cfg, resume=False):
    from kv_cache_readout.io import run_lock
    from .runtime import Runtime
    root = path(cfg['output'])
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        manifest = deployment.verify(cfg)
        records, documents, plan = inputs(cfg, manifest)
        by_id = {r['record_id']: r for r in records}
        identity = dict(schema=2, config=cfg, manifest=manifest, code=code_identity(),
                        schedule_sha=digest(plan), documents_sha=digest(documents),
                        corpus_sha=sha_file(path(cfg['background']['document_path'])))
        snapshot = None
        if resume:
            state = read_json(root/'state.json')
            if state['identity'] != digest(identity) or read_json(root/'identity.json') != identity:
                raise ValueError('Resume identity changed: config/code/model/checkpoint/documents')
            if read_json(root/'schedule.json') != plan or read_json(root/'background_documents.json') != documents:
                raise ValueError('Frozen workload changed')
            snapshot = check_commits(root, state)
        else:
            if (root/'state.json').exists() or (root/'identity.json').exists():
                raise ValueError('Run exists; resume it or use a new output directory')
            atomic_json(root/'identity.json', identity)
            atomic_json(root/'schedule.json', plan)
            atomic_json(root/'background_documents.json', documents)
            state = dict(schema=2, identity=digest(identity), cursor=0, status='prepared')
            atomic_json(root/'state.json', state)
        remove_stale_checkpoints(root, snapshot)
        if state['cursor'] == len(plan):
            return summarize(root, state['cursor'], cfg, 'complete')
        pause_file = root/'pause.request'
        pause_file.unlink(missing_ok=True)
        stopped = []
        previous = {s: signal.signal(s, lambda *_: stopped.append(True)) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            runtime = Runtime(cfg, manifest)
            if snapshot:
                runtime.restore(snapshot)
            history = [w for step in plan[:state['cursor']] for w in step['writers']]
            # Count every request type, including Readers, for source distance.
            sequence = [e for step in plan for e in ordered_events(step)]
            positions = {event['id']: i for i, event in enumerate(sequence)}
            registry = {event['id']: event for event in sequence}
            material = cfg['materials'][0]
            last_completion = None
            def send(event):
                nonlocal last_completion
                delay = cfg['reader_schedule']['submit_delay_seconds']
                if last_completion is not None and delay:
                    time.sleep(max(0, last_completion+delay-time.monotonic()))
                record = by_id[event['record_id']] if event['role'] == 'writer' else None
                doc = documents[event['document_index']] if event['role'] == 'background' else material
                request = runtime.request(event, record, doc)
                result = runtime.generate([request])
                if event['role'] == 'reader':
                    result['request_public_interval'] = {k: request[k] for k in ('public_start', 'public_end')}
                last_completion = time.monotonic()
                progress('REQUEST', id=event['id'], role=event['role'],
                         phase=event.get('phase', event['role']), seconds=result['seconds'])
                return result
            for step in plan[state['cursor']:]:
                index = step['index']
                folder = root/f'round_{index:06d}'
                if folder.exists():
                    folder.rename(root/(folder.name+'.uncommitted-'+uuid.uuid4().hex))
                attempt = root/(folder.name+'.attempt-'+uuid.uuid4().hex)
                attempt.mkdir()
                writers, backgrounds, readers, events = [], [], [], []
                started = time.monotonic()
                progress('GROUP', number=index+1, writers=len(step['writers']),
                         background_total=step['background_count'],
                         reader_gaps=step['reader_gap_background_counts'],
                         cooldown=step['cooldown_background_count'], timing=step['timing'])
                for event in step['writers']:
                    result = send(event)
                    writers.append(result)
                    events.extend(result['events'])
                writer_finished_at = writers[-1]['rows'][0]['batch_completed_at']
                history.extend(step['writers'])
                writer_by_request = {w['id']: w['record_id'] for w in history}
                background_before_reader = 0
                for event in ordered_events(step)[len(step['writers']):]:
                    if event['role'] == 'background':
                        result = send(event)
                        backgrounds.append(result)
                        events.extend(result['events'])
                        background_before_reader += 1
                        continue
                    if event['role'] != 'reader':
                        raise ValueError('No Writer may be inserted in the Reader/cooldown phase')
                    reader = event
                    reader_index = len(readers)
                    result = send(reader)
                    response = result['rows'][0]
                    events.extend(result['events'])
                    report = reader_cache_report(result['request_public_interval'], response['audit'], registry)
                    log_reader_cache(reader, report)
                    scored = score(reader, response['text'], response['audit'], history, by_id)
                    for distance in scored['source_distances']:
                        distance['scheduled_intervening_requests'] = positions[reader['id']]-positions[distance['request_id']]-1
                    loaded = sorted({writer_by_request[s] for s in response['audit']['loaded_sources'] if s in writer_by_request})
                    attributed = sorted({writer_by_request[s] for s in scored['attributed_requests']})
                    readers.append(dict(round=index, reader_index=reader_index, reader=reader,
                        response=response, score=scored, cache_report=report, gap_bucket=step['gap_bucket'],
                        background_count=step['background_count'], loaded_writer_records=loaded,
                        attributed_writer_records=attributed, seconds=result['seconds'],
                        writer_to_reader_seconds=response['submitted_at']-writer_finished_at,
                        intervening_requests=positions[reader['id']]-positions[step['writers'][-1]['id']]-1,
                        background_before_reader=background_before_reader,
                        cooldown_background_count=step['cooldown_background_count'], timing=step['timing']))
                atomic_json(attempt/'result.json', dict(round=index, group=index, readers=readers,
                    timing=step['timing'], reader_gap_background_counts=step['reader_gap_background_counts'],
                    cooldown_background_count=step['cooldown_background_count'],
                    seconds=time.monotonic()-started, inventory=runtime.inventory()))
                atomic_json(attempt/'writers.json', writers)
                atomic_json(attempt/'backgrounds.json', backgrounds)
                atomic_json(attempt/'events.json', events)
                cache_file = root/f'cache_{index:06d}_{uuid.uuid4().hex}.pt'
                runtime.snapshot(cache_file)
                checkpoint = dict(file=cache_file.name, round=index, sha256=sha_file(cache_file))
                atomic_json(attempt/'commit.json', dict(checkpoint=checkpoint,
                    files={p.name: sha_file(p) for p in attempt.iterdir() if p.is_file()}))
                attempt.rename(folder)
                state.update(cursor=index+1, checkpoint=checkpoint, status='running')
                atomic_json(root/'state.json', state)
                remove_stale_checkpoints(root, cache_file)
                current = summarize(root, state['cursor'], cfg, 'running')
                progress('ROUND_DONE', round=index+1, background=step['background_count'],
                         cooldown=step['cooldown_background_count'],
                         readers=len(readers), completed_queries=current['completed_queries'],
                         readout_rate=current['readout_rate'])
                if stopped or pause_file.exists():
                    state['status'] = 'paused'
                    atomic_json(root/'state.json', state)
                    return summarize(root, state['cursor'], cfg, 'paused')
            state['status'] = 'complete'
            state.pop('error', None)
            atomic_json(root/'state.json', state)
            return summarize(root, state['cursor'], cfg, 'complete')
        except BaseException as exc:
            state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            atomic_json(root/'state.json', state)
            summarize(root, state['cursor'], cfg, 'failed')
            raise
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'preflight', 'run', 'resume', 'pause', 'status'])
    parser.add_argument('--config', default='configs/e2e_hr_workload_server1.json')
    args = parser.parse_args()
    cfg = load_settings(args.config)
    if args.action in ('run', 'resume') and os.environ.get('PYTHONHASHSEED') != '0':
        os.environ['PYTHONHASHSEED'] = '0'
        os.execv(sys.executable, [sys.executable, '-m', 'e2e_workload', *sys.argv[1:]])
    if args.action == 'prepare':
        deployment.prepare(cfg)
        print(f"Deployment exported to {path(cfg['deployment'])}")
    elif args.action == 'preflight':
        manifest = deployment.verify(cfg)
        records, documents, plan = inputs(cfg, manifest)
        print(json.dumps(dict(status='ready', native_gpu_check=False, records=len(records),
            rounds=len(plan), readers=sum(len(s['readers']) for s in plan),
            business_groups=len(plan), timing=cfg['reader_schedule'].get('timing', 'background_then_read'),
            reader_gap_backgrounds=sum(sum(s['reader_gap_background_counts']) for s in plan),
            cooldown_backgrounds=sum(s['cooldown_background_count'] for s in plan),
            backgrounds=sum(s['background_count'] for s in plan), document_pool=len(documents),
            schedule_sha=digest(plan), documents_sha=digest(documents)), indent=2))
    elif args.action in ('status', 'pause'):
        root = path(cfg['output'])
        if not (root/'state.json').exists():
            raise ValueError('Run has no state file')
        if args.action == 'pause':
            atomic_json(root/'pause.request', dict(requested_at=time.time()))
        print(json.dumps(read_json(root/'state.json'), indent=2))
    else:
        print(json.dumps(execute(cfg, resume=args.action == 'resume'), indent=2))


if __name__ == '__main__':
    main()
