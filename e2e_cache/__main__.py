"""Run from the project root with python -m e2e_cache --help.

prepare exports a new deployment. run/resume execute a real local vLLM engine.
pause is cooperative at round boundaries. status never loads a GPU model.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid

from .common import (ROOT, atomic_json, digest, load_settings, path, read_json,
                     schedule, score, sha_file)

CHECKPOINT_INTERVAL = 10


def _display(value):
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, float):
        return f'{value:.2f}'
    text = str(value).replace('\r', '\\r').replace('\n', '\\n')
    return text if len(text) <= 96 else text[:93] + '...'


def progress(event, **values):
    """Write stable human-readable progress lines amid native engine logs."""
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    details = ' | '.join(f'{key}={_display(value)}' for key, value in values.items())
    print(f'[E2E] {stamp} | {event}' + (f' | {details}' if details else ''), flush=True)


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def cache_sources(audit, history):
    """Resolve native cache source request IDs to scheduled record IDs."""
    by_request = {event['id']: event for event in history}
    requests = sorted(set(audit.get('loaded_sources', [])))
    records = sorted({by_request[source]['record_id'] for source in requests
                      if source in by_request})
    return requests, records


def cache_block_summary(audit):
    """Summarize unique native public-cache chunks without counting layers."""
    blocks = audit.get('loaded_blocks', [])
    cached_tokens = sum(block['public_tokens'] for block in blocks)
    labels = []
    for block in blocks:
        key = block['key']
        short_key = key if len(key) <= 16 else key[:8] + '..' + key[-6:]
        labels.append(f'{block["prompt_start"]}:{block["prompt_end"]}'
                      f'={block["public_tokens"]}t:{block.get("region", "C1")}@'
                      f'{short_key}x{block["layer_blocks"]}L')
    return cached_tokens, ';'.join(labels) or '-'


def verified_checkpoint(root, state):
    """Resolve and verify the single checkpoint referenced by committed state."""
    checkpoint = state.get('checkpoint')
    if (not isinstance(checkpoint, dict)
            or set(checkpoint) != {'file', 'round', 'sha256'}
            or type(checkpoint['round']) is not int
            or not 0 <= checkpoint['round'] < state['cursor']
            or Path(checkpoint['file']).name != checkpoint['file']
            or not checkpoint['file'].startswith('cache_')
            or not checkpoint['file'].endswith('.pt')):
        raise ValueError('Run state has an invalid rolling cache checkpoint reference')
    filename = root / checkpoint['file']
    if not filename.is_file() or sha_file(filename) != checkpoint['sha256']:
        raise ValueError('Rolling cache checkpoint is missing or corrupted')
    return filename


def remove_stale_checkpoints(root, keep):
    """Remove only unreferenced rolling checkpoint files inside this run."""
    keep = None if keep is None else Path(keep).name
    for filename in root.glob('cache_*.pt'):
        if filename.name != keep:
            filename.unlink()


def replay_signature(row):
    """Stable request evidence used to verify cache-rebuilding replay."""
    audit = row['audit']
    return {
        'request_id': row.get('request_id'),
        'text': row['text'],
        'raw_text': row.get('raw_text'),
        'generated_token_ids': row.get('generated_token_ids'),
        'finish_reason': row.get('finish_reason'),
        'generated_tokens': row.get('generated_tokens'),
        'audit': {key: audit.get(key) for key in (
            'loaded_keys', 'loaded_blocks', 'hit_blocks', 'loaded_sources', 'unknown_sources',
            'recomputation', 'store_calls', 'retrieve_calls')},
    }


def rebuild_cache(runtime, cfg, plan, start, stop, by_id, material, root):
    """Replay committed requests after the latest periodic checkpoint."""
    for step in plan[start:stop]:
        expected_writers = read_json(root / f'round_{step["index"]:06d}' / 'writers.json')
        cap = cfg['writer_batch_size']
        replayed_batches = []
        for offset in range(0, len(step['writers']), cap):
            batch = step['writers'][offset:offset + cap]
            requests = [runtime.request(w, by_id[w['record_id']], material[w['material']])
                        for w in batch]
            replayed_batches.append(runtime.generate(requests))
        if [[replay_signature(row) for row in result['rows']] for result in replayed_batches] != [
                [replay_signature(row) for row in result['rows']] for result in expected_writers]:
            raise RuntimeError(f'Writer replay diverged while rebuilding cache at round {step["index"] + 1}')
        if cfg['cache']['sharing'] == 'cold':
            runtime.clear()
        reader = step['reader']
        replayed = runtime.generate([
            runtime.request(reader, None, material[reader['material']])])['rows'][0]
        expected = read_json(root / f'round_{step["index"]:06d}' / 'result.json')['response']
        if replay_signature(replayed) != replay_signature(expected):
            raise RuntimeError(f'Reader replay diverged while rebuilding cache at round {step["index"] + 1}')


def code_identity():
    from kv_cache_readout.io import code_identity as existing_identity
    return dict(existing_identity(), **{
        'e2e_cache/' + p.name: sha_file(p) for p in Path(__file__).parent.glob('*.py')})


def load_inputs(cfg, manifest):
    from kv_cache_readout.config import load_config, fingerprint
    from kv_cache_readout.data import dataset_identity, load_split
    experiment = load_config(path(cfg['experiment_config']))
    if fingerprint(experiment) != manifest['experiment_sha']:
        raise ValueError('Experiment changed since deployment export')
    if dataset_identity(experiment) != manifest['dataset']:
        raise ValueError('Frozen dataset identity changed')
    records = load_split(experiment, 'test', training=False)
    return records


def summary(root, cursor, total, status):
    rows = []
    for index in range(cursor):
        rows.append(read_json(root / f'round_{index:06d}' / 'result.json'))
    success = sum(row['score']['success'] for row in rows)
    loaded = sum(row['score']['loaded'] for row in rows)
    complete = all(row['score']['audit_complete'] for row in rows)
    result = dict(status=status, planned_queries=total, completed_queries=cursor,
        successes=success, cache_loads=loaded, audit_complete=complete,
        any_writer_leak_rate=success / total if cursor == total and complete else None,
        completed_query_leak_rate=success / cursor if cursor and complete else None,
        cache_load_rate=loaded / cursor if cursor else None,
        recovery_given_load=success / loaded if loaded and complete else None,
        ambiguous_values=sum(row['score']['ambiguous_value'] for row in rows),
        sources_with_field=sum(row['score']['source_has_field'] for row in rows),
        source_field_readout_rate=(success / sum(row['score']['source_has_field'] for row in rows)
                                  if complete and any(row['score']['source_has_field'] for row in rows) else None),
        target_successes=sum(row['score']['target_success'] for row in rows),
        scope='native vLLM/LMCache in-process engine; one reader per round',
        position_policy='native destination positions and native blending',
        resumed=any(row['resumed'] for row in rows))
    atomic_json(root / 'summary.json', result)
    temporary = root / 'predictions.csv.tmp'
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['round', 'field', 'answer', 'success',
            'loaded', 'audit_complete', 'ambiguous_value', 'attributed_requests', 'seconds',
            'source_has_field', 'target_success', 'target_source_loaded', 'expected_answer', 'source_answers'])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(round=row['round'], field=row['reader']['field'],
                answer=row['response']['text'], seconds=row['seconds'],
                **{k: row['score'][k] for k in ('success', 'loaded', 'audit_complete', 'ambiguous_value')},
                **{k: row['score'][k] for k in ('source_has_field','target_success','target_source_loaded','expected_answer')},
                source_answers=json.dumps(row['score']['source_answers'], ensure_ascii=False),
                attributed_requests=json.dumps(row['score']['attributed_requests'])))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, root / 'predictions.csv')
    return result


def execute(cfg, resume=False, restart_failed_startup=False):
    from kv_cache_readout.io import run_lock
    from .deployment import verify
    root = path(cfg['output'])
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        manifest = verify(cfg)
        records = load_inputs(cfg, manifest)
        by_id = {r['record_id']: r for r in records}
        material = {m['id']: m for m in cfg['materials']}
        plan = schedule(cfg, records)
        identity = dict(config=cfg, manifest=manifest, code=code_identity(), schedule_sha=digest(plan),
                        runtime={'pythonhashseed': '0'})
        state_file = root / 'state.json'
        if resume:
            state = read_json(state_file)
            if restart_failed_startup:
                if state['cursor'] != 0 or state['status'] != 'failed':
                    raise ValueError('Startup retry requires a failed run with zero completed rounds')
                if list(root.glob('round_*/commit.json')) or list(root.glob('round_*/result.json')):
                    raise ValueError('Startup retry refuses a run containing round results or cache commits')
                previous = read_json(root / state.get('identity_file', 'identity.json'))
                if digest(previous) != state['identity']:
                    raise ValueError('Stored startup identity is corrupted')
                stable = lambda value: {k: v for k, v in value.items() if k not in ('code', 'runtime')}
                if stable(previous) != stable(identity) or read_json(root / 'schedule.json') != plan:
                    raise ValueError('Startup retry requires unchanged configuration, deployment, data and schedule')
                atomic_json(root / ('startup_failure_' + uuid.uuid4().hex + '.json'), state)
                identity_file = 'identity_' + digest(identity) + '.json'
                atomic_json(root / identity_file, identity)
                state = dict(identity=digest(identity), identity_file=identity_file, cursor=0, status='prepared')
                atomic_json(state_file, state)
            if state['identity'] != digest(identity):
                raise ValueError('Resume identity mismatch: configuration, data, deployment or code changed')
            if read_json(root / 'schedule.json') != plan:
                raise ValueError('Frozen request sequence changed')
            if state['cursor']:
                for index in range(state['cursor']):
                    committed = root / f'round_{index:06d}'
                    commit = read_json(committed / 'commit.json')
                    for name, expected in commit['files'].items():
                        if sha_file(committed / name) != expected:
                            raise ValueError(f'Committed round artifact changed: {index}/{name}')
                if state.get('checkpoint'):
                    checkpoint_file = verified_checkpoint(root, state)
                    checkpoint_round = state['checkpoint']['round']
                    committed = read_json(root / f'round_{checkpoint_round:06d}' / 'commit.json')
                    if committed.get('checkpoint') != state['checkpoint']:
                        raise ValueError('Committed round and rolling cache checkpoint disagree')
                    remove_stale_checkpoints(root, checkpoint_file)
                else:
                    if state['cursor'] >= CHECKPOINT_INTERVAL:
                        raise ValueError('Periodic rolling cache checkpoint is missing')
                    remove_stale_checkpoints(root, None)
            else:
                remove_stale_checkpoints(root, None)
        else:
            if state_file.exists() or (root / 'identity.json').exists():
                raise ValueError('Run exists; use resume or a new output directory')
            atomic_json(root / 'identity.json', identity)
            atomic_json(root / 'schedule.json', plan)
            state = dict(identity=digest(identity), cursor=0, status='prepared')
            atomic_json(state_file, state)
        if state['cursor'] == len(plan):
            return summary(root, state['cursor'], len(plan), 'complete')
        progress('START', mode='resume' if resume else 'run',
                 completed=f'{state["cursor"]}/{len(plan)}',
                 writers=f'{cfg["writers_min"]}-{cfg["writers_max"]}',
                 method=f'{cfg["recomputation"]["method"]}:{cfg["recomputation"]["ratio"]:.0%}',
                 check_layer=cfg['recomputation']['check_layer'],
                 evaluation_override=cfg.get('evaluation_override', {}).get('reason', '-'),
                 use_u=cfg['use_u'], use_a=cfg['use_a'],
                 checkpoint_every=CHECKPOINT_INTERVAL, output=root)
        pause_file = root / 'pause.request'
        pause_file.unlink(missing_ok=True)
        interrupted = []
        previous = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, lambda *_: interrupted.append(True))
        try:
            from .runtime import Runtime
            progress('ENGINE', state='loading', gpu=cfg['engine']['device'])
            runtime = Runtime(cfg, manifest)
            progress('ENGINE', state='ready', gpu=cfg['engine']['device'])
            replay_start = 0
            if state.get('checkpoint'):
                checkpoint_file = verified_checkpoint(root, state)
                runtime.restore(checkpoint_file)
                replay_start = state['checkpoint']['round'] + 1
                progress('CACHE', state='restored', round=replay_start,
                         checkpoint=checkpoint_file.name)
            if replay_start < state['cursor']:
                progress('CACHE', state='rebuilding',
                         rounds=f'{replay_start + 1}-{state["cursor"]}')
                rebuild_cache(runtime, cfg, plan, replay_start, state['cursor'],
                              by_id, material, root)
                progress('CACHE', state='rebuilt', completed=state['cursor'])
            history = [w for step in plan[:state['cursor']] for w in step['writers']]
            session_started = time.monotonic()
            session_start_cursor = state['cursor']
            for step in plan[state['cursor']:]:
                index = step['index']
                final = root / f'round_{index:06d}'
                # Preserve incomplete attempts for diagnosis. Only state.json
                # commits a round; replay starts from the previous cache image.
                if final.exists():
                    final.rename(root / (final.name + '.uncommitted-' + uuid.uuid4().hex))
                attempt = root / (final.name + '.attempt-' + uuid.uuid4().hex)
                attempt.mkdir()
                started = time.monotonic()
                events, writer_results = [], []
                reused_writers = 0
                cap = cfg['writer_batch_size']
                progress('ROUND', number=f'{index + 1}/{len(plan)}', phase='writers',
                         writers=len(step['writers']), reader_field=step['reader']['field'])
                for offset in range(0, len(step['writers']), cap):
                    batch = step['writers'][offset:offset + cap]
                    requests = [runtime.request(w, by_id[w['record_id']], material[w['material']]) for w in batch]
                    batch_started = time.monotonic()
                    result = runtime.generate(requests)
                    writer_results.append(result)
                    events.extend(result['events'])
                    reused_writers += sum(
                        row['audit']['store_calls'] == 0 for row in result['rows'])
                    progress('WRITERS', round=f'{index + 1}/{len(plan)}',
                             completed=f'{min(offset + len(batch), len(step["writers"]))}/{len(step["writers"])}',
                             seconds=time.monotonic() - batch_started,
                             u_checks=result['mapping_checks']['u'],
                             cache_reused=reused_writers)
                history.extend(step['writers'])
                if cfg['cache']['sharing'] == 'cold':
                    runtime.clear()
                reader = step['reader']
                progress('READER', round=f'{index + 1}/{len(plan)}',
                         field=reader['field'], state='querying')
                reader_request = runtime.request(reader, None, material[reader['material']])
                result = runtime.generate([reader_request])
                response = result['rows'][0]
                events.extend(result['events'])
                source_requests, source_records = cache_sources(response['audit'], history)
                cached_tokens, block_summary = cache_block_summary(response['audit'])
                expected_tokens = reader_request['public_end'] - reader_request['public_start']
                hit_blocks = response['audit'].get('hit_blocks', response['audit'].get('loaded_blocks', []))
                c1_blocks = response['audit'].get('loaded_blocks', [])
                c1_regions = sorted({block.get('region', 'C1') for block in c1_blocks})
                layer0 = next((entry for entry in response['audit']['recomputation']
                               if entry['layer'] == 0), {})
                progress('CACHE_HIT', round=f'{index + 1}/{len(plan)}',
                         hit=bool(hit_blocks), c1_hit=bool(c1_blocks),
                         c1_cached_tokens=cached_tokens, c1_expected_tokens=expected_tokens,
                         full_c1_hit=cached_tokens == expected_tokens,
                         c1_blocks=len(c1_blocks), non_c1_blocks=len(hit_blocks) - len(c1_blocks),
                         c1_regions=','.join(c1_regions) or '-',
                         block_detail=block_summary,
                         layer_keys=len(response['audit']['loaded_keys']),
                         check_layer=cfg['recomputation']['check_layer'],
                         layer0_updated=layer0.get('updated_tokens', 0),
                         layer0_queries=layer0.get('query_tokens', 0),
                         source_requests=','.join(source_requests) or '-',
                         source_records=','.join(source_records) or '-',
                         unknown_sources=len(response['audit'].get('unknown_sources', [])))
                scored = score(reader, response['text'], response['audit'], history, by_id)
                atomic_json(attempt / 'result.json', dict(round=index, reader=reader,
                    response=response, score=scored, seconds=time.monotonic() - started,
                    reader_batch_seconds=result['seconds'], resumed=resume,
                    mapping_checks=result['mapping_checks']))
                atomic_json(attempt / 'writers.json', writer_results)
                atomic_json(attempt / 'events.json', events)
                pause_requested = bool(interrupted or pause_file.exists())
                checkpoint_due = (index + 1) % CHECKPOINT_INTERVAL == 0 or pause_requested
                checkpoint = None
                snapshot = None
                checkpoint_file = None
                if checkpoint_due:
                    checkpoint_file = root / f'cache_{index:06d}_{uuid.uuid4().hex}.pt'
                    snapshot = runtime.snapshot(checkpoint_file)
                    checkpoint = dict(file=checkpoint_file.name, round=index,
                                      sha256=sha_file(checkpoint_file))
                atomic_json(attempt / 'commit.json', dict(snapshot=snapshot, checkpoint=checkpoint,
                    files={p.name: sha_file(p) for p in attempt.iterdir() if p.is_file()}))
                attempt.rename(final)
                previous_checkpoint = state.get('checkpoint', {}).get('file')
                state.update(cursor=index + 1, status='running')
                if checkpoint is not None:
                    state['checkpoint'] = checkpoint
                atomic_json(state_file, state)
                if (checkpoint_file is not None and previous_checkpoint
                        and previous_checkpoint != checkpoint_file.name):
                    (root / previous_checkpoint).unlink(missing_ok=True)
                if checkpoint is not None:
                    progress('CHECKPOINT', round=f'{index + 1}/{len(plan)}',
                             reason='pause' if pause_requested else 'interval',
                             entries=snapshot['entries'], file=checkpoint_file.name)
                current = summary(root, state['cursor'], len(plan), 'running')
                completed_this_session = state['cursor'] - session_start_cursor
                average = (time.monotonic() - session_started) / completed_this_session
                eta = average * (len(plan) - state['cursor'])
                progress('ROUND_DONE', number=f'{index + 1}/{len(plan)}',
                         cache_hit=scored['loaded'], leak=scored['success'],
                         source_has_field=scored['source_has_field'], target_success=scored['target_success'],
                         target_source_loaded=scored['target_source_loaded'],
                         attributed=len(scored['attributed_requests']),
                         leak_sources=','.join(scored['attributed_requests']) or '-',
                         answer=response['text'], finish_reason=response['finish_reason'],
                         generated_tokens=response['generated_tokens'],
                         token_preview=','.join(response.get('generated_token_pieces', [])[:8]) or '-',
                         seconds=time.monotonic() - started,
                         leak_rate=f'{current["completed_query_leak_rate"] or 0:.2%}',
                         cache_load_rate=f'{current["cache_load_rate"] or 0:.2%}', eta=duration(eta))
                if pause_requested:
                    state['status'] = 'paused'
                    atomic_json(state_file, state)
                    progress('PAUSED', completed=f'{state["cursor"]}/{len(plan)}')
                    return summary(root, state['cursor'], len(plan), 'paused')
            state['status'] = 'complete'
            atomic_json(state_file, state)
            progress('COMPLETE', completed=f'{state["cursor"]}/{len(plan)}')
            return summary(root, state['cursor'], len(plan), 'complete')
        except BaseException as error:
            state.update(status='failed', error=f'{type(error).__name__}: {error}')
            atomic_json(state_file, state)
            summary(root, state['cursor'], len(plan), 'failed')
            progress('FAILED', completed=f'{state["cursor"]}/{len(plan)}', error=state['error'])
            raise
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'preflight', 'run', 'resume', 'pause', 'status'])
    parser.add_argument('--config', default='configs/e2e_hr_native.json')
    parser.add_argument('--restart-failed-startup', action='store_true',
                        help='Resume a zero-round initialization failure after code repair; preserve prior identity')
    args = parser.parse_args()
    if args.restart_failed_startup and args.action != 'resume':
        parser.error('--restart-failed-startup requires resume')
    cfg = load_settings(args.config)
    if args.action in ('run', 'resume') and os.environ.get('PYTHONHASHSEED') != '0':
        # CPython reads this variable at process startup. Re-exec before
        # loading native libraries so the coordinator and workers agree.
        os.environ['PYTHONHASHSEED'] = '0'
        os.execv(sys.executable, [sys.executable, '-m', 'e2e_cache', *sys.argv[1:]])
    root = path(cfg['output'])
    if args.action in ('pause', 'status'):
        if not (root / 'state.json').exists():
            raise ValueError('Run has no state file')
        if args.action == 'pause':
            atomic_json(root / 'pause.request', {'requested': time.time()})
        print(json.dumps(read_json(root / 'state.json'), ensure_ascii=False, indent=2))
    elif args.action == 'prepare':
        from .deployment import prepare
        prepare(cfg)
        print(f'Deployment exported to {path(cfg["deployment"])}')
    elif args.action == 'preflight':
        from .deployment import verify
        manifest = verify(cfg)
        records = load_inputs(cfg, manifest)
        print(json.dumps(dict(scene=manifest['experiment'].get('scene', 'medical'),
            records=len(records), rounds=cfg['rounds'],
            schedule_sha=digest(schedule(cfg, records)), native_gpu_check=False), indent=2))
    else:
        print(json.dumps(execute(cfg, resume=args.action == 'resume',
                                 restart_failed_startup=args.restart_failed_startup), indent=2))


if __name__ == '__main__':
    main()
