from __future__ import annotations

import json
import math
import random
import signal
import time
from contextlib import nullcontext

import torch

from .backend import Backend
from .config import fingerprint, resolve
from .data import dataset_identity, load_split
from .distributed import Runtime
from .io import atomic_json, atomic_torch, code_identity, read_checkpoint, run_lock
from .optim import parameters_for, assign_gradients, make_scheduler, rng_state, restore_rng, seed_all
from .suffix import preservation_loss
from .progress import emit

CHECKPOINT_SCHEMA = 'aligned_readout_v3'


def qa_phase(backend, prompt, samples, denominator, epoch):
    emit(backend.cfg,'writer_c1',records=len(samples),state='begin')
    contexts = backend.blend_many(samples, prompt, epoch=epoch)
    emit(backend.cfg,'writer_c1',records=len(samples),state='done')
    questions = [r['qa'] for r in samples]
    values = backend.answer_components_many(contexts, questions)
    other, focal = prompt.sum() * 0, prompt.sum() * 0
    selected = set(backend.cfg['focal_fields'])
    active = [False, False, False]
    for qs, components in zip(questions, values):
        if 'readout' in backend.cfg:
            from .readout import combine_tasks
            pair,routes=combine_tasks(backend.cfg,qs,components)
            other=other+pair[0]/denominator
            focal=focal+pair[1]/denominator
            active[0] |= 'other' in routes
            active[1] |= 'focal' in routes
            continue
        if 'joint_loss' in backend.cfg:
            from .joint import active_groups
            if len(qs) != 1:
                raise ValueError('Joint training requires one question per sample')
            other = other+components[0][0]/denominator
            focal = focal+components[0][1]/denominator
            routes = active_groups(qs[0], backend.cfg['joint_loss'])
            active[0] |= 'other' in routes
            active[1] |= 'focal' in routes
            continue
        groups = {}
        for qa, pair in zip(qs, components):
            groups.setdefault(qa.get('group', qa['field']), []).append(pair)
        for name, pairs in groups.items():
            if name in selected:
                focal = focal + torch.stack([p[1] for p in pairs]).mean() / len(groups) / denominator
                active[1] = True
            else:
                other = other + torch.stack([p[0] for p in pairs]).mean() / len(groups) / denominator
                active[0] = True
    emit(backend.cfg,'backward_other',state='begin')
    go = torch.autograd.grad(other, prompt, retain_graph=True)[0]
    emit(backend.cfg,'backward_other',state='done')
    emit(backend.cfg,'backward_focal',state='begin')
    gf = torch.autograd.grad(focal, prompt)[0]
    emit(backend.cfg,'backward_focal',state='done')
    split = backend.cfg['prompt_partition']['other_length']
    end = split + backend.cfg['prompt_partition']['focal_length']
    gradient = torch.cat((go[:split], gf[split:end], torch.zeros_like(prompt[end:]))).detach()
    return gradient, float((other+focal).detach()), active


def keep_phase(backend, prompt, samples, denominator, epoch):
    emit(backend.cfg,'keep_sources',records=len(samples),state='begin')
    with torch.no_grad():
        _, references = backend.source_many(samples, None, epoch=epoch)
    _, students = backend.source_many(samples, prompt, epoch=epoch)
    emit(backend.cfg,'keep_sources',state='done')
    loss = prompt.sum() * 0
    for index,(student, reference) in enumerate(zip(students, references)):
        emit(backend.cfg,'keep_recompute',record=index+1,total=len(students),state='begin')
        value, _ = preservation_loss(backend, student, reference,
                                    ratio=backend.cfg['native_contract']['recompute_ratio'])
        loss = loss + value * backend.cfg['scoped_cache']['preservation_weight'] / denominator
    emit(backend.cfg,'backward_keep',state='begin')
    gradient = torch.autograd.grad(loss, prompt)[0]
    emit(backend.cfg,'backward_keep',state='done')
    keep = backend.cfg['prompt_partition']['keep_length']
    gradient = torch.cat((torch.zeros_like(prompt[:-keep]), gradient[-keep:])).detach()
    return gradient, float(loss.detach())


def batch_gradients(backend, parameters, samples, denominator, epoch):
    """Function boundaries release QA graphs before constructing S graphs."""
    gradient = torch.zeros_like(torch.cat(parameters))
    total = 0.
    active = [False, False, False]
    cap = backend.cfg['training']['source_batch_records']
    for offset in range(0, len(samples), cap):
        g, value, flags = qa_phase(backend, torch.cat(parameters), samples[offset:offset+cap], denominator, epoch)
        gradient.add_(g); total += value
        active = [a or b for a, b in zip(active, flags)]
    if backend.cfg['scoped_cache']['preservation_weight'] > 0:
        cap = backend.cfg['training']['preservation_batch_records']
        for offset in range(0, len(samples), cap):
            g, value = keep_phase(backend, torch.cat(parameters), samples[offset:offset+cap], denominator, epoch)
            gradient.add_(g); total += value
            active[2] = True
    if not torch.isfinite(gradient).all() or not math.isfinite(total):
        raise FloatingPointError('Non-finite two-phase gradient/loss')
    return gradient, total, active


def validate_checkpoint(state, cfg, backend, dataset, *, training=False, world=1):
    if state.get('schema_version') != CHECKPOINT_SCHEMA:
        raise ValueError(f'Expected {CHECKPOINT_SCHEMA}; v1 optimizer checkpoints cannot resume v2')
    for key, expected in [('config_sha256', fingerprint(cfg)), ('dataset', dataset),
                          ('native_contract', cfg['native_contract']), ('model', backend.identity),
                          ('code', code_identity())]:
        if state.get(key) != expected:
            raise ValueError(f'Checkpoint identity mismatch: {key}')
    if training and len(state.get('rng_by_rank', [])) != world:
        raise ValueError('Exact training resume requires the original world size')
    if not torch.isfinite(state['prompt']).all():
        raise ValueError('Checkpoint prompt contains nonfinite values')
    if tuple(state['prompt'].shape) != (cfg['prompt_length'], backend.model.config.hidden_size):
        raise ValueError('Checkpoint prompt shape differs from the configured U')


def train(cfg, *, resume=False, resume_from=None, stop_after_epochs=None, max_updates=None):
    from .identity import resume_compatibility, local_batch, remap_seed
    runtime = Runtime(cfg)
    output = resolve(cfg['output_directory'])
    latest = output / 'latest.pt'
    seed_all(cfg['seed'])
    dataset = dataset_identity(cfg)
    records = load_split(cfg, 'train', training=True)
    if any(len(r['qa']) > cfg['training']['questions_per_record'] for r in records):
        raise ValueError('questions_per_record must cover every question; component subsets are not allowed')
    if stop_after_epochs is not None and not 1 <= stop_after_epochs <= cfg['training']['epochs']:
        raise ValueError('stop_after_epochs must be within the original training schedule')
    lock = run_lock(output) if runtime.rank == 0 else nullcontext()
    error = None
    try:
        lock.__enter__()
    except Exception as exc:
        error = str(exc)
    errors = runtime.objects(error)
    if any(errors):
        runtime.close()
        raise RuntimeError(str(errors))
    interrupted, previous = [], {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda *_: interrupted.append(True))
    try:
        emit(cfg,'model_load',state='begin')
        backend = Backend.load(dict(cfg, device=str(runtime.device)))
        backend.cfg = cfg
        emit(cfg,'model_load',state='done')
        state = read_checkpoint(resolve(resume_from) if resume_from else latest) if (resume or resume_from) else None
        if state:
            branch = resume_compatibility(state, cfg, dataset, backend.identity, code_identity())
            if branch and not resume_from:
                raise ValueError('A scientific condition change requires the automatic branch launcher')
            if tuple(state['prompt'].shape) != (cfg['prompt_length'], backend.model.config.hidden_size):
                raise ValueError('Checkpoint prompt shape mismatch')
            initial = state['prompt'].to(backend.device, torch.float32)
        else:
            if latest.exists():
                raise ValueError('Run already exists; use --resume')
            initial = backend.initial_prompt().to(backend.device)
        parameters = parameters_for(initial, cfg)
        optimizer = torch.optim.AdamW(parameters, lr=cfg['training']['learning_rate'],
                                      weight_decay=cfg['training']['weight_decay'])
        total_steps = math.ceil(len(records) / cfg['training']['batch_records']) * cfg['training']['epochs']
        scheduler = make_scheduler(optimizer, total_steps, cfg['training']['warmup_fraction'])
        epoch = cursor = step = 0
        if state:
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            epoch, cursor, step = (state['progress'][k] for k in ('epoch', 'cursor', 'step'))
            if len(state.get('rng_by_rank', [])) == runtime.world:
                restore_rng(state['rng_by_rank'][runtime.rank])
            else:
                seed_all(remap_seed(cfg['seed'], epoch, cursor, step, runtime.rank, runtime.world))
                if runtime.rank == 0:
                    print('[RESUME] GPU count changed: global cursor and optimizer preserved; RNG deterministically remapped', flush=True)
            if branch:
                # Preserve Adam moments and scheduler position; use the new explicit schedule/LR.
                scheduler.base_lrs = [cfg['training']['learning_rate']] * len(optimizer.param_groups)
                from .optim import schedule_factor
                factor = schedule_factor(step, total_steps, cfg['training']['warmup_fraction'])
                for group in optimizer.param_groups:
                    group['initial_lr'] = cfg['training']['learning_rate']
                    group['lr'] = cfg['training']['learning_rate'] * factor
                    group['weight_decay'] = cfg['training']['weight_decay']

        def save(status, extra=None):
            rng = runtime.objects(rng_state())
            if runtime.rank == 0:
                checkpoint = dict(schema_version=CHECKPOINT_SCHEMA, method='soft',
                    prompt=torch.cat(parameters).detach().cpu(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), rng_by_rank=rng,
                    progress=dict(epoch=epoch, cursor=cursor, step=step, status=status),
                    config=cfg, config_sha256=fingerprint(cfg), dataset=dataset,
                    parent_checkpoint=str(resume_from) if resume_from else None,
                    execution_policy='aligned_native_operators_without_inference_engine',
                    resume_policy=dict(world_size=runtime.world,
                        previous_world_size=len(state.get('rng_by_rank',[])) if state else None,
                        rng='restored' if state and len(state.get('rng_by_rank',[]))==runtime.world else 'seeded_or_remapped'),
                    model=backend.identity, native_contract=cfg['native_contract'], code=code_identity())
                atomic_torch(latest, checkpoint)
                if extra:
                    atomic_torch(output / extra, checkpoint)
                atomic_json(output / 'summary.json', checkpoint['progress'])
            runtime.barrier()

        target = stop_after_epochs or cfg['training']['epochs']
        if epoch > target:
            raise ValueError('Checkpoint has already passed the requested epoch boundary')
        effective_limit = max_updates if max_updates is not None else cfg['training']['max_steps']
        if effective_limit is not None and effective_limit < 1:
            raise ValueError('max_updates must be positive')
        if effective_limit is not None and step >= effective_limit:
            return dict(status='paused', epoch=epoch, cursor=cursor, step=step)
        pause = output / 'pause.request'
        if runtime.rank == 0:
            pause.unlink(missing_ok=True)
        emit(cfg,'initial_barrier',state='begin')
        runtime.barrier()
        emit(cfg,'initial_barrier',state='done')
        while epoch < target:
            order = list(range(len(records)))
            random.Random(f"{cfg['seed']}|epoch|{epoch}").shuffle(order)
            if not 0 <= cursor <= len(order):
                raise ValueError('Invalid checkpoint cursor')
            while cursor < len(order):
                batch = order[cursor:cursor + cfg['training']['batch_records']]
                samples = [records[i] for i in local_batch(batch, runtime.rank, runtime.world)]
                emit(cfg,'update',step=step+1,epoch=epoch+1,records=len(samples),
                     questions=sum(len(record['qa']) for record in samples),state='begin')
                optimizer.zero_grad(set_to_none=True)
                started = time.monotonic()
                error = None
                try:
                    gradient, loss, active = batch_gradients(backend, parameters, samples, len(batch), epoch)
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                emit(cfg,'gradient_sync',state='begin')
                errors = runtime.objects(error)
                if any(errors):
                    raise RuntimeError(f'No optimizer update committed: {errors}')
                gradient = runtime.sum(gradient)
                flags = runtime.sum(torch.tensor(active, device=backend.device, dtype=torch.int))
                loss = runtime.sum(torch.tensor(loss, device=backend.device)).item()
                emit(cfg,'gradient_sync',state='done')
                assign_gradients(parameters, gradient, flags.tolist())
                norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['training']['grad_clip'], error_if_nonfinite=True)
                used_lr = optimizer.param_groups[0]['lr']
                optimizer.step(); scheduler.step()
                cursor += len(batch); step += 1
                if runtime.rank == 0:
                    row = dict(epoch=epoch+1, cursor=cursor, step=step, loss=loss,
                               check_layer=cfg['native_contract']['check_layer'],
                               recompute_ratio=cfg['native_contract']['recompute_ratio'],
                               gradient_norm=float(norm), learning_rate=used_lr,
                               seconds=time.monotonic()-started, active_segments=flags.bool().tolist())
                    if 'execution' in cfg:
                        row.update(execution=cfg['execution']['profile'],
                                   global_batch=len(batch),world_size=runtime.world,
                                   requests_per_forward=1,teacher_forcing=cfg['execution']['teacher_forcing'])
                        if backend.native.fast:
                            row.pop('requests_per_forward')
                            row.update(writer_requests_per_forward=1,
                                       sample_batch_size=cfg['training'].get('sample_batch_size',cfg['training']['source_batch_records']),
                                       questions_per_sample_batch=cfg['training'].get('questions_per_sample_batch',1),
                                       local_records=len(samples),
                                       kv_workspace_allocations=backend.native.workspace.allocations)
                    if 'readout' in cfg:
                        from .readout import task_weights
                        row.update(train_mode=cfg['readout']['train_mode'],
                                   task_weights=task_weights(cfg),eval_mode=cfg['readout']['eval_mode'])
                    if 'c1_selection' in cfg:
                        row.update(selection_method=cfg['c1_selection']['method'],selection_seed=cfg['c1_selection']['seed'])
                    with (output / 'training.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(row) + '\n')
                    print('[TRAIN] ' + ' | '.join(f'{k}={v}' for k, v in row.items()), flush=True)
                limited = effective_limit is not None and step >= effective_limit
                stopping = runtime.any(interrupted or pause.exists() or limited)
                if step % cfg['training']['checkpoint_every_steps'] == 0 or stopping:
                    save('paused' if stopping else 'running')
                if stopping:
                    return dict(status='paused', epoch=epoch, cursor=cursor, step=step)
            epoch += 1; cursor = 0
            save('running', f'epoch_{epoch:03d}.pt')
        status = 'complete' if epoch == cfg['training']['epochs'] else 'stage_reached'
        save(status, 'final.pt' if status == 'complete' else f'epoch_{epoch:03d}.pt')
        return dict(status=status, epoch=epoch, cursor=cursor, step=step)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        lock.__exit__(None, None, None)
        runtime.close()
