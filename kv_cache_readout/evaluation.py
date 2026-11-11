"""Record-atomic native evaluation and explicitly labelled legacy controls."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from types import SimpleNamespace

import torch

from .backend import Backend, Context
from .config import resolve, fingerprint, file_hash
from .data import dataset_identity, load_split
from .io import atomic_json, read_checkpoint, code_identity, run_lock
from .training import validate_checkpoint
from .suffix import pair_for, mismatch_for
from .legacy_cache import Context as LegacyContext
from .legacy_recomputation import recompute_context

REPEAT = ('Repeat the preceding passage verbatim. Output only the passage, preserving its '
          'original wording, punctuation, and paragraph breaks. Do not add explanations.')


def conditions():
    result = [dict(id='c1_native_r30', task='c1', algorithm='native', ratio=.3, use_u=True)]
    for method in ('random', 'epic', 'cacheblend'):
        for ratio in (.1, .3):
            result.append(dict(id=f'c1_legacy_{method}_r{int(ratio*100)}', task='c1',
                               algorithm=method, ratio=ratio, use_u=True))
            for use_u in (True, False):
                result.append(dict(id=f's_legacy_{method}_r{int(ratio*100)}_u{int(use_u)}', task='suffix',
                                   algorithm=method, ratio=ratio, use_u=use_u))
    return result


def legacy_context(backend, layers, start, tokens, record_id, condition):
    adapter = SimpleNamespace(model=backend.model, cfg=dict(backend.cfg, attention_implementation='eager'),
                              public_ids=tokens, device=backend.device, embedding=backend.embedding,
                              tensor=backend.tensor, forward=backend.forward, dtype=backend.dtype)
    old = LegacyContext(layers, start+len(tokens), start, len(tokens))
    updated, audit = recompute_context(adapter, old, method=condition['algorithm'],
                                      ratio=condition['ratio'], seed=backend.cfg['seed'],
                                      record_id=record_id, token_ids=tokens if condition['task']=='suffix' else None)
    return Context(updated.layers, updated.next_position, dict(audit, legacy_question_boundary=True))


def reconstruction_rows(record, rows):
    by_field = {r['field']: r for r in rows}
    extra = []
    for parent, parts, joiner in [('name', ('name_first', 'name_middle', 'name_last'), ' '),
                                  ('birth_date', ('birth_year', 'birth_month', 'birth_day'), '-')]:
        if not all(part in by_field for part in parts):
            continue
        predictions = [by_field[p]['prediction'].strip() for p in parts]
        if parent == 'name':
            predictions = [value for value in predictions if value]
        prediction = joiner.join(predictions)
        extra.append(dict(record_id=record['record_id'], field=parent, group=parent,
                          gold=record['fields'][parent], prediction=prediction,
                          em=float(prediction == record['fields'][parent]),
                          nll=statistics.mean(by_field[p]['nll'] for p in parts), prediction_kind='reconstructed'))
    return extra


def metrics(rows):
    scoring = [r for r in rows if r.get('prediction_kind') != 'generated_component'
               and r['field'] != 'name_full_direct']
    fields = defaultdict(list)
    for row in scoring:
        fields[row['field']].append(row)
    per_field = {field: dict(count=len(values), em=statistics.mean(r['em'] for r in values),
                            nll=statistics.mean(r['nll'] for r in values)) for field, values in fields.items()}
    generated = [r for r in rows if r.get('prediction_kind') != 'reconstructed']
    result = dict(questions=len(generated), records=len({r['record_id'] for r in rows}),
                macro_em=statistics.mean(v['em'] for v in per_field.values()) if per_field else None,
                macro_nll=statistics.mean(v['nll'] for v in per_field.values()) if per_field else None,
                blank_rate=statistics.mean(not r['prediction'].strip() for r in generated),
                per_field=per_field,
                direct_name_em=(statistics.mean(r['em'] for r in rows if r['field']=='name_full_direct')
                                if any(r['field']=='name_full_direct' for r in rows) else None))
    if any('readout' in r for r in rows):
        result['suffix'] = dict(exact_match=statistics.mean(r['em'] for r in rows),
            contains_original_rate=statistics.mean(r['contains_original'] for r in rows),
            edit_similarity=statistics.mean(r['edit_similarity'] for r in rows),
            truncated_rate=statistics.mean(r['truncated'] for r in rows))
        result['readout'] = {f'{stage}_{control}_{metric}': statistics.mean(r['readout'][stage][control][metric] for r in rows)
                             for stage in ('pre','post') for control in ('correct','wrong','clean')
                             for metric in ('nll','token_accuracy','first8_accuracy','first_stop_probability',
                                            'first_gold_probability','first_gold_rank')}
    return result


def edit_similarity(a,b):
    a,b = sorted((a,b), key=len)
    distance = len(b)
    if a:
        equal = {}
        for i,c in enumerate(a):
            equal[c] = equal.get(c,0) | (1 << i)
        positive,negative = ~0,0
        high = 1 << (len(a)-1)
        distance = len(a)
        for c in b:
            eq = equal.get(c,0)
            vertical = eq | negative
            horizontal = (((eq & positive) + positive)^positive) | eq
            plus,minus = negative | ~(horizontal | positive), positive & horizontal
            distance += bool(plus & high) - bool(minus & high)
            plus,minus = (plus << 1)|1, minus << 1
            positive,negative = minus | ~(vertical | plus), plus & vertical
    return 1-distance/max(len(a),len(b)) if a or b else 1.


def _json_scores(backend, context, qa):
    scores = backend.conditional_scores(context, qa, content_only=True)
    scores.pop('first_logits')
    return scores


@torch.no_grad()
def _record(backend, record, prompt, condition):
    effective = prompt if condition['use_u'] else None
    if condition['algorithm'] == 'native':
        context = backend.blend_many([record], effective)[0]
    else:
        writers, suffixes = backend.source_many([record], effective)
        suffix = suffixes[0]
        if condition['task'] == 'c1':
            start = suffix.start - (backend.cfg['prompt_partition']['keep_length'] if effective is not None else 0)
            start -= len(backend.delimiter_ids) + len(backend.public_ids)
            context = legacy_context(backend, writers[0], start, backend.public_ids, record['record_id'], condition)
        else:
            context = legacy_context(backend, suffix.layers, suffix.start, list(suffix.token_ids), record['record_id'], condition)
    questions = record['qa']
    diagnostics = None
    if condition['task'] == 'suffix':
        target = pair_for(record, backend.cfg)
        donor = mismatch_for(record, backend.cfg)
        qa = dict(field='suffix_verbatim', question=REPEAT, answer=target['text'], group='suffix_verbatim')
        questions = [qa]
        _, wrongs = backend.source_many([dict(record, _suffix_fixture=donor)], effective)
        wrong = wrongs[0]
        _, clean_layers = backend.forward(ids=target['ids'], position_start=suffix.start)
        raw = dict(correct=Context(suffix.layers, suffix.start+suffix.length, {'legacy_question_boundary': True}),
                   wrong=Context(wrong.layers, wrong.start+wrong.length, {'legacy_question_boundary': True}),
                   clean=Context(clean_layers, suffix.start+suffix.length, {'legacy_question_boundary': True}))
        post = dict(correct=context,
                    wrong=legacy_context(backend, wrong.layers, wrong.start, list(wrong.token_ids), record['record_id'], condition),
                    clean=legacy_context(backend, clean_layers, suffix.start, target['ids'], record['record_id'], condition))
        diagnostics = dict(target_suffix_id=target['suffix_id'], wrong_suffix_id=donor['suffix_id'],
                           donor_token_length_delta=len(donor['ids'])-len(target['ids']),
                           pre={k: _json_scores(backend, v, qa) for k,v in raw.items()},
                           post={k: _json_scores(backend, v, qa) for k,v in post.items()})
    rows = []
    for qa in questions:
        prediction, generation = backend.generate(context, qa, max_new_tokens=256 if diagnostics else None)
        scores = _json_scores(backend, context, qa)
        ce, _ = backend.answer_loss_components(context, qa)
        row = dict(record_id=record['record_id'], field=qa['field'], group=qa.get('group', qa['field']),
                   question=qa['question'], gold=qa['answer'], prediction=prediction,
                   em=float(prediction.strip()==qa['answer'].strip()), nll=float(ce),
                   prediction_kind=qa.get('prediction_kind', 'generated_field'),
                   conditional=scores, audit=context.audit, **generation)
        if diagnostics:
            row.update(readout=diagnostics, em=float(prediction==qa['answer']),
                       contains_original=qa['answer'] in prediction,
                       edit_similarity=edit_similarity(prediction,qa['answer']))
        rows.append(row)
    if condition['task'] == 'c1':
        rows.extend(reconstruction_rows(record, rows))
    return rows


def _csv(output, rows):
    columns = sorted(set().union(*(row.keys() for row in rows)))
    temporary = output / 'predictions.csv.tmp'
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list)) else v
                          for k,v in row.items()} for row in rows)
    temporary.replace(output / 'predictions.csv')


def evaluate(cfg, checkpoint_path, output_path, *, limit=None, resume=False, condition=None, backend=None):
    condition = condition or conditions()[0]
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    checkpoint_file = resolve(checkpoint_path)
    state = read_checkpoint(checkpoint_file)
    backend = backend or Backend.load(cfg)
    dataset = dataset_identity(cfg)
    validate_checkpoint(state, cfg, backend, dataset)
    records = load_split(cfg, 'test', training=False)
    records = records if limit is None else records[:limit]
    identity = dict(checkpoint_sha256=file_hash(checkpoint_file), config=fingerprint(cfg),
                    dataset=dataset, code=code_identity(), condition=condition,
                    records=[r['record_id'] for r in records])
    output = resolve(output_path)
    if output.exists() and not resume:
        raise ValueError('Evaluation exists; use --resume or a new output')
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        identity_file = output / 'identity.json'
        if identity_file.exists():
            if json.loads(identity_file.read_text(encoding='utf-8')) != identity:
                raise ValueError('Evaluation resume identity changed')
        else:
            if resume:
                raise ValueError('Evaluation resume identity is missing')
            atomic_json(identity_file, identity)
        prompt = state['prompt'].to(backend.device, torch.float32)
        rows = []
        for index, record in enumerate(records):
            filename = output / f'record_{index:06d}.json'
            if filename.exists():
                committed = json.loads(filename.read_text(encoding='utf-8'))
                if committed['sha256'] != fingerprint(committed['rows']):
                    raise ValueError(f'Committed prediction changed: {filename}')
                current = committed['rows']
                expected_fields = [q['field'] for q in record['qa']] if condition['task']=='c1' else ['suffix_verbatim']
                actual_fields = [r['field'] for r in current if r.get('prediction_kind')!='reconstructed']
                if actual_fields != expected_fields or any(r['record_id']!=record['record_id'] for r in current):
                    raise ValueError('Committed record is incomplete or has the wrong identity')
            else:
                current = _record(backend, record, prompt, condition)
                atomic_json(filename, dict(rows=current, sha256=fingerprint(current)))
            rows.extend(current)
            _csv(output, rows)
            atomic_json(output / 'summary.json', dict(status='running', completed_records=index+1,
                                                     condition=condition, **metrics(rows)))
            print(f'[EVAL] condition={condition["id"]} records={index+1}/{len(records)}', flush=True)
        summary = dict(status='complete', condition=condition, **metrics(rows))
        atomic_json(output / 'summary.json', summary)
        return summary


def evaluate_suite(cfg, checkpoint_path, output_path, *, limit=None, resume=False):
    backend = Backend.load(cfg)
    output = resolve(output_path)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for condition in conditions():
        target = output / condition['id']
        summaries.append(evaluate(cfg, checkpoint_path, target, limit=limit,
                                  resume=resume and target.exists(), condition=condition, backend=backend))
        atomic_json(output / 'suite_summary.json', summaries)
    # Explicitly paired S/no-U comparison for each legacy policy, preserving
    # each record's complete free response alongside its reference response.
    for method in ('random','epic','cacheblend'):
        for ratio in (10,30):
            prefix = f's_legacy_{method}_r{ratio}'
            def read(name):
                return [json.loads(p.read_text(encoding='utf-8'))['rows'][0]
                        for p in sorted((output/name).glob('record_*.json'))]
            student,reference = read(prefix+'_u1'),read(prefix+'_u0')
            paired = [dict(record_id=a['record_id'],gold=a['gold'],prediction=a['prediction'],
                           reference_prediction=b['prediction'],em=a['em'],reference_em=b['em'],
                           baseline_correct_to_wrong=bool(b['em'] and not a['em'])) for a,b in zip(student,reference)]
            if any(a['record_id']!=b['record_id'] or a['gold']!=b['gold'] for a,b in zip(student,reference)):
                raise ValueError('S/no-U comparison records differ')
            atomic_json(output/(prefix+'_comparison.json'),paired)
    return summaries
