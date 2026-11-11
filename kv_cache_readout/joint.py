"""Fixed-schema joint readout, field-weighted losses and strict generation scoring."""
import json
import math

FIELDS = ('name', 'identity_number', 'address', 'email', 'birth_date', 'position',
          'department', 'salary', 'leave_duration', 'leave_reason')
PROMPT = '''Recover the target employee's record from the retained cached context.
Return exactly one JSON object with these keys in this order:
name, identity_number, address, email, birth_date, position, department, salary, leave_duration, leave_reason.
Use the recorded values exactly, preserving spelling, digits, punctuation, and date formats.
Return the complete employee name and complete birth date. Use JSON null for a field that is not provided.
Represent every non-null value as a string. Do not guess, add extra keys, or include explanations or Markdown.'''


def target(fields):
    parts = ['{']; spans = []
    for index, field in enumerate(FIELDS):
        parts.append((',' if index else '')+json.dumps(field)+':')
        start = sum(map(len, parts))
        value = fields.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError('Joint targets require string values')
        encoded = json.dumps(value, ensure_ascii=False)
        parts.append(encoded)
        spans.append(dict(field=field, present=field in fields, span=[start, start+len(encoded)]))
    return ''.join(parts)+'}', spans


def question(fields):
    answer, spans = target(fields)
    return dict(field='all_fields', group='all_fields', question=PROMPT, answer=answer,
                template_id='hr_joint_json_v1', prediction_kind='joint_json', joint_fields=spans)


def default_settings():
    return dict(default=dict(type='ce', weight=1., u_group='other'),
                focal_defaults=dict(gamma=2., alpha=1.),
                fields={f: dict(type='focal', weight=1., u_group='focal')
                        for f in ('identity_number', 'address', 'email', 'birth_date')},
                format=dict(type='ce', weight=.1, u_group='other'),
                missing=dict(type='ce', weight=.1, u_group='other'))


def options(settings, field):
    result = dict(settings['focal_defaults'])
    result.update(settings['default'])
    result.update(settings[field] if field in ('format', 'missing') else settings['fields'].get(field, {}))
    return result


def validate_settings(settings):
    if set(settings) != {'default', 'focal_defaults', 'fields', 'format', 'missing'}:
        raise ValueError('Invalid joint_loss keys')
    if set(settings['default']) != {'type', 'weight', 'u_group'}:
        raise ValueError('joint_loss.default requires type, weight, u_group')
    if set(settings['focal_defaults']) != {'gamma', 'alpha'}:
        raise ValueError('joint_loss.focal_defaults requires gamma and alpha')
    if set(settings['fields']) - set(FIELDS):
        raise ValueError('Unknown joint_loss field')
    for key in ('format', 'missing', *FIELDS):
        override = settings[key] if key in ('format', 'missing') else settings['fields'].get(key, {})
        if set(override) - {'type', 'weight', 'u_group', 'gamma', 'alpha', 'reduction'}:
            raise ValueError('Unknown joint_loss option')
        if 'reduction' in override and (key != 'birth_date' or override['reduction'] not in ('token_mean','ymd_mean')):
            raise ValueError('Only birth_date supports token_mean/ymd_mean reduction')
        opt = options(settings, key)
        if opt['type'] not in ('ce', 'focal') or opt['u_group'] not in ('other', 'focal'):
            raise ValueError('Invalid joint loss type or U group')
        for name in ('weight', 'gamma', 'alpha'):
            v = opt[name]
            if type(v) not in (float, int) or not math.isfinite(v) or v < 0:
                raise ValueError('Joint loss weights/parameters must be finite and nonnegative')


def layout(backend, qa):
    """Assign every answer token (including EOS) once, using actual prompt+answer offsets."""
    cache = getattr(backend, '_joint_layout_cache', None)
    if cache is None:
        cache = backend._joint_layout_cache = {}
    key = (qa['question'], qa['answer'], json.dumps(qa['joint_fields'], sort_keys=True))
    if key in cache:
        return cache[key]
    present_values = {k: v for k, v in json.loads(qa['answer']).items() if v is not None}
    canonical, spans = target(present_values)
    if qa['answer'] != canonical or qa['joint_fields'] != spans:
        raise ValueError('Joint answer/spans differ from canonical schema')
    prefix = backend.question_text(qa['question'])
    encoded = backend.tokenizer(prefix+qa['answer'], add_special_tokens=False, return_offsets_mapping=True)
    question_ids, target_ids = backend.question_and_target(qa)
    if encoded['input_ids'] != question_ids+target_ids[:-1]:
        raise ValueError('Joint tokenizer boundary differs from the actual model targets')
    owners = []
    for start, end in encoded['offset_mapping'][len(question_ids):]:
        start -= len(prefix); end -= len(prefix)
        overlaps = [max(0, min(end, s['span'][1])-max(start, s['span'][0])) for s in spans]
        owners.append(overlaps.index(max(overlaps)) if max(overlaps) else -1)
    owners.append(-1)  # EOS is a format token.
    for i in range(len(spans)):
        if i not in owners:
            raise ValueError('Joint field has no supervised tokens')
    result = (owners, spans)
    cache[key] = result
    return result


def date_component_owners(backend,qa):
    """Map digit tokens to year/month/day; punctuation is never a date component."""
    import datetime
    import re
    structured=qa.get('prediction_kind')=='joint_json'
    if structured:
        value=json.loads(qa['answer'])['birth_date']
        span=next(s['span'] for s in qa['joint_fields'] if s['field']=='birth_date')
        start=span[0]+1  # skip the JSON string quote
    else:
        value=qa['answer'];start=0
    if not isinstance(value,str) or not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}',value):
        raise ValueError('ymd_mean requires a complete YYYY-MM-DD target')
    datetime.date.fromisoformat(value)
    cache=getattr(backend,'_date_layout_cache',None)
    if cache is None:cache=backend._date_layout_cache={}
    key=(qa['question'],qa['answer'],start)
    if key in cache:return cache[key]
    ranges=[(start,start+4),(start+5,start+7),(start+8,start+10)]
    prefix=backend.question_text(qa['question'])
    encoded=backend.tokenizer(prefix+qa['answer'],add_special_tokens=False,return_offsets_mapping=True)
    question_ids,target_ids=backend.question_and_target(qa)
    if encoded['input_ids']!=question_ids+target_ids[:-1]:
        raise ValueError('Date tokenization differs from actual model targets')
    owners=[]
    for a,b in encoded['offset_mapping'][len(question_ids):]:
        a-=len(prefix);b-=len(prefix)
        overlaps=[max(0,min(b,y)-max(a,x)) for x,y in ranges]
        if sum(v>0 for v in overlaps)>1:
            raise ValueError('A token crosses multiple date components')
        owners.append(overlaps.index(max(overlaps)) if max(overlaps) else -1)
    owners.append(-1)
    if not {0,1,2} <= set(owners):raise ValueError('Missing date component token supervision')
    cache[key]=owners
    return owners


def reduce_losses(backend, qa, ce):
    import torch
    owners, spans = layout(backend, qa)
    if len(owners) != ce.numel():
        raise ValueError('Joint token loss/mask length mismatch')
    owners = torch.tensor(owners, device=ce.device)
    settings = backend.cfg['joint_loss']
    routed = [ce.sum()*0, ce.sum()*0]
    denominator = sum(options(settings, s['field'])['weight'] for s in spans if s['present'])

    def add(mask, opt, scale=1.):
        if not mask.any() or not opt['weight']:
            return
        values = ce[mask]
        if opt['type'] == 'focal':
            values = opt['alpha']*(-torch.expm1(-values)).clamp_min(torch.finfo(values.dtype).tiny).pow(opt['gamma'])*values
        index = int(opt['u_group'] == 'focal')
        routed[index] = routed[index]+values.mean()*opt['weight']*scale

    absent = owners < -1  # initially empty
    format_mask=owners == -1
    for i, span in enumerate(spans):
        if span['present']:
            opt=options(settings,span['field'])
            if span['field']=='birth_date' and opt.get('reduction')=='ymd_mean':
                components=torch.tensor(date_component_owners(backend,qa),device=ce.device)
                if ((components>=0)&(owners!=i)).any():
                    raise ValueError('Date component lies outside date field')
                for part in range(3):add(components==part,opt,1./(3*denominator) if denominator else 0.)
                format_mask |= (owners==i)&(components<0)
            else:
                add(owners == i,opt,1./denominator if denominator else 0.)
        else:
            absent |= owners == i
    add(absent, options(settings, 'missing'))
    add(format_mask, options(settings, 'format'))
    return torch.stack(routed)


def active_groups(qa, settings):
    names = [s['field'] for s in qa['joint_fields'] if s['present']]+['format']
    if any(not s['present'] for s in qa['joint_fields']):
        names.append('missing')
    return {options(settings, name)['u_group'] for name in names if options(settings, name)['weight'] > 0}


def score_rows(rows):
    """Parse each free generation once; never repair or coerce a prediction."""
    from collections import defaultdict
    import re
    per_field = defaultdict(list); expanded = [];date_parts=defaultdict(list)
    valid = schema = exact = missing = missing_wrong = hallucinated = truncated = 0
    for row in rows:
        gold = json.loads(row['gold']); duplicates = []
        def object_pairs(pairs):
            d = {}
            for k, v in pairs:
                if k in d: duplicates.append(k)
                d[k] = v
            return d
        try:
            def reject_constant(value):
                raise ValueError('Nonstandard JSON constant: '+value)
            pred = json.loads(row['prediction'], object_pairs_hook=object_pairs,
                              parse_constant=reject_constant)
            if not isinstance(pred, dict) or duplicates:
                raise ValueError('Not a unique-key JSON object')
            valid += 1
        except (ValueError, TypeError):
            pred = None
        good_schema = pred is not None and set(pred) == set(FIELDS) and all(v is None or isinstance(v, str) for v in pred.values())
        schema += good_schema
        exact += good_schema and pred == gold
        truncated += bool(row.get('truncated'))
        if isinstance(gold.get('birth_date'),str) and re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}',gold['birth_date']):
            actual=pred.get('birth_date') if pred is not None else None
            components=(actual.split('-') if isinstance(actual,str) and re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}',actual) else [None]*3)
            for part,value,expected in zip(('year','month','day'),components,gold['birth_date'].split('-')):
                date_parts[part].append(float(value==expected))
        for field in FIELDS:
            exists = pred is not None and field in pred
            correct = exists and type(pred[field]) is type(gold[field]) and pred[field] == gold[field]
            present = gold[field] is not None
            if present: per_field[field].append(float(correct))
            else:
                missing += 1; missing_wrong += not correct
                hallucinated += exists and pred[field] is not None
            expanded.append(dict(record_id=row['record_id'], field=field, gold=gold[field],
                prediction=pred.get(field) if pred is not None else None, present=present,
                prediction_key_present=exists, em=float(correct), json_valid=pred is not None))
    n = len(rows)
    stats = {k: dict(count=len(v), em=sum(v)/len(v)) for k, v in per_field.items()}
    return dict(records=n, questions=n, json_valid_rate=valid/n if n else None,
        schema_valid_rate=schema/n if n else None, record_em=exact/n if n else None,
        macro_field_em=sum(v['em'] for v in stats.values())/len(stats) if stats else None,
        per_field=stats, missing_fields=missing, missing_error_rate=missing_wrong/missing if missing else None,
        missing_false_fill_rate=hallucinated/missing if missing else None,
        truncated_rate=truncated/n if n else None,
        birth_date_component_em={k:dict(count=len(v),em=sum(v)/len(v)) for k,v in date_parts.items()}), expanded
