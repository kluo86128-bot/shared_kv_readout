"""Build source-backed 3-2-4 IDs and one joint JSON question per HR sample."""
from pathlib import Path
import argparse
import ast
import copy
import hashlib
import json
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kv_cache_readout.joint import question, default_settings


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def normalized(value):
    if re.fullmatch(r'[\d\s.\-/]+',value):
        digits=re.sub(r'\D','',value)
        if len(digits)==9:
            return digits[:3]+'-'+digits[3:5]+'-'+digits[5:]
    return None


def replace_identity(record,value,donor):
    old=record['fields']['identity_number']; text=record['private_prefix']
    result=copy.deepcopy(record)
    edits=[m.span() for m in re.finditer(re.escape(old),text)]
    assert edits
    for field,a in record['field_annotations'].items():
        if field=='identity_number':continue
        s,e=a['evidence_span']
        assert not any(max(s,x)<min(e,y) for x,y in edits),(record['record_id'],field)
    def offset(p):
        delta=0
        for a,b in edits:
            if p<=a:break
            if p<b:return a+delta+min(p-a,len(value))
            delta+=len(value)-(b-a)
        return p+delta
    # Exact literal replacement also updates repeated mentions of the same ID.
    new=text.replace(old,value)
    result['private_prefix']=new
    result['fields']['identity_number']=value
    for field,a in result['field_annotations'].items():
        s,e=record['field_annotations'][field]['evidence_span']
        a['evidence_span']=[offset(s),offset(e)]
        if field=='identity_number':
            a['value']=value
            # End of a replaced span maps to the new end even with length changes.
            a['evidence_span']=[offset(s),offset(s)+len(value)]
            a['normalization']=dict(method='source_ssn_replacement' if donor else 'nine_digit_separator_normalization',
                parent_source_ref=record['field_annotations'][field]['source_ref'],
                parent_value_sha256=sha(old.encode()))
            if donor:
                a.update(source_ref=donor['source_ref'],source_evidence=value,
                         origin='source_backed_synthetic_replacement',derived_serialization=False)
                a['normalization'].update(donor)
            else:
                a['derived_serialization']=old!=value
        a,b=a['evidence_span'];assert new[a:b]==result['fields'][field],(record['record_id'],field)
    for block in result.get('augmentation_blocks',[]):
        block['span']=[offset(p) for p in block['span']]
    end=offset(record['native_text_length'])
    result.update(native_text_length=end,native_text_sha256=sha(new[:end].encode()),
                  private_prefix_sha256=sha(new.encode()),dataset_id='hr_v5_joint_json_id324')
    result['related_source_keys']=sorted({a['source_ref'] for a in result['field_annotations'].values()})
    result['native_target_count']=sum(a['origin']=='native' for a in result['field_annotations'].values())
    result['joint_derivation']=dict(parent_prefix_sha256=sha(text.encode()),
        identity_action='source_replacement' if donor else ('unchanged' if old==value else 'format_only'))
    qa=question(result['fields'])
    for name in ('qa','training_qa','evaluation_qa'):result[name]=[copy.deepcopy(qa)]
    return result


def main():
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-directory',required=True)
    args=p.parse_args()
    raw=Path(args.raw_directory).resolve()
    dest=ROOT/'data/dataset_hr_v5_joint'
    if dest.exists():raise FileExistsError(dest)
    parent=ROOT/'data/dataset_hr_v4_reduced'
    source={s:[json.loads(line) for line in (parent/f'{s}.jsonl').open(encoding='utf-8')]
            for s in ('train','validation','test')}
    refs={ref for rows in source.values() for r in rows for ref in r['related_source_keys']}
    used={re.sub(r'\D','',r['fields']['identity_number']) for rows in source.values() for r in rows}
    tok=Tokenizer.from_file(str(ROOT/'data/tokenizer/Qwen3-4B/tokenizer.json'))
    manifest=dict(dataset_id='hr_v5_joint_json_id324',schema_version='multiscene_privacy_v1',
        prepared_date='2026-11-08',scene='hr',split_sizes={},files={},parent_files={},
        parent_dataset_directory='data/dataset_hr_v4_reduced',parent_dataset_id='hr_v4_reduced_position',
        source_urls={'Gretel-PII':'https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1/tree/e06eb1499ca8d54470f085021cd8e54f9efac7fd'},
        donor_files={}, donor_policy='Unique literal ssn entities, selected from matching original source split; exclude every HR parent source UID and every existing identity numeric value.',
        joint_prompt_version='hr_joint_json_v1',seed=17)
    report={};new_numbers={}
    pinned=json.loads((ROOT/'data/dataset_multiscene_v3_hr_balanced/manifest.json').read_text(encoding='utf-8'))
    expected_raw={item['name']:item['sha256'] for item in pinned['source_files']}
    for split,rows in source.items():
        raw_file=raw/f'native_gretel_pii_{split}.parquet'
        raw_sha=sha(raw_file.read_bytes())
        if raw_sha != expected_raw.get(raw_file.name):
            raise ValueError('Raw donor file differs from pinned original source: '+raw_file.name)
        donors={}
        for batch in pq.ParquetFile(raw_file).iter_batches(columns=['uid','entities','text']):
            for item in batch.to_pylist():
                ref='Gretel-PII:'+item['uid']
                if ref in refs:continue
                for entity in ast.literal_eval(item['entities']):
                    v=entity['entity']
                    if 'ssn' not in entity['types'] or not re.fullmatch(r'\d{3}-\d{2}-\d{4}',v):continue
                    if v not in item['text'] or v.replace('-','') in used:continue
                    donors.setdefault(v,dict(source_ref=ref,source_split=split,
                        source_document_sha256=sha(item['text'].encode()),source_file=raw_file.name))
        candidates=iter(sorted(donors,key=lambda v:sha(('17|'+split+'|'+v).encode())))
        counts=dict(unchanged=0,format_only=0,source_replacement=0);records=[];sizes=[];new_numbers[split]=set()
        for row in rows:
            old=row['fields']['identity_number'];v=normalized(old);donor=None
            if v is None:
                v=next(candidates)
                while v.replace('-','') in used:v=next(candidates)
                donor=donors[v];used.add(v.replace('-',''));refs.add(donor['source_ref'])
                new_numbers[split].add(v)
            result=replace_identity(row,v,donor)
            counts[result['joint_derivation']['identity_action']]+=1
            records.append(result)
            sizes.append(len(tok.encode(result['qa'][0]['answer'],add_special_tokens=False).ids))
        path=dest/f'{split}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
        manifest['files'][path.name]=sha(path.read_bytes());manifest['split_sizes'][split]=len(records)
        manifest['parent_files'][path.name]=sha((parent/path.name).read_bytes())
        manifest['donor_files'][raw_file.name]=raw_sha
        report[split]=dict(samples=len(records),identity_actions=counts,
            answer_tokens=dict(mean=statistics.mean(sizes),max=max(sizes)))
    assert not (new_numbers['train']&new_numbers['test'] or new_numbers['train']&new_numbers['validation'] or new_numbers['test']&new_numbers['validation'])
    manifest['statistics']=report
    write(dest/'manifest.json',manifest)
    write(dest/'public_text_128.json',json.loads((parent/'public_text_128.json').read_text(encoding='utf-8')))
    for suffix in ('','_server2'):
        cfg=json.loads((ROOT/f'configs/train_hr_v4_c128{suffix}.json').read_text(encoding='utf-8'))
        cfg.update(dataset_directory='data/dataset_hr_v5_joint',joint_loss=default_settings(),
                   output_directory='runs/hr_joint_c128_'+('s2' if suffix else 's1'))
        cfg['training'].update(batch_records=4,sample_batch_size=1,questions_per_sample_batch=1)
        cfg['generation']['max_new_tokens']=256
        write(ROOT/f'configs/train_hr_joint_c128{suffix}.json',cfg)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
