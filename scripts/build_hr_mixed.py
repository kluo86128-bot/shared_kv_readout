"""Derive switchable fieldwise/joint/mixed tasks from normalized HR v5 records."""
from pathlib import Path
import copy
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kv_cache_readout.joint import question
from kv_cache_readout.readout import field_questions


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    source=ROOT/'data/dataset_hr_v5_joint';dest=ROOT/'data/dataset_hr_v6_mixed'
    if dest.exists():raise FileExistsError(dest)
    manifest=dict(dataset_id='hr_v6_mixed_id324',schema_version='multiscene_privacy_v1',
        prepared_date='2026-11-08',parent_dataset_directory='data/dataset_hr_v5_joint',
        parent_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
        split_sizes={},files={},parent_files={},statistics={},
        modes=['fieldwise','joint','mixed'],default_evaluation='joint',
        fields_values_and_prefixes_unchanged=True)
    for split in ('train','validation','test'):
        original=source/f'{split}.jsonl';records=[];field_tasks=0
        for line in original.open(encoding='utf-8'):
            record=json.loads(line)
            record.update(dataset_id=manifest['dataset_id'],fieldwise_qa=field_questions(record),
                          joint_qa=[question(record['fields'])])
            record['training_qa']=copy.deepcopy(record['fieldwise_qa']+record['joint_qa'])
            record['evaluation_qa']=copy.deepcopy(record['joint_qa'])
            record['qa']=copy.deepcopy(record['training_qa'])
            field_tasks+=len(record['fieldwise_qa']);records.append(record)
        target=dest/f'{split}.jsonl';target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
        manifest['files'][target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
        manifest['parent_files'][original.name]=hashlib.sha256(original.read_bytes()).hexdigest()
        manifest['split_sizes'][split]=len(records)
        manifest['statistics'][split]=dict(samples=len(records),fieldwise_tasks=field_tasks,
            joint_tasks=len(records),mixed_tasks=field_tasks+len(records),evaluation_tasks=len(records))
    write(dest/'manifest.json',manifest)
    write(dest/'public_text_128.json',json.loads((source/'public_text_128.json').read_text(encoding='utf-8')))
    for suffix in ('','_server2'):
        cfg=json.loads((ROOT/f'configs/train_hr_joint_c128{suffix}.json').read_text(encoding='utf-8'))
        cfg.update(dataset_directory='data/dataset_hr_v6_mixed',
            output_directory='runs/hr_mixed_c128_'+('s2' if suffix else 's1'),
            readout=dict(train_mode='mixed',eval_mode='joint',fieldwise_weight=.5,joint_weight=.5))
        cfg['training'].update(batch_records=8,sample_batch_size=2,questions_per_sample_batch=1)
        write(ROOT/f'configs/train_hr_mixed_c128{suffix}.json',cfg)
    print(json.dumps(manifest['statistics'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
