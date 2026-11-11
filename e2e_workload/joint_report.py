"""Joint JSON metrics grounded only in verified, unique direct Writer sources."""
from .source_csv import source_rows,_atomic_csv,_atomic_json
from kv_cache_readout.joint import score_rows


def write_joint_report(root,rows,registry):
    selected=[];source_ids={};total=0
    for row in rows:
        if row['reader']['field']!='all_fields':continue
        total+=1
        base,_=source_rows(row,registry)
        if base['source_status']!='writer_source':continue
        query=row['reader']['id'];source_ids[query]=base['record_id']
        selected.append(dict(record_id=query,gold=base['gold'],prediction=base['answer'],
            truncated=row['response'].get('finish_reason')=='length'))
    if not total:return None
    summary,fields=score_rows(selected)
    for row in fields:
        row['query_id']=row['record_id'];row['record_id']=source_ids[row['query_id']]
    summary.update(total_joint_queries=total,verified_unique_writer_queries=len(selected),
        excluded_queries=total-len(selected),gold_policy='actual_verified_unique_direct_writer_only')
    _atomic_csv(root/'joint_field_predictions.csv',fields)
    _atomic_json(root/'joint_summary.json',summary)
    return summary
