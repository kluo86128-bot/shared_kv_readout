from __future__ import annotations

import json
from pathlib import Path

from .config import file_hash, fingerprint, resolve, ROOT


def _dataset_root(cfg: dict) -> Path:
    root = resolve(cfg["dataset_directory"])
    if not root.is_dir():
        raise ValueError(f"Dataset directory is missing: {root}")
    return root


def dataset_identity(cfg: dict) -> dict:
    root = _dataset_root(cfg)
    names = ["manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl"]
    files = {}
    for name in names:
        filename = root / name
        if not filename.is_file():
            raise ValueError(f"Frozen dataset file is missing: {filename}")
        files[name] = {"bytes": filename.stat().st_size, "sha256": file_hash(filename)}
    result = {"directory": cfg["dataset_directory"], "files": files,
              "identity_sha256": fingerprint(files)}
    result['frozen_tokenizer'] = {p.name: file_hash(p) for p in sorted((ROOT/'data/tokenizer/Qwen3-4B').glob('*.json'))}
    if "scoped_cache" in cfg:
        from .suffix import suffix_identity
        result["suffix"] = suffix_identity(cfg)
    return result


def questions_for(record: dict, *, training: bool, cfg=None) -> list[dict]:
    key = "training_qa" if training else "evaluation_qa"
    questions = record.get(key) or record.get("qa") or []
    if cfg is not None and 'readout' in cfg:
        from .readout import select_questions
        questions=select_questions(record,cfg,training)
    result = []
    for qa in questions:
        if not all(name in qa for name in ("field", "question", "answer")):
            raise ValueError(f"Malformed QA in {record.get('record_id')}")
        result.append({k: qa[k] for k in qa if k in {
            "field", "group", "question", "answer", "template_id",
            "empty_target", "prediction_kind", "scoring_target", "joint_fields",
        }})
    if not result:
        raise ValueError(f"Record has no usable questions: {record.get('record_id')}")
    return result


def load_split(cfg: dict, split: str, *, training: bool | None = None) -> list[dict]:
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unsupported split: {split}")
    path = _dataset_root(cfg) / f"{split}.jsonl"
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("scene") != cfg["scene"]:
                continue
            record = dict(record)
            mode = split == "train" if training is None else training
            record["qa"] = questions_for(record, training=mode, cfg=cfg)
            joint = [q.get('prediction_kind') == 'joint_json' for q in record['qa']]
            if 'readout' in cfg:
                pass  # select_questions validates the requested task families.
            elif 'joint_loss' in cfg:
                if len(joint) != 1 or not all(joint):
                    raise ValueError('Joint mode requires exactly one joint_json QA per record')
            elif any(joint):
                raise ValueError('Joint dataset requires joint_loss configuration')
            records.append(record)
    if not records:
        raise ValueError(f"No {cfg['scene']} records in {path}")
    ids = [row["record_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate record IDs in {split}")
    return records

