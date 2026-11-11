from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "native_cache_readout_v1"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def fingerprint(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(value: str | Path) -> str:
    result = hashlib.sha256()
    with Path(value).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_config(value: str | Path) -> dict:
    cfg = json.loads(resolve(value).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    required = {
        "schema_version", "model_name_or_path", "device", "dtype",
        "dataset_directory", "output_directory", "scene", "seed",
        "public_text", "delimiter", "question_format", "prompt_length",
        "initialization_words", "prompt_partition", "focal_fields", "loss",
        "scoped_cache", "native_contract", "training", "generation",
    }
    if set(cfg) - {'execution', 'joint_loss', 'readout', 'c1_selection'} != required:
        missing, extra = required - set(cfg), set(cfg) - required
        raise ValueError(f"Invalid config keys; missing={sorted(missing)}, extra={sorted(extra)}")
    if 'c1_selection' in cfg:
        from .c1_selection import validate
        validate(cfg['c1_selection'])
        if cfg['c1_selection']['method']=='none' and cfg['native_contract']['recompute_ratio']!=0:
            raise ValueError('none selection requires recompute_ratio=0')
    if 'joint_loss' in cfg:
        from .joint import validate_settings
        validate_settings(cfg['joint_loss'])
    if 'readout' in cfg:
        from .readout import validate_readout
        validate_readout(cfg['readout'])
        if 'joint_loss' not in cfg:
            raise ValueError('Explicit readout modes require joint_loss field settings')
    if 'execution' in cfg:
        from .native_model import EXECUTION,FAST_EXECUTION
        if cfg['execution'] not in (EXECUTION,FAST_EXECUTION):
            raise ValueError('Unsupported native execution profile')
    if cfg["schema_version"] != SCHEMA:
        raise ValueError(f"schema_version must be {SCHEMA}")
    if cfg["scene"] != "hr":
        raise ValueError("The refactored v1 experiment is intentionally scoped to HR")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(cfg["prompt_length"]) is not int or cfg["prompt_length"] < 1:
        raise ValueError("prompt_length must be positive")
    if not cfg["public_text"] or not cfg["delimiter"]:
        raise ValueError("public_text and delimiter must be nonempty")
    if cfg["question_format"] not in ("chat", "plain"):
        raise ValueError("question_format must be chat or plain")
    if not cfg["initialization_words"]:
        raise ValueError("initialization_words must be nonempty")
    partition = cfg["prompt_partition"]
    if set(partition) != {"other_length", "focal_length", "keep_length"}:
        raise ValueError("prompt_partition requires other/focal/keep lengths")
    if any(type(value) is not int or value < 1 for value in partition.values()):
        raise ValueError("All prompt partitions must be positive")
    if sum(partition.values()) != cfg["prompt_length"]:
        raise ValueError("Prompt partitions must sum to prompt_length")
    if not cfg["focal_fields"] or cfg["loss"] != {"type": "field_focal_split", "gamma": 2.0, "alpha": 1.0}:
        raise ValueError("v1 preserves the field_focal_split objective")
    scoped = cfg["scoped_cache"]
    expected_scoped = {"suffix_directory", "suffix_max_tokens", "preservation_weight", "epsilon",
                       "suffix_recomputation"}
    if set(scoped) != expected_scoped or not scoped["suffix_recomputation"]:
        raise ValueError("scoped_cache must preserve the latest suffix-recomputation objective")
    if scoped["suffix_max_tokens"] != 128 or scoped["preservation_weight"] != 0.1:
        raise ValueError("v1 preserves suffix_max_tokens=128 and preservation_weight=0.1")

    contract = cfg["native_contract"]
    contract.setdefault('cached_key_policy', 'relocate_source_to_destination')
    contract.setdefault('pre_check_cache_policy', 'retain_relocated_writer')
    expected_contract_keys = {
        "vllm_version", "lmcache_version", "c1_tokens", "check_layer",
        "recompute_ratio", "position_policy", "topk_policy",
        "attention_policy", "writer_layout", "reader_layout",
        'cached_key_policy', 'pre_check_cache_policy',
    }
    if set(contract) != expected_contract_keys:
        raise ValueError("native_contract has missing or unexpected keys")
    if contract["vllm_version"] != "0.18.0" or contract["lmcache_version"] != "0.4.2":
        raise ValueError("v1 is pinned to vLLM 0.18.0 and LMCache 0.4.2")
    if (type(contract["c1_tokens"]) is not int or contract["c1_tokens"] not in (56, 128)
            or type(contract["check_layer"]) is not int
            or contract["check_layer"] < 0):
        raise ValueError("Requires C1=56 or 128 and a nonnegative check_layer")
    if not isinstance(contract["recompute_ratio"], (int,float)) or not math.isfinite(contract["recompute_ratio"]) or not 0 <= contract["recompute_ratio"] <= 1:
        raise ValueError("recompute_ratio must be finite and in [0,1]")
    fixed = {
        "position_policy": "reader_destination_zero",
        "topk_policy": "floor_min_one_fixed_after_check",
        "attention_policy": "flash_bottom_right_causal",
        'cached_key_policy': 'relocate_source_to_destination',
        'pre_check_cache_policy': 'retain_relocated_writer',
        "writer_layout": "P+delimiter+U_other32+U_focal16+delimiter+C1+delimiter+U_keep16+S+question",
        "reader_layout": "C1+delimiter+question",
    }
    for key, expected in fixed.items():
        if contract[key] != expected:
            raise ValueError(f"native_contract.{key} must be {expected}")

    training = cfg["training"]
    if 'sample_batch_size' in training:
        b = training['sample_batch_size']; q = training['questions_per_sample_batch']
        if type(b) is not int or b < 1 or type(q) is not int or q < 1:
            raise ValueError('Both batch limits must be positive integers')
        training.update(source_batch_records=b, preservation_batch_records=1,
                        question_batch_size=b*q, c1_batch_size=b, question_order='per_sample')
        from .native_model import EXECUTION
        if cfg.get('execution') == EXECUTION and (b != 1 or q != 1):
            raise ValueError('native-reference requires sample_batch_size=questions_per_sample_batch=1')
    training.setdefault('source_batch_records', 2)
    training.setdefault('preservation_batch_records', 1)
    training.setdefault('question_batch_size', 4)
    training.setdefault('checkpoint_questions', True)
    expected_training = {
        "epochs", "batch_records", "questions_per_record", "learning_rate",
        "weight_decay", "grad_clip", "warmup_fraction", "checkpoint_every_steps", "max_steps",
        'source_batch_records', 'preservation_batch_records', 'question_batch_size', 'checkpoint_questions',
    }
    if 'sample_batch_size' in training:
        expected_training.update(('sample_batch_size','questions_per_sample_batch'))
    # Opt-in only: do not change the identity or behavior of existing runs.
    if 'c1_batch_size' in training:
        from .native_model import FAST_EXECUTION, EXECUTION
        if (cfg.get('execution') not in (FAST_EXECUTION, EXECUTION)
                or type(training['c1_batch_size']) is not int or training['c1_batch_size'] < 1):
            raise ValueError('c1_batch_size requires the fast native profile and a positive integer')
        expected_training.add('c1_batch_size')
    if 'question_order' in training:
        from .native_model import FAST_EXECUTION, EXECUTION
        if (cfg.get('execution') not in (FAST_EXECUTION, EXECUTION)
                or training['question_order'] not in ('record_major', 'round_robin', 'per_sample')):
            raise ValueError('question_order requires the fast profile and record_major/round_robin')
        expected_training.add('question_order')
    if set(training) != expected_training:
        raise ValueError("training has missing or unexpected keys")
    for key in ("epochs", "batch_records", "questions_per_record", "checkpoint_every_steps"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ('source_batch_records', 'preservation_batch_records', 'question_batch_size'):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f'training.{key} must be positive')
    if type(training['checkpoint_questions']) is not bool:
        raise ValueError('checkpoint_questions must be boolean')
    if training["max_steps"] is not None and (
        type(training["max_steps"]) is not int or training["max_steps"] < 1
    ):
        raise ValueError("training.max_steps must be null or positive")
    for key in ("learning_rate", "weight_decay", "grad_clip", "warmup_fraction"):
        if not isinstance(training[key], (int, float)) or not math.isfinite(training[key]):
            raise ValueError(f"training.{key} must be finite")
    if not 0 <= training["warmup_fraction"] < 1:
        raise ValueError("training.warmup_fraction must be in [0, 1)")
    if training['learning_rate'] <= 0 or training['weight_decay'] < 0 or training['grad_clip'] <= 0:
        raise ValueError('learning_rate/grad_clip must be positive and weight_decay nonnegative')

    generation = cfg["generation"]
    if (set(generation) != {"max_new_tokens"} or type(generation["max_new_tokens"]) is not int
            or generation['max_new_tokens'] < 1):
        raise ValueError("generation requires integer max_new_tokens")

