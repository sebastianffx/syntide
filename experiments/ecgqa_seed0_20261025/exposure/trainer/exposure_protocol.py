"""CPU-testable contract for the fixed original-QA-exposure ablation.

The original stream deliberately repeats the archived trainer's NumPy
permutation-and-wrap algorithm. Generated sampling never advances that RNG.
"""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager

import numpy as np

ORIGINAL_ROWS = 82829
GENERATED_ROWS = 11672
ORIGINAL_BATCH = 8
ORIGINAL_ACCUMULATION = 2
GENERATED_BATCH = 2
PRIMARY_STEP = 12000
LR_SCHEDULE_STEPS = 15531
WARMUP_STEPS = math.ceil(LR_SCHEDULE_STEPS * 0.1)
BASELINE_TRAIN_SHA256 = "e150bb0ec1564a8aa7aa7506c62e22b79ad243ea63950c7b8a03756fcd74219a"
BASELINE_TRAINER_SHA256 = "16bf861ea252e68dd2adc452bab6abfa0d37e29103337c2cab0d94bd3c075128"
BASE_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
CHECKPOINT_POLICY = "fixed_original_exposure"


def task_ids_sha256(rows):
    """Match the archived evaluator's ordered task-ID identity convention."""
    return hashlib.sha256(json.dumps([row["task_id"] for row in rows]).encode()).hexdigest()


def validate_sources(rows, manifest):
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported source manifest schema")
    if manifest.get("original_train_parquet_sha256") != BASELINE_TRAIN_SHA256:
        raise ValueError("Original baseline parquet identity differs")
    if len(rows) != ORIGINAL_ROWS + GENERATED_ROWS:
        raise ValueError("Expected precisely baseline original plus frozen generated rows")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("Training task IDs must be unique across both sources")
    sources = {}
    for name, start, count in (("original", 0, ORIGINAL_ROWS),
                               ("generated", ORIGINAL_ROWS, GENERATED_ROWS)):
        spec = manifest[name]
        if spec.get("start") != start or spec.get("count") != count:
            raise ValueError(f"Wrong {name} source boundary")
        source = rows[start:start + count]
        if spec.get("task_ids_sha256") != task_ids_sha256(source):
            raise ValueError(f"Changed {name} source row order/task IDs")
        sources[name] = source
    return sources


def validate_train_arguments(args):
    required = {
        "seed": 0, "max_steps": PRIMARY_STEP, "batch_size": ORIGINAL_BATCH,
        "gradient_accumulation": ORIGINAL_ACCUMULATION, "val_every": 1000,
        "eval_batch_size": 256, "learning_rate": 0.0002, "max_length": 1536,
        "max_new_tokens": 384, "base_revision": BASE_REVISION,
        "selection_method": "option-nll", "selection_limit": 0,
    }
    for name, expected in required.items():
        if getattr(args, name) != expected:
            raise ValueError(f"Frozen exposure protocol requires {name}={expected!r}")


def validate_baseline(config, baseline):
    if baseline.get("trainer_source_sha256") != BASELINE_TRAINER_SHA256:
        raise ValueError("Baseline trainer fingerprint differs")
    if baseline.get("max_steps") != LR_SCHEDULE_STEPS:
        raise ValueError("Baseline LR horizon differs")
    if baseline.get("data_files", {}).get("train.parquet") != BASELINE_TRAIN_SHA256:
        raise ValueError("Baseline source corpus differs")
    fields = (
        "base_revision", "batch_size", "gradient_accumulation", "eval_batch_size",
        "gradient_clip", "init_sha256", "learning_rate", "max_length", "max_new_tokens",
        "metrics_source_sha256", "model", "opentslm_commit", "packages", "precision",
        "prompt_version", "seed", "selection_limit", "selection_method",
        "selection_questions", "total_parameters", "trainable_parameters", "val_every",
        "warmup_fraction", "weight_decay",
    )
    for name in fields:
        if config.get(name) != baseline.get(name):
            raise ValueError(f"Ablation differs from baseline at {name}")
    if config["data_files"]["selection.parquet"] != baseline["data_files"]["selection.parquet"]:
        raise ValueError("Selection question order differs from baseline")


def lr_scale(step):
    if not 1 <= step <= PRIMARY_STEP:
        raise ValueError("Step outside the frozen original-exposure schedule")
    return (step / WARMUP_STEPS if step <= WARMUP_STEPS
            else (LR_SCHEDULE_STEPS - step) / (LR_SCHEDULE_STEPS - WARMUP_STEPS))


class PermutationStream:
    """One independent, no-drop epoch stream with an auditable draw prefix."""
    def __init__(self, rows, seed=0):
        if not rows:
            raise ValueError("Cannot sample an empty source")
        self.rows = rows
        self.generator = np.random.default_rng(seed)
        self.order = self.generator.permutation(len(rows))
        self.offset = 0
        self.draws = 0
        self.completed_epochs = 0
        self.digest = hashlib.sha256()
        self.first_task_ids = []

    def take(self, count):
        if type(count) is not int or count < 1:
            raise ValueError("Draw count must be a positive integer")
        indices = []
        while len(indices) < count:
            available = min(count - len(indices), len(self.order) - self.offset)
            indices.extend(self.order[self.offset:self.offset + available].tolist())
            self.offset += available
            if self.offset == len(self.order):
                self.order, self.offset = self.generator.permutation(len(self.order)), 0
                self.completed_epochs += 1
        rows = [self.rows[i] for i in indices]
        for row in rows:
            # JSON string framing makes ID boundaries unambiguous.
            self.digest.update((json.dumps(row["task_id"]) + "\n").encode())
            if len(self.first_task_ids) < 32:
                self.first_task_ids.append(row["task_id"])
        self.draws += len(rows)
        return rows

    def audit(self):
        return {
            "draws": self.draws, "completed_epochs": self.completed_epochs,
            "epoch_offset": self.offset,
            "draw_task_ids_sha256": self.digest.hexdigest(),
            "draw_hash_format": "UTF-8 json.dumps(task_id) plus LF, concatenated in draw order",
            "first_32_task_ids": self.first_task_ids,
        }


def weighted_microbatch_loss(losses, mask, microbatch_rows, total_rows=18):
    """Preserve archived token normalization, then weight by microbatch rows.

    This is not a claim that variable-length answers receive equal token weight.
    The original baseline itself averaged two token-normalized batch losses.
    """
    if microbatch_rows <= 0 or total_rows < microbatch_rows:
        raise ValueError("Invalid microbatch/source weight")
    return losses.sum() / mask.sum() * (microbatch_rows / total_rows)


@contextmanager
def generated_forward_rng(torch_module, step, devices):
    """Synthetic dropout draws cannot consume the original CPU/CUDA RNG stream."""
    with torch_module.random.fork_rng(devices=devices):
        torch_module.manual_seed(1_000_000 + step)
        yield


def validate_primary_checkpoint(config, completed, payload, history):
    if config.get("checkpoint_policy") != CHECKPOINT_POLICY:
        raise ValueError("Checkpoint policy is not fixed original exposure")
    if config.get("max_steps") != PRIMARY_STEP or completed.get("max_steps") != PRIMARY_STEP:
        raise ValueError("Wrong completed optimizer-update budget")
    if config.get("learning_rate_schedule_steps") != LR_SCHEDULE_STEPS:
        raise ValueError("Changed baseline LR chronology")
    if config.get("warmup_steps") != WARMUP_STEPS:
        raise ValueError("Changed baseline warmup")
    if completed.get("primary_step") != PRIMARY_STEP or payload.get("step") != PRIMARY_STEP:
        raise ValueError("Early selection peak cannot replace the exposure-matched checkpoint")
    if [row["step"] for row in history] != list(range(1000, PRIMARY_STEP + 1, 1000)):
        raise ValueError("Diagnostic selection history is incomplete or reordered")
    if payload.get("selection_accuracy") != history[-1]["accuracy"]:
        raise ValueError("Primary checkpoint differs from its diagnostic selection event")
    expected = {"original": PRIMARY_STEP * 16, "generated": PRIMARY_STEP * GENERATED_BATCH}
    for source, draws in expected.items():
        if completed.get("source_exposure", {}).get(source, {}).get("draws") != draws:
            raise ValueError(f"Wrong actual {source} exposure at primary checkpoint")
