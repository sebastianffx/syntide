"""Load finished runs back from disk and re-check the run contract, whichever trainer produced them.

A run is accepted only if:
- ``trial.json`` is unedited and ``result.json`` belongs to it and is complete;
- every checkpoint still has the sha256 that ``finish`` recorded;
- its predictions cover every record of the split x the 5 classes for each checkpoint (validation: best and
  final; test: best), at the recorded steps;
- the predictions reproduce the score the run reported (``best_val_auroc``, or ``test/auroc_overall``).

Records are sorted by ``record_id``, so runs of one split line up row by row.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from syntide.data.schema import SUPERCLASSES
from syntide.eval.stats import macro_auroc
from syntide.trial import TrialError, TrialSpec, load_spec

PLAN_FILE = "plan.json"
RESULT_FILE = "result.json"
CLAIM_FILE = "test_claim.json"
PREDICTIONS = {"val": "predictions_val.parquet", "test": "predictions_test.parquet"}
CHECKPOINTS = {"val": ("best", "final"), "test": ("best",)}
TEST_SCORES_FILE = "test_scores.json"


class RunError(RuntimeError):
    """A run directory breaks the run contract: edited, incomplete, or its outputs do not agree."""


@dataclass(frozen=True)
class Run:
    """One finished trial's predictions on one split, plus what it recorded about itself."""

    run_dir: Path
    spec: TrialSpec
    result: dict[str, Any]
    split: str
    record_id: np.ndarray  # (n,) sorted
    subject_uid: np.ndarray  # (n,)
    labels: np.ndarray  # (n, 5) bool
    is_artifact: np.ndarray  # (n,) bool
    probs: dict[str, np.ndarray]  # checkpoint -> (n, 5)
    steps: dict[str, int]  # checkpoint -> optimizer step

    @property
    def key(self) -> tuple[str, str]:
        return (self.spec.regime, self.spec.arm)

    @property
    def seed(self) -> int:
        return self.spec.replicate_seed

    @property
    def fingerprint(self) -> str:
        return self.result["trainer"]["fingerprint"]

    @property
    def init_sha256(self) -> str:
        return self.result["trainer"]["init_sha256"]

    @property
    def exposure(self) -> dict[str, int]:
        return self.result["exposure"]


def load_run(run_dir: str | Path, split: str) -> Run:
    run_dir = Path(run_dir)
    if split not in PREDICTIONS:
        raise ValueError(f"unknown split {split!r}; choose one of {', '.join(PREDICTIONS)}")
    try:
        spec = load_spec(run_dir)
    except TrialError as error:
        raise RunError(str(error)) from error
    result_path = run_dir / RESULT_FILE
    if not result_path.exists():
        raise RunError(f"{run_dir} is not finished (no {RESULT_FILE})")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    identity = (result.get("trial_id"), result.get("regime"), result.get("arm"), result.get("replicate_seed"))
    if result.get("status") != "complete" or identity != (
        spec.trial_id,
        spec.regime,
        spec.arm,
        spec.replicate_seed,
    ):
        raise RunError(f"{result_path} does not hold the finished trial {spec.trial_id[:12]}")
    checkpoints = result["provenance"]["checkpoints"]
    for checkpoint in checkpoints.values():
        path = run_dir / checkpoint["path"]
        if not path.is_file() or _sha256(path) != checkpoint["sha256"]:
            raise RunError(f"{path} is missing or changed after training; its scores no longer stand")

    path = run_dir / PREDICTIONS[split]
    if not path.exists():
        raise RunError(f"{run_dir} has no {PREDICTIONS[split]}")
    parquet = pq.read_table(path)
    table = {name: parquet.column(name).to_numpy(zero_copy_only=False) for name in parquet.column_names}
    unknown = sorted(set(np.unique(table["checkpoint"])) - set(CHECKPOINTS[split]))
    if unknown:
        raise RunError(f"{path} holds checkpoints {unknown}; a {split} file holds only {CHECKPOINTS[split]}")
    for column, expected in (
        ("trial_id", spec.trial_id),
        ("arm", spec.arm),
        ("regime", spec.regime),
        ("seed", spec.replicate_seed),
    ):
        if not (table[column] == expected).all():
            raise RunError(f"{path}: column {column} is not {expected!r} throughout")

    probs, steps, records = {}, {}, None
    for name in CHECKPOINTS[split]:
        rows = table["checkpoint"] == name
        if not rows.any():
            raise RunError(f"{path} has no rows for the {name} checkpoint")
        block = _block(path, name, {key: values[rows] for key, values in table.items()})
        if block["step"] != checkpoints[name]["step"]:
            raise RunError(
                f"{path}: the {name} rows are at step {block['step']}, not {checkpoints[name]['step']}"
            )
        if records is None:
            records = block
        elif not all(np.array_equal(block[k], records[k]) for k in ("record_id", "labels", "is_artifact")):
            raise RunError(f"{path}: the checkpoints were scored on different records")
        probs[name], steps[name] = block["probs"], block["step"]
    assert records is not None  # CHECKPOINTS[split] is never empty

    run = Run(
        run_dir=run_dir,
        spec=spec,
        result=result,
        split=split,
        record_id=records["record_id"],
        subject_uid=records["subject_uid"],
        labels=records["labels"],
        is_artifact=records["is_artifact"],
        probs=probs,
        steps=steps,
    )
    _check_reported_score(run)
    return run


def load_runs(protocol_dir: str | Path, split: str) -> tuple[list[Run], list[str]]:
    """Every planned run finished with outputs for ``split``, and the planned run dirs that are not.

    For the test split, every loaded run must also be in ``test_claim.json`` with its claimed checkpoint.
    """
    protocol_dir = Path(protocol_dir)
    plan_path = protocol_dir / PLAN_FILE
    if not plan_path.exists():
        raise RunError(f"{plan_path} does not exist; plan the protocol first")
    claimed = None
    if split == "test":
        claim_path = protocol_dir / CLAIM_FILE
        if not claim_path.exists():
            raise RunError(f"{claim_path} does not exist; score the test fold first")
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        claimed = {trial["run_dir"]: trial for trial in claim["trials"]}
    runs, pending = [], []
    for entry in json.loads(plan_path.read_text(encoding="utf-8"))["trials"]:
        run_dir = protocol_dir / entry["run_dir"]
        ready = (run_dir / RESULT_FILE).exists() and (split == "val" or (run_dir / TEST_SCORES_FILE).exists())
        if not ready:
            pending.append(entry["run_dir"])
            continue
        run = load_run(run_dir, split)
        if run.spec.trial_id != entry["trial_id"]:
            raise RunError(
                f"{run_dir} holds trial {run.spec.trial_id[:12]}, but the plan names {entry['trial_id'][:12]}"
            )
        if claimed is not None:
            trial = claimed.get(entry["run_dir"])
            best = run.result["provenance"]["checkpoints"]["best"]["sha256"]
            if trial is None or trial["trial_id"] != run.spec.trial_id or trial["best_sha256"] != best:
                raise RunError(f"{run_dir} is not the run test_claim.json scored")
        runs.append(run)
    return runs, pending


def _block(path: Path, name: str, rows: dict[str, np.ndarray]) -> dict[str, Any]:
    """One checkpoint's rows -> per-record arrays sorted by record_id (rows are record-major)."""
    k = len(SUPERCLASSES)
    if len(rows["record_id"]) % k:
        raise RunError(f"{path}: the {name} rows are not a whole number of records x {k} classes")
    n = len(rows["record_id"]) // k
    shaped = {key: values.reshape(n, k) for key, values in rows.items()}
    if not (shaped["class_id"] == np.asarray(SUPERCLASSES)).all():
        raise RunError(f"{path}: the {name} rows do not list the classes {SUPERCLASSES} per record")
    for key in ("record_id", "subject_uid", "is_artifact", "step"):
        if not (shaped[key] == shaped[key][:, :1]).all():
            raise RunError(f"{path}: {key} changes within a record's {k} rows")
    record_id = shaped["record_id"][:, 0].astype(str)
    if len(np.unique(record_id)) != n:
        raise RunError(f"{path}: the {name} rows list a record twice")
    probs = shaped["y_prob"].astype(np.float64)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise RunError(f"{path}: the {name} probabilities are not all in [0, 1]")
    order = np.argsort(record_id, kind="stable")
    steps = np.unique(shaped["step"])
    return {
        "record_id": record_id[order],
        "subject_uid": shaped["subject_uid"][:, 0].astype(str)[order],
        "labels": shaped["y_true"].astype(bool)[order],
        "is_artifact": shaped["is_artifact"][:, 0].astype(bool)[order],
        "probs": probs[order],
        "step": int(steps[0]),
    }


def _check_reported_score(run: Run) -> None:
    """The predictions must give back the score the run reported for its best checkpoint."""
    if run.split == "val":
        reported, what = run.result["best_val_auroc"], "result.json best_val_auroc"
    else:
        scores = json.loads((run.run_dir / TEST_SCORES_FILE).read_text(encoding="utf-8"))
        if scores["trial_id"] != run.spec.trial_id:
            raise RunError(f"{run.run_dir / TEST_SCORES_FILE} belongs to another trial")
        reported, what = scores["test/auroc_overall"], f"{TEST_SCORES_FILE} test/auroc_overall"
    recomputed = macro_auroc(run.labels, run.probs["best"])
    if not math.isclose(recomputed, reported, rel_tol=1e-9, abs_tol=1e-12):
        raise RunError(
            f"{run.run_dir}: the predictions give AUROC {recomputed:.6f}; {what} is {reported:.6f}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
