"""Evaluate the fixed set of completed trials on test data once.

Trainers never see the test split. SYNTIDE loads it here, after training is
over:

1. **Freeze.** Every trial in ``plan.json`` must be finished, trained by one
   trainer, and its best checkpoint (chosen on validation) must still have
   the content hash its ``result.json`` recorded.
2. **Claim.** The first call writes ``test_claim.json`` next to
   ``plan.json`` (``open(path, "x")``), naming every trial and checkpoint it
   scores. A later call may only finish that same list: a changed, added or
   retrained trial is refused, so the test fold is spent once, on the whole
   experiment.
3. **Load.** Each manifest passes the leakage gate again, and its ``test``
   role (fold 10) is loaded through the adapter. Every manifest must name
   the same test records.
4. **Score.** Each best checkpoint goes through the ``predictor`` its
   trainer named. The run gets ``predictions_test.parquet`` and
   ``test_scores.json``; a run that already has its scores is never scored
   again. ``test/auroc_{overall,artifact,clean}`` and ``test/loss`` are
   computed exactly as the validation scores are
   (``syntide.eval.stats.val_scores``).
"""

import argparse
import hashlib
import importlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from syntide import __version__
from syntide.data.schema import SUPERCLASSES, Samples
from syntide.eval.stats import val_scores
from syntide.trial import Checkpoint, Predictor, TrialSpec, load_spec

log = logging.getLogger(__name__)

PLAN_FILE = "plan.json"
CLAIM_FILE = "test_claim.json"
PREDICTIONS_FILE = "predictions_test.parquet"
SCORES_FILE = "test_scores.json"


class HoldoutError(RuntimeError):
    """Unfinished or changed runs prevent test evaluation."""


@dataclass(frozen=True)
class FrozenRun:
    run_dir: Path
    spec: TrialSpec
    result: dict[str, Any]
    best: Checkpoint
    best_sha256: str


def frozen_runs(protocol_dir: str | Path) -> list[FrozenRun]:
    """Return planned runs after all finish without input changes."""
    protocol_dir = Path(protocol_dir)
    plan_path = protocol_dir / PLAN_FILE
    if not plan_path.exists():
        raise HoldoutError(
            f"{plan_path} does not exist; plan the protocol first"
        )
    runs, unfinished = [], []
    for entry in json.loads(plan_path.read_text(encoding="utf-8"))["trials"]:
        run_dir = protocol_dir / entry["run_dir"]
        result_path = run_dir / "result.json"
        if not result_path.exists():
            unfinished.append(entry["run_dir"])
            continue
        spec = load_spec(run_dir)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete" or not (
            result["trial_id"] == spec.trial_id == entry["trial_id"]
        ):
            raise HoldoutError(
                f"{run_dir} does not hold the finished trial "
                f"{entry['trial_id'][:12]}"
            )
        best = result["provenance"]["checkpoints"]["best"]
        path = run_dir / best["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != best["sha256"]:
            raise HoldoutError(
                f"{path} changed after training; its validation choice no "
                "longer holds"
            )
        runs.append(
            FrozenRun(
                run_dir, spec, result, Checkpoint(path, best["step"]), digest
            )
        )
    if unfinished:
        raise HoldoutError(
            f"{len(unfinished)} planned trials are not finished ("
            f"{', '.join(unfinished[:4])}"
            f"{(', ...' if len(unfinished) > 4 else '')}); the test fold "
            "is scored once every trial is done"
        )
    if not runs:
        raise HoldoutError(f"{plan_path} plans no trial")
    trainers = {run.result["trainer"]["fingerprint"] for run in runs}
    if len(trainers) > 1:
        raise HoldoutError(
            f"the runs come from {len(trainers)} trainer fingerprints; "
            "compared arms must share one"
        )
    return runs


def claim(protocol_dir: str | Path, runs: Sequence[FrozenRun]) -> Path:
    """Record exactly which completed trials may use the test split.

    Repeating the same set is allowed. A different set is rejected.
    """
    protocol_dir = Path(protocol_dir)
    trials = [
        {
            "run_dir": str(run.run_dir.relative_to(protocol_dir)),
            "trial_id": run.spec.trial_id,
            "best_sha256": run.best_sha256,
        }
        for run in runs
    ]
    path = protocol_dir / CLAIM_FILE
    payload = {
        "claimed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "syntide_version": __version__,
    }
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump({**payload, "trials": trials}, handle, indent=1)
            handle.write("\n")
    except FileExistsError:
        claimed = json.loads(path.read_text(encoding="utf-8"))
        if claimed["trials"] != trials:
            raise HoldoutError(
                f"the test fold of {protocol_dir} was claimed on "
                f"{claimed['claimed_at']} for another set of trials or "
                "checkpoints; it is never scored twice. Plan a new "
                "protocol to test again."
            ) from None
    return path


def load_test(
    runs: Sequence[FrozenRun], registry: str | Path | None = None
) -> Samples:
    """Check patient separation and load the held-out test split."""
    from syntide.data.adapter import PtbXlAdapter
    from syntide.data.challenge import load_manifest
    from syntide.data.leakage import check_manifest

    spec = runs[0].spec
    adapter = PtbXlAdapter(registry or spec.registry, spec.dataset)
    index, source = adapter.index(), adapter.source()
    test_ids = set()
    for manifest_path, manifest_sha in sorted(
        {(run.spec.manifest, run.spec.manifest_sha256) for run in runs}
    ):
        manifest = load_manifest(Path(manifest_path))
        if manifest.sha256 != manifest_sha:
            raise HoldoutError(
                f"{manifest_path} changed since its trials were planned"
            )
        check_manifest(manifest, index, source)  # Gate 1
        test_ids.add(manifest.roles["test"])
    if {run.spec.source_sha256 for run in runs} != {source["sha256"]}:
        raise HoldoutError(
            "the local data is not what every trial was planned on"
        )
    if len(test_ids) != 1:
        raise HoldoutError(
            "the manifests name different test records; their runs "
            "cannot be compared"
        )
    return adapter.load(list(test_ids.pop()))


def import_predictor(name: str) -> Predictor:
    """Import the trainer prediction function when evaluation starts."""
    module, _, function = name.partition(":")
    return getattr(importlib.import_module(module), function)


def score_run(
    run: FrozenRun,
    test: Samples | None,
    load_predictor: Callable[[str], Predictor] = import_predictor,
) -> dict[str, Any]:
    """Score the selected checkpoint once, reusing saved test results."""
    scores_path = run.run_dir / SCORES_FILE
    if scores_path.exists():
        return json.loads(scores_path.read_text(encoding="utf-8"))
    if test is None:
        raise HoldoutError(
            f"{run.run_dir} is not scored yet and no test fold was loaded"
        )
    predict = load_predictor(run.result["trainer"]["predictor"])
    probs = np.asarray(
        predict(run.best, test.x, run.result["trainer"]["device"]),
        dtype=np.float64,
    )
    scores = {
        key.replace("val/", "test/", 1): value
        for key, value in val_scores(test, probs).items()
    }
    _write_predictions(run, test, probs)
    spec = run.spec
    payload = {
        "trial_id": spec.trial_id,
        "regime": spec.regime,
        "arm": spec.arm,
        "seed": spec.replicate_seed,
        "checkpoint": "best",
        "step": run.best.step,
        "n_test": len(test),
        "n_test_artifact": int(test.is_artifact.sum()),
        **scores,
        "scored_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with scores_path.open(
        "x", encoding="utf-8"
    ) as handle:  # last: scores exist only for a scored run
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    return payload


def _write_predictions(
    run: FrozenRun, test: Samples, probs: np.ndarray
) -> None:
    n, k = test.labels.shape
    spec = run.spec
    table = pa.table(
        {
            "trial_id": np.repeat(spec.trial_id, n * k),
            "arm": np.repeat(spec.arm, n * k),
            "regime": np.repeat(spec.regime, n * k),
            "seed": np.repeat(spec.replicate_seed, n * k),
            "checkpoint": np.repeat("best", n * k),
            "step": np.repeat(run.best.step, n * k),
            "record_id": np.repeat(test.record_id, k),
            "subject_uid": np.repeat(test.subject_uid, k),
            "class_id": np.tile(np.asarray(SUPERCLASSES), n),
            "y_true": test.labels.reshape(-1),
            "y_prob": probs.reshape(-1),
            "is_artifact": np.repeat(test.is_artifact, k),
        }
    )
    pq.write_table(table, run.run_dir / PREDICTIONS_FILE, compression="zstd")


def evaluate(
    protocol_dir: str | Path,
    *,
    test: Samples | None = None,
    registry: str | Path | None = None,
    load_predictor: Callable[[str], Predictor] = import_predictor,
) -> list[dict[str, Any]]:
    """Check fixed inputs, record test access, and score the planned trials.

    When test is supplied, use those samples instead of loading the dataset.
    """
    runs = frozen_runs(protocol_dir)
    claim(protocol_dir, runs)
    if test is None and any(
        not (run.run_dir / SCORES_FILE).exists() for run in runs
    ):
        test = load_test(runs, registry)
    return [score_run(run, test, load_predictor) for run in runs]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m syntide.eval.holdout",
        description=(
            "Score every finished trial of a protocol on the test fold, once."
        ),
    )
    parser.add_argument(
        "protocol_dir",
        help="the protocol's runs directory, e.g. runs/ptbxl-artifact-50",
    )
    parser.add_argument(
        "--registry",
        help="override the registry path the trials were planned with",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
    try:
        rows = evaluate(args.protocol_dir, registry=args.registry)
    except HoldoutError as error:
        raise SystemExit(f"test fold not scored: {error}") from error
    for row in rows:
        log.info(
            (
                "%-11s %-3s seed%d  test AUROC overall %.4f  noisy %.4f  "
                "clean %.4f"
            ),
            row["regime"],
            row["arm"],
            row["seed"],
            row["test/auroc_overall"],
            row["test/auroc_artifact"],
            row["test/auroc_clean"],
        )


if __name__ == "__main__":
    main()
