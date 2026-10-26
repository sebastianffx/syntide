"""Prepare classification trials and validate trainer results.

A trainer uses this module and ``syntide.eval.tracking`` to read a fixed
``trial.json``, consume training batches, and save checked results.
Initialize the model with ``trial.spec.init_seed`` and record its initial
parameter hash. Consume ``trial.batches()`` exactly once, evaluate
validation data every ``val_every`` steps, and pass both the best and final
checkpoints to ``trial.finish``.

Finishing a trial checks that every planned batch was consumed, validation
scores can be reproduced, and the selected checkpoint is the earliest one
with the highest selection score. Trainers receive no test data. Later
evaluation loads the test split through the trainer's declared prediction
function.
"""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from syntide import __version__
from syntide.data.schema import SUPERCLASSES, Samples
from syntide.eval.stats import SELECTION_METRIC, val_scores
from syntide.eval.tracking import Tracker, TrackerConfig, make_tracker
from syntide.generate.ledger import LedgerWriter
from syntide.generate.policy import CalibrationError, load_calibration
from syntide.generate.stream import Batch, TrainStream, init_seed

__all__ = [
    "Batch",
    "Checkpoint",
    "Predictor",
    "TrainerInfo",
    "Trial",
    "TrialError",
    "TrialSpec",
    "is_complete",
    "load_spec",
    "make_spec",
    "open_trial",
    "prepare",
]

TRIAL_FILE = "trial.json"
RESULT_FILE = "result.json"
EXPOSURE_FILE = "exposure.json"
LEDGER_FILE = "recipes.parquet"
PREDICTIONS_FILE = "predictions_val.parquet"

# Paths say where inputs live, not what they are, so they stay out of the trial
# id.
_NOT_IN_ID = ("trial_id", "registry", "manifest", "calibration")


class TrialError(Exception):
    """A trial cannot be prepared, or a trainer's results break the protocol."""


@dataclass(frozen=True)
class TrialSpec:
    """Fixed inputs and settings for one treatment, data pool, and seed."""

    protocol: str
    regime: str
    arm: str
    replicate_seed: int
    source_role: str  # depleted | natural_mix | pool
    transform_family: str  # none | generic | editor
    sampling_policy: str  # none | fixed | coverage_matched | gap_targeted
    generator: str | None  # e.g. per_lead_aug@1; None for the none family
    validators: list[str]
    transform: (
        dict  # the family's settings (A1 envelope, editor config); {} for none
    )
    dataset: str
    source_sha256: str
    manifest_sha256: str
    calibration_sha256: str | None
    batch_size: int
    max_steps: int
    val_every: int
    transform_fraction: float  # 0 for arms without transform slots (C1, C2)
    init_seed: int  # trainers seed model initialization with it
    syntide_version: str
    registry: str
    manifest: str
    calibration: str | None
    trial_id: str

    @property
    def run_name(self) -> str:
        return f"{self.regime}/{self.arm}/seed{self.replicate_seed}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _trial_id(content: Mapping[str, Any]) -> str:
    identity = {
        key: value for key, value in content.items() if key not in _NOT_IN_ID
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def make_spec(**content: Any) -> TrialSpec:
    """Build a trial specification and derive its identity fields."""
    content = {
        **content,
        "init_seed": init_seed(content["replicate_seed"]),
        "syntide_version": __version__,
        "validators": list(content["validators"]),
        "transform": dict(content["transform"]),
    }
    return TrialSpec(**content, trial_id=_trial_id(content))


def load_spec(run_dir: str | Path) -> TrialSpec:
    """Read ``trial.json`` and verify it was not edited after planning."""
    path = Path(run_dir) / TRIAL_FILE
    if not path.exists():
        raise TrialError(
            f"{path} does not exist; plan the trial with python -m "
            "syntide.generate mode=plan"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {f.name for f in fields(TrialSpec)}
    if set(payload) != expected:
        raise TrialError(
            f"{path} has fields {sorted(set(payload) ^ expected)} that a "
            "TrialSpec does not"
        )
    if _trial_id(payload) != payload["trial_id"]:
        raise TrialError(
            f"{path} does not match its trial_id; re-plan it instead of "
            "editing it"
        )
    return TrialSpec(**payload)


def write_spec(run_dir: str | Path, spec: TrialSpec) -> bool:
    """Save a trial specification without replacing a different trial.

    Args:
        run_dir: Directory in which to save trial.json.
        spec: Validated specification to save.

    Returns:
        True if a file was written, or False if the same trial already
        exists.

    Raises:
        TrialError: The directory already contains a different trial.
    """
    path = Path(run_dir) / TRIAL_FILE
    if path.exists():
        existing = load_spec(run_dir)
        if existing.trial_id != spec.trial_id:
            raise TrialError(
                f"{path} already holds trial {existing.trial_id[:12]}, not "
                f"{spec.trial_id[:12]}. Plan into a new runs_dir, or "
                "delete that run directory on purpose."
            )
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, spec.to_dict())
    return True


@dataclass(frozen=True)
class TrainerInfo:
    """Trainer identity and model settings used to compare treatments."""

    name: str
    version: str
    # every setting that can change the model; not logging cadence or paths
    config: dict
    device: str
    # "module:function" satisfying Predictor; syntide.eval calls it for the test
    # split
    predictor: str

    @property
    def fingerprint(self) -> str:
        content = {
            "name": self.name,
            "version": self.version,
            "config": self.config,
            "device": self.device,
        }
        canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    step: int


class Predictor(Protocol):
    def __call__(
        self, checkpoint: Checkpoint, x: np.ndarray, device: str
    ) -> np.ndarray:
        """Return class probabilities for a batch of electrocardiograms.

        Args:
            checkpoint: Saved model parameters to load.
            x: Signals in millivolts, shaped (recordings, 12 leads, time
                samples).
            device: Device on which to run prediction.

        Returns:
            Probabilities shaped (recordings, 5 diagnostic classes).
        """
        ...


def is_complete(run_dir: str | Path, trainer: TrainerInfo) -> bool:
    """Return whether this trainer has finished the planned trial.

    Raises:
        TrialError: Existing results belong to another trial or trainer.
    """
    path = Path(run_dir) / RESULT_FILE
    if not path.exists():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    _check_same_run(path, result, load_spec(run_dir), trainer)
    return result.get("status") == "complete"


def _check_same_run(
    path: Path, result: Mapping, spec: TrialSpec, trainer: TrainerInfo
) -> None:
    if result["trial_id"] != spec.trial_id:
        raise TrialError(
            f"{path} belongs to trial {result['trial_id'][:12]}, not "
            f"{spec.trial_id[:12]}"
        )
    if result["trainer"]["fingerprint"] != trainer.fingerprint:
        raise TrialError(
            f"{path} was trained by {result['trainer']['name']} with "
            f"fingerprint {result['trainer']['fingerprint'][:12]}, not "
            f"{trainer.fingerprint[:12]}. A different trainer or trainer "
            "config needs its own runs_dir."
        )


def prepare(
    run_dir: str | Path,
    *,
    tracker: TrackerConfig,
    registry: str | Path | None = None,
) -> "Trial":
    """Validate local inputs and open a planned trial for training.

    Check patient separation and load the planned training and validation
    data. Test data is never loaded here.

    Args:
        run_dir: Directory containing the fixed trial specification.
        tracker: Configuration for recording training progress.
        registry: Optional local registry path overriding the planned path.

    Returns:
        An open trial, to be used as a context manager.

    Raises:
        TrialError: The trial is complete or its inputs differ from the
        plan.
    """
    from syntide.data.adapter import PtbXlAdapter
    from syntide.data.challenge import load_manifest
    from syntide.data.leakage import check_manifest, check_recipe_parents

    run_dir = Path(run_dir)
    spec = load_spec(run_dir)
    if spec.syntide_version != __version__:
        raise TrialError(
            f"{run_dir} was planned with syntide {spec.syntide_version}, "
            f"this is {__version__}"
        )
    if (run_dir / RESULT_FILE).exists():
        raise TrialError(
            f"{run_dir} is already finished; results are never overwritten"
        )

    adapter = PtbXlAdapter(registry or spec.registry, spec.dataset)
    index, source = adapter.index(), adapter.source()
    if source["sha256"] != spec.source_sha256:
        raise TrialError(
            f"the local data ({source['sha256'][:12]}) is not what "
            f"{run_dir} was planned on"
        )
    manifest = load_manifest(Path(spec.manifest))
    if manifest.sha256 != spec.manifest_sha256:
        raise TrialError(f"{spec.manifest} changed since {run_dir} was planned")
    check_manifest(manifest, index, source)
    if spec.transform_family != "none":
        check_recipe_parents(manifest.roles[spec.source_role], manifest)
    calibration = None
    if spec.calibration is not None:
        try:
            calibration = load_calibration(
                spec.calibration, spec.calibration_sha256
            )
        except CalibrationError as error:
            raise TrialError(f"{run_dir}: {error}") from error
        if calibration["manifest_sha256"] != spec.manifest_sha256:
            raise TrialError(
                f"{spec.calibration} was fit on another manifest than "
                f"{spec.manifest}"
            )

    parents = adapter.load(manifest.roles[spec.source_role])
    val = adapter.load(manifest.roles["val"])
    board = make_tracker(
        tracker, run_dir, spec.run_name, spec.protocol, spec.to_dict()
    )
    return open_trial(
        spec, parents, val, board, run_dir, calibration=calibration
    )


def open_trial(
    spec: TrialSpec,
    parents: Samples,
    val: Samples,
    tracker: Tracker,
    run_dir: str | Path,
    *,
    calibration: dict[str, Any] | None = None,
) -> "Trial":
    """Open a training stream from samples already loaded into memory."""
    if len(val) == 0:
        raise TrialError("the validation set is empty")
    return Trial(spec, parents, val, tracker, Path(run_dir), calibration)


class Trial:
    """An open training trial that closes its tracker on exit."""

    def __init__(
        self,
        spec: TrialSpec,
        parents: Samples,
        val: Samples,
        tracker: Tracker,
        run_dir: Path,
        calibration: dict[str, Any] | None = None,
    ):
        self.spec = spec
        self.val = val
        self.tracker = tracker
        self.run_dir = run_dir
        self._ledger = LedgerWriter(run_dir / LEDGER_FILE)
        self._stream = TrainStream(
            spec, parents, tracker, self._ledger, calibration
        )
        self._open = True

    def batches(self) -> Iterator[Batch]:
        """Yield the planned number of training batches, once.

        Yields:
            Each training batch in its fixed order, for spec.max_steps
            updates.
        """
        return iter(self._stream)

    @property
    def exposure(self) -> dict[str, int]:
        return self._stream.exposure

    def __enter__(self) -> "Trial":
        return self

    def __exit__(self, *exc: object) -> None:
        if self._open:
            self._ledger.discard()
            self.tracker.close()
            self._open = False

    def finish(
        self,
        trainer: TrainerInfo,
        best: Checkpoint,
        final: Checkpoint,
        val_history: Sequence[Mapping[str, float]],
        val_probs: Mapping[str, np.ndarray],
        init_sha256: str,
        wall_time_s: float,
    ) -> dict[str, Any]:
        """Validate trainer results and save the completed run.

        Args:
            trainer: Trainer identity and configuration used for this run.
            best: Earliest checkpoint with the highest validation selection
                score.
            final: Checkpoint from the last planned training step.
            val_history: Validation metrics recorded at each planned
                validation step.
            val_probs: Validation probabilities for the best and final
                checkpoints.
            init_sha256: Hash of the model's initial
                parameters and buffers.
            wall_time_s: Elapsed training time in seconds.

        Returns:
            The run result saved to result.json.

        Raises:
            TrialError: The run is closed, incomplete, or inconsistent with
                its plan.
        """
        if not self._open:
            raise TrialError("this trial is already closed")
        spec, history = self.spec, [dict(entry) for entry in val_history]
        result_path = self.run_dir / RESULT_FILE
        if result_path.exists():
            _check_same_run(
                result_path, json.loads(result_path.read_text()), spec, trainer
            )
            raise TrialError(
                f"{self.run_dir} is already finished; results are never "
                "overwritten"
            )
        if not self._stream.consumed:
            raise TrialError(
                f"the trainer consumed {self._stream.steps_served} of "
                f"{spec.max_steps} batches; every arm trains for the full "
                "budget"
            )
        if final.step != spec.max_steps:
            raise TrialError(
                f"the final checkpoint is at step {final.step}, not "
                f"max_steps={spec.max_steps}"
            )
        steps = [int(entry["step"]) for entry in history]
        expected = list(
            range(spec.val_every, spec.max_steps + 1, spec.val_every)
        )
        if steps != expected:
            raise TrialError(
                f"validation must run every {spec.val_every} steps; got "
                f"steps {steps[:8]}..."
            )
        scores = [entry[SELECTION_METRIC] for entry in history]
        best_step = steps[int(np.argmax(scores))]
        if best.step != best_step:
            raise TrialError(
                f"the best checkpoint is at step {best.step}, but the "
                f"first argmax of {SELECTION_METRIC} is step {best_step}"
            )
        if set(val_probs) != {"best", "final"}:
            raise TrialError(
                "val_probs needs exactly 'best' and 'final', got "
                f"{sorted(val_probs)}"
            )
        for name, checkpoint in (("best", best), ("final", final)):
            reported = history[steps.index(checkpoint.step)]
            recomputed = val_scores(self.val, val_probs[name])
            for key, value in recomputed.items():
                if key not in reported or not math.isclose(
                    reported[key], value, rel_tol=1e-9, abs_tol=1e-12
                ):
                    raise TrialError(
                        f"{name} probabilities give {key}={value:.6f}, but "
                        f"the history reports {reported.get(key)} at step "
                        f"{checkpoint.step}"
                    )
            if not Path(checkpoint.path).is_file():
                raise TrialError(
                    f"the {name} checkpoint {checkpoint.path} does not exist"
                )
        if not re.fullmatch(r"[0-9a-f]{64}", init_sha256):
            raise TrialError(
                "init_sha256 must be the sha256 hex digest of the "
                "initial model state"
            )

        exposure = self._stream.exposure
        self._ledger.close()
        _write_json(self.run_dir / EXPOSURE_FILE, exposure)
        self._write_predictions({"best": best, "final": final}, val_probs)
        result = {
            "status": "complete",
            "trial_id": spec.trial_id,
            "protocol": spec.protocol,
            "regime": spec.regime,
            "arm": spec.arm,
            "replicate_seed": spec.replicate_seed,
            "best_step": best.step,
            "best_val_auroc": float(scores[steps.index(best.step)]),
            "final_step": final.step,
            "val_history": history,
            "exposure": exposure,
            "trainer": {
                **asdict(trainer),
                "fingerprint": trainer.fingerprint,
                "init_sha256": init_sha256,
            },
            "wall_time_s": float(wall_time_s),
            "provenance": {
                "dataset": spec.dataset,
                "registry": spec.registry,
                "source_sha256": spec.source_sha256,
                "manifest": spec.manifest,
                "manifest_sha256": spec.manifest_sha256,
                "calibration": spec.calibration,
                "calibration_sha256": spec.calibration_sha256,
                "checkpoints": {
                    name: {
                        "path": self._relative(checkpoint.path),
                        "step": checkpoint.step,
                        "sha256": _file_sha256(Path(checkpoint.path)),
                    }
                    for name, checkpoint in (("best", best), ("final", final))
                },
            },
            "env": _environment(),
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        _write_json(
            result_path, result
        )  # last: result.json exists only for a finished trial
        self.tracker.log_summary(
            {
                "arm": spec.arm,
                "regime": spec.regime,
                "seed": spec.replicate_seed,
                **_flat(trainer.config),
            },
            {
                "summary/best_val_auroc_overall": result["best_val_auroc"],
                "summary/best_step": best.step,
                "summary/wall_time_seconds": float(wall_time_s),
            },
        )
        self.tracker.close()
        self._open = False
        return result

    def _write_predictions(
        self,
        checkpoints: Mapping[str, Checkpoint],
        val_probs: Mapping[str, np.ndarray],
    ) -> None:
        spec, val = self.spec, self.val
        n, k = val.labels.shape
        parts = []
        for name, checkpoint in checkpoints.items():
            parts.append(
                {
                    "checkpoint": np.repeat(name, n * k),
                    "step": np.repeat(checkpoint.step, n * k),
                    "record_id": np.repeat(val.record_id, k),
                    "subject_uid": np.repeat(val.subject_uid, k),
                    "class_id": np.tile(np.asarray(SUPERCLASSES), n),
                    "y_true": val.labels.reshape(-1),
                    "y_prob": np.asarray(
                        val_probs[name], dtype=np.float64
                    ).reshape(-1),
                    "is_artifact": np.repeat(val.is_artifact, k),
                }
            )
        columns = {
            key: np.concatenate([part[key] for part in parts])
            for key in parts[0]
        }
        rows = len(columns["record_id"])
        table = pa.table(
            {
                "trial_id": np.repeat(spec.trial_id, rows),
                "arm": np.repeat(spec.arm, rows),
                "regime": np.repeat(spec.regime, rows),
                "seed": np.repeat(spec.replicate_seed, rows),
                **columns,
            }
        )
        pq.write_table(
            table, self.run_dir / PREDICTIONS_FILE, compression="zstd"
        )

    def _relative(self, path: str | Path) -> str:
        path = Path(path).resolve()
        try:
            return str(path.relative_to(self.run_dir.resolve()))
        except ValueError:
            return str(path)


def _flat(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, Mapping):
            out.update(_flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    """Write atomically, so a crash never leaves a half-written file behind."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=1, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "timenet", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "syntide": __version__,
        "packages": packages,
        "git": _git_state(),
    }


def _git_state() -> dict[str, Any] | None:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"rev": rev.strip(), "dirty": bool(status.stdout.strip())}
