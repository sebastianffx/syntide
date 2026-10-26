"""Prepare question-answer training trials and validate their results.

Each trial uses published training collections, a fixed sampling schedule,
and a fixed checkpoint-selection collection. ``load_any_spec`` reads the
schema version in ``trial.json`` to choose question-answer or classification
loading. Existing classification files retain their original identities.

A trainer initializes its model with ``trial.spec.init_seed``, consumes each
batch once, and evaluates the selection collection every ``val_every``
steps. ``QAExample.student`` contains only inputs the model may read:
signals, question, options, and permitted context. ``QAExample.target``
contains the answer and any explanation used for training.

``finish`` parses the saved model outputs again, recomputes reported scores,
and requires the earliest checkpoint with the highest selection accuracy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from syntide import __version__
from syntide.data.qa import QATask
from syntide.eval import qa_metrics
from syntide.eval.tracking import Tracker, TrackerConfig, make_tracker
from syntide.trial import (
    RESULT_FILE,
    TRIAL_FILE,
    Checkpoint,
    TrainerInfo,
    TrialError,
    TrialSpec,
    _environment,
    _file_sha256,
    _write_json,
    load_spec,
)

QA_TRIAL_SCHEMA = "syntide.qa-trial@1"
EXPOSURE_FILE = "exposure.json"
PREDICTIONS_FILE = "predictions_val.parquet"
SUPERVISION = ("answer_only", "rationale")


@dataclass(frozen=True)
class QATrialSpec:
    schema_version: str
    protocol: str
    arm: str
    supervision: str  # answer_only | rationale
    replicate_seed: int
    corpora: list[
        dict[str, str]
    ]  # [{label, corpus_id, path}], training corpora
    selection: dict[str, str]  # {corpus_id, path}: the frozen selection suite
    curriculum: dict[str, Any]
    curriculum_sha256: str  # the frozen curriculum and its exact pools
    batch_size: int  # examples per optimizer update
    max_steps: int
    val_every: int
    metric: dict[str, str]  # name, direction, tie_break
    parser: str
    decoding: dict[
        str, Any
    ]  # frozen generation / option-scoring settings the trainer must use
    context_policy: str
    budget: dict[
        str, Any
    ]  # primary budget declaration (e.g. updates x examples) and its notes
    init_seed: int
    syntide_version: str
    trial_id: str

    @property
    def run_name(self) -> str:
        return f"{self.arm}/{self.supervision}/seed{self.replicate_seed}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _identity(content: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude storage paths from the content used to identify a trial."""
    out = {k: v for k, v in content.items() if k != "trial_id"}
    out["corpora"] = [
        {k: v for k, v in c.items() if k != "path"} for c in out["corpora"]
    ]
    out["selection"] = {
        k: v for k, v in out["selection"].items() if k != "path"
    }
    return out


def _trial_id(content: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        _identity(content), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def make_qa_spec(**content: Any) -> QATrialSpec:
    from syntide.generate.stream import init_seed

    if content["supervision"] not in SUPERVISION:
        raise TrialError(f"supervision must be one of {SUPERVISION}")
    content = {
        **content,
        "schema_version": QA_TRIAL_SCHEMA,
        "init_seed": init_seed(content["replicate_seed"]),
        "syntide_version": __version__,
    }
    return QATrialSpec(**content, trial_id=_trial_id(content))


def load_qa_spec(run_dir: str | Path) -> QATrialSpec:
    path = Path(run_dir) / TRIAL_FILE
    if not path.exists():
        raise TrialError(
            f"{path} does not exist; plan it with python -m "
            "syntide.generate.qa plan"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {f.name for f in fields(QATrialSpec)}
    if (
        payload.get("schema_version") != QA_TRIAL_SCHEMA
        or set(payload) != expected
    ):
        raise TrialError(f"{path} is not a {QA_TRIAL_SCHEMA} trial")
    if _trial_id(payload) != payload["trial_id"]:
        raise TrialError(
            f"{path} does not match its trial_id; re-plan it instead of "
            "editing it"
        )
    return QATrialSpec(**payload)


def load_any_spec(run_dir: str | Path) -> TrialSpec | QATrialSpec:
    """Load a trial according to its declared schema."""
    payload = json.loads(
        (Path(run_dir) / TRIAL_FILE).read_text(encoding="utf-8")
    )
    return (
        load_qa_spec(run_dir)
        if payload.get("schema_version") == QA_TRIAL_SCHEMA
        else load_spec(run_dir)
    )


def write_qa_spec(run_dir: str | Path, spec: QATrialSpec) -> bool:
    path = Path(run_dir) / TRIAL_FILE
    if path.exists():
        existing = load_any_spec(run_dir)
        if existing.trial_id != spec.trial_id:
            raise TrialError(
                f"{path} already holds trial {existing.trial_id[:12]}, not "
                f"{spec.trial_id[:12]}"
            )
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, spec.to_dict())
    return True


@dataclass(frozen=True)
class QAExample:
    task_id: str
    # Model inputs: signals, recording references, question, options, context.
    student: dict[str, Any]
    # privileged: the supervised text (answer line, with the rationale in
    # rationale mode)
    target: str
    target_mode: str
    family: str
    stage: str
    corpus: str


@dataclass(frozen=True)
class QABatch:
    step: int
    examples: list[QAExample]


class QAPredictor(Protocol):
    def __call__(
        self, checkpoint: Checkpoint, examples: Sequence[QAExample], device: str
    ) -> list[str]:
        """Return one output per example, ending with an Answer: line."""
        ...


def example(
    task: QATask,
    signal_of: Any,
    *,
    target_mode: str | None,
    stage: str = "",
    corpus: str = "",
    signals: Sequence[np.ndarray] | None = None,
) -> QAExample:
    """Build model inputs and, optionally, a training target.

    Set target_mode to None when preparing inputs for prediction without
    answers.
    """
    view = task.student_view()
    view["signals"] = (
        list(signals)
        if signals is not None
        else [signal_of(ref.waveform_id) for ref in task.inputs]
    )
    return QAExample(
        task.task_id,
        view,
        task.target(target_mode) if target_mode else "",
        target_mode or "",
        task.family,
        stage,
        corpus,
    )


def is_qa_complete(run_dir: str | Path, trainer: TrainerInfo) -> bool:
    path = Path(run_dir) / RESULT_FILE
    if not path.exists():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    spec = load_qa_spec(run_dir)
    if (
        result["trial_id"] != spec.trial_id
        or result["trainer"]["fingerprint"] != trainer.fingerprint
    ):
        raise TrialError(
            f"{path} belongs to another trial or trainer; use a new runs_dir"
        )
    return result.get("status") == "complete"


def prepare_qa(run_dir: str | Path, *, tracker: TrackerConfig) -> QATrial:
    """Validate collections and open a question-answer trial."""
    from syntide.generate.qa.corpus import load_corpus

    run_dir = Path(run_dir)
    spec = load_qa_spec(run_dir)
    if spec.syntide_version != __version__:
        raise TrialError(
            f"{run_dir} was planned with syntide {spec.syntide_version}, "
            f"this is {__version__}"
        )
    if (run_dir / RESULT_FILE).exists():
        raise TrialError(
            f"{run_dir} is already finished; results are never overwritten"
        )
    corpora = {}
    for entry in spec.corpora:
        corpus = load_corpus(entry["path"])
        if corpus.corpus_id != entry["corpus_id"]:
            raise TrialError(
                f"{entry['path']} holds {corpus.corpus_id}, not the "
                f"planned {entry['corpus_id']}"
            )
        corpora[entry["label"]] = corpus
    selection = load_corpus(spec.selection["path"])
    if selection.corpus_id != spec.selection["corpus_id"]:
        raise TrialError(
            f"{spec.selection['path']} holds {selection.corpus_id}, not "
            f"{spec.selection['corpus_id']}"
        )
    board = make_tracker(
        tracker, run_dir, spec.run_name, spec.protocol, spec.to_dict()
    )
    return open_qa_trial(spec, corpora, selection, board, run_dir)


def check_trial_inputs(corpora: Mapping[str, Any], selection: Any) -> None:
    """Validate collection roles and patient separation.

    Training collections must permit training. The selection collection must
    be reserved for checkpoint selection and share no patient with training
    data.
    """
    from syntide.data.leakage import check_disjoint_subjects

    for label, corpus in corpora.items():
        if corpus.purpose != "training":
            raise TrialError(
                f"corpus {label} ({corpus.corpus_id}) is a {corpus.purpose}"
                " corpus; it cannot train"
            )
    if (
        selection.purpose != "evaluation"
        or selection.info["identity"]["config"].get("suite_role") != "selection"
    ):
        raise TrialError(f"{selection.corpus_id} is not a selection suite")
    subjects = {
        label: {s for ref in c.inputs.values() for s in ref.root_subject_uids}
        for label, c in corpora.items()
    }
    train = set().union(*subjects.values())
    check_disjoint_subjects(
        {
            "training corpora": train,
            "selection suite": {
                s
                for ref in selection.inputs.values()
                for s in ref.root_subject_uids
            },
        }
    )


def open_qa_trial(
    spec: QATrialSpec,
    corpora: Mapping[str, Any],
    selection: Any,
    tracker: Tracker,
    run_dir: str | Path,
) -> QATrial:
    """Open a trial from collections already loaded into memory."""
    from syntide.generate.qa.curriculum import (
        Curriculum,
        CurriculumConfig,
        StageConfig,
    )

    check_trial_inputs(corpora, selection)
    cfg = CurriculumConfig(
        **{
            **spec.curriculum,
            "stages": [StageConfig(**s) for s in spec.curriculum["stages"]],
        }
    )
    curriculum = Curriculum(
        cfg,
        {label: c.tasks for label, c in corpora.items()},
        batch_size=spec.batch_size,
        max_steps=spec.max_steps,
        supervision=spec.supervision,
    )
    if curriculum.sha256 != spec.curriculum_sha256:
        raise TrialError(
            "the curriculum or its corpora changed since the trial was planned"
        )
    return QATrial(spec, corpora, selection, curriculum, tracker, Path(run_dir))


class QATrial:
    def __init__(
        self,
        spec: QATrialSpec,
        corpora: Mapping[str, Any],
        selection: Any,
        curriculum: Any,
        tracker: Tracker,
        run_dir: Path,
    ) -> None:
        self.spec, self.corpora, self.selection, self.curriculum = (
            spec,
            dict(corpora),
            selection,
            curriculum,
        )
        self.tracker, self.run_dir = tracker, run_dir
        self._tasks = {
            t.task_id: (label, t)
            for label, c in self.corpora.items()
            for t in c.tasks
        }
        self._slots: list[Any] = []
        self._started = False
        self.steps_served = 0
        self._open = True

    @property
    def selection_tasks(self) -> list[QATask]:
        return list(self.selection.tasks)

    def selection_examples(self) -> list[QAExample]:
        """Return selection inputs without answers, in their fixed order."""
        return [
            example(t, self.selection.signal, target_mode=None)
            for t in self.selection.tasks
        ]

    def batches(self) -> Iterator[QABatch]:
        if self._started:
            raise RuntimeError("a QA trial's batches can be iterated only once")
        self._started = True
        return self._iter()

    def _iter(self) -> Iterator[QABatch]:
        for slots in self.curriculum.schedule():
            examples = []
            for slot in slots:
                label, task = self._tasks[slot.task_id]
                examples.append(
                    example(
                        task,
                        self.corpora[label].signal,
                        target_mode=slot.target_mode,
                        stage=slot.stage,
                        corpus=label,
                    )
                )
            self._slots.extend(slots)
            self.steps_served = slots[0].step
            if slots[0].step % self.spec.val_every == 0:
                self.tracker.log_scalars(
                    {"budget/examples": len(self._slots)}, slots[0].step
                )
            yield QABatch(slots[0].step, examples)

    @property
    def exposure(self) -> dict[str, Any]:
        from syntide.generate.qa.curriculum import exposure

        return exposure(
            self._slots, {tid: task for tid, (_, task) in self._tasks.items()}
        )

    def __enter__(self) -> QATrial:
        return self

    def __exit__(self, *exc: object) -> None:
        if self._open:
            self.tracker.close()
            self._open = False

    def finish(
        self,
        trainer: TrainerInfo,
        best: Checkpoint,
        final: Checkpoint,
        val_history: Sequence[Mapping[str, float]],
        val_outputs: Mapping[str, Sequence[str]],
        init_sha256: str,
        wall_time_s: float,
        token_counts: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
        """Validate trainer results and save the completed run."""
        if not self._open:
            raise TrialError("this trial is already closed")
        spec, history = self.spec, [dict(entry) for entry in val_history]
        result_path = self.run_dir / RESULT_FILE
        if result_path.exists():
            raise TrialError(
                f"{self.run_dir} is already finished; results are never "
                "overwritten"
            )
        if self.steps_served != spec.max_steps:
            raise TrialError(
                f"the trainer consumed {self.steps_served} of "
                f"{spec.max_steps} batches"
            )
        if final.step != spec.max_steps:
            raise TrialError(
                f"the final checkpoint is at step {final.step}, not "
                f"max_steps={spec.max_steps}"
            )
        steps = [int(e["step"]) for e in history]
        if steps != list(
            range(spec.val_every, spec.max_steps + 1, spec.val_every)
        ):
            raise TrialError(
                f"validation must run every {spec.val_every} steps; got "
                f"{steps[:8]}"
            )
        metric = spec.metric["name"]
        scores = [e[metric] for e in history]
        best_step = steps[
            int(np.argmax(scores))
        ]  # first argmax: ties go to the earliest step
        if best.step != best_step:
            raise TrialError(
                f"the best checkpoint is at step {best.step}, but the "
                f"first argmax of {metric} is {best_step}"
            )
        if set(val_outputs) != {"best", "final"}:
            raise TrialError("val_outputs needs exactly 'best' and 'final'")
        tasks = self.selection_tasks
        rows = {}
        for name, checkpoint in (("best", best), ("final", final)):
            outputs = list(val_outputs[name])
            recomputed = qa_metrics.val_scores(tasks, outputs)
            reported = history[steps.index(checkpoint.step)]
            for key, value in recomputed.items():
                if key not in reported or not math.isclose(
                    reported[key], value, rel_tol=1e-9, abs_tol=1e-12
                ):
                    raise TrialError(
                        f"{name} outputs give {key}={value:.6f}, the "
                        f"history reports {reported.get(key)}"
                    )
            if not Path(checkpoint.path).is_file():
                raise TrialError(
                    f"the {name} checkpoint {checkpoint.path} does not exist"
                )
            rows[name] = qa_metrics.score_outputs(tasks, outputs)
        if not re.fullmatch(r"[0-9a-f]{64}", init_sha256):
            raise TrialError(
                "init_sha256 must be the sha256 hex digest of the "
                "initial model state"
            )
        exposure = {
            **self.exposure,
            "trainer_token_counts": dict(token_counts or {}),
        }
        _write_json(self.run_dir / EXPOSURE_FILE, exposure)
        self._write_predictions(rows, {"best": best, "final": final})
        result = {
            "status": "complete",
            "schema_version": QA_TRIAL_SCHEMA,
            "trial_id": spec.trial_id,
            "protocol": spec.protocol,
            "arm": spec.arm,
            "supervision": spec.supervision,
            "replicate_seed": spec.replicate_seed,
            "best_step": best.step,
            "best_val": float(scores[steps.index(best.step)]),
            "final_step": final.step,
            "metric": spec.metric,
            "val_history": history,
            "exposure": exposure,
            "trainer": {
                **asdict(trainer),
                "fingerprint": trainer.fingerprint,
                "init_sha256": init_sha256,
            },
            "wall_time_s": float(wall_time_s),
            "provenance": {
                "corpora": spec.corpora,
                "selection": spec.selection,
                "curriculum_sha256": spec.curriculum_sha256,
                "checkpoints": {
                    name: {
                        "path": _relative(self.run_dir, cp.path),
                        "step": cp.step,
                        "sha256": _file_sha256(Path(cp.path)),
                    }
                    for name, cp in (("best", best), ("final", final))
                },
            },
            "env": _environment(),
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        _write_json(result_path, result)
        self.tracker.log_summary(
            {
                "arm": spec.arm,
                "supervision": spec.supervision,
                "seed": spec.replicate_seed,
            },
            {
                "summary/best_val_qa_accuracy": result["best_val"],
                "summary/best_step": best.step,
            },
        )
        self.tracker.close()
        self._open = False
        return result

    def _write_predictions(
        self,
        rows: Mapping[str, list[dict[str, Any]]],
        checkpoints: Mapping[str, Checkpoint],
    ) -> None:
        columns: dict[str, list] = {"checkpoint": [], "step": []}
        for name, block in rows.items():
            for row in block:
                columns["checkpoint"].append(name)
                columns["step"].append(checkpoints[name].step)
                for key, value in row.items():
                    columns.setdefault(key, []).append(value)
        n = len(columns["checkpoint"])
        table = pa.table(
            {
                "trial_id": [self.spec.trial_id] * n,
                "arm": [self.spec.arm] * n,
                "supervision": [self.spec.supervision] * n,
                "seed": [self.spec.replicate_seed] * n,
                **{k: pa.array(v) for k, v in columns.items()},
            }
        )
        pq.write_table(
            table, self.run_dir / PREDICTIONS_FILE, compression="zstd"
        )


def recompute_from_predictions(
    run_dir: str | Path, tasks: Sequence[QATask]
) -> dict[str, dict[str, float]]:
    """Recompute checkpoint scores from saved outputs using the fixed parser."""
    table = pq.read_table(Path(run_dir) / PREDICTIONS_FILE).to_pylist()
    by_id = {t.task_id: t for t in tasks}
    out = {}
    for name in ("best", "final"):
        block = [row for row in table if row["checkpoint"] == name]
        if [row["task_id"] for row in block] != [t.task_id for t in tasks]:
            raise TrialError(
                f"{run_dir}: the {name} predictions do not cover the "
                "selection suite in order"
            )
        out[name] = qa_metrics.val_scores(
            [by_id[r["task_id"]] for r in block], [r["output"] for r in block]
        )
    return out


def _relative(run_dir: Path, path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(run_dir.resolve()))
    except ValueError:
        return str(path)


@dataclass
class QAArmConfig:
    corpora: list[str] = field(
        default_factory=list
    )  # labels of the protocol's corpora


@dataclass
class QAProtocolConfig:
    """Shared training budget and sampling schedule."""

    id: str = "ecgqa-cot-qa"
    corpora: dict[str, str] = field(
        default_factory=dict
    )  # label -> published corpus path
    selection: str = ""  # published selection suite path
    arms: dict[str, QAArmConfig] = field(default_factory=dict)
    supervision: list[str] = field(default_factory=lambda: ["rationale"])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    batch_size: int = 16
    max_steps: int = 15531
    val_every: int = 1000
    curriculum: dict[str, Any] = field(
        default_factory=lambda: {
            "name": "uniform-passes@1",
            "seed": 0,
            "stages": [{"name": "all"}],
        }
    )
    decoding: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)


def plan_qa_trials(
    protocol: QAProtocolConfig,
    runs_dir: str | Path,
    *,
    arms: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[Path]:
    """Validate collections and save the planned trial specifications.

    Create one trial.json for each treatment, supervision mode, and random
    seed.
    """
    from syntide.generate.qa.corpus import load_corpus
    from syntide.generate.qa.curriculum import (
        Curriculum,
        CurriculumConfig,
        StageConfig,
    )
    from syntide.generate.qa.verify import CONTEXT_POLICY

    loaded = {
        label: load_corpus(path) for label, path in protocol.corpora.items()
    }
    selection = load_corpus(protocol.selection)
    written = []
    entries = []
    for arm in arms or list(protocol.arms):
        labels = protocol.arms[arm].corpora
        unknown = set(labels) - set(loaded)
        if not labels or unknown:
            raise TrialError(
                f"arm {arm}: corpora {sorted(unknown) or labels} are not "
                "in the protocol"
            )
        corpora = {label: loaded[label] for label in labels}
        check_trial_inputs(corpora, selection)
        for supervision in protocol.supervision:
            stages = [StageConfig(**s) for s in protocol.curriculum["stages"]]
            cfg = CurriculumConfig(**{**protocol.curriculum, "stages": stages})
            curriculum = Curriculum(
                cfg,
                {label: c.tasks for label, c in corpora.items()},
                batch_size=protocol.batch_size,
                max_steps=protocol.max_steps,
                supervision=supervision,
            )
            for seed in seeds or protocol.seeds:
                spec = make_qa_spec(
                    protocol=protocol.id,
                    arm=arm,
                    supervision=supervision,
                    replicate_seed=seed,
                    corpora=[
                        {
                            "label": label,
                            "corpus_id": c.corpus_id,
                            "path": str(c.path),
                        }
                        for label, c in corpora.items()
                    ],
                    selection={
                        "corpus_id": selection.corpus_id,
                        "path": str(selection.path),
                    },
                    curriculum={
                        **protocol.curriculum,
                        "stages": [asdict(s) for s in stages],
                    },
                    curriculum_sha256=curriculum.sha256,
                    batch_size=protocol.batch_size,
                    max_steps=protocol.max_steps,
                    val_every=protocol.val_every,
                    metric={
                        "name": qa_metrics.SELECTION_METRIC,
                        "direction": qa_metrics.METRIC_DIRECTION,
                        "tie_break": qa_metrics.TIE_BREAK,
                        "answerability": qa_metrics.ANSWERABILITY_POLICY,
                        "numeric_tolerance": qa_metrics.TOLERANCE_POLICY,
                    },
                    parser=qa_metrics.PARSER,
                    decoding=dict(protocol.decoding),
                    context_policy=CONTEXT_POLICY,
                    budget=dict(protocol.budget),
                )
                run_dir = (
                    Path(runs_dir)
                    / protocol.id
                    / arm
                    / supervision
                    / f"seed{seed}"
                )
                if write_qa_spec(run_dir, spec):
                    written.append(run_dir)
                entries.append(
                    {
                        "run_dir": str(
                            run_dir.relative_to(Path(runs_dir) / protocol.id)
                        ),
                        "trial_id": spec.trial_id,
                        "arm": arm,
                        "supervision": supervision,
                        "seed": seed,
                    }
                )
    plan_path = Path(runs_dir) / protocol.id / "plan.json"
    existing = (
        json.loads(plan_path.read_text())["trials"]
        if plan_path.exists()
        else []
    )
    merged = {e["run_dir"]: e for e in [*existing, *entries]}
    _write_json(
        plan_path,
        {
            "protocol": protocol.id,
            "schema_version": QA_TRIAL_SCHEMA,
            "trials": [merged[k] for k in sorted(merged)],
        },
    )
    return written
