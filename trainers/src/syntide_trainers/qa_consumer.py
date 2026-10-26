"""Exercise question-answer trial training with a small offline model.

For categorical questions, learn the most frequent answer by question family
and wording. For numeric questions, fit a least-squares model using four
waveform measurements: duration, a repeating-pattern rate, slow-wave
amplitude, and robust signal spread. Training reads only the batches
supplied by Syntide.

Evaluate the selection collection every val_every steps and keep the
earliest checkpoint with the highest accuracy. Save the final model at the
last step. Checkpoints are JSON files. This NumPy implementation checks the
training interface; it is not the model used for the reported experiment.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from syntide.eval import qa_metrics
from syntide.eval.tracking import TrackerConfig
from syntide.qa_trial import QAExample, QATrial, is_qa_complete, prepare_qa
from syntide.trial import Checkpoint, TrainerInfo, TrialError
from syntide_trainers import __version__

log = logging.getLogger(__name__)
NAME = "syntide_trainers.qa_consumer"
PREDICTOR = f"{NAME}:predict"
_ANSWER = re.compile(r"answer\s*:\s*(.*)\s*$", re.I | re.S)


def features(signal: np.ndarray, sample_rate: float) -> list[float]:
    x = np.asarray(signal, dtype=np.float64)
    n = x.shape[-1]
    energy = np.abs(np.diff(x, axis=-1)).mean(axis=0)
    energy = energy - energy.mean()
    lo, hi = int(0.3 * sample_rate), min(int(2.0 * sample_rate), n - 2)
    rate = 0.0
    if hi > lo and np.any(energy):
        ac = np.correlate(energy, energy, mode="full")[n - 1 :]
        lag = lo + int(np.argmax(ac[lo:hi]))
        rate = 60.0 * sample_rate / lag
    width = max(1, int(sample_rate))
    slow = np.stack(
        [np.convolve(lead, np.ones(width) / width, mode="same") for lead in x]
    )
    return [
        1.0,
        n / sample_rate,
        rate,
        float((0.5 * np.ptp(slow, axis=-1)).max()),
        float(
            (
                1.4826
                * np.median(
                    np.abs(x - np.median(x, axis=-1, keepdims=True)), axis=-1
                )
            ).max()
        ),
    ]


def _answer(target: str) -> str:
    found = _ANSWER.search(target.rsplit("\n", 1)[-1]) or _ANSWER.search(target)
    return found.group(1).strip() if found else ""


class Model:
    def __init__(self) -> None:
        self.answers: dict[str, Counter] = defaultdict(Counter)
        self.numeric: dict[str, list[list[float]]] = defaultdict(list)

    def update(self, example: QAExample) -> None:
        answer = _answer(example.target)
        key = f"{example.family}\x00{example.student['question']}"
        self.answers[key][answer] += 1
        self.answers[example.family][answer] += 1
        if not example.student["options"] and re.fullmatch(
            r"-?\d+(\.\d+)?", answer
        ):
            spec = example.student["inputs"][0]["signal"]
            self.numeric[example.family].append(
                [
                    *features(
                        example.student["signals"][0], spec["sample_rate"]
                    ),
                    float(answer),
                ]
            )

    def state(self) -> dict:
        coef = {}
        for family, rows in self.numeric.items():
            data = np.asarray(rows)
            coef[family] = (
                np.linalg.lstsq(data[:, :-1], data[:, -1], rcond=None)[
                    0
                ].tolist()
                if len(rows) >= 2
                else None
            )
        return {
            "answers": {k: dict(v) for k, v in sorted(self.answers.items())},
            "coef": coef,
        }


def predict_state(state: dict, examples: Sequence[QAExample]) -> list[str]:
    outputs = []
    for ex in examples:
        votes = (
            state["answers"].get(f"{ex.family}\x00{ex.student['question']}")
            or state["answers"].get(ex.family)
            or {}
        )
        coef = state["coef"].get(ex.family)
        if (
            not ex.student["options"]
            and coef is not None
            and not (
                votes and max(votes, key=votes.get) == "insufficient evidence"
            )
        ):
            spec = ex.student["inputs"][0]["signal"]
            value = float(
                np.dot(
                    coef,
                    features(ex.student["signals"][0], spec["sample_rate"]),
                )
            )
            outputs.append(f"Answer: {round(value, 3)}")
        elif votes:
            outputs.append(
                "Answer: "
                f"{sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]}"
            )
        else:
            outputs.append("Answer: insufficient evidence")
    return outputs


def _save(state: dict, path: Path, step: int, trial_id: str) -> Checkpoint:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(
            {"state": state, "step": step, "trial_id": trial_id}, sort_keys=True
        )
        + "\n"
    )
    os.replace(partial, path)
    return Checkpoint(path, step)


def predict(
    checkpoint: Checkpoint | str | Path,
    examples: Sequence[QAExample],
    device: str,
) -> list[str]:
    """The ``QAPredictor`` syntide calls for held-out suites."""
    path = checkpoint.path if isinstance(checkpoint, Checkpoint) else checkpoint
    return predict_state(json.loads(Path(path).read_text())["state"], examples)


def trainer_info() -> TrainerInfo:
    return TrainerInfo(
        name=NAME,
        version=__version__,
        config={"model": "family-majority+linear-numeric"},
        device="cpu",
        predictor=PREDICTOR,
    )


def fit(trial: QATrial, info: TrainerInfo) -> dict:
    spec, started = trial.spec, time.monotonic()
    model = Model()
    init_sha256 = hashlib.sha256(
        json.dumps(model.state(), sort_keys=True).encode()
    ).hexdigest()
    selection = trial.selection_examples()
    tasks = trial.selection_tasks
    history, best, best_score, best_outputs, outputs = (
        [],
        None,
        -np.inf,
        None,
        None,
    )
    tokens = Counter()
    for batch in trial.batches():
        for ex in batch.examples:
            model.update(ex)
            tokens["target_whitespace"] += len(ex.target.split())
        if batch.step % spec.val_every == 0:
            state = model.state()
            outputs = predict_state(state, selection)
            scores = qa_metrics.val_scores(tasks, outputs)
            history.append({"step": batch.step, **scores})
            trial.tracker.log_scalars(scores, batch.step)
            if scores[qa_metrics.SELECTION_METRIC] > best_score:
                best_score, best_outputs = (
                    scores[qa_metrics.SELECTION_METRIC],
                    outputs,
                )
                best = _save(
                    state,
                    trial.run_dir / "best.json",
                    batch.step,
                    spec.trial_id,
                )
    final = _save(
        model.state(),
        trial.run_dir / "final.json",
        spec.max_steps,
        spec.trial_id,
    )
    return trial.finish(
        info,
        best=best,
        final=final,
        val_history=history,
        val_outputs={"best": best_outputs, "final": outputs},
        init_sha256=init_sha256,
        wall_time_s=time.monotonic() - started,
        token_counts=dict(tokens),
    )


def train(run_dir: str | Path, tracker: TrackerConfig) -> dict | None:
    info = trainer_info()
    if is_qa_complete(run_dir, info):
        return None
    with prepare_qa(run_dir, tracker=tracker) as trial:
        return fit(trial, info)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m syntide_trainers.qa_consumer",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--tracker", default="conf/tracker/files.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
    tracker = TrackerConfig.from_yaml(args.tracker)
    dirs = [
        Path(m)
        for p in args.run_dirs
        for m in (sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
        if (Path(m) / "trial.json").is_file()
    ]
    for n, run_dir in enumerate(dirs, 1):
        try:
            result = train(run_dir, tracker)
        except TrialError as error:
            raise SystemExit(
                f"[{n}/{len(dirs)}] {run_dir}: refused: {error}"
            ) from error
        log.info(
            "[%d/%d] %s: %s",
            n,
            len(dirs),
            run_dir,
            "already complete"
            if result is None
            else (
                f"best val/qa_accuracy {result['best_val']:.4f} at "
                f"{result['best_step']}"
            ),
        )


if __name__ == "__main__":
    main()
