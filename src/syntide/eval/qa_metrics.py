"""Parse and score question-answer model outputs consistently.

Parser ``strict-answer-line@1`` (the RunPod baseline's primary parser): the
last case-insensitive ``Answer:`` marker; the rest of that line is
lower-cased, stripped of punctuation (decimal points in numbers are kept)
and whitespace-collapsed, then matched exactly to a declared option.
Anything else is invalid, and an invalid output is wrong and stays in every
denominator. Text before the marker (the rationale) never counts: naming the
right label in a rationale does not rescue a wrong final answer.

Correctness by answer type: options must match exactly; sets (multi-choice,
lead sets; ``none`` is the empty set) must be equal; numbers must be within
the task's frozen tolerance (``|pred - answer| <= max(abs, rel *
|answer|)``). An unanswerable task is right only when the answer is
``insufficient evidence``; that answer on an answerable task is wrong.
Answerability is scored as its own binary task (abstention precision and
recall) as well.

Rationales are scored separately (``rationale_contradictions``): a number
with a unit (bpm, millivolts, ms, s) that no cited evidence supports within
tolerance, or a stated conclusion that differs from the final answer, is a
checkable contradiction. Answer metrics never look at it.
"""

from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from syntide.data.qa import INSUFFICIENT, Evidence, QATask

PARSER = "strict-answer-line@1"
SELECTION_METRIC = "val/qa_accuracy"
METRIC_DIRECTION = "max"
TIE_BREAK = "earliest"
ANSWERABILITY_POLICY = (
    "insufficient-evidence-is-an-answer@1"  # right only on unanswerable tasks
)
TOLERANCE_POLICY = (
    "per-task-frozen-in-corpus@1"  # |pred - answer| <= max(abs, rel * |answer|)
)
_MARKER = re.compile(r"answer\s*:", re.I)
_PUNCT = str.maketrans(
    "", "", "".join(c for c in string.punctuation if c not in ".-")
)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_UNIT_NUMBER = re.compile(r"(-?\d+(?:\.\d+)?)\s*(bpm|mv|ms|s)\b", re.I)


def normalize(text: str) -> str:
    text = text.lower().translate(_PUNCT)
    text = re.sub(
        r"(?<!\d)\.|\.(?!\d)", " ", text
    )  # sentence periods go; decimal points stay
    return " ".join(text.split())


@dataclass(frozen=True)
class Parsed:
    # the text after the last Answer: marker (first line), or "" without one
    raw: str
    value: Any  # the parsed answer, or None when invalid
    valid: bool


def parse(output: str, task: QATask) -> Parsed:
    matches = list(_MARKER.finditer(output or ""))
    if not matches:
        return Parsed("", None, False)
    raw = (
        output[matches[-1].end() :].strip().splitlines()[0].strip()
        if output[matches[-1].end() :].strip()
        else ""
    )
    text = normalize(raw)
    if text == normalize(INSUFFICIENT):
        return Parsed(raw, INSUFFICIENT, True)
    kind = (
        task.answer_type
        if task.answer_type != "insufficient"
        else _native_type(task)
    )
    options = {normalize(o): o for o in task.options}
    if kind in ("binary", "choice"):
        return Parsed(raw, options.get(text), text in options)
    if kind in ("multi_choice", "lead_set"):
        if text == "none":
            return Parsed(raw, [], True)
        parts = [
            normalize(p) for p in re.split(r",|\band\b", raw) if normalize(p)
        ]
        values = [options.get(p) for p in parts]
        if not parts or any(v is None for v in values):
            return Parsed(raw, None, False)
        return Parsed(raw, sorted(set(values), key=task.options.index), True)
    if kind == "numeric":
        found = _NUMBER.search(text)
        return Parsed(
            raw, float(found.group()) if found else None, found is not None
        )
    return Parsed(raw, None, False)


def _native_type(task: QATask) -> str:
    """Determine the expected answer format from the question options."""
    if not task.options:
        return "numeric"
    return "binary" if set(task.options) == {"yes", "no"} else "choice"


def is_correct(parsed: Parsed, task: QATask) -> bool:
    if not parsed.valid:
        return False
    if task.answerability != "answerable":
        return parsed.value == INSUFFICIENT
    if parsed.value == INSUFFICIENT:
        return False
    if task.answer_type == "numeric":
        tol = task.tolerance or {"abs": 0.0, "rel": 0.0}
        return abs(parsed.value - float(task.answer)) <= max(
            tol.get("abs", 0.0), tol.get("rel", 0.0) * abs(float(task.answer))
        )
    if task.answer_type in ("multi_choice", "lead_set"):
        return set(parsed.value) == set(task.answer)
    return parsed.value == task.answer


def score_outputs(
    tasks: Sequence[QATask], outputs: Sequence[str]
) -> list[dict[str, Any]]:
    """Parse model outputs and measure correctness for each task."""
    if len(tasks) != len(outputs):
        raise ValueError(f"{len(outputs)} outputs for {len(tasks)} tasks")
    rows = []
    for task, output in zip(tasks, outputs, strict=True):
        parsed = parse(output, task)
        error = None
        if (
            task.answer_type == "numeric"
            and parsed.valid
            and parsed.value != INSUFFICIENT
        ):
            error = abs(float(parsed.value) - float(task.answer))
        rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "composition": task.composition,
                "answerability": task.answerability,
                "answer_type": task.answer_type,
                "answer": task.answer_text,
                "output": output,
                "parsed": parsed.raw,
                "valid": parsed.valid,
                "correct": is_correct(parsed, task),
                "abstained": parsed.value == INSUFFICIENT,
                "abs_error": error,
                "patients": "|".join(
                    sorted(
                        {
                            s
                            for ref in task.inputs
                            for s in ref.root_subject_uids
                        }
                    )
                ),
                "origin": task.inputs[0].origin,
                "context_reveals_answer": bool(
                    task.verification.get("context_reveals_answer")
                ),
            }
        )
    return rows


def summarize(
    rows: Sequence[Mapping[str, Any]], tasks: Mapping[str, QATask] | None = None
) -> dict[str, Any]:
    """Summarize accuracy, answer validity, and numerical errors."""
    if not rows:
        raise ValueError("no rows to summarize")
    n = len(rows)
    by_family: dict[str, list] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    family_accuracy = {
        f: float(np.mean([r["correct"] for r in rs]))
        for f, rs in by_family.items()
    }
    out: dict[str, Any] = {
        "n": n,
        "accuracy": float(np.mean([r["correct"] for r in rows])),
        "family_macro_accuracy": float(np.mean(list(family_accuracy.values()))),
        "invalid_rate": float(np.mean([not r["valid"] for r in rows])),
        "n_families": len(by_family),
    }
    numeric = [r["abs_error"] for r in rows if r["abs_error"] is not None]
    numeric_rows = [r for r in rows if r["answer_type"] == "numeric"]
    if numeric_rows:
        out["numeric_within_tolerance"] = float(
            np.mean([r["correct"] for r in numeric_rows])
        )
        out["numeric_median_abs_error"] = (
            float(np.median(numeric)) if numeric else None
        )
        out["numeric_parse_failures"] = sum(
            not r["valid"] for r in numeric_rows
        )
    truth = np.array([r["answerability"] != "answerable" for r in rows])
    said = np.array([r["abstained"] for r in rows])
    if truth.any():
        out["abstention_recall"] = float((said & truth).sum() / truth.sum())
        out["abstention_precision"] = (
            float((said & truth).sum() / said.sum()) if said.any() else 0.0
        )
    if tasks is not None:
        out["family_macro_f1"] = _family_macro_f1(rows, tasks)
    return out


def _family_macro_f1(
    rows: Sequence[Mapping[str, Any]], tasks: Mapping[str, QATask]
) -> float | None:
    """Average class precision-recall scores across question families."""
    scores = []
    by_family: dict[str, list] = defaultdict(list)
    for row in rows:
        task = tasks[row["task_id"]]
        if (
            task.answer_type in ("binary", "choice")
            and task.answerability == "answerable"
        ):
            by_family[row["family"]].append((task, row))
    for members in by_family.values():
        classes = sorted({o for task, _ in members for o in task.options})
        tp, fp, fn = Counter(), Counter(), Counter()
        for task, row in members:
            predicted = parse(row["output"], task).value
            if predicted == task.answer:
                tp[task.answer] += 1
            else:
                fn[task.answer] += 1
                if predicted in classes:
                    fp[predicted] += 1
        f1 = [
            2 * tp[c] / (2 * tp[c] + fp[c] + fn[c])
            if tp[c] + fp[c] + fn[c]
            else 0.0
            for c in classes
        ]
        scores.append(float(np.mean(f1)))
    return float(np.mean(scores)) if scores else None


def val_scores(
    tasks: Sequence[QATask], outputs: Sequence[str]
) -> dict[str, float]:
    """Compute the validation scores used for checkpoint selection."""
    summary = summarize(score_outputs(tasks, outputs))
    return {
        SELECTION_METRIC: summary["accuracy"],
        "val/qa_family_macro_accuracy": summary["family_macro_accuracy"],
        "val/qa_invalid_rate": summary["invalid_rate"],
    }


def rationale_contradictions(
    output: str, task: QATask, evidence: Mapping[str, Evidence]
) -> list[str]:
    """Find evidence contradictions before the final answer line."""
    matches = list(_MARKER.finditer(output or ""))
    body = output[: matches[-1].start()] if matches else (output or "")
    final = parse(output, task)
    problems = []
    cited = [evidence[e] for e in task.required_evidence if e in evidence]
    units = {"bpm": "bpm", "mv": "mV", "ms": "ms", "s": "s"}
    for number, unit in _UNIT_NUMBER.findall(body):
        same_unit = [
            e
            for e in cited
            if e.units == units[unit.lower()]
            and isinstance(e.value, int | float)
            and not isinstance(e.value, bool)
        ]
        if same_unit and not any(
            abs(float(number) - e.value) <= max(0.05 * abs(e.value), 0.05)
            for e in same_unit
        ):
            problems.append(f"unsupported_number:{number}{unit}")
    conclusion = r"\b(?:answer is|therefore|so the answer is)\s+"
    for option in task.options:
        stated = re.search(
            conclusion + re.escape(option.lower()) + r"\b", body.lower()
        )
        if stated and final.valid and final.value != option:
            problems.append(f"conclusion_differs:{option}")
    return problems


def cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """Estimate accuracy uncertainty by resampling whole patients."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        groups[row["patients"]].append(bool(row["correct"]))
    keys = sorted(groups)
    sums = np.array([sum(groups[k]) for k in keys], dtype=float)
    counts = np.array([len(groups[k]) for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(
        len(keys), np.full(len(keys), 1 / len(keys)), size=n_boot
    )
    stats = (draws @ sums) / (draws @ counts)
    alpha = (1 - level) / 2
    return {
        "estimate": float(sums.sum() / counts.sum()),
        "low": float(np.quantile(stats, alpha)),
        "high": float(np.quantile(stats, 1 - alpha)),
        "clusters": len(keys),
        "n_boot": n_boot,
    }


def paired_difference(
    a: Sequence[Mapping[str, Any]],
    b: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """Compare accuracy with uncertainty from patient resampling."""
    by_b = {row["task_id"]: row for row in b}
    if {row["task_id"] for row in a} != set(by_b):
        raise ValueError("paired QA contrasts need the same tasks in both arms")
    diff: dict[str, list[float]] = defaultdict(list)
    for row in a:
        diff[row["patients"]].append(
            float(row["correct"]) - float(by_b[row["task_id"]]["correct"])
        )
    keys = sorted(diff)
    sums = np.array([sum(diff[k]) for k in keys])
    counts = np.array([len(diff[k]) for k in keys], dtype=float)
    draws = np.random.default_rng(seed).multinomial(
        len(keys), np.full(len(keys), 1 / len(keys)), size=n_boot
    )
    stats = (draws @ sums) / (draws @ counts)
    alpha = (1 - level) / 2
    return {
        "estimate": float(sums.sum() / counts.sum()),
        "low": float(np.quantile(stats, alpha)),
        "high": float(np.quantile(stats, 1 - alpha)),
        "clusters": len(keys),
        "n_boot": n_boot,
    }
