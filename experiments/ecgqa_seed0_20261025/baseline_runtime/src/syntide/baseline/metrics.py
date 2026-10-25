"""Dependency-free ECG-QA scoring and training-pool majority references.

F1 aggregation follows OpenTSLM 2968f4b's ECG-QA scorer: class macro-F1
within each template (including every declared option, absent classes = 0),
then either an unweighted or sample-weighted mean across templates. The
primary strict answer parser deliberately follows our frozen protocol rather
than upstream's more permissive parser. Every input row stays in the denominator.
"""

from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

UPSTREAM_COMMIT = "2968f4b"
PAPER_REFERENCE = {
    "model": "OpenTSLM-Flamingo Gemma3-270M",
    "f1_percent": 32.71,
    "accuracy_percent": 35.50,
    "table": 2,
    "page": 6,
}
PARSER_DESCRIPTIONS = {
    "strict": "Last case-insensitive Answer: marker required; lowercase, strip punctuation and collapse "
    "whitespace; exact match to a declared template option; invalid predictions count as wrong.",
    "upstream": "OpenTSLM 2968f4b: last case-sensitive 'Answer: ' marker, or entire output when absent; "
    "remove special EOS tokens, lowercase, and strip trailing punctuation (.,!?;:).",
}


def _normalize(text: str, mode: str) -> str:
    text = text.lower().strip()
    if mode == "strict":
        return " ".join(text.translate(str.maketrans("", "", string.punctuation)).split())
    if mode == "upstream":
        return text.rstrip(".,!?;:")
    raise ValueError(f"Unknown answer parser: {mode}")


def _options(options: Sequence[str], mode: str) -> dict[str, str]:
    if isinstance(options, str) or not options:
        raise ValueError("Template answer options must be a non-empty sequence of strings")
    result = {}
    for option in options:
        if not isinstance(option, str):
            raise ValueError("Each template answer option must be a string")
        # Upstream leaves punctuation on option labels, unlike prediction/gold normalization.
        normalized = _normalize(option, mode) if mode == "strict" else option.lower().strip()
        if not normalized or normalized in result:
            raise ValueError(f"Empty or ambiguous normalized template option: {option!r}")
        result[normalized] = option
    return result


def parse_answer(text: str, answer_options: Sequence[str], *, mode: str = "strict") -> str | None:
    """Return a normalized supported answer, or None for an invalid output."""
    options = _options(answer_options, mode)
    if not isinstance(text, str):
        return None
    if mode == "strict":
        markers = list(re.finditer(r"\banswer\s*:\s*", text, flags=re.IGNORECASE))
        if not markers:
            return None
        candidate = text[markers[-1].end() :].strip()
        candidate = re.sub(r"<\|.*?\|>|<eos>\s*$", "", candidate).strip()
    elif mode == "upstream":
        candidate = text
        if "Answer: " in text:
            candidate = text.split("Answer: ")[-1].strip()
            candidate = re.sub(r"<\|.*?\|>|<eos>$", "", candidate).strip()
            candidate = re.sub(r"\.$", "", candidate).strip()
    else:
        raise ValueError(f"Unknown answer parser: {mode}")
    normalized = _normalize(candidate, mode)
    return normalized if normalized in options else None


def _row_options(row: Mapping[str, Any]) -> Sequence[str]:
    return row.get("answer_options", row.get("possible_answers", []))


def _gold(row: Mapping[str, Any], options: Mapping[str, str], parser: str) -> str:
    answer = row.get("answer", row.get("target_answer"))
    if not isinstance(answer, str):
        raise ValueError("ECG-QA baseline requires one string answer per question")
    normalized = _normalize(answer, parser)
    if normalized not in options:
        raise ValueError(f"Ground truth {answer!r} is outside template {row['template_id']} options")
    return normalized


def _summarize(points: list[dict]) -> dict:
    grouped = defaultdict(list)
    for point in points:
        grouped[point["template_id"]].append(point)
    templates = {}
    for template, rows in sorted(grouped.items()):
        options = rows[0]["options"]
        classes = {}
        for label in options:
            tp = sum(p["gold"] == label and p["prediction"] == label for p in rows)
            fp = sum(p["gold"] != label and p["prediction"] == label for p in rows)
            fn = sum(p["gold"] == label and p["prediction"] != label for p in rows)
            classes[label] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            }
        n = len(rows)
        invalid = sum(p["prediction"] is None for p in rows)
        templates[template] = {
            "n": n,
            "accuracy": sum(p["prediction"] == p["gold"] for p in rows) / n,
            "macro_f1": sum(c["f1"] for c in classes.values()) / len(classes),
            "invalid_n": invalid,
            "invalid_rate": invalid / n,
            "per_class": classes,
        }
    n = len(points)
    invalid = sum(p["prediction"] is None for p in points)
    return {
        "n": n,
        "n_templates": len(templates),
        "accuracy": sum(p["prediction"] == p["gold"] for p in points) / n if n else None,
        "template_macro_accuracy": (
            sum(t["accuracy"] for t in templates.values()) / len(templates) if templates else None
        ),
        "template_macro_f1": (
            sum(t["macro_f1"] for t in templates.values()) / len(templates) if templates else None
        ),
        "weighted_template_macro_f1": (
            sum(t["macro_f1"] * t["n"] for t in templates.values()) / n if n else None
        ),
        "invalid_n": invalid,
        "invalid_rate": invalid / n if n else None,
        "invalid_percent": 100 * invalid / n if n else None,
        "per_template": templates,
    }


def score_predictions(records: Iterable[Mapping[str, Any]], *, parser: str = "strict") -> dict:
    """Score distinct exported questions, retaining invalid outputs as errors.

    Required row fields: template_id, answer, answer_options, output_text,
    is_artifact. Aliases possible_answers, generated_answer, and noisy are
    accepted. The exporter owns question deduplication; this scorer never
    silently removes rows. All accuracy/F1 values are fractions in [0, 1].
    Empty noisy/clean slices report null metrics, never a fabricated zero.
    """
    if parser not in PARSER_DESCRIPTIONS:
        raise ValueError(f"Unknown answer parser: {parser}")
    points = []
    template_options = {}
    for row in records:
        template = str(row["template_id"])
        options = _options(_row_options(row), parser)
        if template in template_options and set(template_options[template]) != set(options):
            raise ValueError(f"Inconsistent answer options for template {template}")
        template_options[template] = options
        noisy = row.get("is_artifact", row.get("noisy"))
        if not isinstance(noisy, bool):
            raise ValueError("Every prediction needs a boolean is_artifact/noisy flag")
        output = row.get("output_text", row.get("generated_answer", row.get("generated")))
        points.append(
            {
                "template_id": template,
                "options": options,
                "gold": _gold(row, options, parser),
                "prediction": parse_answer(output, _row_options(row), mode=parser),
                "noisy": noisy,
            }
        )
    if not points:
        raise ValueError("Cannot score an empty prediction set")
    return {
        "parser": parser,
        "parser_description": PARSER_DESCRIPTIONS[parser],
        "f1_definition": "Macro over all declared answer options within each template; "
        "then unweighted or sample-weighted mean across templates; absent classes score zero.",
        "aggregation_source": (
            f"OpenTSLM@{UPSTREAM_COMMIT}/evaluation/opentslm/ecg_qa_cot/parse_ecg_qa_cot_data.py"
        ),
        "groups": {
            "all": _summarize(points),
            "noisy": _summarize([p for p in points if p["noisy"]]),
            "clean": _summarize([p for p in points if not p["noisy"]]),
        },
    }


def fit_majority_answers(train_records: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Fit per-template answer priors on training rows only; resolve ties lexically.

    Preserves rationale-row weighting, matching training exposure. Explicit
    validation/test folds are rejected as a guard against fitting on holdout.
    """
    counts = defaultdict(Counter)
    for row in train_records:
        if "strat_fold" in row and int(row["strat_fold"]) not in range(1, 9):
            raise ValueError("Majority reference may only fit on PTB-XL training folds 1-8")
        if row.get("role", "train") not in {"train", "pool"}:
            raise ValueError("Majority reference may only fit on training records")
        options = _options(_row_options(row), "strict")
        answer = _gold(row, options, "strict")
        counts[str(row["template_id"])][answer] += 1
    if not counts:
        raise ValueError("Cannot fit a majority reference without training records")
    return {template: min(c, key=lambda label: (-c[label], label)) for template, c in sorted(counts.items())}


def majority_predictions(records: Iterable[Mapping[str, Any]], majority: Mapping[str, str]) -> list[dict]:
    """Produce ordinary prediction rows scored with the same strict parser."""
    result = []
    for row in records:
        template = str(row["template_id"])
        if template not in majority:
            raise ValueError(f"No training-pool majority answer for template {template}")
        answer = majority[template]
        options = _options(_row_options(row), "strict")
        if answer not in options:
            raise ValueError(f"Training majority answer is not an option for template {template}")
        result.append(
            {**row, "output_text": f"Answer: {options[answer]}", "predictor": "train_template_majority"}
        )
    return result
