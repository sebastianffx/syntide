"""Check whether artifact measurements match the requested settings.

Both sampling policies use the same source recordings and random draws. By
default, each policy creates 1,000 candidates. For each continuous setting,
measured values are regressed on requested values. Acceptance requires a slope
between 0.9 and 1.1, an intercept within 5 percent of the setting's range, mean
absolute error within 10 percent of that range, and correlation of at least
0.9.
Every candidate must also contain finite values and keep the original shape.

The check saves all candidates and a review sample of 40 accepted recipes,
selected from the lowest and highest artifact severities. The review sample
includes an empty verdict column for expert assessment.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from syntide.data.schema import Samples
from syntide.generate import descriptors, editor
from syntide.generate.editor import (
    ARTIFACTS,
    CONTROLS,
    Recipe,
    control_range,
    edit,
)
from syntide.generate.ledger import LEDGER_SCHEMA, ledger_row
from syntide.generate.policy import POLICIES, U_FIELDS, controls_for
from syntide.generate.stream import rng
from syntide.generate.validate import accepted, evaluate, failed, in_context

NAME = "generator_gate@1"
RECIPES_PER_POLICY = 1000
REVIEW_SAMPLE = 40
# fewer candidates than this for a (type, control) are reported but not judged
MIN_CELL = 20
THRESHOLDS = {
    "slope": [0.9, 1.1],
    "intercept_of_range": 0.05,
    "mae_of_range": 0.10,
    "corr": 0.9,
}
GATE_SCHEMA = LEDGER_SCHEMA.append(pa.field("policy", pa.string()))


def control_fidelity(
    requested: np.ndarray, realized: np.ndarray, low: float, high: float
) -> dict[str, Any]:
    """Check agreement between requested and measured values."""
    requested, realized = (
        np.asarray(requested, dtype=float),
        np.asarray(realized, dtype=float),
    )
    span = high - low
    if (
        len(requested) < MIN_CELL
        or np.ptp(requested) == 0
        or not np.isfinite(realized).all()
    ):
        finite = bool(np.isfinite(realized).all())
        return {
            "n": len(requested),
            "passed": None if finite else False,
            "note": "too few or non-finite",
        }
    slope, intercept = np.polyfit(requested, realized, 1)
    corr = float(np.corrcoef(requested, realized)[0, 1])
    mae = float(np.mean(np.abs(realized - requested)))
    passed = (
        THRESHOLDS["slope"][0] <= slope <= THRESHOLDS["slope"][1]
        and abs(intercept) <= THRESHOLDS["intercept_of_range"] * span
        and mae <= THRESHOLDS["mae_of_range"] * span
        and corr >= THRESHOLDS["corr"]
    )
    return {
        "n": len(requested),
        "slope": float(slope),
        "intercept_of_range": float(intercept / span),
        "mae_of_range": mae / span,
        "corr": corr,
        "passed": bool(passed),
    }


def run_gate(
    parents: Samples,
    calibration: dict[str, Any],
    calibration_sha256: str,
    cfg: dict[str, Any],
    fs: int,
    *,
    recipes: int = RECIPES_PER_POLICY,
    seed: int = 0,
) -> tuple[dict[str, Any], pa.Table]:
    """Validate both artifact sampling policies against the requested settings.

    Args:
        parents: Allowed source recordings for the validation candidates.
        calibration: Measured training coverage and sampling distributions.
        calibration_sha256: Checksum identifying the calibration file.
        cfg: Editor configuration.
        fs: Sampling frequency in hertz.
        recipes: Number of candidates to draw for each policy.
        seed: Seed shared by both sampling policies.

    Returns:
        The validation report and a table of all candidate generation records.
    """
    n_samples = parents.x.shape[-1]
    envelope, record_s = calibration["envelope"], n_samples / fs
    rows, results = [], {}
    for policy in POLICIES:
        cells: dict[tuple[str, str], tuple[list, list]] = defaultdict(
            lambda: ([], [])
        )
        kinds, rejections, n_accepted, finite = Counter(), Counter(), 0, True
        for i in range(recipes):
            draw = rng(
                seed, "gate", i
            )  # Pair the sampling policies by recipe index.
            parent = int(draw.integers(len(parents)))
            u, seed_key = draw.random(len(U_FIELDS)), int(draw.integers(2**63))
            controls = controls_for(policy, u, calibration, n_samples, fs)
            recipe = Recipe(
                str(parents.record_id[parent]),
                editor.GENERATOR,
                editor.VERSION,
                controls,
                seed_key,
                calibration["gap_report_id"],
            )
            candidate = edit(parents.x[parent], recipe, fs)
            checks, realized = evaluate(
                candidate, parents.x[parent], recipe, cfg, fs, full=True
            )
            checks.append(in_context(candidate, recipe, fs))
            finite &= bool(np.isfinite(candidate.x).all())
            ok = accepted(checks)
            n_accepted += ok
            kinds[controls["artifact_type"]] += 1
            if not ok:
                rejections[failed(checks)] += 1
            for name in CONTROLS[controls["artifact_type"]]:
                cells[(controls["artifact_type"], name)][0].append(
                    controls[name]
                )
                cells[(controls["artifact_type"], name)][1].append(
                    realized[name]
                )
            rows.append(
                ledger_row(
                    recipe=recipe,
                    step=i,
                    slot=0,
                    attempt=0,
                    rng_stream=f"gate/{policy}/recipe{i}",
                    parent_subject_uid=str(parents.subject_uid[parent]),
                    label_effect=editor.LABEL_EFFECT,
                    controls_realized=realized,
                    validations=[check.to_json() for check in checks],
                    decision="accepted" if ok else "rejected",
                    reason=failed(checks),
                )
                | {"policy": policy}
            )
        controls_report = {
            f"{kind}.{name}": control_fidelity(
                *cells[(kind, name)],
                *control_range(envelope, kind, name, record_s),
            )
            for kind in ARTIFACTS
            for name in CONTROLS[kind]
            if (kind, name) in cells
        }
        results[policy] = {
            "candidates": recipes,
            "accepted": n_accepted,
            "acceptance_rate": n_accepted / recipes,
            "rejections": dict(rejections),
            "artifact_types": dict(kinds),
            "all_finite": finite,
            "controls": controls_report,
        }
    passed = all(
        result["all_finite"]
        and all(
            cell["passed"] is not False for cell in result["controls"].values()
        )
        for result in results.values()
    )
    report = {
        "passed": passed,
        "gate": NAME,
        "generator": f"{editor.GENERATOR}@{editor.VERSION}",
        "descriptors": f"{descriptors.NAME}@{descriptors.VERSION}",
        "calibration_sha256": calibration_sha256,
        "gap_report_id": calibration["gap_report_id"],
        "recipes_per_policy": recipes,
        "seed": seed,
        "thresholds": THRESHOLDS,
        "policies": results,
    }
    return report, pa.Table.from_pylist(rows, schema=GATE_SCHEMA)


def review_sample(
    gate_recipes: pa.Table, envelope: dict[str, Any], n: int = REVIEW_SAMPLE
) -> list[dict[str, Any]]:
    """Select accepted recipes from both extremes of each artifact severity."""
    rows = [
        row for row in gate_recipes.to_pylist() if row["decision"] == "accepted"
    ]
    sample, seen = [], set()
    per_type = [
        n // len(ARTIFACTS) + (k < n % len(ARTIFACTS))
        for k in range(len(ARTIFACTS))
    ]
    for kind, quota in zip(ARTIFACTS, per_type, strict=True):
        low, high = envelope[kind]["amplitude_mv"]
        of_kind = []
        for row in rows:
            controls = json.loads(row["controls_requested"])
            if (
                controls["artifact_type"] == kind
                and row["recipe_id"] not in seen
            ):
                of_kind.append(
                    (
                        (controls["amplitude_mv"] - low) / (high - low),
                        row,
                        controls,
                    )
                )
        of_kind.sort(key=lambda item: (item[0], item[1]["recipe_id"]))
        chosen = (
            of_kind[: quota // 2]
            + of_kind[len(of_kind) - (quota - quota // 2) :]
            if len(of_kind) > quota
            else of_kind
        )
        for position, row, controls in chosen:
            seen.add(row["recipe_id"])
            sample.append(
                {
                    "recipe_id": row["recipe_id"],
                    "policy": row["policy"],
                    "artifact_type": kind,
                    "electrode": controls["electrode"],
                    "amplitude_mv": controls["amplitude_mv"],
                    "severity_position": round(position, 4),
                    "parent_record_id": row["parent_record_id"],
                    "seed_key": row["seed_key"],
                    "gap_report_id": row["gap_report_id"],
                    "controls": row["controls_requested"],
                    "verdict": "",
                    "reviewer": "",
                    "notes": "",
                }
            )
    return sample


def write_review_sample(sample: list[dict[str, Any]], path: Path) -> bool:
    """Write a review sample without overwriting existing expert annotations."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample[0]))
        writer.writeheader()
        writer.writerows(sample)
    return True
