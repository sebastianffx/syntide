"""Choose artifact settings from measured training-data coverage.

Calibration uses only recordings marked ``retained_artifact``. For each
artifact type, it records the observed frequency and severity distribution.
Each recipe identifies the calibration report by its content checksum.

The ``coverage_matched`` policy follows the observed artifact frequencies and
severity quantiles. The ``gap_targeted`` policy gives more weight to uncommon
artifact types and severities, with smoothing to avoid division by zero.

Both policies use the same random draws for electrode, frequency, onset, and
duration. Severity is selected separately for each artifact type within its
configured range.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from syntide.data.challenge import Manifest
from syntide.data.leakage import check_calibration_ids
from syntide.data.schema import ARTIFACT_TYPES, Samples
from syntide.generate import descriptors, editor
from syntide.generate.editor import (
    ARTIFACTS,
    CONTROLS,
    ELECTRODES,
    EditorConfig,
    control_range,
)
from syntide.generate.ledger import canonical_json

POLICIES = ("coverage_matched", "gap_targeted")
# Ordered uniform draws shared by both artifact sampling policies.
U_FIELDS = (
    "artifact_type",
    "severity",
    "electrode",
    "freq_hz",
    "onset_s",
    "duration_s",
)
GRID_POINTS = 256


class CalibrationError(ValueError):
    """A calibration file is missing, changed, or from another trial."""


def fit_calibration(
    manifest: Manifest, retained: Samples, cfg: EditorConfig, fs: int
) -> dict[str, Any]:
    """Measure artifact coverage in the permitted retained recordings."""
    check_calibration_ids(retained.record_id, manifest)
    records, severities = [], {kind: [] for kind in ARTIFACTS}
    for row in range(len(retained)):
        flagged = [
            kind
            for kind in ARTIFACTS
            if retained.artifacts[row, ARTIFACT_TYPES.index(kind)]
        ]
        measured = {
            kind: descriptors.severity(kind, retained.x[row], fs)
            for kind in flagged
        }
        for kind, value in measured.items():
            severities[kind].append(value)
        records.append(
            {
                "record_id": str(retained.record_id[row]),
                "subject_uid": str(retained.subject_uid[row]),
                "severity": measured,
            }
        )
    report = gap_report(severities, cfg)
    content = {
        "challenge_id": manifest.challenge_id,
        "manifest_sha256": manifest.sha256,
        "descriptors": f"{descriptors.NAME}@{descriptors.VERSION}",
        "generator": f"{editor.GENERATOR}@{editor.VERSION}",
        **report,
        "records": records,
    }
    return {"gap_report_id": _gap_report_id(content), **content}


def gap_report(
    severities: dict[str, list[float]], cfg: EditorConfig
) -> dict[str, Any]:
    """Summarize artifact frequencies and severities for both policies."""
    envelope = asdict(cfg.envelope)
    counts = {kind: len(severities.get(kind, [])) for kind in ARTIFACTS}
    total = sum(counts.values())
    if total == 0:
        raise CalibrationError(
            "no retained record carries a v1 artifact type; nothing "
            "to calibrate on"
        )
    frequency = {kind: counts[kind] / total for kind in ARTIFACTS}
    targeted = {
        kind: (1 / len(ARTIFACTS)) / (frequency[kind] + cfg.type_smoothing)
        for kind in ARTIFACTS
    }
    types = {}
    for kind in ARTIFACTS:
        low, high = envelope[kind]["amplitude_mv"]
        grid = np.linspace(low, high, GRID_POINTS)
        values = np.sort(np.asarray(severities.get(kind, []), dtype=np.float64))
        prior = np.full(GRID_POINTS, 1 / (high - low))
        retained = _kde(values, grid, high - low)
        target = prior / (retained + cfg.density_floor * prior)
        target /= np.trapezoid(target, grid)
        cdf = np.concatenate(
            [[0.0], np.cumsum(np.diff(grid) * (target[1:] + target[:-1]) / 2)]
        )
        types[kind] = {
            "n_retained": counts[kind],
            "frequency": frequency[kind],
            "severities": values.tolist(),
            "grid": grid.tolist(),
            "prior_density": prior.tolist(),
            "retained_density": retained.tolist(),
            "targeted_density": target.tolist(),
            "targeted_cdf": (cdf / cdf[-1]).tolist(),
        }
    return {
        "envelope": envelope,
        "type_smoothing": cfg.type_smoothing,
        "density_floor": cfg.density_floor,
        "type_weights": {
            "coverage_matched": frequency,
            "gap_targeted": {
                kind: w / sum(targeted.values()) for kind, w in targeted.items()
            },
        },
        "types": types,
    }


def _kde(values: np.ndarray, grid: np.ndarray, width: float) -> np.ndarray:
    """Estimate a smooth severity distribution from observed values.

    The Gaussian kernel uses Silverman's bandwidth rule, with a minimum of two
    percent of the permitted range. Empty observations produce zero density.
    """
    if len(values) == 0:
        return np.zeros_like(grid)
    spread = (
        min(values.std(), np.subtract(*np.percentile(values, [75, 25])) / 1.34)
        if len(values) > 1
        else 0.0
    )
    bandwidth = max(0.9 * spread * len(values) ** -0.2, 0.02 * width)
    z = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * z**2).sum(axis=1) / (
        len(values) * bandwidth * np.sqrt(2 * np.pi)
    )


def controls_for(
    policy: str,
    u: np.ndarray,
    calibration: dict[str, Any],
    n_samples: int,
    fs: int,
) -> dict:
    """Convert ordered uniform random draws into one artifact recipe."""
    if policy not in POLICIES:
        raise ValueError(
            f"unknown sampling policy {policy!r}; choose one of {POLICIES}"
        )
    draw = dict(zip(U_FIELDS, (float(v) for v in u), strict=True))
    weights = np.array(
        [calibration["type_weights"][policy][kind] for kind in ARTIFACTS]
    )
    cumulative = np.cumsum(weights) / weights.sum()
    kind = ARTIFACTS[
        min(
            int(
                np.searchsorted(cumulative, draw["artifact_type"], side="right")
            ),
            len(ARTIFACTS) - 1,
        )
    ]
    envelope, record_s = calibration["envelope"], n_samples / fs
    controls: dict[str, Any] = {
        "artifact_type": kind,
        "electrode": ELECTRODES[
            min(int(draw["electrode"] * len(ELECTRODES)), len(ELECTRODES) - 1)
        ],
        "amplitude_mv": _severity(
            policy, draw["severity"], calibration["types"][kind], envelope[kind]
        ),
    }
    for name in CONTROLS[kind][1:]:
        if name == "onset_s":
            continue
        low, high = control_range(envelope, kind, name, record_s)
        controls[name] = low + draw[name] * (high - low)
    if "onset_s" in CONTROLS[kind]:
        controls["onset_s"] = draw["onset_s"] * (
            record_s - controls["duration_s"]
        )
    return controls


def _severity(
    policy: str,
    u: float,
    calibrated: dict[str, Any],
    envelope: dict[str, list[float]],
) -> float:
    low, high = envelope["amplitude_mv"]
    if policy == "coverage_matched":
        if not calibrated["severities"]:
            raise CalibrationError(
                "coverage_matched cannot draw a type with no retained records"
            )
        return float(
            np.clip(np.quantile(calibrated["severities"], u), low, high)
        )
    return float(np.interp(u, calibrated["targeted_cdf"], calibrated["grid"]))


def save_calibration(calibration: dict[str, Any], path: str | Path) -> str:
    """Save an unchanged calibration and return its file checksum."""
    return save_frozen_json(path, calibration, "calibration")


def load_calibration(
    path: str | Path, sha256: str | None = None
) -> dict[str, Any]:
    """Read a calibration and verify its content and optional file checksum."""
    path = Path(path)
    if not path.exists():
        raise CalibrationError(
            f"{path} does not exist; run python -m syntide.generate "
            f"mode=calibrate"
        )
    raw = path.read_bytes()
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
        raise CalibrationError(f"{path} changed since the trial was planned")
    calibration = json.loads(raw)
    content = {k: v for k, v in calibration.items() if k != "gap_report_id"}
    if _gap_report_id(content) != calibration.get("gap_report_id"):
        raise CalibrationError(
            f"{path} does not match its gap_report_id; recalibrate "
            f"instead of editing it"
        )
    return calibration


def save_frozen_json(path: str | Path, payload: Any, what: str) -> str:
    """Write JSON data without replacing different existing content.

    Args:
        path: Destination file.
        payload: JSON-serializable content to save.
        what: Description used in conflict messages.

    Returns:
        The saved file's SHA-256 checksum.
    """
    path = Path(path)
    text = json.dumps(payload, indent=1) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(
            f"{path} already holds a different {what}. It is frozen "
            f"once trials use it: delete it on "
            "purpose to rebuild it, and re-plan the trials that depend on it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def _gap_report_id(content: dict[str, Any]) -> str:
    return (
        "gap-"
        + hashlib.sha256(canonical_json(content).encode()).hexdigest()[:16]
    )
