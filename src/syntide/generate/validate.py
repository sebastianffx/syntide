"""Check edited signals and measure the effects of each artifact.

Acceptance requires finite values, an unchanged shape, bounded amplitude, and
agreement between requested and measured artifact severity. Recordings already
above the amplitude limit may keep their original peak value.

The artifact is measured on the strongest affected lead after subtracting the
source signal. A separate report measures the complete edited waveform,
including normal electrical activity, for comparison with natural recordings.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from syntide.generate import descriptors
from syntide.generate.editor import Candidate, Recipe


@dataclass(frozen=True)
class Check:
    name: str
    version: str
    value: float
    passed: bool | None  # None for report-only checks
    blocking: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "value": self.value,
            "pass": self.passed,
        }


def realized_controls(
    candidate: Candidate,
    controls: dict[str, Any],
    fs: int,
    *,
    full: bool = False,
) -> dict:
    """Measure an edit, optionally including frequency, onset, and duration."""
    lead = candidate.residual[candidate.strongest_lead]
    kind = controls["artifact_type"]
    if kind == "burst_noise":
        found = descriptors.burst(lead, fs)
        realized = {"amplitude_mv": found["burst_rms_mv"]}
        if full:
            realized |= {
                "onset_s": found["burst_onset_s"],
                "duration_s": found["burst_duration_s"],
            }
        return realized
    realized = {"amplitude_mv": descriptors.severity(kind, lead, fs)}
    if full and kind == "baseline_drift":
        realized["freq_hz"] = sinusoid_hz(lead, fs)
    return realized


def sinusoid_hz(signal: np.ndarray, fs: int) -> float:
    """Estimate sinusoid frequency from at least one complete cycle."""
    above = np.signbit(signal)
    edges = np.flatnonzero(above[1:] != above[:-1])
    if len(edges) < 2:
        return float("nan")
    times = (edges + signal[edges] / (signal[edges] - signal[edges + 1])) / fs
    return float((len(times) - 1) / (2 * (times[-1] - times[0])))


def evaluate(
    candidate: Candidate,
    parent_x: np.ndarray,
    recipe: Recipe,
    cfg: dict[str, Any],
    fs: int,
    *,
    full: bool = False,
) -> tuple[list[Check], dict[str, float]]:
    """Return acceptance checks and measured settings for an edited signal."""
    x = candidate.x
    finite = bool(np.isfinite(x).all()) and x.shape == parent_x.shape
    peak = float(np.abs(x).max()) if finite else float("inf")
    limit = max(cfg["mv_limit"], float(np.abs(parent_x).max()))
    realized = realized_controls(candidate, recipe.controls, fs, full=full)
    requested = recipe.controls["amplitude_mv"]
    error = abs(realized["amplitude_mv"] - requested) / requested
    checks = [
        Check("bounds", "1", peak, finite and peak <= limit, blocking=True),
        Check(
            "alignment",
            "1",
            error,
            error <= cfg["alignment_tol"],
            blocking=True,
        ),
    ]
    return checks, realized


def in_context(candidate: Candidate, recipe: Recipe, fs: int) -> Check:
    kind = recipe.controls["artifact_type"]
    return Check(
        "in_context",
        "1",
        descriptors.severity(kind, candidate.x, fs),
        None,
        blocking=False,
    )


def accepted(checks: list[Check]) -> bool:
    return all(check.passed for check in checks if check.blocking)


def failed(checks: list[Check]) -> str:
    return ",".join(
        f"{check.name}@{check.version}"
        for check in checks
        if check.blocking and not check.passed
    )
