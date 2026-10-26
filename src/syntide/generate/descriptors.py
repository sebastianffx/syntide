"""Measure electrode artifacts in electrocardiogram recordings.

Calibration and validation use the same measurements, independent of the
settings requested from the generator. Measurements are computed per lead and
reported for the lead with the strongest artifact:

* ``wander_mv`` measures slow baseline movement below 0.7 hertz.
* ``hf_rms_mv`` measures high-frequency noise above 15 hertz using a robust
  amplitude estimate: 1.4826 times the median absolute deviation.
* ``burst_rms_mv``, ``burst_onset_s``, and ``burst_duration_s`` describe the
  strongest short interval of high-frequency noise.

The signal's normal electrical activity also contributes to these measurements,
so a recording can have a higher value than its isolated artifact.
"""

from functools import cache

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

NAME = "ecg.electrode_artifacts"
VERSION = "1"
WANDER_CUTOFF_HZ = 0.7
HF_CUTOFF_HZ = 15.0
BURST_WINDOW_S = 0.5
_BURST_HOP_S = 0.02
_EDGE_POWER_S = 0.1
_MAD_TO_SD = 1.4826


@cache
def _sos(kind: str, cutoff_hz: float, fs: int) -> np.ndarray:
    return butter(4, cutoff_hz, btype=kind, fs=fs, output="sos")


def lowpass(x: np.ndarray, fs: int) -> np.ndarray:
    return sosfiltfilt(
        _sos("lowpass", WANDER_CUTOFF_HZ, fs), np.atleast_2d(x), axis=-1
    )


def highpass(x: np.ndarray, fs: int) -> np.ndarray:
    return sosfiltfilt(
        _sos("highpass", HF_CUTOFF_HZ, fs), np.atleast_2d(x), axis=-1
    )


def robust_sd(x: np.ndarray, axis: int = -1) -> np.ndarray:
    median = np.median(x, axis=axis, keepdims=True)
    return _MAD_TO_SD * np.median(np.abs(x - median), axis=axis)


def wander_mv(x: np.ndarray, fs: int) -> float:
    low = lowpass(x, fs)
    return float((0.5 * (low.max(axis=-1) - low.min(axis=-1))).max())


def hf_rms_mv(x: np.ndarray, fs: int) -> float:
    return float(robust_sd(highpass(x, fs)).max())


def burst(x: np.ndarray, fs: int) -> dict[str, float]:
    """Measure the strongest short interval of high-frequency noise.

    The search uses half-second windows of robust amplitude estimates. It grows
    an interval around the strongest window and refines the edges using a
    moving
    power estimate. This reduces the influence of individual heartbeats.

    Args:
        x: Signal with shape (leads, samples), in millivolts.
        fs: Sampling frequency in hertz.

    Returns:
        Noise amplitude in millivolts, onset in seconds, and duration in
        seconds.
    """
    hp = highpass(x, fs)
    n_samples = hp.shape[-1]
    width, hop = round(BURST_WINDOW_S * fs), round(_BURST_HOP_S * fs)
    frames = sliding_window_view(hp, width, axis=-1)[:, ::hop]
    envelope = _MAD_TO_SD * np.median(
        np.abs(frames), axis=-1
    )  # (leads, windows); window k starts at k * hop
    power = uniform_filter1d(
        hp**2, size=round(_EDGE_POWER_S * fs), axis=-1, mode="constant"
    )
    best = {"burst_rms_mv": 0.0, "burst_onset_s": 0.0, "burst_duration_s": 0.0}
    for lead in range(hp.shape[0]):
        env = envelope[lead]
        peak = int(np.argmax(env))
        if env[peak] <= 0:
            continue
        background = float(np.median(env))
        loud = env >= background + 0.25 * (env[peak] - background)
        lo = (
            0
            if loud[: peak + 1].all()
            else peak - int(np.argmin(loud[peak::-1])) + 1
        )
        hi = (
            len(env) - 1
            if loud[peak:].all()
            else peak + int(np.argmin(loud[peak:])) - 1
        )
        centre = peak * hop + width // 2
        covered = hp[lead, lo * hop : hi * hop + width]
        reached = power[lead] >= 0.5 * float(np.mean(covered**2))
        scan_start, scan_stop = (
            max(0, lo * hop - width),
            min(n_samples, hi * hop + 2 * width),
        )
        before, after = (
            reached[scan_start : centre + 1],
            reached[centre:scan_stop][::-1],
        )
        onset = (
            scan_start + int(np.argmax(before))
            if before.any()
            else lo * hop + width // 2
        )
        offset = (
            scan_stop - int(np.argmax(after))
            if after.any()
            else hi * hop + width // 2 + 1
        )
        rms = float(robust_sd(hp[lead, onset:offset]))
        if rms > best["burst_rms_mv"]:
            best = {
                "burst_rms_mv": rms,
                "burst_onset_s": onset / fs,
                "burst_duration_s": (offset - onset) / fs,
            }
    return best


def severity(artifact_type: str, x: np.ndarray, fs: int) -> float:
    """Return artifact severity in millivolts for the strongest lead."""
    if artifact_type == "baseline_drift":
        return wander_mv(x, fs)
    if artifact_type == "static_noise":
        return hf_rms_mv(x, fs)
    if artifact_type == "burst_noise":
        return burst(x, fs)["burst_rms_mv"]
    raise ValueError(
        f"no severity descriptor for artifact type {artifact_type!r}"
    )


def describe(x: np.ndarray, fs: int) -> dict[str, float]:
    """Measure every supported artifact descriptor for one recording."""
    return {
        "wander_mv": wander_mv(x, fs),
        "hf_rms_mv": hf_rms_mv(x, fs),
        **burst(x, fs),
    }
