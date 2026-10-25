"""``ecg.electrode_artifacts@1``: the coverage space for ECG artifacts.

Descriptors are named, unit-bearing measurements of a signal. Calibration measures natural artifact
ECGs with them, and the validators measure edits with them, so the policies never see generator
knobs (MVE: "coverage lives in descriptor space, not control space"). Each descriptor is computed
per lead and reported for the strongest lead:

- ``wander_mv``: half the peak-to-peak of a zero-phase low-pass below 0.7 Hz (baseline drift);
- ``hf_rms_mv``: the robust RMS (1.4826 x MAD) of a zero-phase high-pass above 15 Hz (static noise);
  the MAD ignores the few QRS samples that leak above 15 Hz, so no beat detection is needed;
- ``burst_rms_mv``, ``burst_onset_s``, ``burst_duration_s``: the loudest high-frequency segment, found on
  a 0.5 s robust-RMS envelope of the same high-pass (burst noise).

Physiological content leaks into these bands (T waves below 0.7 Hz, QRS above 15 Hz), so a natural
ECG's descriptors are biased upward compared with an artifact measured alone.
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
    return sosfiltfilt(_sos("lowpass", WANDER_CUTOFF_HZ, fs), np.atleast_2d(x), axis=-1)


def highpass(x: np.ndarray, fs: int) -> np.ndarray:
    return sosfiltfilt(_sos("highpass", HF_CUTOFF_HZ, fs), np.atleast_2d(x), axis=-1)


def robust_sd(x: np.ndarray, axis: int = -1) -> np.ndarray:
    median = np.median(x, axis=axis, keepdims=True)
    return _MAD_TO_SD * np.median(np.abs(x - median), axis=axis)


def wander_mv(x: np.ndarray, fs: int) -> float:
    low = lowpass(x, fs)
    return float((0.5 * (low.max(axis=-1) - low.min(axis=-1))).max())


def hf_rms_mv(x: np.ndarray, fs: int) -> float:
    return float(robust_sd(highpass(x, fs)).max())


def burst(x: np.ndarray, fs: int) -> dict[str, float]:
    """The loudest high-frequency segment of the strongest lead: its robust RMS, onset and duration.

    1. A robust envelope (1.4826 x median |x| of the high-pass, 0.5 s windows every 0.02 s) finds the
       loudest window. QRS ringing fills a 0.25 s window at 100 Hz but not a 0.5 s one, so the
       envelope follows noise, not beats.
    2. The segment grows around it while the envelope stays a quarter of the way from the record's
       median level up to the peak, so noisy dips inside a burst do not split it.
    3. Its edges are refined on a 0.1 s moving power trace: scanning in from outside towards the
       loudest window, the edge is the first sample reaching half of the mean power under the
       segment's windows.
    """
    hp = highpass(x, fs)
    n_samples = hp.shape[-1]
    width, hop = round(BURST_WINDOW_S * fs), round(_BURST_HOP_S * fs)
    frames = sliding_window_view(hp, width, axis=-1)[:, ::hop]
    envelope = _MAD_TO_SD * np.median(np.abs(frames), axis=-1)  # (leads, windows); window k starts at k * hop
    power = uniform_filter1d(hp**2, size=round(_EDGE_POWER_S * fs), axis=-1, mode="constant")
    best = {"burst_rms_mv": 0.0, "burst_onset_s": 0.0, "burst_duration_s": 0.0}
    for lead in range(hp.shape[0]):
        env = envelope[lead]
        peak = int(np.argmax(env))
        if env[peak] <= 0:
            continue
        background = float(np.median(env))
        loud = env >= background + 0.25 * (env[peak] - background)
        lo = 0 if loud[: peak + 1].all() else peak - int(np.argmin(loud[peak::-1])) + 1
        hi = len(env) - 1 if loud[peak:].all() else peak + int(np.argmin(loud[peak:])) - 1
        centre = peak * hop + width // 2
        covered = hp[lead, lo * hop : hi * hop + width]
        reached = power[lead] >= 0.5 * float(np.mean(covered**2))
        scan_start, scan_stop = max(0, lo * hop - width), min(n_samples, hi * hop + 2 * width)
        before, after = reached[scan_start : centre + 1], reached[centre:scan_stop][::-1]
        onset = scan_start + int(np.argmax(before)) if before.any() else lo * hop + width // 2
        offset = scan_stop - int(np.argmax(after)) if after.any() else hi * hop + width // 2 + 1
        rms = float(robust_sd(hp[lead, onset:offset]))
        if rms > best["burst_rms_mv"]:
            best = {
                "burst_rms_mv": rms,
                "burst_onset_s": onset / fs,
                "burst_duration_s": (offset - onset) / fs,
            }
    return best


def severity(artifact_type: str, x: np.ndarray, fs: int) -> float:
    """The descriptor that measures how severe an artifact of this type is, in mV."""
    if artifact_type == "baseline_drift":
        return wander_mv(x, fs)
    if artifact_type == "static_noise":
        return hf_rms_mv(x, fs)
    if artifact_type == "burst_noise":
        return burst(x, fs)["burst_rms_mv"]
    raise ValueError(f"no severity descriptor for artifact type {artifact_type!r}")


def describe(x: np.ndarray, fs: int) -> dict[str, float]:
    """Every descriptor of one ECG (leads, T)."""
    return {"wander_mv": wander_mv(x, fs), "hf_rms_mv": hf_rms_mv(x, fs), **burst(x, fs)}
