"""Signal measurements detect added artifacts without counting heartbeats."""

import numpy as np
import pytest

from syntide.generate import descriptors

FS = 100
T = np.arange(1000) / FS


def _ecg_like() -> np.ndarray:
    """Create twelve signal leads with 72 sharp heartbeat pulses per minute."""
    beat = np.zeros_like(T)
    for r in np.arange(0.3, 10, 1 / 1.2):
        beat += np.exp(-0.5 * ((T - r) / 0.01) ** 2) + 0.2 * np.exp(
            -0.5 * ((T - r - 0.25) / 0.06) ** 2
        )
    return np.outer(np.linspace(0.5, 1.0, 12), beat)


def test_a_clean_ecg_reads_as_artifact_free():
    found = descriptors.describe(_ecg_like(), FS)
    assert found["wander_mv"] < 0.05
    assert (
        found["hf_rms_mv"] < 0.005
    )  # the QRS fills too few samples to move the MAD
    assert found["burst_rms_mv"] < 0.005  # ...or a 0.5 s window's median


def test_injected_artifacts_are_recovered_in_their_units_despite_the_beats():
    ecg, rng = _ecg_like(), np.random.default_rng(0)
    assert descriptors.wander_mv(
        ecg + 0.8 * np.sin(2 * np.pi * 0.25 * T + 1.0), FS
    ) == pytest.approx(0.8, rel=0.05)

    # White noise of SD s keeps about 70% of its power above 15 Hz, so it reads
    # as ~0.84 s.
    noise = 0.06 * rng.standard_normal(ecg.shape)
    alone = descriptors.hf_rms_mv(noise, FS)
    assert alone == pytest.approx(0.06 * np.sqrt(35 / 50), rel=0.06)
    assert descriptors.hf_rms_mv(ecg + noise, FS) == pytest.approx(
        alone, rel=0.2
    )

    burst = np.where(
        (T >= 3.0) & (T < 5.0), 0.24 * rng.standard_normal(ecg.shape), 0.0
    )
    found = descriptors.burst(ecg + burst, FS)
    assert found["burst_rms_mv"] == pytest.approx(
        descriptors.burst(burst, FS)["burst_rms_mv"], rel=0.2
    )
    assert found["burst_onset_s"] == pytest.approx(3.0, abs=0.1)
    assert found["burst_duration_s"] == pytest.approx(2.0, abs=0.15)
