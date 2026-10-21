"""Fixtures shared by every phase's tests: a synthetic PTB-XL index, synthetic ECGs and trial specs."""

from collections.abc import Callable
from dataclasses import asdict

import numpy as np
import pyarrow as pa
import pytest

from syntide.data.schema import ARTIFACT_TYPES, LEADS, SUPERCLASSES, Samples
from syntide.generate.editor import EditorConfig
from syntide.generate.generic import GenericAugConfig
from syntide.generate.policy import gap_report
from syntide.trial import TrialSpec, make_spec

ARMS = {  # arm -> (source_role, transform_family, sampling_policy, generator)
    "A0": ("depleted", "none", "none", None),
    "A1": ("depleted", "generic", "fixed", "per_lead_aug@1"),
    "A2": ("depleted", "editor", "coverage_matched", "ecg.electrode_artifact@1.0.0"),
    "A3": ("depleted", "editor", "gap_targeted", "ecg.electrode_artifact@1.0.0"),
    "C1": ("natural_mix", "none", "none", None),
}
# Retained severities for a test calibration: drift and static noise are common, bursts are rare.
RETAINED_SEVERITIES = {
    "baseline_drift": [0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2],
    "static_noise": [0.008, 0.01, 0.012, 0.013, 0.015, 0.016, 0.018, 0.02, 0.022, 0.025, 0.03, 0.04],
    "burst_noise": [0.03, 0.05],
}


@pytest.fixture(scope="session")
def synthetic_index() -> pa.Table:
    """A PTB-XL-shaped index: 1,500 patients over 10 patient-level folds, 1-3 ECGs each, ~22% artifacts."""
    rng = np.random.default_rng(1234)
    rows = {"record_id": [], "subject_uid": [], "strat_fold": [], "is_artifact": []}
    ecg_id = 0
    for patient in range(1500):
        fold = int(rng.integers(1, 11))
        for _ in range(int(rng.integers(1, 4))):
            ecg_id += 1
            rows["record_id"].append(f"ptbxl-{ecg_id}")
            rows["subject_uid"].append(f"ptb-xl:{patient}")
            rows["strat_fold"].append(fold)
            rows["is_artifact"].append(bool(rng.random() < 0.22))
    return pa.table(rows)


@pytest.fixture(scope="session")
def make_samples() -> Callable[..., Samples]:
    """Random labelled ECGs; half carry an artifact flag, so val slices have both labels per class."""

    def build(n: int, *, seed: int = 0, n_samples: int = 100, prefix: str = "rec") -> Samples:
        rng = np.random.default_rng(seed)
        artifacts = np.zeros((n, len(ARTIFACT_TYPES)), dtype=bool)
        artifacts[::2, 0] = True
        return Samples(
            record_id=np.array([f"{prefix}-{i}" for i in range(n)]),
            subject_uid=np.array([f"ptb-xl:{prefix}{i}" for i in range(n)]),
            labels=rng.random((n, len(SUPERCLASSES))) < 0.5,
            artifacts=artifacts,
            x=(0.3 * rng.standard_normal((n, len(LEADS), n_samples))).astype(np.float32),
        )

    return build


@pytest.fixture(scope="session")
def make_test_spec() -> Callable[..., TrialSpec]:
    """A planned-looking TrialSpec (8 per batch, 30 steps, val every 10) without running the planner.

    Editor arms (A2, A3) pair with the ``editor_calibration`` fixture."""

    def build(arm: str = "A1", **overrides) -> TrialSpec:
        role, family, policy, generator = ARMS[arm]
        transform = {"generic": asdict(GenericAugConfig()), "editor": asdict(EditorConfig())}.get(family, {})
        content = {
            "protocol": "test-protocol",
            "regime": "scarcity25",
            "arm": arm,
            "replicate_seed": 0,
            "source_role": role,
            "transform_family": family,
            "sampling_policy": policy,
            "generator": generator,
            "validators": [],
            "transform": transform,
            "dataset": "physionet/ptb-xl@1.0.3",
            "source_sha256": "0" * 64,
            "manifest_sha256": "1" * 64,
            "calibration_sha256": "2" * 64 if family == "editor" else None,
            "batch_size": 8,
            "max_steps": 30,
            "val_every": 10,
            "transform_fraction": 0.5 if role == "depleted" else 0.0,
            "registry": "datasets/timenet-registry",
            "manifest": "manifests/test/pool25.json",
            "calibration": "manifests/test/calibration/pool25.json" if family == "editor" else None,
        }
        return make_spec(**{**content, **overrides})

    return build


@pytest.fixture(scope="session")
def editor_calibration() -> dict:
    """A calibration fit from RETAINED_SEVERITIES with the default editor settings."""
    return {
        "gap_report_id": "gap-test",
        "manifest_sha256": "1" * 64,
        **gap_report(RETAINED_SEVERITIES, EditorConfig()),
    }
