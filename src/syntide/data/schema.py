"""Shared PTB-XL constants and arrays of labelled recordings.

Training and evaluation can use these structures without importing TimeNet.
"""

from dataclasses import dataclass

import numpy as np

# Clinical lead order. PTB-XL headers spell the augmented leads in upper case.
LEADS = (
    "I",
    "II",
    "III",
    "AVR",
    "AVL",
    "AVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
# Diagnostic superclasses in alphabetical order; column k of ``Samples.labels``
# is SUPERCLASSES[k].
SUPERCLASSES = ("CD", "HYP", "MI", "NORM", "STTC")
# PTB-XL signal-quality columns; column k of ``Samples.artifacts`` is
# ARTIFACT_TYPES[k].
ARTIFACT_TYPES = (
    "baseline_drift",
    "static_noise",
    "burst_noise",
    "electrodes_problems",
)
FS_HZ = 100


def subject_uid(patient_id: int) -> str:
    """Qualify a patient identifier with its dataset name."""
    return f"ptb-xl:{patient_id}"


@dataclass(frozen=True)
class Samples:
    """Labelled electrocardiograms with matching array row order."""

    record_id: np.ndarray  # (n,) str
    subject_uid: np.ndarray  # (n,) str
    labels: np.ndarray  # (n, len(SUPERCLASSES)) bool
    artifacts: np.ndarray  # (n, len(ARTIFACT_TYPES)) bool
    x: np.ndarray  # (n, len(LEADS), T) float32, millivolts

    def __post_init__(self) -> None:
        n = len(self.record_id)
        expected = {
            "subject_uid": (n,),
            "labels": (n, len(SUPERCLASSES)),
            "artifacts": (n, len(ARTIFACT_TYPES)),
            "x": (n, len(LEADS)),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual[: len(shape)] != shape:
                raise ValueError(
                    f"Samples.{name} has shape {actual}, expected {shape}"
                )

    @property
    def is_artifact(self) -> np.ndarray:
        """Whether each record carries any PTB-XL signal-quality flag."""
        return self.artifacts.any(axis=1)

    def __len__(self) -> int:
        return len(self.record_id)
