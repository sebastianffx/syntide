"""Round-trip contract: PTB-XL rows -> connector -> TimeNet registry -> adapter."""

import numpy as np
import pytest
from timenet.client import TimeNet

from syntide.data.schema import ARTIFACT_TYPES, SUPERCLASSES


def test_signals_round_trip_in_millivolts_in_clinical_lead_order(built_ptbxl):
    # TimeNet returns a record's leads sorted by name (AVF, AVL, AVR, I, ...); the adapter must
    # restore clinical order. Requested record order must be preserved too.
    samples = built_ptbxl.adapter.load(["ptbxl-2", "ptbxl-1"])

    assert samples.record_id.tolist() == ["ptbxl-2", "ptbxl-1"]
    for row, ecg_id in enumerate((2, 1)):
        np.testing.assert_allclose(
            samples.x[row], built_ptbxl.signals[ecg_id].T, rtol=0, atol=built_ptbxl.adc_step[ecg_id]
        )


def test_diagnostic_labels_follow_superclass_rules_and_official_split(built_ptbxl):
    index = built_ptbxl.adapter.index()
    labels = dict(zip(index["record_id"].to_pylist(), index["labels"].to_pylist(), strict=True))

    # Rhythm codes are ignored; two MI codes collapse to one MI label.
    assert labels == {"ptbxl-1": ["NORM"], "ptbxl-2": ["MI", "STTC"], "ptbxl-4": ["CD", "HYP"]}
    # A record with no diagnostic superclass has no task and cannot be loaded for training.
    with pytest.raises(KeyError, match="ptbxl-3"):
        built_ptbxl.adapter.load(["ptbxl-3"])

    samples = built_ptbxl.adapter.load(["ptbxl-2"])
    assert [SUPERCLASSES[k] for k in np.flatnonzero(samples.labels[0])] == ["MI", "STTC"]

    # TimeNet consumers see the PTB-XL authors' split on each task.
    tasks = TimeNet(built_ptbxl.registry).load("physionet/ptb-xl", auto_build=False).tasks
    assert {task.id: task.metadata["split"] for task in tasks} == {
        "ptbxl-1-dx": "train",
        "ptbxl-2-dx": "train",
        "ptbxl-4-dx": "test",
    }


def test_index_namespaces_subjects_and_counts_only_real_noise_flags(built_ptbxl):
    rows = {row["record_id"]: row for row in built_ptbxl.adapter.index().to_pylist()}

    assert rows["ptbxl-1"]["subject_uid"] == rows["ptbxl-2"]["subject_uid"] == "ptb-xl:101"
    assert rows["ptbxl-4"]["subject_uid"] == "ptb-xl:303"
    assert rows["ptbxl-1"]["artifact_types"] == ["static_noise"]
    assert rows["ptbxl-2"]["artifact_types"] == ["burst_noise"]
    # A whitespace-only baseline_drift value is not a flag.
    assert rows["ptbxl-4"]["artifact_types"] == []
    assert [rows[r]["is_artifact"] for r in ("ptbxl-1", "ptbxl-2", "ptbxl-4")] == [True, True, False]

    samples = built_ptbxl.adapter.load(["ptbxl-2", "ptbxl-4"])
    assert samples.artifacts[0].tolist() == [t == "burst_noise" for t in ARTIFACT_TYPES]
    assert samples.is_artifact.tolist() == [True, False]
