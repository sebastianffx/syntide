"""Question-answering changes preserve classification records."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import fields
from pathlib import Path

import pytest

from syntide.explore.plans import METHODS
from syntide.explore.qa import TASK_COLUMNS
from syntide.generate import editor, generic
from syntide.generate.full.diffusets import NATIVE_SPEC
from syntide.qa_trial import QATrialSpec, load_any_spec
from syntide.trial import TrialSpec, _trial_id, load_spec

FIXTURES = Path(__file__).parents[1] / "fixtures"
LEGACY = {
    "legacy_trial_ptbxl_artifact_50_A3_seed0.json": (
        "e3608c7fff4a21c03510183c531e7352a00cc904cac8aedd097453e32846184a"
    ),
    "legacy_trial_ptbxl_artifact_50_C2_full100_seed1.json": (
        "fb2ab1c270590a4b7a509ffa6b2beb8c10121d0384fc6cb4ebbd54b561e69563"
    ),
}


@pytest.mark.parametrize(("name", "trial_id"), sorted(LEGACY.items()))
def test_real_legacy_trial_files_load_with_their_original_ids(
    name, trial_id, tmp_path
):
    shutil.copy(FIXTURES / name, tmp_path / "trial.json")
    payload = json.loads((tmp_path / "trial.json").read_text())
    assert "schema_version" not in payload  # the legacy shape is untouched
    assert set(payload) == {f.name for f in fields(TrialSpec)}
    assert _trial_id(payload) == trial_id
    legacy, dispatched = load_spec(tmp_path), load_any_spec(tmp_path)
    assert isinstance(dispatched, TrialSpec) and dispatched == legacy
    assert dispatched.trial_id == trial_id


def test_trial_spec_fields_are_unchanged():
    assert [f.name for f in fields(TrialSpec)] == [
        "protocol",
        "regime",
        "arm",
        "replicate_seed",
        "source_role",
        "transform_family",
        "sampling_policy",
        "generator",
        "validators",
        "transform",
        "dataset",
        "source_sha256",
        "manifest_sha256",
        "calibration_sha256",
        "batch_size",
        "max_steps",
        "val_every",
        "transform_fraction",
        "init_seed",
        "syntide_version",
        "registry",
        "manifest",
        "calibration",
        "trial_id",
    ]
    assert "schema_version" in {f.name for f in fields(QATrialSpec)}


def test_artifact_generators_plan_methods_and_ecgqa_columns_are_unchanged():
    assert (editor.GENERATOR, editor.VERSION) == (
        "ecg.electrode_artifact",
        "1.0.0",
    )
    assert (generic.GENERATOR, generic.VERSION) == ("per_lead_aug", "1")
    assert editor.LABEL_EFFECT == {
        "diagnosis": "preserves",
        "artifact_type": "determines",
        "signal_quality": "invalidates",
    }
    assert METHODS == {
        "augmentation": "per_lead_aug@1",
        "editing": "ecg.electrode_artifact@1.0.0",
        "diffusion": "diffusets@1",
    }
    assert TASK_COLUMNS == (
        "task_id",
        "record_id",
        "subject_uid",
        "strat_fold",
        "is_artifact",
        "artifact_types",
        "cot_split",
        "question_type",
        "template_id",
        "question",
        "answer",
        "answer_options",
        "clinical_context",
        "rationale",
        "has_rationale",
    )
    assert (NATIVE_SPEC.sample_rate, NATIVE_SPEC.num_samples) == (102.4, 1024)


def test_ecgqa_source_fingerprint_is_the_documented_algorithm(qa_registry):
    view = qa_registry.view
    digest = hashlib.sha256(view.signals.source_sha256().encode())
    columns = view.tasks().to_pydict()
    for row in zip(*(columns[name] for name in TASK_COLUMNS), strict=True):
        digest.update(repr(row).encode())
    assert view.source_sha256() == digest.hexdigest()


def test_qa_trial_files_are_not_classification_trials(tmp_path):
    from syntide.trial import TrialError

    (tmp_path / "trial.json").write_text(
        json.dumps({"schema_version": "syntide.qa-trial@1", "arm": "x"})
    )
    with pytest.raises(TrialError):
        load_spec(
            tmp_path
        )  # the classification loader never accepts a QA trial
    with pytest.raises(TrialError):
        load_any_spec(
            tmp_path
        )  # and a malformed QA trial is refused, not reinterpreted
