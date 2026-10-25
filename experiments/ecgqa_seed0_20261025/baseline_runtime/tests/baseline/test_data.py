"""Focused preparation checks; synthetic arrays only, no model or downloaded dataset.

Run these on the RunPod worker with the baseline's normal Python dependencies.
"""

import json
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from syntide.baseline import data
from syntide.data.challenge import ChallengeSpec, build_manifest, save_manifest
from syntide.data.leakage import LeakageError


def _task(record, subject, task_id, *, template=1, answer="no", noisy=False):
    return {
        "task_id": task_id,
        "record_id": record,
        "subject_uid": subject,
        "strat_fold": 9,
        "is_artifact": noisy,
        "artifact_types": ["static_noise"] if noisy else [],
        "cot_split": "train",
        "question_type": "single-verify",
        "template_id": template,
        "question": f"Template {template} question?",
        "answer": answer,
        "answer_options": ["no", "yes"],
        "clinical_context": "Same permitted clinical context.",
        "rationale": f"Rationale variant {task_id}. Answer: {answer}",
        "has_rationale": True,
    }


def test_deduplication_retains_lowest_task_and_keeps_question_on_other_ecg():
    rows = [
        _task("ptbxl-1", "p1", "z", answer="yes"),
        _task("ptbxl-1", "p1", "a", answer="yes"),
        _task("ptbxl-2", "p2", "b", answer="no"),
    ]
    tasks = pa.Table.from_pylist(rows)
    data.check_question_truth(tasks)
    chosen = data.distinct_questions(tasks)
    assert chosen["task_id"].to_pylist() == ["a", "b"]
    assert chosen["answer"].to_pylist() == ["yes", "no"]
    assert chosen["rationale"][0].as_py() == rows[1]["rationale"]


@pytest.mark.parametrize("change", [{"answer": "yes"}, {"template_id": 2}, {"answer_options": ["no"]}])
def test_conflicting_truth_within_ecg_question_is_rejected(change):
    first = _task("ptbxl-1", "p1", "a")
    second = {**first, "task_id": "b", **change}
    with pytest.raises(ValueError, match="conflicting answer/template/options"):
        data.check_question_truth(pa.Table.from_pylist([first, second]))


def test_answer_option_order_does_not_make_duplicate_conflict():
    first = _task("ptbxl-1", "p1", "a")
    second = {**first, "task_id": "b", "answer_options": ["yes", "no"]}
    data.check_question_truth(pa.Table.from_pylist([first, second]))


def _selection_tasks():
    return pa.Table.from_pylist(
        [
            _task(
                f"record-{patient}",
                f"patient-{patient}",
                f"q-{template}-{noisy}-{patient}-{variant}",
                template=template,
                noisy=noisy,
            )
            for template in (1, 2)
            for noisy in (False, True)
            for patient in range(8)
            for variant in range(3)
        ]
    )


def test_selection_balances_strata_and_draws_patients_before_extra_questions():
    validation = _selection_tasks()
    selected = data.selection_subset(validation, size=20, seed=7).to_pylist()
    strata = Counter((row["template_id"], row["is_artifact"]) for row in selected)
    assert set(strata.values()) == {5}
    for template, noisy in strata:
        patients = [
            row["subject_uid"]
            for row in selected
            if (row["template_id"], row["is_artifact"]) == (template, noisy)
        ]
        assert len(patients) == len(set(patients))
    assert len({row["task_id"] for row in selected}) == 20


def test_selection_is_answer_blind_deterministic_and_order_independent():
    validation = _selection_tasks()
    expected = set(data.selection_subset(validation, size=25, seed=11)["task_id"].to_pylist())
    changed = validation.to_pylist()[::-1]
    for row in changed:
        row.update(answer="yes", rationale="Replacement rationale", clinical_context="Changed context")
        row["answer_options"] = ["changed", "options"]
    actual = data.selection_subset(pa.Table.from_pylist(changed), size=25, seed=11)
    assert set(actual["task_id"].to_pylist()) == expected
    assert set(data.selection_subset(validation, size=25, seed=12)["task_id"].to_pylist()) != expected
    assert data.selection_subset(validation, size=10_000, seed=11).num_rows == validation.num_rows


@pytest.fixture
def fake_export(tmp_path, monkeypatch):
    from syntide.data.ecgqa import adapter as adapter_module

    index_rows = [
        {"record_id": f"ptbxl-{i}", "subject_uid": f"patient-{i}", "strat_fold": fold, "is_artifact": False}
        for i, fold in enumerate((1, 2, 9, 10), start=1)
    ]
    rows = []
    for record in index_rows:
        for template, variant in ((1, "b"), (1, "a"), (2, "a")):
            row = _task(
                record["record_id"],
                record["subject_uid"],
                f"{record['record_id']}-{template}-{variant}",
                template=template,
            )
            row["strat_fold"] = record["strat_fold"]
            rows.append(row)
    source = {
        "dataset": "physionet/ecg-qa-cot",
        "version": "1.0.0",
        "sha256": "questions",
        "signals": {"dataset": "physionet/ptb-xl", "version": "1.0.3", "sha256": "signals"},
    }

    class FakeAdapter:
        def __init__(self):
            self.index_table = pa.Table.from_pylist(index_rows)
            self.tasks_table = pa.Table.from_pylist(rows)
            self.source_data = source
            self.signal_loads = 0

        def tasks(self):
            return self.tasks_table

        def index(self):
            return self.index_table

        def source(self):
            return self.source_data

        def tasks_of(self, record_ids):
            return self.tasks_table.filter(
                pa.array([record in record_ids for record in self.tasks_table["record_id"].to_pylist()])
            )

        def dropped(self):
            return {}

        def load(self, record_ids):
            self.signal_loads += 1
            values = np.array([int(record.rsplit("-", 1)[1]) for record in record_ids], dtype=np.float32)
            return SimpleNamespace(
                record_id=np.array(record_ids),
                x=np.broadcast_to(values[:, None, None], (len(record_ids), 12, 1000)),
            )

    adapter = FakeAdapter()
    manifest = build_manifest(adapter.index(), ChallengeSpec(id="test-baseline"), 1.0, source)
    manifest_path = tmp_path / "source.json"
    save_manifest(manifest, manifest_path)
    verified = []
    monkeypatch.setattr(adapter_module, "EcgQaAdapter", lambda registry: adapter)
    monkeypatch.setattr(data, "_verify_signal_source", lambda registry, source: verified.append(source))
    return SimpleNamespace(
        adapter=adapter,
        manifest_path=manifest_path,
        output=tmp_path / "prepared",
        verified=verified,
        registry=tmp_path / "registry",
    )


def _export(fixture, **kwargs):
    return data.export_dataset(
        fixture.registry, fixture.manifest_path, fixture.output, selection_size=2, **kwargs
    )


def test_export_preserves_training_variants_and_aligns_signals_with_questions(fake_export):
    result = _export(fake_export)
    assert fake_export.verified == [fake_export.adapter.source()]
    assert result["splits"]["train"]["questions"] == 6
    assert result["splits"]["validation"]["questions"] == 2
    assert result["splits"]["test"]["questions"] == 2
    assert result["splits"]["train"]["patients"] == 2
    assert result["deduplication"] == {"validation_rows_removed": 1, "test_rows_removed": 1}
    records = json.loads((fake_export.output / "records.json").read_text())
    signals = np.load(fake_export.output / "signals.npy")
    assert signals.shape == (4, 12, 1000)
    for split in ("train", "selection", "validation", "test"):
        for row in pq.read_table(fake_export.output / f"{split}.parquet").to_pylist():
            assert records[row["signal_index"]] == row["record_id"]
            assert np.all(signals[row["signal_index"]] == int(row["record_id"].rsplit("-", 1)[1]))
    assert data.load_export_manifest(fake_export.output)["sha256"] == result["sha256"]
    assert _export(fake_export)["sha256"] == result["sha256"]
    assert fake_export.adapter.signal_loads == 1  # identical export reuses files, never rewrites
    with pytest.raises(FileExistsError, match="different source or preparation"):
        _export(fake_export, selection_seed=1)


@pytest.mark.parametrize("fault", ["source", "patient", "question"])
def test_export_stops_source_and_patient_question_leakage_before_loading_signals(fake_export, fault):
    adapter = fake_export.adapter
    if fault == "source":
        adapter.source_data = {**adapter.source_data, "sha256": "changed"}
    elif fault == "patient":
        rows = adapter.index_table.to_pylist()
        rows[2]["subject_uid"] = rows[0]["subject_uid"]
        adapter.index_table = pa.Table.from_pylist(rows)
    else:
        rows = adapter.tasks_table.to_pylist()
        rows[0]["subject_uid"] = "patient-4"
        adapter.tasks_table = pa.Table.from_pylist(rows)
    with pytest.raises(LeakageError):
        _export(fake_export)
    assert adapter.signal_loads == 0
    assert not fake_export.output.exists()


def test_export_rejects_manifest_tampering_before_source_access(fake_export):
    payload = json.loads(fake_export.manifest_path.read_text())
    payload["roles"]["pool"].append("ptbxl-4")
    fake_export.manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="content does not match"):
        _export(fake_export)
    assert fake_export.adapter.signal_loads == 0


def test_record_overlap_is_rejected_even_if_manifest_has_been_rehashed(fake_export, monkeypatch):
    from syntide.data import challenge

    manifest = challenge.load_manifest(fake_export.manifest_path)
    leaked = replace(manifest, roles={**manifest.roles, "pool": (*manifest.roles["pool"], "ptbxl-3")})
    monkeypatch.setattr(challenge, "load_manifest", lambda path: leaked)
    with pytest.raises(LeakageError, match="records are in both pool and val"):
        _export(fake_export)
    assert fake_export.adapter.signal_loads == 0


def test_physical_source_corruption_stops_export_despite_matching_logical_fingerprint(
    fake_export, monkeypatch
):
    def corrupted(registry, source):
        raise ValueError("physical source checksum mismatch")

    monkeypatch.setattr(data, "_verify_signal_source", corrupted)
    with pytest.raises(ValueError, match="physical source checksum mismatch"):
        _export(fake_export)
    assert fake_export.adapter.signal_loads == 0
    assert not fake_export.output.exists()


def test_training_verification_does_not_read_test_table_but_final_verification_rejects_corruption(
    fake_export,
):
    _export(fake_export)
    (fake_export.output / "test.parquet").write_bytes(b"corrupted held-out artifact")
    data.verify_prepared(fake_export.output, splits=("train", "selection"))
    with pytest.raises(ValueError, match=r"test\.parquet"):
        data.verify_prepared(fake_export.output, splits=("test",))
    with pytest.raises(ValueError, match=r"test\.parquet"):
        data.load_export_manifest(fake_export.output)


def test_export_manifest_tampering_is_rejected_even_with_unchanged_files(fake_export):
    _export(fake_export)
    path = fake_export.output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["splits"]["train"]["questions"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest checksum mismatch"):
        data.verify_prepared(fake_export.output)


@pytest.mark.parametrize("signal_fault", ["shape", "nonfinite"])
def test_invalid_signal_never_publishes_partial_export(fake_export, monkeypatch, signal_fault):
    def invalid(record_ids):
        shape = (len(record_ids), 12, 999 if signal_fault == "shape" else 1000)
        x = np.zeros(shape, dtype=np.float32)
        if signal_fault == "nonfinite":
            x[0, 0, 0] = np.nan
        return SimpleNamespace(record_id=np.array(record_ids), x=x)

    monkeypatch.setattr(fake_export.adapter, "load", invalid)
    with pytest.raises(ValueError, match=r"unexpected 100 Hz ECG shape|nonfinite signal"):
        _export(fake_export)
    assert not fake_export.output.exists()
    assert not list(fake_export.output.parent.glob(f".{fake_export.output.name}-*"))
