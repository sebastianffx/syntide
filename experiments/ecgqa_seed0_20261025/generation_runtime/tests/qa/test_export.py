"""Prepared exports preserve baseline evaluation, ancestry, signal bytes and source exposure."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.qa.fixtures import memory_source

from syntide.generate.qa import export as exporter
from syntide.generate.qa.acquisition import AcquisitionConfig
from syntide.generate.qa.build import BuildConfig, PairsConfig, build_corpus
from syntide.generate.qa.corpus import load_corpus


def write_manifest(path, content):
    content = {k: v for k, v in content.items() if k not in ("sha256", "files")}
    content["files"] = {
        name: {"sha256": exporter.file_sha256(path / name), "bytes": (path / name).stat().st_size}
        for name in exporter.FILES
    }
    result = {**content, "sha256": exporter.canonical_sha256(content)}
    (path / "manifest.json").write_text(json.dumps(result))
    return result


def baseline_for(path, source):
    path.mkdir()
    rows = []
    for i, row in enumerate(source.rows):
        for question in row.questions:
            rows.append(
                {
                    **question,
                    "subject_uid": row.subject_uid,
                    "strat_fold": 1,
                    "artifact_types": row.artifact_types,
                    "has_rationale": True,
                    "signal_index": i,
                }
            )
    train = pa.Table.from_pylist(rows)
    pq.write_table(train, path / "train.parquet")
    heldout = {
        **rows[0],
        "record_id": "heldout",
        "subject_uid": "heldout-patient",
        "signal_index": len(source.rows),
    }
    for split in ("selection", "validation", "test"):
        pq.write_table(pa.Table.from_pylist([heldout], schema=train.schema), path / f"{split}.parquet")
    records = [r.record_id for r in source.rows] + ["heldout"]
    (path / "records.json").write_text(json.dumps(records))
    signal = np.stack([x for _, x in source.load(records[:-1])] + [np.zeros((12, 1000), np.float32)])
    np.save(path / "signals.npy", signal)
    return write_manifest(
        path,
        {
            "schema_version": 1,
            "source": source.provenance["source"],
            "source_manifest": {"sha256": source.provenance["manifest_sha256"]},
            "preparation": {"train_role": "pool"},
            "signal": {"shape": list(signal.shape), "dtype": "float32", "fs_hz": 100},
            "splits": {"train": {"questions": len(rows)}, "selection": {"questions": 1}},
        },
    )


@pytest.fixture
def data(tmp_path):
    source = memory_source(4)
    source.provenance.update(
        source={"dataset": "fixture", "sha256": "source-hash"}, manifest_sha256="split-hash"
    )
    baseline = tmp_path / "baseline"
    baseline_for(baseline, source)
    corpus = tmp_path / "corpus"
    build_corpus(
        BuildConfig(
            name="export-fixture",
            components=["source", "grounded", "acquisition"],
            acquisition=AcquisitionConfig(operations=["crop", "gain"], per_waveform=1),
            pairs=PairsConfig(natural_per_unit=2, variation_fraction=0),
        ),
        source,
        corpus,
    )
    return baseline, corpus, tmp_path / "prepared"


def test_export_preserves_baseline_and_exact_signals_with_explicit_exclusions(data, monkeypatch):
    baseline, corpus, output = data
    real_read = pq.read_table

    def read_without_test_labels(path, *args, **kwargs):
        assert str(path).endswith(("train.parquet", "selection.parquet")), "held-out labels were opened"
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", read_without_test_labels)
    info = exporter.export_prepared([corpus], baseline, output)
    exported = real_read(output / "train.parquet")
    original = real_read(baseline / "train.parquet")
    assert exported.slice(0, original.num_rows).equals(original)
    for split in ("selection", "validation", "test"):
        assert (output / f"{split}.parquet").read_bytes() == (baseline / f"{split}.parquet").read_bytes()
    old_signals, new_signals = np.load(baseline / "signals.npy"), np.load(output / "signals.npy")
    assert np.array_equal(new_signals[: len(old_signals)], old_signals)
    records = json.loads((output / "records.json").read_text())
    assert records[: len(old_signals)] == json.loads((baseline / "records.json").read_text())
    loaded = load_corpus(corpus)
    new_tasks = {t.task_id: t for t in loaded.tasks}
    rows = exported.slice(original.num_rows).to_pylist()
    assert rows
    for row in rows:
        task = new_tasks[row["task_id"]]
        assert len(task.inputs) == 1
        assert task.inputs[0].signal.num_samples == 1000
        assert np.array_equal(new_signals[row["signal_index"]], loaded.signal(row["record_id"]))
        assert row["clinical_context"] == task.context["clinical_context"][0]
        assert row["answer"] == task.answer_text
        assert row["answer_options"] == list(task.options)
    assert any(row["answer_options"] == [] for row in rows)
    counts = info["qa_export"]["corpora"][0]
    assert counts["counts"]["original_tasks_already_in_baseline"] == original.num_rows
    assert counts["counts"]["numeric_tasks"] > 0
    assert counts["excluded"]["multiple_ecg_inputs"] > 0
    assert counts["excluded"]["incompatible_signal_length"] > 0
    assert info["qa_export"]["appended_waveforms"]
    assert info["sha256"] != exporter.verify_prepared(baseline)["sha256"]
    assert exporter.export_prepared([corpus], baseline, output) == info


def test_export_refuses_changed_source_and_patient_lineage(data, monkeypatch):
    baseline, corpus, output = data
    iterate = exporter._tasks

    def wrong_patient(path, info):
        for task in iterate(path, info):
            refs = tuple(replace(ref, root_subject_uids=("heldout-patient",)) for ref in task.inputs)
            yield replace(task, inputs=refs)

    monkeypatch.setattr(exporter, "_tasks", wrong_patient)
    with pytest.raises(PermissionError, match="lineage is outside"):
        exporter.export_prepared([corpus], baseline, output)
    assert not output.exists()
    monkeypatch.setattr(exporter, "_tasks", iterate)
    manifest = json.loads((baseline / "manifest.json").read_text())
    manifest["source_manifest"]["sha256"] = "different-split"
    write_manifest(baseline, manifest)
    with pytest.raises(PermissionError, match="split manifest differs"):
        exporter.export_prepared([corpus], baseline, output)


def test_export_refuses_corrupt_or_changed_existing_output(data):
    baseline, corpus, output = data
    exporter.export_prepared([corpus], baseline, output)
    with (output / "train.parquet").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="file length mismatch"):
        exporter.export_prepared([corpus], baseline, output)
    assert (output / "train.parquet").read_bytes().endswith(b"tampered")


def test_export_never_drops_unrepresentable_context(data, monkeypatch):
    baseline, corpus, output = data
    iterate = exporter._tasks

    def extra_context(path, info):
        for task in iterate(path, info):
            yield replace(task, context={**task.context, "supplied_rule": "essential context"})

    monkeypatch.setattr(exporter, "_tasks", extra_context)
    with pytest.raises(ValueError, match="unsupported student context"):
        exporter.export_prepared([corpus], baseline, output)
    assert not output.exists()


def test_real_waveform_must_match_baseline_exactly(data):
    baseline, corpus, output = data
    signals = np.load(baseline / "signals.npy")
    signals[0, 0, 0] += 1
    np.save(baseline / "signals.npy", signals)
    write_manifest(baseline, json.loads((baseline / "manifest.json").read_text()))
    with pytest.raises(ValueError, match="corpus waveform differs from baseline"):
        exporter.export_prepared([corpus], baseline, output)
    assert not output.exists()


def test_question_only_augmentation_keeps_signal_and_record_files_identical(tmp_path):
    source = memory_source(2)
    source.provenance.update(
        source={"dataset": "fixture", "sha256": "source-hash"}, manifest_sha256="split-hash"
    )
    baseline, corpus, output = (tmp_path / name for name in ("baseline", "paraphrase", "prepared"))
    baseline_for(baseline, source)
    build_corpus(
        BuildConfig(
            name="paraphrase",
            components=["paraphrase"],
            teacher={"provider": "template", "store": str(tmp_path / "teacher.jsonl"), "mode": "record"},
        ),
        source,
        corpus,
    )
    info = exporter.export_prepared([corpus], baseline, output)
    assert info["qa_export"]["added_training_rows"] == 4
    assert info["qa_export"]["appended_waveforms"] == []
    for name in ("signals.npy", "records.json"):
        assert (output / name).read_bytes() == (baseline / name).read_bytes()
    rows = pq.read_table(output / "train.parquet").to_pylist()
    assert rows[:4] == pq.read_table(baseline / "train.parquet").to_pylist()
    assert all(row["task_id"].startswith("qa-") for row in rows[4:])


def test_failed_publication_does_not_leave_a_partial_prepared_folder(data, monkeypatch):
    baseline, corpus, output = data

    def interrupted_copy(*_args, **_kwargs):
        raise OSError("simulated interrupted copy")

    monkeypatch.setattr(exporter.shutil, "copyfile", interrupted_copy)
    with pytest.raises(OSError, match="simulated interrupted copy"):
        exporter.export_prepared([corpus], baseline, output)
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}-*"))
