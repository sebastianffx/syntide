"""Dataset exploration preserves question links and waveform origins."""

import csv
import json
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest
import wfdb
from tests.data import conftest as data_fixtures
from timenet.engine import run_pipeline
from timenet_connectors.datasets.physionet.ecg_qa_cot.connector import (
    EcgQaCotConnector,
    EcgQaCotSource,
)

from syntide.data.challenge import ChallengeSpec, build_manifest, save_manifest
from syntide.data.leakage import LeakageError
from syntide.data.schema import LEADS
from syntide.explore.data import open_source
from syntide.explore.qa import EcgQaView, check_qa_tasks

build_ptbxl_registry = data_fixtures.build_ptbxl_registry
built_ptbxl = data_fixtures.built_ptbxl


@pytest.fixture(scope="module")
def ecgqa(built_ptbxl, tmp_path_factory):
    """Build a small question-answering registry with the TimeNet connector."""
    root = tmp_path_factory.mktemp("explorer-qa")
    signal_root = root / "records500"
    bucket = signal_root / "00000"
    bucket.mkdir(parents=True)
    for ecg_id in range(1, 5):
        times = np.arange(500) / 500
        values = np.stack(
            [np.sin(times * (lead + 1)) for lead in range(len(LEADS))], axis=1
        )
        wfdb.wrsamp(
            f"{ecg_id:05d}_hr",
            fs=500,
            units=["mV"] * 12,
            sig_name=list(LEADS),
            p_signal=values,
            fmt=["16"] * 12,
            write_dir=str(bucket),
        )
    answers = root / "answers.csv"
    with answers.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["template_id", "classes"])
        writer.writerows(
            [(4, "['none', 'myocardial infarction']"), (27, "['no', 'yes']")]
        )
    qa_rows = [
        (
            "train",
            2,
            4,
            "single-choose",
            "Which diagnostic symptom does this ECG show?",
            "none",
        ),
        (
            "train",
            1,
            27,
            "single-verify",
            "Does this ECG show static noise?",
            "yes",
        ),
        (
            "train",
            1,
            4,
            "single-choose",
            "Which diagnostic symptom does this ECG show?",
            "none",
        ),
        (
            "validation",
            3,
            27,
            "single-verify",
            "Does this ECG show static noise?",
            "no",
        ),
        (
            "test",
            4,
            27,
            "single-verify",
            "Does this ECG show static noise?",
            "no",
        ),
    ]
    paths = []
    for split in ("train", "validation", "test"):
        path = root / f"{split}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "ecg_id",
                    "template_id",
                    "question_type",
                    "question",
                    "answer",
                    "rationale",
                    "clinical_context",
                ],
            )
            writer.writeheader()
            for (
                row_split,
                ecg_id,
                template,
                question_type,
                question,
                answer,
            ) in qa_rows:
                if row_split == split:
                    writer.writerow(
                        {
                            "ecg_id": f"[{ecg_id}]",
                            "template_id": template,
                            "question_type": question_type,
                            "question": question,
                            "answer": answer,
                            "rationale": (
                                f"Looking at ECG {ecg_id}.\nAnswer: {answer}"
                            ),
                            "clinical_context": (
                                f"Patient {ecg_id}.\nSignal quality: noted."
                            ),
                        }
                    )
        paths.append((split, path))

    async def local_download(self, cache_dir):
        return [
            EcgQaCotSource(
                records_root=signal_root,
                answers_path=answers,
                cot_csvs=tuple(paths),
            )
        ]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(EcgQaCotConnector, "download_async", local_download)
        run_pipeline(
            EcgQaCotConnector(),
            built_ptbxl.registry,
            cache_dir=tmp_path_factory.mktemp("explorer-qa-cache"),
            keep_cache=True,
        )
    return EcgQaView(built_ptbxl.registry)


def _catalog(tmp_path, rows):
    path = tmp_path / "catalog.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return {"dataset": "jsonl", "input": str(path)}


def _record(tmp_path, record_id="a", **kwargs):
    np.save(
        tmp_path / f"{record_id}.npy",
        np.arange(16, dtype=np.float32).reshape(2, 8),
    )
    return {
        "record_id": record_id,
        "signal_path": f"{record_id}.npy",
        "sample_rate": 100,
        "channels": ["left", "right"],
        "units": "mV",
        "subject_uid": f"patient:{record_id}",
        "split": "train",
        "labels": ["rare"],
        **kwargs,
    }


def test_generic_qa_rows_group_recordings_and_only_train_arrays_are_loaded(
    tmp_path, monkeypatch
):
    a = _record(tmp_path, question="What rhythm?", answer="sinus", text_id="q1")
    b = _record(tmp_path, "b", text="Sinus rhythm")
    test = _record(tmp_path, "heldout", split="test")
    # Discovery needs only held-out metadata to enforce the split; its waveform
    # need not exist.
    (tmp_path / "heldout.npy").unlink()
    config = _catalog(
        tmp_path,
        [
            a,
            {**a, "question": "Any noise?", "answer": "no", "text_id": "q2"},
            b,
            test,
        ],
    )
    real_load = np.load
    seen = []

    def read(path, **kwargs):
        seen.append(str(path))
        return real_load(path, **kwargs)

    monkeypatch.setattr(np, "load", read)
    source = open_source(config)
    assert not seen
    assert [row["record_id"] for row in source.rows] == ["a", "b"]
    assert source.provenance["population_records"] == 2
    assert source.provenance["population_texts"] == 3
    assert source.provenance["population_distinct_questions"] == 2
    assert len(source.rows[0]["texts"]) == 2
    assert source.rows[0]["texts"][0]["text"] == ""
    assert source.rows[1]["texts"][0]["text"] == "Sinus rhythm"
    loaded = source.load(["b", "a"])
    assert [row["record_id"] for row in loaded] == ["b", "a"]
    assert [row["signal"].shape for row in loaded] == [(2, 8), (2, 8)]
    assert len(seen) == 2 and all("heldout" not in value for value in seen)
    with pytest.raises(ValueError, match="outside the discovery population"):
        source.load(["heldout"])


def test_generic_detects_subject_leakage_without_loading_signals(tmp_path):
    a = _record(tmp_path)
    b = _record(tmp_path, "b", subject_uid=a["subject_uid"], split="validation")
    with pytest.raises(LeakageError, match="multiple dataset splits"):
        open_source(_catalog(tmp_path, [a, b]))


@pytest.mark.parametrize(
    "role", ["val", "validation", "test", "withheld", "heldout", "unused"]
)
def test_heldout_roles_cannot_be_discovery_sources(role):
    with pytest.raises(
        ValueError, match="cannot be used for generation discovery"
    ):
        open_source({"dataset": "jsonl", "role": role})


def test_generic_needs_explicit_split_and_positive_sample_rate(tmp_path):
    row = _record(tmp_path)
    del row["split"]
    with pytest.raises(ValueError, match="split"):
        open_source(_catalog(tmp_path, [row]))
    row["split"] = "train"
    row["sample_rate"] = 0
    with pytest.raises(ValueError, match="sample_rate"):
        open_source(_catalog(tmp_path, [row]))


def test_generic_repeated_record_must_have_consistent_identity(tmp_path):
    row = _record(tmp_path)
    with pytest.raises(ValueError, match="conflicting recording metadata"):
        open_source(_catalog(tmp_path, [row, {**row, "split": "test"}]))
    with pytest.raises(ValueError, match="conflicting recording metadata"):
        open_source(_catalog(tmp_path, [row, {**row, "sample_rate": 200}]))


def test_generic_duplicate_task_ids_deduplicate_exact_rows_and_reject_conflicts(
    tmp_path,
):
    row = _record(tmp_path, question="Rhythm?", answer="sinus", task_id="q1")
    source = open_source(_catalog(tmp_path, [row, row]))
    assert source.provenance["population_records"] == 1
    assert source.provenance["population_texts"] == 1
    with pytest.raises(ValueError, match="reused for different text"):
        open_source(_catalog(tmp_path, [row, {**row, "answer": "other"}]))


def test_fingerprint_rejects_waveform_changes_during_analysis(
    tmp_path,
):
    row = _record(tmp_path)
    config = _catalog(tmp_path, [row])
    before = open_source(config)
    assert open_source(config).provenance == before.provenance
    np.save(tmp_path / "a.npy", np.ones((2, 8), dtype=np.float32))
    after = open_source(config)
    assert (
        before.provenance["source_sha256"] != after.provenance["source_sha256"]
    )
    assert (
        before.provenance["catalog_sha256"]
        == after.provenance["catalog_sha256"]
    )
    with pytest.raises(ValueError, match="changed after cataloging"):
        before.load(["a"])


def test_generic_supports_nested_texts_and_explicit_npz_key(tmp_path):
    row = _record(
        tmp_path,
        signal_path="signals.npz",
        signal_key="waveform",
        texts=[
            {
                "text_id": "sentence",
                "text": "Sinus rhythm",
                "metadata": {"language": "en"},
            },
            {
                "text_id": "qa",
                "question": "Rhythm?",
                "answer": "sinus",
                "rationale": "Regular rhythm",
            },
        ],
    )
    np.savez(
        tmp_path / "signals.npz",
        waveform=np.ones((2, 4)),
        another=np.zeros((2, 4)),
    )
    source = open_source(_catalog(tmp_path, [row]))
    assert source.rows[0]["texts"][0]["metadata"] == {"language": "en"}
    assert source.rows[0]["texts"][1]["rationale"] == "Regular rhythm"
    assert source.load(["a"])[0]["signal"].shape == (2, 4)


@pytest.mark.parametrize(
    "signal,error",
    [
        (np.zeros((8, 2)), "channels do not match"),
        (np.zeros(8), "expected non-empty"),
        (np.full((2, 8), np.nan), "non-finite"),
    ],
)
def test_generic_rejects_malformed_waveforms_on_load(tmp_path, signal, error):
    row = _record(tmp_path)
    np.save(tmp_path / "a.npy", signal)
    source = open_source(_catalog(tmp_path, [row]))
    with pytest.raises(ValueError, match=error):
        source.load(["a"])


def test_catalog_reads_rhythm_labels_without_waveforms(
    built_ptbxl, monkeypatch, tmp_path
):
    from timenet.reader import TimeFReader

    statements = tmp_path / "scp_statements.csv"
    statements.write_text(
        ",description,diagnostic,form,rhythm,diagnostic_class\n"
        "SR,Sinus rhythm,,,1.0,\nNORM,Normal ECG,1.0,,,NORM\n"
    )

    def no_waveforms(*args, **kwargs):
        raise AssertionError(
            "Catalog construction attempted to load a waveform"
        )

    with monkeypatch.context() as patch:
        patch.setattr(TimeFReader, "_load_signal", no_waveforms)
        source = open_source(
            {
                "dataset": "ptbxl",
                "registry": str(built_ptbxl.registry),
                "scp_statements": str(statements),
            }
        )
    assert [row["record_id"] for row in source.rows] == ["ptbxl-1", "ptbxl-2"]
    assert {
        "scp:SR",
        "scp:NORM",
        "superclass:NORM",
        "artifact:static_noise",
    } <= set(source.rows[0]["labels"])
    assert (
        source.rows[0]["metadata"]["label_descriptions"]["scp:SR"]
        == "Sinus rhythm"
    )
    assert source.provenance["label_definitions"]["scp:SR"]["categories"] == [
        "rhythm"
    ]
    loaded = source.load(["ptbxl-2"])[0]
    assert loaded["signal"].shape == (12, 100)
    assert loaded["sample_rate"] == 100 and loaded["units"] == "mV"
    assert loaded["split"] == "train"


def test_ecgqa_groups_questions_and_uses_ptbxl_split(ecgqa, built_ptbxl):
    source = open_source(
        {"dataset": "ecgqa", "registry": str(built_ptbxl.registry)}
    )
    assert [row["record_id"] for row in source.rows] == ["ptbxl-1", "ptbxl-2"]
    assert source.provenance["population_records"] == 2
    assert source.provenance["population_texts"] == 3
    first = source.rows[0]["texts"][0]
    assert first["question"] and first["answer"] and first["rationale"]
    assert first["text"] == ""
    assert first["metadata"]["template_id"] == 27
    assert source.provenance["source"] == ecgqa.source()
    assert "unreviewed" in source.provenance["label_effect_status"]


def test_ecgqa_release_split_cannot_move_a_ptbxl_test_record_into_discovery(
    ecgqa, built_ptbxl, monkeypatch
):
    original = EcgQaView._cot_splits

    def mismatched(self):
        return {**original(self), "ptbxl-1": "test", "ptbxl-4": "train"}

    monkeypatch.setattr(EcgQaView, "_cot_splits", mismatched)
    source = open_source(
        {"dataset": "ecgqa", "registry": str(built_ptbxl.registry)}
    )
    assert [row["record_id"] for row in source.rows] == ["ptbxl-1", "ptbxl-2"]
    assert source.rows[0]["texts"][0]["metadata"]["cot_split"] == "test"


def test_private_qa_manifest_gate_and_compatible_source(
    ecgqa, built_ptbxl, tmp_path
):
    manifest = build_manifest(
        ecgqa.index(), ChallengeSpec(), 1.0, ecgqa.source()
    )
    path = tmp_path / "qa-manifest.json"
    save_manifest(manifest, path)
    source = open_source(
        {
            "dataset": "ecgqa",
            "registry": str(built_ptbxl.registry),
            "manifest": str(path),
        }
    )
    assert source.provenance["source"] == ecgqa.source()
    assert (
        source.provenance["source"]["signals"] == built_ptbxl.adapter.source()
    )
    assert source.provenance["population_texts"] == 3


@pytest.mark.parametrize(
    "damage,match",
    [
        ("duplicate", "task ids are repeated"),
        ("record", "outside the index"),
        ("patient", "another patient"),
        ("split", "both training roles and val/test"),
    ],
)
def test_private_qa_gate_rejects_broken_task_joins(ecgqa, damage, match):
    index = ecgqa.index()
    manifest = build_manifest(index, ChallengeSpec(), 1.0, ecgqa.source())
    tasks = (
        ecgqa.tasks()
        .select(["task_id", "record_id", "subject_uid"])
        .to_pylist()
    )
    if damage == "duplicate":
        tasks[1]["task_id"] = tasks[0]["task_id"]
    elif damage == "record":
        tasks[0]["record_id"] = "unknown"
    elif damage == "patient":
        tasks[0]["subject_uid"] = "another-patient"
    else:
        manifest = replace(
            manifest,
            roles={
                **manifest.roles,
                "pool": (*manifest.roles["pool"], manifest.roles["test"][0]),
            },
        )
    with pytest.raises(LeakageError, match=match):
        check_qa_tasks(pa.Table.from_pylist(tasks), manifest, index)


def test_manifest_default_is_depleted_and_validates_source(
    built_ptbxl, tmp_path
):
    adapter = built_ptbxl.adapter
    manifest = build_manifest(
        adapter.index(), ChallengeSpec(), 1.0, adapter.source()
    )
    path = tmp_path / "pool.json"
    save_manifest(manifest, path)
    source = open_source(
        {
            "dataset": "ptbxl",
            "registry": str(built_ptbxl.registry),
            "manifest": str(path),
        }
    )
    assert source.provenance["role"] == "depleted"
    assert source.provenance["manifest_sha256"] == manifest.sha256
    assert set(row["record_id"] for row in source.rows) == set(
        manifest.roles["depleted"]
    )
    wrong = build_manifest(
        adapter.index(), ChallengeSpec(), 1.0, {"sha256": "wrong"}
    )
    save_manifest(wrong, tmp_path / "wrong.json")
    with pytest.raises(LeakageError, match="not from the local data"):
        open_source(
            {
                "dataset": "ptbxl",
                "registry": str(built_ptbxl.registry),
                "manifest": str(tmp_path / "wrong.json"),
            }
        )


def test_pool_role_cannot_reintroduce_withheld_records(
    synthetic_index, tmp_path, monkeypatch
):
    source = {"dataset": "test", "sha256": "test"}
    manifest = build_manifest(synthetic_index, ChallengeSpec(), 0.5, source)
    assert manifest.roles["withheld"]
    path = tmp_path / "pool.json"
    save_manifest(manifest, path)

    class Adapter:
        def __init__(self, *args):
            pass

        def index(self):
            return synthetic_index

        def source(self):
            return source

    monkeypatch.setattr("syntide.explore.data.PtbXlAdapter", Adapter)
    with pytest.raises(LeakageError, match="includes withheld or held-out"):
        open_source(
            {
                "dataset": "ptbxl",
                "registry": str(tmp_path),
                "manifest": str(path),
                "role": "pool",
            }
        )
