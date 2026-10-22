"""Bounded, restartable, atomic corpus construction; immutable outputs; replayable teacher responses."""

from __future__ import annotations

import json

import pytest
from tests.qa.fixtures import memory_source

from syntide.generate.qa.build import BuildConfig, build_corpus
from syntide.generate.qa.corpus import CorpusError, load_corpus, verify_corpus_files
from syntide.generate.qa.teacher import TeacherError


def config(**overrides) -> BuildConfig:
    return BuildConfig(name="c", components=["source", "grounded", "acquisition"], unit_size=3, **overrides)


def test_an_interrupted_build_publishes_nothing_and_resumes_to_the_same_corpus(tmp_path):
    source = memory_source(8)
    reference = build_corpus(config(), source, tmp_path / "reference")
    out = tmp_path / "resumed"
    assert build_corpus(config(), source, out, stop_after_units=1) is None
    assert not out.exists() and (tmp_path / "resumed.staging").exists()
    with pytest.raises(CorpusError, match="another configuration"):
        build_corpus(config(seed=1), source, out)  # a different build never mixes into the staged units
    info = build_corpus(config(), source, out)
    assert info["corpus_id"] == reference["corpus_id"]
    assert not (tmp_path / "resumed.staging").exists()
    with pytest.raises(FileExistsError):
        build_corpus(config(), source, out)  # published corpora are immutable


def test_publication_is_verified_and_edits_are_detected(tmp_path):
    info = build_corpus(config(), memory_source(4), tmp_path / "c")
    assert verify_corpus_files(tmp_path / "c")["corpus_id"] == info["corpus_id"]
    catalog = tmp_path / "c" / "catalog.jsonl"
    catalog.write_text(catalog.read_text().replace("ptbxl-1", "ptbxl-9", 1))
    with pytest.raises(CorpusError, match="changed since publication"):
        load_corpus(tmp_path / "c")


def test_each_waveform_is_stored_once_however_many_questions_it_has(tmp_path):
    info = build_corpus(config(), memory_source(4), tmp_path / "c")
    corpus = load_corpus(tmp_path / "c")
    signals = list((tmp_path / "c" / "signals").glob("*.npy"))
    assert len(signals) == info["counts"]["waveforms"] == len(corpus.inputs)
    assert info["counts"]["qa_examples"] > 5 * len(signals)
    per_waveform = {}
    for task in corpus.tasks:
        per_waveform.setdefault(task.inputs[0].waveform_id, 0)
        per_waveform[task.inputs[0].waveform_id] += 1
    catalog = [json.loads(line) for line in (tmp_path / "c" / "catalog.jsonl").read_text().splitlines()]
    assert len(catalog) == len(corpus.inputs)
    assert {row["record_id"]: len(row["texts"]) for row in catalog if row["texts"]} == per_waveform
    counts = info["counts"]
    for key in (
        "unique_patients",
        "root_recordings",
        "waveforms",
        "semantic_tasks",
        "qa_examples",
        "paraphrases",
        "attempts",
        "accepted",
        "rejected",
        "tokens",
    ):
        assert key in counts
    assert counts["unique_patients"] == counts["root_recordings"] == 4


def test_teacher_responses_replay_to_the_same_task_ids(tmp_path):
    store = tmp_path / "teacher.jsonl"
    record = BuildConfig(
        name="t",
        components=["source", "paraphrase"],
        unit_size=4,
        teacher={"provider": "template", "store": str(store), "mode": "record"},
    )
    first = build_corpus(record, memory_source(4), tmp_path / "recorded")
    assert first["counts"]["paraphrases"] > 0
    replay = BuildConfig(
        **{**record.__dict__, "teacher": {"provider": "template", "store": str(store), "mode": "replay"}}
    )
    second = build_corpus(replay, memory_source(4), tmp_path / "replayed")
    ids = lambda path: sorted(t.task_id for t in load_corpus(path).tasks)  # noqa: E731
    assert ids(tmp_path / "recorded") == ids(tmp_path / "replayed")
    assert first["corpus_id"] == second["corpus_id"]
    paraphrased = [t for t in load_corpus(tmp_path / "replayed").tasks if t.paraphrase != "seed"]
    assert all(t.generation["paraphrase_teacher"]["response_sha256"] for t in paraphrased)
    sources = {t.semantic_key for t in load_corpus(tmp_path / "replayed").tasks if t.paraphrase == "seed"}
    assert {t.semantic_key for t in paraphrased} <= sources  # a paraphrase is the same semantic task
    empty = BuildConfig(
        **{
            **record.__dict__,
            "teacher": {"provider": "template", "store": str(tmp_path / "none.jsonl"), "mode": "replay"},
        }
    )
    with pytest.raises(TeacherError, match="replay never calls"):
        build_corpus(empty, memory_source(4), tmp_path / "missing")


def test_training_and_evaluation_purposes_do_not_mix(tmp_path):
    with pytest.raises(PermissionError):
        build_corpus(config(purpose="evaluation", suite_role="selection"), memory_source(2), tmp_path / "x")
    held_out = memory_source(2, purpose="evaluation")
    with pytest.raises(ValueError, match="evaluation suites hold"):
        build_corpus(config(purpose="evaluation", suite_role="selection"), held_out, tmp_path / "y")
    with pytest.raises(PermissionError):
        build_corpus(BuildConfig(name="z", components=["source"]), held_out, tmp_path / "z")
