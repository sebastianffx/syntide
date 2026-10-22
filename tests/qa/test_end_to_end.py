"""Offline end to end: source records -> evidence -> QA tasks -> verification -> accepted export -> explorer
-> frozen QA trials -> offline consumer -> recomputed validation -> frozen held-out evaluation."""

from __future__ import annotations

import json
from dataclasses import replace

import pyarrow.parquet as pq
import pytest

from syntide.data.leakage import LeakageError
from syntide.eval import qa_holdout
from syntide.eval.tracking import TrackerConfig
from syntide.explore.data import open_source as open_explorer_source
from syntide.generate.qa.build import BuildConfig, build_corpus
from syntide.generate.qa.corpus import load_corpus
from syntide.generate.qa.sources import open_source
from syntide.qa_trial import (
    QAArmConfig,
    QAProtocolConfig,
    load_any_spec,
    plan_qa_trials,
    recompute_from_predictions,
)


@pytest.fixture(scope="module")
def corpora(qa_registry, tmp_path_factory):
    out = tmp_path_factory.mktemp("qa-corpora")
    reg, manifest = qa_registry.registry, qa_registry.manifest
    train = open_source(reg, manifest, "pool", purpose="training")
    built = {"registry": reg, "manifest": manifest}
    for name, components in (("real", ["source"]), ("grounded", ["grounded"]), ("acq", ["acquisition"])):
        cfg = BuildConfig(name=name, components=components, unit_size=4)
        build_corpus(cfg, train, out / name)
        built[name] = out / name
    for name, role, suite_role in (("selection", "val", "selection"), ("test", "test", "test")):
        source = open_source(reg, manifest, role, purpose="evaluation")
        cfg = BuildConfig(
            name=name,
            purpose="evaluation",
            suite_role=suite_role,
            components=["source", "grounded"],
            source_dedupe=True,
        )
        build_corpus(cfg, source, out / name)
        built[name] = out / name
    return built


def _registry_paths(corpora):
    return corpora["registry"], corpora["manifest"]


def test_sources_respect_roles_and_evidence_matches_the_known_rates(qa_registry):
    train = open_source(qa_registry.registry, qa_registry.manifest, "pool")
    with pytest.raises(PermissionError):
        open_source(qa_registry.registry, qa_registry.manifest, "test", purpose="training")
    with pytest.raises(PermissionError):
        train.load(["ptbxl-17"])  # a test-fold record is outside the training population
    with pytest.raises(LeakageError):
        train.check_ancestry([("synthetic:x", ("ptbxl-17",), ("ptb-xl:509",))])
    from syntide.generate.qa.evidence import EvidenceExtractor

    for ref, x in train.load([row.record_id for row in train.rows[:6]]):
        ev = {e.attribute: e for e in EvidenceExtractor().extract(x, ref)}
        assert ev["heart_rate_bpm"].value == pytest.approx(qa_registry.rates[ref.waveform_id], abs=3)


def test_accepted_export_reopens_in_the_explorer_with_one_point_per_waveform(corpora):
    corpus = load_corpus(corpora["grounded"])
    source = open_explorer_source({"dataset": "jsonl", "input": str(corpora["grounded"] / "catalog.jsonl")})
    assert len(source.rows) == len(corpus.inputs)
    assert sum(len(row["texts"]) for row in source.rows) == len(corpus.tasks)
    pairs = [t for t in corpus.tasks if len(t.inputs) == 2]
    assert pairs, "natural comparisons are part of the grounded corpus"
    texts = [text for row in source.rows for text in row["texts"]]
    for task in pairs:  # a pair is one task, attached once, to its first input
        assert sum(text["text_id"] == task.task_id for text in texts) == 1
    loaded = source.load([source.rows[0]["record_id"]])[0]
    assert loaded["signal"].shape == (12, 1000)


def test_frozen_qa_trials_train_recompute_and_evaluate_once(corpora, tmp_path):
    from syntide_trainers import qa_consumer

    protocol = QAProtocolConfig(
        id="qa-e2e",
        corpora={
            "real": str(corpora["real"]),
            "grounded": str(corpora["grounded"]),
            "acq": str(corpora["acq"]),
        },
        selection=str(corpora["selection"]),
        arms={
            "real_qa": QAArmConfig(["real"]),
            "grounded": QAArmConfig(["real", "grounded"]),
            "grounded_acq": QAArmConfig(["real", "grounded", "acq"]),
        },
        supervision=["answer_only", "rationale"],
        seeds=[0, 1],
        batch_size=4,
        max_steps=6,
        val_every=3,
        decoding={"method": "test-consumer"},
        budget={"primary": "6 updates x 4 examples"},
    )
    runs_dir = tmp_path / "runs"
    written = plan_qa_trials(protocol, runs_dir)
    assert len(written) == 3 * 2 * 2
    assert plan_qa_trials(protocol, runs_dir) == []  # the same plan is already there
    tracker = TrackerConfig("files")
    results = {}
    for run_dir in written:
        result = qa_consumer.train(run_dir, tracker)
        assert result["status"] == "complete"
        spec = load_any_spec(run_dir)
        selection = load_corpus(spec.selection["path"]).tasks
        recomputed = recompute_from_predictions(run_dir, selection)
        history = {entry["step"]: entry for entry in result["val_history"]}
        assert recomputed["best"]["val/qa_accuracy"] == pytest.approx(
            history[result["best_step"]]["val/qa_accuracy"]
        )
        results[run_dir] = result
        assert qa_consumer.train(run_dir, tracker) is None  # complete runs are skipped

    exposure = json.loads((written[0] / "exposure.json").read_text())
    assert exposure["examples"] == 6 * 4

    protocol_dir = runs_dir / "qa-e2e"
    rules = qa_holdout.QAEvalRules(
        suites={"test": str(corpora["test"])},
        primary_suite="test",
        primary=["grounded/rationale", "real_qa/rationale"],
    )
    with pytest.raises(qa_holdout.QAFreezeError, match="does not exist"):
        qa_holdout.score(protocol_dir, rules)  # never scored before the rules are frozen
    qa_holdout.freeze(protocol_dir, rules)
    changed = replace(rules, n_boot=500)
    with pytest.raises(qa_holdout.QAFreezeError, match="changed"):
        qa_holdout.score(protocol_dir, changed)
    assert not (protocol_dir / qa_holdout.CLAIM_FILE).exists()
    written_scores = qa_holdout.score(protocol_dir, rules)
    assert (protocol_dir / qa_holdout.CLAIM_FILE).exists()
    assert any("__context_only" in p.name for p in written_scores)
    assert any("__waveform_swap" in p.name for p in written_scores)
    assert qa_holdout.score(protocol_dir, rules) == []  # scored once
    report = qa_holdout.report(protocol_dir, rules)
    suite = report["suites"]["test"]
    assert set(suite["arms"]) == {f"{a}/{s}" for a in protocol.arms for s in protocol.supervision}
    assert suite["contrasts"] and suite["contrasts"][0]["primary"]
    first = written[0]
    table = pq.read_table(first / "qa_test.parquet").to_pylist()
    assert {row["task_id"] for row in table} == {t.task_id for t in load_corpus(corpora["test"]).tasks}
    val = qa_holdout.report(protocol_dir, rules, split="val")
    assert "selection" in val["suites"]


def test_a_checkpoint_changed_after_freezing_is_refused_before_scoring(corpora, tmp_path):
    from syntide_trainers import qa_consumer

    protocol = QAProtocolConfig(
        id="qa-tamper",
        corpora={"real": str(corpora["real"])},
        selection=str(corpora["selection"]),
        arms={"real_qa": QAArmConfig(["real"])},
        seeds=[0],
        batch_size=2,
        max_steps=2,
        val_every=1,
    )
    run_dir = plan_qa_trials(protocol, tmp_path)[0]
    qa_consumer.train(run_dir, TrackerConfig("files"))
    rules = qa_holdout.QAEvalRules(suites={"test": str(corpora["test"])}, primary_suite="test")
    qa_holdout.freeze(tmp_path / "qa-tamper", rules)
    best = run_dir / "best.json"
    original = best.read_text()
    best.write_text(original.replace('"step"', '"step" ', 1))
    with pytest.raises(qa_holdout.QAFreezeError, match="changed after training"):
        qa_holdout.score(tmp_path / "qa-tamper", rules)
    best.write_text(original)
    other = BuildConfig(name="test", purpose="evaluation", suite_role="test", components=["source"])
    moved = tmp_path / "other-test-suite"
    build_corpus(other, open_source(*_registry_paths(corpora), "test", purpose="evaluation"), moved)
    changed = replace(rules, suites={"test": str(moved)})  # another dataset version at the frozen name
    with pytest.raises(qa_holdout.QAFreezeError, match="suites"):
        qa_holdout.score(tmp_path / "qa-tamper", changed)
    assert not (tmp_path / "qa-tamper" / qa_holdout.CLAIM_FILE).exists()
