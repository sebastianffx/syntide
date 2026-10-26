"""Generation retries preserve completed and rejected candidates."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from syntide.generate.full.api import GeneratedSignal
from syntide.generate.qa.generated import load_bundle


@pytest.fixture
def driver():
    path = Path(__file__).parents[1] / "scripts/runpod/diffusets_generate.py"
    spec = importlib.util.spec_from_file_location("diffusets_generate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(driver, tmp_path):
    rows = [
        dict(
            slot_id=i,
            source_record_id="ptbxl-3",
            report="Sinus rhythm",
            age=50,
            sex="M",
            heart_rate=60 + 20 * i,
            seed=100 + i,
        )
        for i in range(3)
    ]
    plan = {
        "schema_version": 1,
        "plan_seed": 0,
        "manifest_sha256": driver.POOL_SHA,
        "candidates": rows,
    }
    np.save(tmp_path / "vector.npy", np.ones(1536, dtype=np.float32))
    entry = {
        "report": "Sinus rhythm",
        "age": 50,
        "sex": "M",
        "subject_uid": "ptb-xl:7",
        "text_embedding_path": "vector.npy",
        "text_embedding_model": driver.EMBEDDING_MODEL,
        "text_embedding_sha256": driver.sha(tmp_path / "vector.npy"),
        "embedding_source": (
            "pinned archive :: ptbxl_text_embed.csv :: ecg_id=3"
        ),
        "prompt_provenance": "unknown_legacy_input",
    }
    index = tmp_path / "embedding-index.json"
    driver.write(index, {"schema_version": 1, "entries": {"ptbxl-3": entry}})
    conditions, subjects, fingerprint = driver.load_conditions(rows, index)
    return plan, index, conditions, subjects, fingerprint


class FakeBackend:
    def __init__(self, fail_seed=None, malformed_seed=None):
        self.calls = []
        self.fail_seed = fail_seed
        self.malformed_seed = malformed_seed

    def validate_request(self, request):
        assert request.signal.num_samples == 1000

    def provenance(self):
        return {
            "generator": "diffusets@1",
            "device": "cpu",
            "num_inference_steps": 1000,
            "checkpoint_sha256": {"unet": "frozen"},
        }

    def generate(self, request):
        self.calls.append(request.seed)
        if request.seed == self.fail_seed:
            raise RuntimeError("simulated interruption")
        signal = (
            np.random.default_rng(request.seed)
            .normal(size=(12, 1000))
            .astype(np.float32)
        )
        if request.seed == self.malformed_seed:
            signal[0, 0] = np.nan
        return GeneratedSignal(signal, request.signal, {"test": True})


def review(stage, record_id, row, condition, subjects, extractor):
    accepted = row["seed"] != 101
    return {
        "status": "verified" if accepted else "rejected",
        "reasons": [] if accepted else ["low_confidence:heart_rate_bpm"],
        "evidence": [],
        "clinical_validation": "not_performed",
    }


def run(driver, inputs, output, backend):
    plan, _, conditions, subjects, fingerprint = inputs
    return driver.execute(
        plan,
        conditions,
        subjects,
        fingerprint,
        output,
        backend,
        expected_count=3,
        reviewer=review,
    )


def test_plan_enforces_fixed_count_unique_seed_and_training_ancestry(
    driver, inputs
):
    plan = inputs[0]
    assert len(driver.validate_plan(plan, {"ptbxl-3"}, expected_count=3)) == 3
    with pytest.raises(ValueError, match="exactly 5000"):
        driver.validate_plan(plan, {"ptbxl-3"})
    for field, value, match in [
        ("seed", 100, "unique"),
        ("source_record_id", "held-out", "outside"),
        ("age", 300, "Invalid age"),
        ("sex", "unknown", "Sex"),
    ]:
        bad = copy.deepcopy(plan)
        bad["candidates"][1][field] = value
        with pytest.raises(ValueError, match=match):
            driver.validate_plan(bad, {"ptbxl-3"}, expected_count=3)


def test_legacy_embeddings_never_claim_known_prompt(driver, inputs):
    plan, index, conditions, *_ = inputs
    assert all("embedding_input" not in c for c in conditions)
    assert all(c["embedding_source"] for c in conditions)
    data = driver.read(index)
    data["entries"]["ptbxl-3"]["prompt_provenance"] = "declared_matching"
    driver.write(index, data)
    with pytest.raises(ValueError, match="unknown original input"):
        driver.load_conditions(plan["candidates"], index)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("report", "Another report", "report differs"),
        ("age", 51, "demographics differ"),
        ("text_embedding_sha256", "bad", "hash/model mismatch"),
    ],
)
def test_embedding_binding_refuses_wrong_report_demographics_or_hash(
    driver, inputs, field, value, match
):
    plan, index, *_ = inputs
    data = driver.read(index)
    data["entries"]["ptbxl-3"][field] = value
    driver.write(index, data)
    with pytest.raises(ValueError, match=match):
        driver.load_conditions(plan["candidates"], index)


def test_resume_preserves_slots_and_rejections_without_topup(
    driver, inputs, tmp_path
):
    output = tmp_path / "generation"
    first = FakeBackend(fail_seed=102)
    with pytest.raises(RuntimeError, match="interruption"):
        run(driver, inputs, output, first)
    assert first.calls == [100, 101, 102]
    hashes = [
        driver.sha(output / "candidates" / f"{i:05d}" / "result.json")
        for i in range(2)
    ]
    assert not (output / "summary.json").exists()
    second = FakeBackend()
    summary = run(driver, inputs, output, second)
    assert second.calls == [102]
    assert (
        summary["candidate_count"] == 3
        and summary["accepted"] == 2
        and summary["rejected"] == 1
    )
    assert hashes == [
        driver.sha(output / "candidates" / f"{i:05d}" / "result.json")
        for i in range(2)
    ]
    assert len(list((output / "claims").glob("*.json"))) == 3
    assert driver.accepted_bundles(output, inputs[0], expected_count=3) == [
        output / "candidates/00000",
        output / "candidates/00002",
    ]
    third = FakeBackend()
    assert run(driver, inputs, output, third) == summary
    assert third.calls == []
    sample = load_bundle(output / "candidates/00000")[0]
    assert sample.source_record_ids == ("ptbxl-3",) and sample.seed == 100
    assert sample.load().shape == (12, 1000)
    assert "embedding_input" not in sample.conditioning


def test_changed_completed_signal_cannot_be_silently_regenerated(
    driver, inputs, tmp_path
):
    output = tmp_path / "generation"
    run(driver, inputs, output, FakeBackend())
    signal = output / "candidates/00000/signals/00000000.npy"
    signal.write_bytes(b"changed")
    backend = FakeBackend()
    with pytest.raises(ValueError, match="artifact changed"):
        run(driver, inputs, output, backend)
    assert backend.calls == []


def test_wrong_plan_cannot_resume_existing_output(driver, inputs, tmp_path):
    output = tmp_path / "generation"
    run(driver, inputs, output, FakeBackend())
    inputs[0]["candidates"][0]["heart_rate"] = 81
    with pytest.raises(ValueError, match="Frozen artifact differs"):
        run(driver, inputs, output, FakeBackend())


def test_malformed_candidate_consumes_slot_without_replacement(
    driver, inputs, tmp_path
):
    output = tmp_path / "generation"
    backend = FakeBackend(malformed_seed=100)
    summary = run(driver, inputs, output, backend)
    assert backend.calls == [100, 101, 102]
    assert summary["status_counts"] == {
        "rejected_technical": 1,
        "rejected": 1,
        "verified": 1,
    }
    assert not (output / "candidates/00000/run.json").exists()
    assert driver.accepted_bundles(output, inputs[0], expected_count=3) == [
        output / "candidates/00002"
    ]
    backend = FakeBackend()
    assert run(driver, inputs, output, backend) == summary
    assert backend.calls == []


def test_completion_manifest_is_required_before_preparation(
    driver, inputs, tmp_path
):
    output = tmp_path / "generation"
    with pytest.raises(RuntimeError):
        run(driver, inputs, output, FakeBackend(fail_seed=102))
    with pytest.raises(FileNotFoundError):
        driver.accepted_bundles(output, inputs[0], expected_count=3)


def test_real_measurement_gate_evidence_is_written_and_reloadable(
    driver, inputs, tmp_path
):
    plan, _, conditions, subjects, fingerprint = inputs
    output = tmp_path / "real-review"
    summary = driver.execute(
        plan,
        conditions,
        subjects,
        fingerprint,
        output,
        FakeBackend(),
        expected_count=3,
    )
    assert summary["candidate_count"] == 3
    for i in range(3):
        review_data = json.loads(
            (output / "candidates" / f"{i:05d}" / "acceptance.json").read_text()
        )
        assert review_data["clinical_validation"] == "not_performed"
        assert {
            "heart_rate_bpm",
            "beat_count",
            "rr_regularity",
            "amplitude_mv",
        } <= {e["attribute"] for e in review_data["evidence"]}
        assert {
            s for e in review_data["evidence"] for s in e["root_subject_uids"]
        } == {"ptb-xl:7"}


def test_plan_accepts_separate_content_and_raw_manifest_hashes(driver, inputs):
    plan = copy.deepcopy(inputs[0])
    plan["manifest_content_sha256"] = driver.POOL_SHA
    plan["manifest_sha256"] = "raw-file-bytes-hash"
    assert (
        driver.validate_plan(plan, {"ptbxl-3"}, expected_count=3)
        == plan["candidates"]
    )


def test_verified_candidate_bundles_compile_to_exact_four_measured_rhythm_tasks(
    driver, inputs, tmp_path
):
    from tests.qa.fixtures import fixture_ecg

    from syntide.generate.qa.build import build_corpus
    from syntide.generate.qa.sources import QASource, SourceRow

    class PulseBackend(FakeBackend):
        def generate(self, request):
            # Deliberately differs from requested HR: supervision must come from
            # the waveform.
            signal = fixture_ecg(70, fs=100, seed=request.seed)
            return GeneratedSignal(signal, request.signal, {"test": True})

    plan, _, conditions, subjects, fingerprint = inputs
    output = tmp_path / "generation"
    summary = driver.execute(
        plan,
        conditions,
        subjects,
        fingerprint,
        output,
        PulseBackend(),
        expected_count=3,
    )
    assert summary["accepted"] == 3
    bundles = driver.accepted_bundles(output, plan, expected_count=3)

    def forbidden_load(ids):
        raise AssertionError(
            "Generated-only corpus must not read any original waveform"
        )

    source = QASource(
        [SourceRow("ptbxl-3", "ptb-xl:7", [], [])],
        forbidden_load,
        {"fingerprint": "training-fixture"},
        purpose="training",
    )
    info = build_corpus(driver.qa_config(bundles), source, tmp_path / "qa")
    assert info["counts"]["generated_waveforms"] == 3
    assert info["counts"]["waveforms"] == 3
    assert info["counts"]["qa_examples"] == 12
    assert set(info["counts"]["by_family"]) == {
        "rhythm.heart_rate",
        "rhythm.beat_count",
        "rhythm.regularity",
        "rhythm.rate_band",
    }
    measured = driver.read(output / "measurement-summary.json")
    assert (
        measured["distributions"]["accepted"]["measured_hr"]["median"]
        != measured["distributions"]["accepted"]["requested_hr"]["median"]
    )
