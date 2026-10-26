"""Generated recordings retain channel, sampling, and source information."""

from __future__ import annotations

import json

import numpy as np
import pytest
from tests.qa.fixtures import fixture_ecg, memory_source

from syntide.data.qa import SignalInfo
from syntide.data.schema import LEADS
from syntide.generate.full import registry
from syntide.generate.full.api import GeneratedSignal
from syntide.generate.full.diffusets import CHANNELS
from syntide.generate.full.runner import run_generation
from syntide.generate.qa import generated
from syntide.generate.qa.build import BuildConfig, GeneratedConfig, build_corpus
from syntide.generate.qa.corpus import CorpusError, CorpusWriter, load_corpus
from syntide.generate.qa.evidence import EvidenceExtractor

TARGET = SignalInfo(LEADS, 100.0, "mV", 1000)


class FixtureBackend:
    """Synthetic test signals with the shape and channel names of DiffuSETS."""

    def __init__(self, config):
        self.rate_bias = config.get("rate_bias", 0)

    def validate_request(self, request):
        if tuple(request.signal.channels) != CHANNELS:
            raise ValueError("DiffuSETS lead names expected")

    def generate(self, request):
        # The measured rate differs from the requested one: conditioning is not
        # ground truth.
        hr = request.conditioning["heart_rate"] + self.rate_bias
        x = fixture_ecg(hr, fs=102.4, seed=request.seed)
        return GeneratedSignal(x, request.signal, {"fixture": True})

    def provenance(self):
        return {"model": "fixture", "weights_sha256": "0" * 64}


@pytest.fixture
def bundles(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "_REGISTERED", {})
    monkeypatch.setattr(registry, "entry_points", lambda **_: [])
    monkeypatch.setattr(
        registry, "_BUILTINS", {"diffusets@1": FixtureBackend}
    )  # the real one needs weights
    plan_body = {
        "schema_version": "syntide.generation-plan@1",
        "analysis_id": "a",
        "source": {},
        "seed": 5,
        "targets": {
            "record_ids": ["ptbxl-1", "ptbxl-2"],
            "text_ids": [],
            "labels": ["superclass:NORM"],
        },
        "requests": [
            {
                "method": "diffusion",
                "count": 2,
                "generator": "diffusets@1",
                "modality": "ecg",
                "parameters": {},
            }
        ],
    }
    from syntide.explore.plans import content_id

    plan = {**plan_body, "plan_id": content_id(plan_body)}
    signal = {
        "modality": "ecg",
        "channels": list(CHANNELS),
        "sample_rate": 102.4,
        "num_samples": 1024,
        "units": "mV",
    }
    out = []
    for rate in (
        60,
        100,
    ):  # same plan seed and request index: the same per-sample seeds
        config = {
            "signal": signal,
            "rate_bias": 4,
            "conditions": [
                {
                    "conditioning": {
                        "report": "normal",
                        "heart_rate": rate,
                        "age": 50,
                        "sex": "F",
                    },
                    "source_record_ids": ["ptbxl-1"],
                },
                {
                    "conditioning": {
                        "report": "normal",
                        "heart_rate": rate,
                        "age": 50,
                        "sex": "F",
                    },
                    "source_record_ids": ["ptbxl-2"],
                },
            ],
        }
        path = tmp_path / f"bundle-{rate}"
        run_generation(plan, config, path, request_index=0)
        out.append(path)
    return out


def test_aliases_rate_conversion_and_ancestry_are_explicit(bundles):
    sample = generated.load_bundle(bundles[0])[0]
    assert sample.status == generated.PENDING and sample.channels == CHANNELS
    assert generated.normalize_channels(["aVR", "I"]) == ("AVR", "I")
    with pytest.raises(generated.GeneratedInputError):
        generated.normalize_channels(["aVR", "AVR"])
    subjects = {"ptbxl-1": "ptb-xl:1001", "ptbxl-2": "ptb-xl:1002"}
    with pytest.raises(generated.GeneratedInputError, match="allow_resample"):
        generated.to_input(sample, TARGET, subjects, allow_resample=False)
    ref, x = generated.to_input(sample, TARGET, subjects, allow_resample=True)
    assert (
        x.shape == (12, 1000)
        and ref.lineage["resampling"]["up"] == 125
        and ref.lineage["resampling"]["down"] == 128
    )
    assert ref.lineage["channel_aliases"] == {
        "aVR": "AVR",
        "aVL": "AVL",
        "aVF": "AVF",
    }
    assert ref.lineage[
        "review_status"
    ] == "pending" and ref.root_subject_uids == ("ptb-xl:1001",)
    native = np.load(sample.signal_path)
    assert np.allclose(
        x[0].mean(), native[0].mean(), atol=0.02
    )  # resampled, not truncated or relabelled
    with pytest.raises(generated.GeneratedInputError, match="not in the index"):
        generated.to_input(sample, TARGET, {}, allow_resample=True)


def test_requested_controls_never_become_answers_and_verification_is_required(
    bundles,
):
    sample = generated.load_bundle(bundles[0])[0]
    ref, x = generated.to_input(
        sample, TARGET, {"ptbxl-1": "ptb-xl:1001"}, allow_resample=True
    )
    verified, evidence, review = generated.verify_generated(
        ref, x, EvidenceExtractor(), ["heart_rate_bpm"]
    )
    assert (
        review["status"] == "verified"
        and verified.lineage["review_status"] == "verified"
    )
    measured = next(e for e in evidence if e.attribute == "heart_rate_bpm")
    assert measured.value == pytest.approx(
        64, abs=3
    )  # what the waveform shows, not the requested 60
    assert verified.lineage["requested_controls"]["heart_rate"] == 60
    flat = np.zeros_like(x)
    flat_ref = ref.__class__(
        **{
            **{f: getattr(ref, f) for f in ref.__dataclass_fields__},
            "waveform_sha256": generated.waveform_sha256(flat),
        }
    )
    still_pending, _, failed = generated.verify_generated(
        flat_ref, flat, EvidenceExtractor(), ["heart_rate_bpm"]
    )
    assert (
        failed["status"] == "rejected"
        and still_pending.lineage["review_status"] == "rejected"
    )


def test_pending_generated_waveforms_cannot_enter_an_accepted_export(
    bundles, tmp_path
):
    from tests.qa.test_evidence_and_tasks import _task

    sample = generated.load_bundle(bundles[0])[0]
    ref, x = generated.to_input(
        sample, TARGET, {"ptbxl-1": "ptb-xl:1001"}, allow_resample=True
    )
    task, store, _ = _task(x)
    task = task.__class__(
        **{
            **{f: getattr(task, f) for f in task.__dataclass_fields__},
            "inputs": (ref,),
            "verification": {"status": "accepted"},
        }
    )
    writer = CorpusWriter(
        tmp_path / "corpus", {"purpose": "training", "name": "pending"}
    )
    writer.put_signal(ref, x)
    evidence = [
        e.__class__(
            **{
                **{f: getattr(e, f) for f in e.__dataclass_fields__},
                "record_id": ref.waveform_id,
                "root_record_ids": ref.root_record_ids,
                "root_subject_uids": ref.root_subject_uids,
            }
        )
        for e in store.values()
    ]
    writer.write_unit("u0", [task], evidence, [], [ref])
    with pytest.raises(CorpusError, match="pending"):
        writer.publish(expected_units=["u0"])
    assert not (tmp_path / "corpus").exists()


def test_generated_build_checks_evidence_and_source_records(
    bundles, tmp_path, monkeypatch
):
    source = memory_source(4)
    cfg = BuildConfig(
        name="gen",
        components=["generated"],
        generated=GeneratedConfig(
            bundles=[str(p) for p in bundles], allow_resample=True
        ),
    )
    info = build_corpus(cfg, source, tmp_path / "gen")
    corpus = load_corpus(tmp_path / "gen")
    assert info["counts"]["generated_waveforms"] == 4
    assert all(
        ref.lineage["review_status"] == "verified"
        for ref in corpus.inputs.values()
    )
    rates = [t for t in corpus.tasks if t.family == "rhythm.heart_rate"]
    assert rates and all(
        t.answer not in (60, 100) for t in rates
    )  # measured, never the requested value
    pairs = [t for t in corpus.tasks if len(t.inputs) == 2]
    assert pairs and all(
        t.lineage["pair"]["variation"] == "diffusets.heart_rate_condition@1"
        for t in pairs
    )
    for task in pairs:  # the faster recording, by measurement, is the answer
        fast = max(
            task.inputs,
            key=lambda r: r.lineage["requested_controls"]["heart_rate"],
        )
        assert task.answer == "AB"[task.inputs.index(fast)]

    outside = memory_source(
        4, start=50
    )  # the conditioning sources are not in this role
    monkeypatch.setattr(
        generated.GeneratedSample,
        "load",
        lambda self: pytest.fail("a waveform was read before ancestry"),
    )
    build_corpus(
        BuildConfig(
            name="gen2",
            components=["generated"],
            generated=GeneratedConfig(
                bundles=[str(bundles[0])], allow_resample=True
            ),
        ),
        outside,
        tmp_path / "gen2",
    )
    ledger = json.loads((tmp_path / "gen2" / "ledger.json").read_text())
    assert ledger and all(
        row["reason"].startswith("ancestry") for row in ledger
    )
