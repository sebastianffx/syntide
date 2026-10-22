"""Evidence identity and caching, inconclusive measurements, acceptance, and the student/privileged split."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest
from tests.qa.fixtures import fixture_ecg

from syntide.data.qa import INSUFFICIENT, ContractError, Evidence, InputRef, QATask, SignalInfo
from syntide.data.schema import LEADS
from syntide.generate.qa.evidence import (
    EvidenceCache,
    EvidenceError,
    EvidenceExtractor,
    ExtractorConfig,
    SidecarConfig,
    waveform_sha256,
)
from syntide.generate.qa.templates import TEMPLATES, View, build_task
from syntide.generate.qa.verify import verify_task
from syntide.qa_trial import example

SPEC = SignalInfo(LEADS, 100.0, "mV", 1000)


def ref_for(x: np.ndarray, record_id: str = "ptbxl-1", subject: str = "ptb-xl:1") -> InputRef:
    return InputRef(
        record_id, waveform_sha256(x), SPEC, "real", (), (record_id,), (subject,), "physionet/ptb-xl@1.0.3"
    )


def facts(x, ref, extractor=None, annotations=None):
    return {
        (e.attribute, e.leads if e.attribute in ("wander_mv", "hf_rms_mv") else ()): e
        for e in (extractor or EvidenceExtractor()).extract(x, ref, annotations)
    }


@pytest.mark.parametrize("rate", [48, 64, 88, 130, 175])
def test_heart_rate_matches_fixture_beats(rate):
    x = fixture_ecg(rate, seed=rate)
    found = facts(x, ref_for(x))[("heart_rate_bpm", ())]
    assert found.status == "present" and found.kind == "measured"
    assert found.value == pytest.approx(rate, abs=2.5)


def test_inconclusive_beats_are_unknown_not_guessed():
    x = (0.4 * np.random.default_rng(0).standard_normal((12, 1000))).astype(np.float32)
    evidence = facts(x, ref_for(x))
    for name in ("heart_rate_bpm", "beat_count", "rr_regularity"):
        assert evidence[(name, ())].status == "unknown" and evidence[(name, ())].value is None
    task, _ = build_task(
        TEMPLATES["rhythm.heart_rate@1"],
        {},
        [View(ref_for(x), list(evidence.values()))],
        {"clinical_context": [""]},
        {"generator": "test"},
    )
    assert task.answer == INSUFFICIENT and task.answerability == "insufficient_evidence"
    assert INSUFFICIENT != "none"


def test_flat_leads_are_unavailable_and_their_lead_facts_unobservable():
    x = fixture_ecg(70, flat=("V2", "V3"))
    evidence = facts(x, ref_for(x))
    assert evidence[("flat_leads", ())].value == ["V2", "V3"]
    assert evidence[("wander_mv", ("V2",))].status == "unobservable"
    assert evidence[("wander_mv", ("II",))].usable


def test_annotations_are_annotation_evidence_of_real_recordings_only():
    x = fixture_ecg(70)
    real = facts(x, ref_for(x), annotations={"labels": ["MI"], "artifact_types": ["static_noise"]})
    assert (
        real[("annotated_static_noise", ())].kind == "annotation"
        and real[("annotated_static_noise", ())].value
    )
    assert real[("diagnostic_superclasses", ())].value == ["MI"]
    edited = InputRef(
        "acq-1", waveform_sha256(x), SPEC, "transformed", ("ptbxl-1",), ("ptbxl-1",), ("ptb-xl:1",), "x"
    )
    assert ("annotated_static_noise", ()) not in facts(
        x, edited, annotations={"labels": ["MI"], "artifact_types": []}
    )


def test_evidence_cache_invalidates_on_waveform_bytes_settings_and_sidecar(tmp_path):
    cache = EvidenceCache(tmp_path / "cache")
    x = fixture_ecg(70)
    first = EvidenceExtractor(cache=cache).extract(x, ref_for(x))
    again = EvidenceExtractor(cache=cache).extract(x, ref_for(x))
    assert (cache.hits, cache.misses) == (1, 1) and [e.evidence_id for e in again] == [
        e.evidence_id for e in first
    ]
    y = x.copy()
    y[0, 0] += 1e-3  # one sample changes
    EvidenceExtractor(cache=cache).extract(y, ref_for(y))
    EvidenceExtractor(ExtractorConfig(flat_mv=0.01), cache=cache).extract(x, ref_for(x))
    assert cache.misses == 3
    with pytest.raises(EvidenceError, match="bytes do not match"):
        EvidenceExtractor(cache=cache).extract(y, ref_for(x))

    sidecar = tmp_path / "features.csv"
    sidecar.write_text("ecg_id,QRS_Dur_Global\n1,96\n")

    def with_sidecar(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return ExtractorConfig(
            sidecar=SidecarConfig(
                path=str(path),
                sha256=digest,
                extractor="GE 12SL",
                extractor_version="23",
                columns={"qrs_duration_ms": ["QRS_Dur_Global", "ms"]},
            )
        )

    one = facts(x, ref_for(x), EvidenceExtractor(with_sidecar(sidecar), cache=cache))[("qrs_duration_ms", ())]
    assert one.value == 96 and one.kind == "annotation" and one.provenance["sidecar_sha256"]
    sidecar.write_text("ecg_id,QRS_Dur_Global\n1,104\n")
    two = facts(x, ref_for(x), EvidenceExtractor(with_sidecar(sidecar), cache=cache))[("qrs_duration_ms", ())]
    assert two.value == 104 and two.evidence_id != one.evidence_id
    stale = with_sidecar(sidecar)
    sidecar.write_text("ecg_id,QRS_Dur_Global\n1,90\n")
    with pytest.raises(EvidenceError, match="does not match the declared"):
        EvidenceExtractor(stale)


def _task(x=None, template="rhythm.heart_rate@1", params=None):
    x = fixture_ecg(70) if x is None else x
    ref = ref_for(x)
    evidence = EvidenceExtractor().extract(x, ref)
    task, cited = build_task(
        TEMPLATES[template],
        params or {},
        [View(ref, evidence)],
        {"clinical_context": ["ctx"]},
        {"generator": "test"},
    )
    return task, {e.evidence_id: e for e in [*evidence, *cited]}, x


def test_unsupported_missing_or_inconclusive_evidence_never_passes():
    task, store, _ = _task()
    assert verify_task(task, store).verification["status"] == "accepted"
    assert verify_task(task, {}).verification["status"] == "rejected"  # missing evidence is not a pass
    (eid,) = task.required_evidence
    requested = replace(store[eid], kind="requested_control")
    assert verify_task(task, {**store, eid: requested}).verification["status"] == "rejected"
    unknown = replace(store[eid], status="unknown", value=None, confidence=None)
    assert verify_task(task, {**store, eid: unknown}).verification["status"] == "rejected"
    lying = replace(task, rationale=task.rationale.replace(task.claims[0].text.split()[-2], "999"))
    assert verify_task(lying, store).verification["status"] == "rejected"
    leaky = replace(task, context={"clinical_context": ["requested heart_rate=60"]})
    assert verify_task(leaky, store).verification["status"] == "rejected"


def test_task_ids_follow_content_and_edited_tasks_are_refused():
    task, _, _ = _task()
    assert QATask.from_json(task.to_json()) == task
    edited = task.to_json()
    edited["answer"] = 100.0
    with pytest.raises(ContractError, match="edited"):
        QATask.from_json(edited)
    with pytest.raises(ContractError):
        replace(task, answer=INSUFFICIENT)  # an answerable task cannot answer "insufficient evidence"
    e = Evidence.from_json(next(iter(_task()[1].values())).to_json())
    assert e.evidence_id.startswith("ev-")


def test_student_inputs_do_not_change_when_only_privileged_supervision_changes():
    task, _, x = _task()
    other = replace(task, answer=task.answer + 7, rationale="A different rationale.", claims=())
    a, b = (
        example(task, lambda _: x, target_mode="rationale"),
        example(other, lambda _: x, target_mode="rationale"),
    )
    assert a.target != b.target
    assert a.student.keys() == b.student.keys()
    assert {k: v for k, v in a.student.items() if k != "signals"} == {
        k: v for k, v in b.student.items() if k != "signals"
    }
    assert all(np.array_equal(s, t) for s, t in zip(a.student["signals"], b.student["signals"], strict=True))
    text = repr({k: v for k, v in a.student.items() if k != "signals"})
    assert task.rationale not in text and "Answer:" not in text
