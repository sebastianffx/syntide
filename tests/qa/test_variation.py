"""Recording changes update answers and preserve comparison order."""

from __future__ import annotations

import numpy as np
import pytest
from tests.qa.fixtures import fixture_ecg
from tests.qa.test_evidence_and_tasks import ref_for

from syntide.data.qa import INSUFFICIENT
from syntide.generate.editor import EditorConfig
from syntide.generate.qa import acquisition, pairs
from syntide.generate.qa.acquisition import (
    AcquisitionConfig,
    UnsupportedTransformation,
)
from syntide.generate.qa.evidence import EvidenceExtractor
from syntide.generate.qa.templates import TEMPLATES, View, build_task
from syntide.generate.qa.verify import verify_task

CFG = AcquisitionConfig()


def seeded(x, template, params=None):
    ref = ref_for(x)
    ev = EvidenceExtractor().extract(x, ref)
    task, cited = build_task(
        TEMPLATES[template],
        params or {},
        [View(ref, ev)],
        {"clinical_context": ["Signal quality: ok."]},
        {"generator": "test"},
    )
    return verify_task(task, {e.evidence_id: e for e in [*ev, *cited]}), ref, ev


def variant(x, ref, op, params):
    recipe = {
        "generator": "ecg.acquisition@1",
        "op": op,
        "params": params,
        "seed_key": 7,
    }
    out = acquisition.apply(x, ref, recipe, CFG)
    return out, EvidenceExtractor().extract(out.x, out.ref)


def test_lead_mask_invalidates_unavailable_channel_facts():
    x = fixture_ecg(72, wander=0.4)
    flat_task, ref, _ = seeded(x, "acquisition.flat_leads@1")
    wander_task, _, _ = seeded(x, "artifact.wander_in_lead@1", {"lead": "V2"})
    v, vev = variant(x, ref, "lead_mask", {"leads": ["V1", "V2"]})
    assert (
        v.ok and v.ref.signal.channels == ref.signal.channels
    )  # lead identities and order are kept
    carried, decision, _ = acquisition.carry(
        flat_task, v, vev, "Signal quality: ok."
    )
    assert decision == "remeasured_updated" and carried.answer == ["V1", "V2"]
    carried, decision, _ = acquisition.carry(
        wander_task, v, vev, "Signal quality: ok."
    )
    assert (
        decision == "insufficient_after_transform"
        and carried.answer == INSUFFICIENT
    )
    assert carried.context == {
        "clinical_context": [""]
    }  # the parent's signal-quality sentence is dropped
    assert (
        v.ref.root_subject_uids == ref.root_subject_uids
        and v.ref.origin == "transformed"
    )


def test_preserved_attributes_reject_changed_answers(
    monkeypatch,
):
    x = fixture_ecg(64)
    rate_task, ref, _ = seeded(x, "rhythm.rate_band@1")
    duration_task, _, _ = seeded(x, "acquisition.duration@1")
    v, vev = variant(x, ref, "gain", {"factor": 2.0})
    assert acquisition.carry(rate_task, v, vev, "")[1] == "preserved"
    crop, cev = variant(x, ref, "crop", {"start_s": 1.0, "duration_s": 5.0})
    carried, decision, _ = acquisition.carry(duration_task, crop, cev, "")
    assert decision == "remeasured_updated" and carried.answer == 5.0
    assert crop.ref.lineage["time_offset_s"] == 1.0
    # If gain claimed to preserve the duration group, a crop-like change would
    # have to be rejected:
    effect = {**acquisition.LABEL_EFFECT["crop"], "duration": "preserves"}
    monkeypatch.setitem(crop.ref.lineage, "label_effect", effect)
    assert acquisition.carry(duration_task, crop, cev, "")[1].startswith(
        "rejected:invariant_violation"
    )


def test_annotations_follow_the_declared_label_effect():
    x = fixture_ecg(70)
    ref = ref_for(x)
    ev = EvidenceExtractor().extract(
        x, ref, {"labels": ["MI"], "artifact_types": ["static_noise"]}
    )
    v, _ = variant(x, ref, "gain", {"factor": 0.5})
    inherited = {
        e.attribute: e
        for e in acquisition.inherit_annotations(ev, v, EvidenceExtractor())
    }
    assert inherited["diagnostic_superclasses"].value == [
        "MI"
    ]  # gain preserves the diagnosis label
    assert (
        inherited["annotated_static_noise"].status == "unknown"
    )  # and invalidates signal quality
    assert all(
        e.waveform_sha256 == v.ref.waveform_sha256 for e in inherited.values()
    )


def test_editor_variant_reuses_the_existing_editor_and_its_validators():
    x = fixture_ecg(70)
    ref = ref_for(x)
    rng = np.random.default_rng(0)
    recipe = acquisition.draw_recipe(
        "electrode_artifact",
        AcquisitionConfig(editor=EditorConfig()),
        rng,
        ref.signal,
    )
    out = acquisition.apply(x, ref, recipe, CFG)
    assert {c["name"] for c in out.checks} >= {
        "bounds",
        "alignment",
        "signal_valid@1",
    }
    assert out.ref.lineage["label_effect"]["diagnosis"] == "preserves"
    with pytest.raises(UnsupportedTransformation):
        acquisition.draw_recipe("time_warp", CFG, rng, ref.signal)


def test_invalid_signals_are_rejected():
    x = fixture_ecg(70)
    out, _ = variant(x, ref_for(x), "gain", {"factor": 1e6})
    assert not out.ok and any(
        c["name"] == "signal_valid@1" and not c["pass"] for c in out.checks
    )


def test_controlled_pair_is_remeasured_and_answers_follow_the_input_order():
    x = fixture_ecg(70)
    ref = ref_for(x)
    ev = EvidenceExtractor().extract(x, ref)
    spec = pairs.variation("amplitude_up@1")
    v, vev = variant(x, ref, "gain", {"factor": 1.5})
    record = pairs.check_pair(spec, View(ref, ev), View(v.ref, vev))
    assert record.status == "accepted", record.reason
    assert (
        record.observed["amplitude_mv"][1] > record.observed["amplitude_mv"][0]
    )
    assert all(check["pass"] for check in record.invariant_checks)
    answers = {}
    for seed in range(
        8
    ):  # both orders occur; the answer always names the louder recording
        accepted, rejected, _ = pairs.comparison_tasks(
            View(ref, ev),
            View(v.ref, vev),
            ["", ""],
            np.random.default_rng(seed),
            pair=record,
            attributes=["amplitude_mv"],
        )
        (task,) = accepted
        louder = "A" if task.inputs[0].waveform_id == v.ref.waveform_id else "B"
        assert task.answer == louder and not rejected
        answers[task.inputs[0].waveform_id] = task.answer
        assert (
            task.lineage["pair"]["claim_scope"]
            == "controlled_signal_comparison"
        )
    assert set(answers.values()) == {"A", "B"}


def test_pairs_reject_inconsistent_or_unsupported_changes():
    slow, fast = fixture_ecg(60, seed=1), fixture_ecg(110, seed=2)
    a, b = ref_for(slow, "ptbxl-1"), ref_for(fast * 1.5, "ptbxl-2")
    record = pairs.check_pair(
        pairs.variation("amplitude_up@1"),
        View(a, EvidenceExtractor().extract(slow, a)),
        View(b, EvidenceExtractor().extract(fast * 1.5, b)),
    )
    assert (
        record.status == "rejected"
        and "invariant_violated:heart_rate_bpm" in record.reason
    )
    for name in ("heart_rate", "diagnosis", "qt_prolongation@1"):
        with pytest.raises(pairs.UnsupportedVariation):
            pairs.variation(name)
