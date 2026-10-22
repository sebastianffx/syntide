"""Error-directed allocation builds new corpora only; frozen curricula reproduce sequences and exposure."""

from __future__ import annotations

from collections import Counter

import pytest
from tests.qa.fixtures import memory_source

from syntide.generate.qa import allocation
from syntide.generate.qa.build import BuildConfig, build_corpus
from syntide.generate.qa.corpus import load_corpus
from syntide.generate.qa.curriculum import (
    Curriculum,
    CurriculumConfig,
    CurriculumError,
    StageConfig,
    exposure,
    sequence_sha256,
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("alloc")
    train = build_corpus(
        BuildConfig(name="train", components=["source", "grounded"]), memory_source(6), out / "train"
    )
    dev_cfg = BuildConfig(
        name="dev", purpose="evaluation", suite_role="development", components=["source", "grounded"]
    )
    build_corpus(dev_cfg, memory_source(6, start=40, purpose="evaluation"), out / "dev")
    sel_cfg = BuildConfig(name="sel", purpose="evaluation", suite_role="selection", components=["grounded"])
    build_corpus(sel_cfg, memory_source(3, start=80, purpose="evaluation"), out / "sel")
    return {"train": out / "train", "dev": out / "dev", "sel": out / "sel", "train_info": train, "out": out}


def _correct(tasks, wrong_family):
    return {t.task_id: t.family != wrong_family for t in tasks}


def _weak(tasks) -> str:
    """The family the fake model gets wrong: the most common evidence-grounded one in the suite."""
    return Counter(t.family for t in tasks if t.composition == "attribute").most_common(1)[0][0]


def test_gap_reports_read_development_suites_only_and_mark_small_slices(built):
    dev, sel = load_corpus(built["dev"]), load_corpus(built["sel"])
    weak = _weak(dev.tasks)
    gap = allocation.gap_report(dev.info, dev.tasks, dev.evidence, _correct(dev.tasks, weak), min_n=5)
    by = {(row["slice"], row["value"]): row for row in gap["slices"]}
    assert by[("family", weak)]["error_rate"] == 1.0
    for row in gap["slices"]:
        assert 1 <= row["patients"] <= 2 * row["n"]  # a comparison task has two patients
        if row["n"] < 5:
            assert not row["reliable"] and row["shrunk_error"] == pytest.approx(
                (gap["overall"]["errors"] + 0.5) / (gap["overall"]["n"] + 1)
            )
    with pytest.raises(allocation.AllocationError, match="development suite"):
        allocation.gap_report(sel.info, sel.tasks, sel.evidence, _correct(sel.tasks, "x"))
    patients = {s for ref in dev.inputs.values() for s in ref.root_subject_uids}
    with pytest.raises(allocation.AllocationError, match="also in the selection"):
        allocation.gap_report(
            dev.info, dev.tasks, dev.evidence, _correct(dev.tasks, "x"), protected_subjects=patients
        )
    with pytest.raises(allocation.AllocationError, match="exactly"):
        allocation.gap_report(dev.info, dev.tasks[1:], dev.evidence, _correct(dev.tasks, "x"))


def test_a_plan_builds_a_new_corpus_and_never_touches_the_parent(built):
    dev = load_corpus(built["dev"])
    weak = _weak(dev.tasks)
    gap = allocation.gap_report(dev.info, dev.tasks, dev.evidence, _correct(dev.tasks, weak), min_n=2)
    plan = allocation.make_plan(gap, 40, parent_corpus_id=built["train_info"]["corpus_id"], seed=0)
    allocation.verify_plan(plan)
    assert sum(plan["family_quotas"].values()) == 40 == sum(plan["template_quotas"].values())
    assert plan["family_weights"][weak] == max(plan["family_weights"].values())
    assert allocation.make_plan(gap, 40, parent_corpus_id=built["train_info"]["corpus_id"], seed=0) == plan
    tampered = {**plan, "budget": 41}
    with pytest.raises(allocation.AllocationError):
        allocation.verify_plan(tampered)
    before = (built["train"] / "corpus.json").read_bytes()
    child = build_corpus(
        BuildConfig(name="train", components=["grounded"]),
        memory_source(6),
        built["out"] / "child",
        plan=plan,
        parent_corpus_id=built["train_info"]["corpus_id"],
    )
    assert child["corpus_id"] != built["train_info"]["corpus_id"]
    assert child["identity"]["plan_id"] == plan["plan_id"]
    assert (built["train"] / "corpus.json").read_bytes() == before
    used = Counter(t.template for t in load_corpus(built["out"] / "child").tasks)
    assert all(used[t] <= n for t, n in plan["template_quotas"].items())
    assert set(used) <= set(plan["template_quotas"])


def test_coverage_uses_measured_evidence(built):
    corpus = load_corpus(built["train"])
    cover = allocation.coverage(corpus.tasks, corpus.evidence)
    assert set(cover) == set(allocation.SLICES)
    assert sum(cover["origin"].values()) == len(corpus.tasks)
    assert set(cover["rate_band"]) <= {"<60", "60-100", ">100", "unknown"}


def _curriculum(built, supervision="rationale", seed=0):
    tasks = load_corpus(built["train"]).tasks
    cfg = CurriculumConfig(
        name="basic-to-composed@1",
        seed=seed,
        stages=[
            StageConfig(
                name="basic", until_step=4, compositions=["attribute", "source"], target_mode="answer_only"
            ),
            StageConfig(
                name="composed",
                until_step=8,
                compositions=["composed", "reasoning", "situation", "comparison"],
                replay=0.5,
            ),
            StageConfig(name="rationale", until_step=10, replay=0.25),
        ],
    )
    return Curriculum(cfg, {"train": tasks}, batch_size=6, max_steps=10, supervision=supervision), {
        t.task_id: t for t in tasks
    }


def test_frozen_curriculum_reproduces_its_sequence_stages_and_exposure(built):
    a, tasks = _curriculum(built)
    b, _ = _curriculum(built)
    assert sequence_sha256(a) == sequence_sha256(b)
    assert sequence_sha256(a) != sequence_sha256(_curriculum(built, seed=1)[0])
    slots = [slot for step in a.schedule() for slot in step]
    assert len(slots) == 60
    basic = [s for s in slots if s.step <= 4]
    assert all(
        tasks[s.task_id].composition in ("attribute", "source") and s.target_mode == "answer_only"
        for s in basic
    )
    composed = [s for s in slots if 4 < s.step <= 8]
    assert any(s.pool.startswith("replay:") for s in composed)
    assert all(
        tasks[s.task_id].composition in ("composed", "reasoning", "situation", "comparison")
        for s in composed
        if not s.pool.startswith("replay:")
    )
    assert all(s.target_mode == "rationale" for s in slots if s.step > 4)
    counted = exposure(slots, tasks)
    assert counted["examples"] == 60 and sum(counted["by_stage"].values()) == 60
    assert counted["whitespace_tokens"]["target"] > 0
    only_answers, _ = _curriculum(built, supervision="answer_only")
    assert {s.target_mode for step in only_answers.schedule() for s in step} == {"answer_only"}
    with pytest.raises(CurriculumError):
        Curriculum(
            CurriculumConfig(stages=[StageConfig(until_step=3)]),
            {"t": list(tasks.values())},
            batch_size=2,
            max_steps=10,
            supervision="rationale",
        )
