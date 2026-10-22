"""QA scoring, the QA trial contract's refusals, and the OpenTSLM prompt adapter."""

from __future__ import annotations

import numpy as np
import pytest
from tests.qa.fixtures import memory_source
from tests.qa.test_evidence_and_tasks import _task

from syntide.data.leakage import LeakageError
from syntide.data.qa import INSUFFICIENT
from syntide.data.schema import LEADS
from syntide.eval import qa_metrics
from syntide.eval.tracking import TrackerConfig, make_tracker
from syntide.generate.qa.build import BuildConfig, build_corpus
from syntide.generate.qa.corpus import load_corpus
from syntide.generate.qa.curriculum import Curriculum, CurriculumConfig, StageConfig
from syntide.generate.qa.sources import source_tasks
from syntide.qa_trial import _trial_id, example, make_qa_spec, open_qa_trial
from syntide.trial import Checkpoint, TrainerInfo, TrialError

TRAINER = TrainerInfo("fake-qa", "1", {}, "cpu", "fake:predict")


def test_a_wrong_final_answer_is_not_rescued_by_its_rationale():
    task, _, _ = _task(template="rhythm.regularity@1")
    assert task.answer == "regular"
    wrong = "The RR intervals look regular overall, so the answer is regular. Answer: irregular"
    rows = qa_metrics.score_outputs([task], [wrong])
    assert rows[0]["valid"] and not rows[0]["correct"]
    assert qa_metrics.rationale_contradictions(
        wrong, task, {}
    )  # the stated conclusion differs from the answer
    assert not qa_metrics.score_outputs([task], ["regular"])[0][
        "valid"
    ]  # no Answer: line is invalid, and wrong
    assert qa_metrics.score_outputs([task], ["Thinking... Answer: Regular."])[0]["correct"]


def test_numeric_sets_and_answerability_are_scored_by_their_contract():
    rate, _, _ = _task()
    tol = max(rate.tolerance["abs"], rate.tolerance["rel"] * rate.answer)
    outputs = [
        f"Answer: {rate.answer + 0.9 * tol:.3f} bpm",
        f"Answer: {rate.answer + 1.5 * tol:.3f}",
        f"Answer: {INSUFFICIENT}",
    ]
    assert [r["correct"] for r in qa_metrics.score_outputs([rate] * 3, outputs)] == [True, False, False]
    flat, _, _ = _task(template="acquisition.flat_leads@1")
    assert flat.answer == []
    assert qa_metrics.score_outputs([flat], ["Answer: none"])[0]["correct"]
    assert not qa_metrics.score_outputs([flat], ["Answer: V2"])[0]["correct"]
    x = (0.4 * np.random.default_rng(0).standard_normal((12, 1000))).astype(np.float32)
    unknown, _, _ = _task(x)
    assert unknown.answerability == "insufficient_evidence"
    assert qa_metrics.score_outputs([unknown], ["Answer: insufficient evidence"])[0]["correct"]
    assert not qa_metrics.score_outputs([unknown], ["Answer: 72"])[0]["correct"]
    lying = f"The heart rate is 140 bpm. Answer: {rate.answer}"
    assert any(
        p.startswith("unsupported_number")
        for p in qa_metrics.rationale_contradictions(lying, rate, _task()[1])
    )


def test_patient_cluster_bootstrap_resamples_patients_not_rows():
    rows = [{"patients": "p1", "correct": True}] * 50 + [{"patients": "p2", "correct": False}]
    ci = qa_metrics.cluster_bootstrap(rows, n_boot=200, seed=0)
    assert ci["clusters"] == 2 and ci["low"] < 0.1 and ci["high"] == 1.0


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    out = tmp_path_factory.mktemp("trial")
    build_corpus(
        BuildConfig(name="train", components=["source", "grounded"]), memory_source(4), out / "train"
    )
    build_corpus(
        BuildConfig(name="sel", purpose="evaluation", suite_role="selection", components=["grounded"]),
        memory_source(2, start=30, purpose="evaluation"),
        out / "sel",
    )
    build_corpus(
        BuildConfig(name="overlap", purpose="evaluation", suite_role="selection", components=["grounded"]),
        memory_source(2, start=1, purpose="evaluation"),
        out / "overlap",
    )
    return {name: load_corpus(out / name) for name in ("train", "sel", "overlap")}


def _spec(train, sel, **overrides):
    curriculum = {"name": "uniform-passes@1", "seed": 0, "stages": [{"name": "all"}]}
    cfg = CurriculumConfig(stages=[StageConfig(name="all")])
    sha = Curriculum(cfg, {"train": train.tasks}, batch_size=2, max_steps=4, supervision="rationale").sha256
    content = dict(
        protocol="p",
        arm="a",
        supervision="rationale",
        replicate_seed=0,
        corpora=[{"label": "train", "corpus_id": train.corpus_id, "path": str(train.path)}],
        selection={"corpus_id": sel.corpus_id, "path": str(sel.path)},
        curriculum=curriculum,
        curriculum_sha256=sha,
        batch_size=2,
        max_steps=4,
        val_every=2,
        metric={"name": qa_metrics.SELECTION_METRIC, "direction": "max", "tie_break": "earliest"},
        parser=qa_metrics.PARSER,
        decoding={},
        context_policy="ecgqa-cot-baseline@1",
        budget={},
    )
    return make_qa_spec(**{**content, **overrides})


def _open(small, run_dir, selection="sel"):
    spec = _spec(small["train"], small[selection])
    board = make_tracker(TrackerConfig("files"), run_dir, spec.run_name, spec.protocol, spec.to_dict())
    return open_qa_trial(spec, {"train": small["train"]}, small[selection], board, run_dir)


def test_qa_trial_ids_ignore_paths_and_follow_corpora(small, tmp_path):
    spec = _spec(small["train"], small["sel"])
    moved = _trial_id({**spec.to_dict(), "corpora": [{**spec.corpora[0], "path": "/elsewhere"}]})
    assert moved == spec.trial_id
    assert (
        _trial_id({**spec.to_dict(), "corpora": [{**spec.corpora[0], "corpus_id": "corpus-other"}]})
        != spec.trial_id
    )


def test_training_corpora_and_the_selection_suite_never_share_patients(small, tmp_path):
    with pytest.raises(LeakageError, match="patients are in both"):
        _open(small, tmp_path, "overlap")
    with pytest.raises(TrialError, match="cannot train"):
        open_qa_trial(
            _spec(small["train"], small["sel"]),
            {"train": small["sel"]},
            small["sel"],
            make_tracker(TrackerConfig("files"), tmp_path, "x", "p", {}),
            tmp_path,
        )


def test_finish_refuses_runs_that_broke_the_qa_protocol(small, tmp_path):
    def run(stop_after=None):
        trial = _open(small, tmp_path / str(stop_after))
        for batch in trial.batches():
            assert all(
                ex.target.endswith(f"Answer: {ex.target.rsplit('Answer: ', 1)[-1]}") for ex in batch.examples
            )
            if batch.step == stop_after:
                break
        tasks = trial.selection_tasks
        good = [f"Answer: {t.answer_text}" for t in tasks]
        bad = ["Answer: ?"] * len(tasks)
        history = [
            {"step": 2, **qa_metrics.val_scores(tasks, bad)},
            {"step": 4, **qa_metrics.val_scores(tasks, good)},
        ]
        for name in ("best", "final"):
            (trial.run_dir / f"{name}.ckpt").write_text(name)
        best, final = Checkpoint(trial.run_dir / "best.ckpt", 4), Checkpoint(trial.run_dir / "final.ckpt", 4)
        return trial, dict(
            trainer=TRAINER,
            best=best,
            final=final,
            val_history=history,
            val_outputs={"best": good, "final": good},
            init_sha256="a" * 64,
            wall_time_s=1.0,
        )

    trial, args = run(stop_after=2)
    with pytest.raises(TrialError, match="consumed 2 of 4"):
        trial.finish(**args)
    trial, args = run()
    with pytest.raises(TrialError, match="first argmax"):
        trial.finish(**{**args, "best": Checkpoint(args["best"].path, 2)})
    with pytest.raises(TrialError, match="history reports"):
        trial.finish(
            **{
                **args,
                "val_outputs": {
                    "best": args["val_outputs"]["best"],
                    "final": ["Answer: ?"] * len(args["val_outputs"]["final"]),
                },
            }
        )
    result = trial.finish(**args)
    assert result["best_val"] == 1.0 and result["exposure"]["examples"] == 8


BASELINE_TEMPLATE = """You are an expert cardiologist analyzing an ECG (electrocardiogram).

Clinical Context: {context}

Your task is to examine the ECG signal and answer the following medical question:

Question: {question}

Instructions:
- Begin by analyzing the time series without assuming a specific answer.
- Think step-by-step about what the observed patterns suggest regarding the cardiac condition.
{paragraph}
- Do **not** mention any final answer until the very end.
- Consider the ECG morphology, intervals, and any abnormalities that relate to the question.

<image> {leads} <|endofchunk|>

Based on your analysis of the ECG data, select your answer from the following options:
{options}

- Make sure that your last word is the answer. You MUST end your response with "Answer: "
"""


def test_opentslm_adapter_reproduces_the_baseline_prompt_and_target_for_source_tasks():
    from syntide_trainers.opentslm_qa import PROMPT_VERSION, loss_spans, prompt

    source = memory_source(1)
    ((ref, x),) = source.load(["ptbxl-1"])
    task = source_tasks(source, ref)[0]
    ex = example(task, lambda _: x, target_mode="rationale")
    out = prompt(ex)
    means, stds = x.mean(axis=-1), x.std(axis=-1)
    leads = "\n".join(
        f"This is ECG Lead {n}, it has mean {float(means[i]):.4f} and std {float(stds[i]):.4f}:"
        for i, n in enumerate(LEADS)
    )
    paragraph = (
        "- Write your rationale as a single, natural paragraph — "
        "do not use bullet points, numbered steps, or section headings."
    )
    expected = BASELINE_TEMPLATE.format(
        paragraph=paragraph,
        context=task.context["clinical_context"][0],
        question=task.question,
        leads=leads,
        options=", ".join(task.options),
    )
    assert out["text"] == expected and out["prompt_version"] == PROMPT_VERSION
    assert np.allclose(out["signals"][0].mean(axis=-1), 0, atol=1e-5)
    assert (
        ex.target == f"The waveform shows it.\nAnswer: {task.answer}"
    )  # the source's own Answer: line appears once
    assert loss_spans(ex)["target"] == " " + ex.target
    pair, _, _ = _task(template="rhythm.heart_rate@1")
    numeric = prompt(example(pair, lambda _: x, target_mode="answer_only"))
    assert numeric["prompt_version"].endswith("-numeric")
