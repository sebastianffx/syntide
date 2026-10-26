"""Trainer outputs must satisfy the saved trial specification.

A NumPy implementation stands in for a trainer, so these tests need neither
PyTorch nor TimeNet.
"""

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pyarrow.parquet as pq
import pytest

from syntide.data.schema import SUPERCLASSES
from syntide.eval.stats import SELECTION_METRIC, val_scores
from syntide.eval.tracking import TrackerConfig, make_tracker
from syntide.trial import (
    Checkpoint,
    TrainerInfo,
    TrialError,
    is_complete,
    load_spec,
    open_trial,
    write_spec,
)

TRAINER = TrainerInfo(
    name="fake",
    version="1",
    config={"width": 1},
    device="cpu",
    predictor="fake:predict",
)
OTHER_TRAINER = replace(TRAINER, config={"width": 2})
QUALITY = {
    10: 0.2,
    20: 0.6,
    30: 0.4,
}  # how well the fake "model" separates val labels; best at step 20


def _open(spec, make_samples, run_dir, parents=None):
    write_spec(run_dir, spec)
    tracker = make_tracker(
        TrackerConfig("files"),
        run_dir,
        spec.run_name,
        spec.protocol,
        spec.to_dict(),
    )
    return open_trial(
        spec,
        parents or make_samples(20),
        make_samples(40, seed=1, prefix="val"),
        tracker,
        run_dir,
    )


def _fake_training(trial, stop_after=None):
    """Create validation scores and checkpoint records for a trial."""
    for batch in trial.batches():
        if batch.step == stop_after:
            break
    probs, history = {}, []
    for step, quality in QUALITY.items():
        noise = np.random.default_rng(step).standard_normal(
            trial.val.labels.shape
        )
        probs[step] = np.clip(
            0.5 + quality * (trial.val.labels - 0.5) + 0.2 * noise, 0, 1
        )
        history.append({"step": step, **val_scores(trial.val, probs[step])})
    best_step = max(history, key=lambda entry: entry[SELECTION_METRIC])["step"]
    checkpoints = {}
    for name, step in (("best", best_step), ("final", 30)):
        path = trial.run_dir / f"{name}.ckpt"
        path.write_bytes(f"{name}-{step}".encode())
        checkpoints[name] = Checkpoint(path, step)
    return {
        "trainer": TRAINER,
        "best": checkpoints["best"],
        "final": checkpoints["final"],
        "val_history": history,
        "val_probs": {"best": probs[best_step], "final": probs[30]},
        "init_sha256": "a" * 64,
        "wall_time_s": 1.0,
    }


def test_a_finished_run_records_exposure_ledger_predictions_and_result(
    make_samples, make_test_spec, tmp_path
):
    spec = make_test_spec("A1")
    parents = make_samples(20)
    with _open(spec, make_samples, tmp_path, parents) as trial:
        args = _fake_training(trial)
        result = trial.finish(**args)

    assert result["best_step"] == 20
    assert result["exposure"]["presentations"] == 30 * 8
    assert result["exposure"]["transform_slots"] == 30 * 4
    assert (
        json.loads((tmp_path / "exposure.json").read_text())
        == result["exposure"]
    )
    assert (
        json.loads((tmp_path / "result.json").read_text())["trainer"][
            "fingerprint"
        ]
        == TRAINER.fingerprint
    )

    ledger = pq.read_table(tmp_path / "recipes.parquet")
    assert ledger.num_rows == 30 * 4
    assert max(ledger["step"].to_pylist()) <= spec.max_steps
    assert set(ledger["parent_record_id"].to_pylist()) <= set(parents.record_id)

    predictions = pq.read_table(
        tmp_path / "predictions_val.parquet"
    ).to_pandas()
    n_val = len(trial.val)
    assert len(predictions) == 2 * n_val * len(SUPERCLASSES)
    assert not predictions.duplicated(
        ["checkpoint", "record_id", "class_id"]
    ).any()
    best = predictions[predictions.checkpoint == "best"]
    assert np.array_equal(
        best.y_prob.to_numpy().reshape(n_val, -1), args["val_probs"]["best"]
    )
    assert np.array_equal(
        best.y_true.to_numpy().reshape(n_val, -1), trial.val.labels
    )

    # The stream logged its budget counters at every validation step.
    budget = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert [
        entry["step"]
        for entry in budget
        if "budget/sample_presentations" in entry
    ] == [10, 20, 30]
    assert is_complete(tmp_path, TRAINER)


def test_controls_without_transform_slots_write_an_empty_ledger(
    make_samples, make_test_spec, tmp_path
):
    with _open(make_test_spec("C1"), make_samples, tmp_path) as trial:
        result = trial.finish(**_fake_training(trial))
    assert result["exposure"]["transform_slots"] == 0
    assert pq.read_table(tmp_path / "recipes.parquet").num_rows == 0


def _set(key, value):
    return lambda args: args.update({key: value})


def _shift_best_probs(args):
    args["val_probs"]["best"] = np.clip(args["val_probs"]["best"] + 0.05, 0, 1)


def _drop_first_validation(args):
    args["val_history"] = args["val_history"][1:]


@pytest.mark.parametrize(
    "stop_after, tamper, message",
    [
        (29, None, "consumed 29 of 30"),
        (
            None,
            lambda args: args.update(final=replace(args["final"], step=20)),
            "final checkpoint is at step 20",
        ),
        (
            None,
            lambda args: args.update(best=replace(args["best"], step=30)),
            "first argmax",
        ),
        (None, _shift_best_probs, "the history reports"),
        (None, _drop_first_validation, "every 10 steps"),
        (None, lambda args: args["best"].path.unlink(), "does not exist"),
    ],
    ids=[
        "partly-consumed",
        "wrong-final-step",
        "best-not-argmax",
        "unreproducible-score",
        "skipped-val",
        "no-ckpt",
    ],
)
def test_finish_refuses_a_run_that_broke_the_protocol(
    make_samples, make_test_spec, tmp_path, stop_after, tamper, message
):
    with _open(make_test_spec("A1"), make_samples, tmp_path) as trial:
        args = _fake_training(trial, stop_after=stop_after)
        if tamper:
            tamper(args)
        with pytest.raises(TrialError, match=message):
            trial.finish(**args)
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "recipes.parquet").exists()
    assert not list(tmp_path.glob("*.partial"))


def test_a_finished_run_is_never_overwritten_and_another_trainer_is_refused(
    make_samples, make_test_spec, tmp_path
):
    spec = make_test_spec("A0")
    assert not is_complete(tmp_path, TRAINER)
    with _open(spec, make_samples, tmp_path) as trial:
        trial.finish(**_fake_training(trial))
    assert is_complete(tmp_path, TRAINER)
    with pytest.raises(TrialError, match="fingerprint"):
        is_complete(tmp_path, OTHER_TRAINER)

    with _open(spec, make_samples, tmp_path) as again:
        args = {**_fake_training(again), "trainer": OTHER_TRAINER}
        with pytest.raises(TrialError, match="fingerprint"):
            again.finish(**args)

    # A result left behind by another trial is refused, not reported as done.
    other = make_test_spec("A0", replicate_seed=1)
    (tmp_path / "trial.json").write_text(json.dumps(other.to_dict()))
    with pytest.raises(TrialError, match="belongs to trial"):
        is_complete(tmp_path, TRAINER)


def test_trial_id_ignores_paths_and_changes_with_every_scientific_input(
    make_test_spec,
):
    base = make_test_spec("A1")
    for paths in (
        {"registry": "/elsewhere"},
        {"manifest": "/copy/pool25.json"},
        {"calibration": "/c.json"},
    ):
        assert make_test_spec("A1", **paths).trial_id == base.trial_id
    for change in (
        {"replicate_seed": 1},
        {"max_steps": 60},
        {"manifest_sha256": "2" * 64},
        {"calibration_sha256": "3" * 64},
        {"transform": {**base.transform, "wander_mv": [0.0, 2.0]}},
    ):
        assert make_test_spec("A1", **change).trial_id != base.trial_id


def test_planned_trial_files_are_frozen(make_test_spec, tmp_path):
    spec = make_test_spec("A1")
    assert write_spec(tmp_path, spec)
    assert not write_spec(tmp_path, spec)  # the same trial again is fine
    with pytest.raises(TrialError, match="already holds trial"):
        write_spec(tmp_path, make_test_spec("A1", max_steps=60))

    payload = json.loads((tmp_path / "trial.json").read_text())
    payload["max_steps"] = 10  # train less without re-planning
    (tmp_path / "trial.json").write_text(json.dumps(payload))
    with pytest.raises(TrialError, match="re-plan"):
        load_spec(tmp_path)


def test_syntide_imports_no_training_framework():
    """Syntide can be imported without a training framework."""
    code = (
        "import importlib, pkgutil, sys, syntide\n"
        "for m in "
        "pkgutil.walk_packages(syntide.__path__, "
        "'syntide.'): importlib.import_module(m.name)\n"
        "print(sorted(n for n in ('torch', "
        "'lightning', 'syntide_trainers') if n in "
        "sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]"
