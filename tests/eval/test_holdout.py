"""Test-once scoring: only after every trial is finished, on unchanged checkpoints, and never twice."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from syntide.data.schema import SUPERCLASSES
from syntide.eval.holdout import CLAIM_FILE, HoldoutError, evaluate
from syntide.eval.stats import SELECTION_METRIC, val_scores
from syntide.eval.tracking import TrackerConfig, make_tracker
from syntide.trial import Checkpoint, TrainerInfo, open_trial, write_spec

TRAINER = TrainerInfo(name="fake", version="1", config={"width": 1}, device="cpu", predictor="fake:predict")
QUALITY = {10: 0.2, 20: 0.6, 30: 0.4}  # the fake model separates val labels best at step 20


def _train(protocol_dir, spec, make_samples):
    """Plan and finish one trial the way an honest trainer would; return its plan.json entry."""
    run_dir = protocol_dir / spec.regime / spec.arm / f"seed{spec.replicate_seed}"
    write_spec(run_dir, spec)
    tracker = make_tracker(TrackerConfig("files"), run_dir, spec.run_name, spec.protocol, spec.to_dict())
    val = make_samples(40, seed=1, prefix="val")
    with open_trial(spec, make_samples(20), val, tracker, run_dir) as trial:
        for _ in trial.batches():
            pass
        probs, history = {}, []
        for step, quality in QUALITY.items():
            noise = np.random.default_rng(step).standard_normal(val.labels.shape)
            probs[step] = np.clip(0.5 + quality * (val.labels - 0.5) + 0.2 * noise, 0, 1)
            history.append({"step": step, **val_scores(val, probs[step])})
        best_step = max(history, key=lambda entry: entry[SELECTION_METRIC])["step"]
        checkpoints = {}
        for name, step in (("best", best_step), ("final", 30)):
            (run_dir / f"{name}.ckpt").write_bytes(f"{spec.arm}-{name}-{step}".encode())
            checkpoints[name] = Checkpoint(run_dir / f"{name}.ckpt", step)
        trial.finish(
            TRAINER,
            best=checkpoints["best"],
            final=checkpoints["final"],
            val_history=history,
            val_probs={"best": probs[best_step], "final": probs[30]},
            init_sha256="a" * 64,
            wall_time_s=1.0,
        )
    return {"run_dir": str(run_dir.relative_to(protocol_dir)), "trial_id": spec.trial_id}


def _write_plan(protocol_dir, entries):
    (protocol_dir / "plan.json").write_text(json.dumps({"protocol": "test-protocol", "trials": entries}))


class FakePredictor:
    """Scores test ECGs by their labels plus checkpoint-specific noise, and counts its calls."""

    def __init__(self, test):
        self.test, self.calls = test, []

    def __call__(self, checkpoint, x, device):
        self.calls.append((checkpoint.path.name, checkpoint.step, device))
        assert x is self.test.x  # the predictor gets the test ECGs, nothing else
        rng = np.random.default_rng(len(self.calls))
        return np.clip(
            0.5 + 0.3 * (self.test.labels - 0.5) + 0.2 * rng.standard_normal(self.test.labels.shape), 0, 1
        )


@pytest.fixture
def finished(tmp_path, make_samples, make_test_spec):
    protocol_dir = tmp_path / "runs" / "test-protocol"
    entries = [_train(protocol_dir, make_test_spec(arm), make_samples) for arm in ("A0", "C1")]
    _write_plan(protocol_dir, entries)
    return protocol_dir, entries


def test_scores_every_frozen_run_once_and_never_again(finished, make_samples):
    protocol_dir, _ = finished
    test = make_samples(30, seed=2, prefix="test")
    predictor = FakePredictor(test)
    rows = evaluate(protocol_dir, test=test, load_predictor=lambda name: predictor)

    assert [(row["arm"], row["checkpoint"], row["step"]) for row in rows] == [
        ("A0", "best", 20),
        ("C1", "best", 20),
    ]
    assert [call[0] for call in predictor.calls] == ["best.ckpt", "best.ckpt"]
    assert {call[2] for call in predictor.calls} == {"cpu"}  # the device the trainer recorded
    for row in rows:
        run_dir = protocol_dir / "scarcity25" / row["arm"] / "seed0"
        predictions = pq.read_table(run_dir / "predictions_test.parquet").to_pandas()
        assert len(predictions) == len(test) * len(SUPERCLASSES)
        probs = predictions.y_prob.to_numpy().reshape(len(test), -1)
        assert row["test/auroc_overall"] == pytest.approx(val_scores(test, probs)["val/auroc_overall"])
        assert json.loads((run_dir / "test_scores.json").read_text()) == row
    claim = json.loads((protocol_dir / CLAIM_FILE).read_text())
    assert [trial["run_dir"] for trial in claim["trials"]] == ["scarcity25/A0/seed0", "scarcity25/C1/seed0"]

    # A second call reads the scores back; nothing is predicted again.
    def refuse(name):
        raise AssertionError("a scored run was scored again")

    assert evaluate(protocol_dir, test=test, load_predictor=refuse) == rows


def test_nothing_is_scored_until_every_planned_trial_is_finished(finished, make_samples, make_test_spec):
    protocol_dir, entries = finished
    pending = make_test_spec("A1")
    write_spec(protocol_dir / "scarcity25" / "A1" / "seed0", pending)
    _write_plan(protocol_dir, [*entries, {"run_dir": "scarcity25/A1/seed0", "trial_id": pending.trial_id}])
    with pytest.raises(HoldoutError, match="1 planned trials are not finished"):
        evaluate(protocol_dir, test=make_samples(30, seed=2), load_predictor=lambda name: None)
    assert not (protocol_dir / CLAIM_FILE).exists()


def test_a_checkpoint_changed_after_training_is_refused(finished, make_samples):
    protocol_dir, _ = finished
    (protocol_dir / "scarcity25" / "C1" / "seed0" / "best.ckpt").write_bytes(b"retrained")
    with pytest.raises(HoldoutError, match="changed after training"):
        evaluate(protocol_dir, test=make_samples(30, seed=2), load_predictor=lambda name: None)


def test_a_claimed_test_fold_is_never_spent_on_another_set_of_trials(finished, make_samples, make_test_spec):
    protocol_dir, entries = finished
    test = make_samples(30, seed=2, prefix="test")
    evaluate(protocol_dir, test=test, load_predictor=lambda name: FakePredictor(test))
    added = _train(protocol_dir, make_test_spec("A1"), make_samples)
    _write_plan(protocol_dir, [*entries, added])
    with pytest.raises(HoldoutError, match="never scored twice"):
        evaluate(protocol_dir, test=test, load_predictor=lambda name: FakePredictor(test))
    assert not (protocol_dir / "scarcity25" / "A1" / "seed0" / "test_scores.json").exists()
