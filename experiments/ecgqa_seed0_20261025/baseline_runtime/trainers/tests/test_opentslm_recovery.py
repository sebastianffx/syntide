"""Artifact-only recovery tests; no model is instantiated or executed."""

import json
from types import SimpleNamespace

import pytest

from syntide.baseline import metrics
from syntide_trainers import opentslm_loop as loop


def _row(task_id="q1", answer="yes"):
    return {
        "task_id": task_id,
        "record_id": "ptbxl-1",
        "subject_uid": "patient-1",
        "template_id": 1,
        "is_artifact": False,
        "question": "Is this normal?",
        "answer": answer,
        "answer_options": ["yes", "no"],
    }


def _prediction(row):
    return {**row, "output_text": "Reasoning. Answer: yes", "parsed_answer": "yes", "invalid": False}


@pytest.mark.parametrize("mutation", ["truth", "order", "parsed", "missing"])
def test_recovery_rejects_changed_or_incomplete_predictions(tmp_path, mutation):
    expected = [_row("q1"), _row("q2")]
    predictions = [_prediction(row) for row in expected]
    if mutation == "truth":
        predictions[0]["answer"] = "no"
    elif mutation == "order":
        predictions.reverse()
    elif mutation == "parsed":
        predictions[0]["parsed_answer"] = "no"
    else:
        predictions.pop()
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in predictions) + "\n")
    with pytest.raises(ValueError):
        loop.read_saved_predictions(path, expected)


def test_metrics_only_recovery_never_loads_model(tmp_path, monkeypatch):
    rows = [_row()]
    config = {
        "seed": 0,
        "max_steps": 10,
        "selection_limit": 0,
        "max_length": 1536,
        "max_new_tokens": 384,
        "manifest_sha256": "manifest-1",
        "prompt_version": loop.PROMPT_VERSION,
        "trainer_source_sha256": loop.sha256(loop.Path(loop.__file__)),
        "metrics_source_sha256": loop.sha256(loop.Path(metrics.__file__)),
        "base_revision": "a" * 40,
    }
    loop.write_json(tmp_path / "config.json", config)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint fixture; torch.load is mocked")
    loop.write_json(
        tmp_path / "train_result.json",
        {
            "status": "trained",
            "max_steps": 10,
            "best_step": 5,
            "best_checkpoint_sha256": loop.sha256(checkpoint),
        },
    )
    loop.write_json(
        tmp_path / "selection_history.json",
        [
            {"step": 5, "accuracy": 0.5},
            {"step": 10, "accuracy": 0.4},
        ],
    )
    payload = {"config": config, "step": 5, "selection_accuracy": 0.5}
    monkeypatch.setattr(loop.torch, "load", lambda *a, **kw: payload)
    monkeypatch.setattr(
        loop,
        "Prepared",
        lambda *a, **kw: SimpleNamespace(
            manifest_sha256="manifest-1",
            rows={"test": rows},
        ),
    )
    calls = []

    def refuse_model(*_args):
        calls.append(True)
        raise RuntimeError("No models allowed in this test")

    monkeypatch.setattr(loop, "load_model", refuse_model)
    args = SimpleNamespace(
        prepared=tmp_path, run_dir=tmp_path, split="test", zero_signal=False, limit=0, batch_size=32
    )
    # The claim is durable before a model is loaded. Simulate the separate crash
    # window after predictions were atomically published but before metrics.
    with pytest.raises(RuntimeError, match="No models allowed"):
        loop.evaluate(args)
    output = tmp_path / "test_predictions.jsonl"
    output.write_text(json.dumps(_prediction(rows[0])) + "\n")
    before = output.read_bytes()
    loop.evaluate(args)
    assert len(calls) == 1
    assert output.read_bytes() == before
    result = json.loads((tmp_path / "test_metrics.json").read_text())
    assert result["metrics_recovered_from_predictions"] is True
    assert result["strict"]["groups"]["all"]["accuracy"] == 1.0
    loop.evaluate(args)
    assert len(calls) == 1
