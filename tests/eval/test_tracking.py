"""Every dashboard backend takes the same calls, and each conf/tracker/*.yaml selects one."""

from pathlib import Path

import pytest
from matplotlib.figure import Figure

from syntide.eval.tracking import BACKENDS, TrackerConfig, make_tracker

CONF = Path(__file__).resolve().parents[2] / "conf" / "tracker"


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_backend_takes_the_same_calls(backend, tmp_path, monkeypatch):
    if backend == "wandb":
        pytest.importorskip("wandb")
        monkeypatch.setenv("WANDB_SILENT", "true")
    tracker = make_tracker(
        TrackerConfig(backend=backend, mode="offline"), tmp_path, "scarcity25/A1/seed0", "test", {"arm": "A1"}
    )
    fig = Figure()
    fig.add_subplot().plot([0, 1], [0, 1])
    tracker.log_scalars({"train/loss": 0.5}, 10)
    tracker.log_scalars({"val/auroc_overall": 0.75, "val/loss": 0.4}, 20)
    tracker.log_figure("examples/original_edit_residual", fig, 20)
    tracker.log_text("evaluation/findings", "# Findings\n\nnone yet", 20)
    tracker.log_summary({"arm": "A1", "model.width": 64, "widths": [1, 2]}, {"summary/best_step": 20})
    tracker.close()

    if backend == "files":
        lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
        assert lines == [
            '{"step": 10, "train/loss": 0.5}',
            '{"step": 20, "val/auroc_overall": 0.75, "val/loss": 0.4}',
        ]
        assert (tmp_path / "figures" / "examples-original_edit_residual_step20.png").exists()
        assert (tmp_path / "summary.json").exists()
    elif backend == "tensorboard":
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        events = EventAccumulator(str(tmp_path))
        events.Reload()
        assert [(e.step, e.value) for e in events.Scalars("val/auroc_overall")] == [(20, 0.75)]


def test_each_tracker_yaml_selects_its_backend():
    for path in sorted(CONF.glob("*.yaml")):
        assert TrackerConfig.from_yaml(path).backend == path.stem
    assert sorted(p.stem for p in CONF.glob("*.yaml")) == sorted(BACKENDS)
    with pytest.raises(ValueError, match="files, tensorboard, wandb"):
        make_tracker(TrackerConfig(backend="mlflow"), Path("."), "run", "group", {})
