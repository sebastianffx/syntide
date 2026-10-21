"""The reference trainer passes SYNTIDE's checks, and its predictor reproduces what it reported."""

import json
from dataclasses import asdict

import numpy as np
import pyarrow.parquet as pq
import torch

from syntide.data.schema import ARTIFACT_TYPES, LEADS, SUPERCLASSES, Samples
from syntide.eval.tracking import TrackerConfig, make_tracker
from syntide.generate.generic import GenericAugConfig
from syntide.trial import make_spec, open_trial, write_spec
from syntide_trainers.torch_loop import (
    ResNet1d,
    fit,
    load_config,
    predict,
    state_sha256,
    train,
    trainer_info,
)


def _samples(n: int, seed: int) -> Samples:
    rng = np.random.default_rng(seed)
    artifacts = np.zeros((n, len(ARTIFACT_TYPES)), dtype=bool)
    artifacts[::2, 0] = True
    return Samples(
        record_id=np.array([f"rec-{seed}-{i}" for i in range(n)]),
        subject_uid=np.array([f"ptb-xl:{seed}-{i}" for i in range(n)]),
        labels=rng.random((n, len(SUPERCLASSES))) < 0.5,
        artifacts=artifacts,
        x=(0.3 * rng.standard_normal((n, len(LEADS), 100))).astype(np.float32),
    )


def _spec():
    return make_spec(
        protocol="test-protocol",
        regime="scarcity25",
        arm="A1",
        replicate_seed=0,
        source_role="depleted",
        transform_family="generic",
        sampling_policy="fixed",
        generator="per_lead_aug@1",
        validators=[],
        transform=asdict(GenericAugConfig()),
        dataset="physionet/ptb-xl@1.0.3",
        source_sha256="0" * 64,
        manifest_sha256="1" * 64,
        calibration_sha256=None,
        batch_size=8,
        max_steps=30,
        val_every=10,
        transform_fraction=0.5,
        registry="unused",
        manifest="unused.json",
        calibration=None,
    )


def test_reference_trainer_run_is_accepted_and_its_predictor_reproduces_best_probs(tmp_path):
    spec = _spec()
    cfg = load_config(overrides=["model.width=4", "device=cpu", "log_every=5"])
    info = trainer_info(cfg)
    write_spec(tmp_path, spec)
    tracker = make_tracker(TrackerConfig("files"), tmp_path, spec.run_name, spec.protocol, {})
    val = _samples(40, seed=1)
    with open_trial(spec, _samples(24, seed=0), val, tracker, tmp_path) as trial:
        result = fit(trial, cfg, info)

    assert result["final_step"] == 30
    assert torch.load(tmp_path / "final.pt", weights_only=True)["step"] == 30
    assert result["best_step"] in (10, 20, 30)
    # Same seed, same initial weights: the hash in result.json is what init_seed produces.
    torch.manual_seed(spec.init_seed)
    assert result["trainer"]["init_sha256"] == state_sha256(ResNet1d(width=4))

    predictions = pq.read_table(tmp_path / "predictions_val.parquet").to_pandas()
    reported = predictions[predictions.checkpoint == "best"].y_prob.to_numpy().reshape(len(val), -1)
    assert np.array_equal(predict(tmp_path / "best.pt", val.x, "cpu"), reported)

    logged = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [e["step"] for e in logged if "train/loss" in e] == [5, 10, 15, 20, 25, 30]
    assert [e["step"] for e in logged if "val/auroc_overall" in e] == [10, 20, 30]

    # A finished trial is skipped without touching the data (the registry path is never opened).
    assert train(tmp_path, cfg, TrackerConfig("files")) is None
