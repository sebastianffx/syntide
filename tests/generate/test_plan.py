"""Planning: one frozen trial.json per selected trial, and nothing at all when the protocol is wrong."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from syntide.data.challenge import ChallengeSpec, build_manifest, load_manifest, save_manifest
from syntide.data.schema import ARTIFACT_TYPES
from syntide.generate.__main__ import GenerateConfig, calibrate_regime
from syntide.generate.plan import (
    ArmConfig,
    PlanError,
    ProtocolConfig,
    RegimeConfig,
    Selection,
    plan_trials,
    write_plan,
)
from syntide.trial import load_spec

SOURCE = {"dataset": "physionet/ptb-xl", "version": "1.0.3", "sha256": "test"}
CONF = Path(__file__).resolve().parents[2] / "conf"


@pytest.fixture
def protocol(synthetic_index, tmp_path) -> ProtocolConfig:
    manifest = tmp_path / "manifests" / "pool25.json"
    save_manifest(build_manifest(synthetic_index, ChallengeSpec(), 0.25, SOURCE), manifest)
    return ProtocolConfig(
        id="test-protocol",
        arms={
            "A0": ArmConfig("depleted", "none", "none"),
            "A1": ArmConfig("depleted", "generic", "fixed"),
            "A3": ArmConfig("depleted", "editor", "gap_targeted"),
            "C1": ArmConfig("natural_mix", "none", "none"),
        },
        regimes={
            "scarcity25": RegimeConfig(
                manifest=str(manifest),
                calibration=str(tmp_path / "manifests" / "calibration" / "pool25.json"),
                generator_gate=str(tmp_path / "manifests" / "generator_gate_pool25.json"),
                max_steps=500,
            )
        },
        seeds=[0, 1, 2],
    )


def _plan(protocol, select, index, runs_dir):
    return write_plan(plan_trials(protocol, select, index=index, source=SOURCE), runs_dir)


def test_plan_writes_frozen_trials_and_reuses_the_ones_already_planned(protocol, synthetic_index, tmp_path):
    runs = tmp_path / "runs"
    gate = Selection(arms=["A0", "A1", "C1"], seeds=[0, 1])
    written = _plan(protocol, gate, synthetic_index, runs)

    assert len(written) == 6
    a0, a1, c1 = (
        load_spec(runs / "test-protocol" / "scarcity25" / arm / "seed0") for arm in ("A0", "A1", "C1")
    )
    assert a0.init_seed == a1.init_seed == c1.init_seed
    assert a0.transform_fraction == a1.transform_fraction == 0.5  # A0-A3 share one slot schedule
    assert c1.transform_fraction == 0.0
    assert _plan(protocol, gate, synthetic_index, runs) == []  # planning again changes nothing

    # The full selection adds only the missing trials; the runs planned first are reused as they are.
    added = _plan(protocol, Selection(arms=["A0", "A1", "C1"]), synthetic_index, runs)
    assert sorted(p.name for p in added) == ["seed2"] * 3
    plan = json.loads((runs / "test-protocol" / "plan.json").read_text())
    assert len(plan["trials"]) == 9

    # A new A1 envelope changes only A1's identity, and a planned A1 trial is never silently replaced.
    protocol.generic = replace(protocol.generic, gain=[0.8, 1.2])
    specs = plan_trials(protocol, gate, index=synthetic_index, source=SOURCE)
    assert {
        s.arm for s in specs if s.trial_id == load_spec(runs / "test-protocol" / s.run_name).trial_id
    } == {
        "A0",
        "C1",
    }
    with pytest.raises(PlanError, match="hold different trials"):
        write_plan(specs, runs)
    assert load_spec(runs / "test-protocol" / "scarcity25" / "A1" / "seed0") == a1


def test_editor_arms_are_planned_against_the_exact_calibration_their_gate_passed(
    protocol, synthetic_index, make_samples, tmp_path
):
    regime = protocol.regimes["scarcity25"]
    manifest = load_manifest(Path(regime.manifest))
    retained = _as_records(
        make_samples(len(manifest.roles["retained_artifact"]), n_samples=1000), manifest, "retained_artifact"
    )
    flags = np.zeros_like(retained.artifacts)
    for row in range(len(retained)):  # spread the retained records over the three v1 types
        flags[row, ARTIFACT_TYPES.index(("baseline_drift", "static_noise", "burst_noise")[row % 3])] = True
    retained = replace(retained, artifacts=flags)
    depleted = _as_records(make_samples(30, n_samples=1000), manifest, "depleted")
    report = calibrate_regime(manifest, retained, depleted, protocol.editor, regime, recipes=60)
    assert report["passed"]

    runs = tmp_path / "runs"
    (a3,) = plan_trials(protocol, Selection(arms=["A3"], seeds=[0]), index=synthetic_index, source=SOURCE)
    assert a3.calibration == regime.calibration and a3.calibration_sha256 == report["calibration_sha256"]
    assert a3.generator == "ecg.electrode_artifact@1.0.0" and a3.validators == ["bounds@1", "alignment@1"]

    # A gate result for another calibration file does not count.
    gate = json.loads(Path(regime.generator_gate).read_text())
    Path(regime.generator_gate).write_text(json.dumps({**gate, "calibration_sha256": "f" * 64}))
    with pytest.raises(PlanError, match="another calibration"):
        _plan(protocol, Selection(arms=["A3"]), synthetic_index, runs)
    assert not runs.exists()


def _as_records(samples, manifest, role):
    return replace(samples, record_id=np.array(manifest.roles[role][: len(samples)]))


def _regime(**change):
    return lambda p: p.regimes.update(scarcity25=replace(p.regimes["scarcity25"], **change))


def _arm(name, **change):
    return lambda p: p.arms.update({name: replace(p.arms.get(name, ArmConfig()), **change)})


def _failed_gate(p):
    regime = p.regimes["scarcity25"]
    Path(regime.calibration).parent.mkdir(parents=True)
    Path(regime.calibration).write_text("{}")
    Path(regime.generator_gate).write_text(json.dumps({"passed": False}))


@pytest.mark.parametrize(
    "change, arms, message",
    [
        (_regime(max_steps=None), ["A0"], "no max_steps"),
        (_regime(max_steps=260), ["A0"], "multiple of val_every"),
        (_arm("A1", source_role="natural_mix"), ["A1"], "depleted parents"),
        (_arm("T0", source_role="test"), ["T0"], "is not one of"),
        (None, ["A3"], "needs the regime's calibration and generator_gate"),
        (_failed_gate, ["A3"], "did not pass"),
        (None, ["A9"], "unknown arms"),
    ],
    ids=[
        "no-budget",
        "budget-off-cadence",
        "transform-on-natural-mix",
        "trains-on-test",
        "no-gate",
        "gate-failed",
        "unknown-arm",
    ],
)
def test_plan_refuses_before_writing_anything(protocol, synthetic_index, tmp_path, change, arms, message):
    if change:
        change(protocol)
    runs = tmp_path / "runs"
    with pytest.raises(PlanError, match=message):
        _plan(protocol, Selection(arms=[*arms, "A0"]), synthetic_index, runs)
    assert not runs.exists()


def test_the_protocol_yaml_defines_the_72_trial_grid():
    import syntide.generate.__main__  # noqa: F401  (registers the schema)

    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        cfg = compose("generate", overrides=["protocol.regimes.scarcity10.max_steps=500"])
    config: GenerateConfig = OmegaConf.to_object(cfg)
    protocol = config.protocol
    assert len(protocol.arms) * len(protocol.regimes) * len(protocol.seeds) == 72
    assert protocol.regimes["scarcity10"].max_steps == 500  # the command line overrides a budget
    # Every budget is set (by the pilot) and lands on the validation cadence.
    assert all(regime.max_steps % protocol.val_every == 0 for regime in protocol.regimes.values())
    assert {arm.source_role for arm in protocol.arms.values()} == {"depleted", "natural_mix", "pool"}
