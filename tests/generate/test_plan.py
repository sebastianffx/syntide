"""Planning: one frozen trial.json per selected trial, and nothing at all when the protocol is wrong."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from syntide.data.challenge import ChallengeSpec, build_manifest, load_manifest, save_manifest
from syntide.data.schema import ARTIFACT_TYPES
from syntide.generate.__main__ import GenerateConfig, calibrate_regime
from syntide.generate.ecg_aug import EcgAugConfig
from syntide.generate.generic import GenericAugConfig
from syntide.generate.plan import (
    ArmConfig,
    PlanError,
    ProtocolConfig,
    RegimeConfig,
    Selection,
    plan_trials,
    write_plan,
)
from syntide.generate.simvea import SimveaConfig
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


def test_a_regime_runs_only_the_arms_it_lists(protocol, synthetic_index, tmp_path):
    # A reference regime that runs one arm only, and needs no calibration for the editor arm it skips.
    protocol.regimes["reference"] = replace(protocol.regimes["scarcity25"], arms=["C1"])
    specs = plan_trials(protocol, Selection(arms=["A0", "C1"]), index=synthetic_index, source=SOURCE)
    assert sorted({(s.regime, s.arm) for s in specs}) == [
        ("reference", "C1"),
        ("scarcity25", "A0"),
        ("scarcity25", "C1"),
    ]
    assert len(specs) == 9  # three seeds each
    (reference,) = plan_trials(
        protocol, Selection(regimes=["reference"], seeds=[0]), index=synthetic_index, source=SOURCE
    )
    assert reference.arm == "C1"
    with pytest.raises(PlanError, match="no selected regime runs"):
        plan_trials(
            protocol, Selection(arms=["A0"], regimes=["reference"]), index=synthetic_index, source=SOURCE
        )


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
        (_regime(arms=["A0", "A9"]), ["A0"], "lists unknown arms"),
        (_arm("A1", transform_fraction=0.3), ["A1"], "whole number of slots"),
        (_arm("C1", transform_fraction=0.5), ["C1"], "only depleted arms"),
        (_arm("A3", generic=GenericAugConfig()), ["A3"], "generic arms only"),
        (_arm("T1", transform_family="ecg_aug", sampling_policy="fixed"), ["T1"], "set the ecg_aug"),
        (
            _arm("T1", transform_family="ecg_aug", sampling_policy="fixed", ecg_aug=EcgAugConfig()),
            ["T1"],
            "switch on at least one",
        ),
        (_arm("A1", ecg_aug=EcgAugConfig(scale=[0.9, 1.1])), ["A1"], "set the ecg_aug"),
        (_arm("T6", transform_family="simvea", sampling_policy="fixed"), ["T6"], "set the simvea"),
        (
            _arm("T6", transform_family="simvea", sampling_policy="fixed", simvea=SimveaConfig(beta=[0, 3])),
            ["T6"],
            "high <= 2",
        ),
        (_arm("A1", simvea=SimveaConfig()), ["A1"], "set the simvea"),
    ],
    ids=[
        "no-budget",
        "budget-off-cadence",
        "transform-on-natural-mix",
        "trains-on-test",
        "no-gate",
        "gate-failed",
        "unknown-arm",
        "regime-lists-unknown-arm",
        "fraction-off-slots",
        "fraction-without-slots",
        "envelope-on-editor-arm",
        "ecg-aug-without-ops",
        "ecg-aug-all-off",
        "ops-on-generic-arm",
        "simvea-without-settings",
        "simvea-beta-beyond-2",
        "simvea-on-generic-arm",
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


def test_the_50_percent_protocol_defines_21_trials_with_one_budget():
    import syntide.generate.__main__  # noqa: F401  (registers the schema)
    from syntide.generate.plan import regime_arms

    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        cfg = compose("generate", overrides=["protocol=ptbxl_artifact_50"])
    protocol = OmegaConf.to_object(cfg).protocol
    runs = {name: regime_arms(protocol, name) for name in protocol.regimes}
    assert runs == {"scarcity50": ["A0", "A1", "A2", "A3", "C1", "C2"], "full100": ["C2"]}
    assert sum(len(arms) for arms in runs.values()) * len(protocol.seeds) == 21
    assert len({regime.max_steps for regime in protocol.regimes.values()}) == 1  # one budget for every trial


def test_dose_and_technique_arms_carry_their_own_fraction_and_transform(protocol, synthetic_index):
    strong = GenericAugConfig(gain=[0.8, 1.2], max_shift_s=1.0, noise_rms_mv=[0.0, 0.2], wander_mv=[0.0, 7.0])
    protocol.arms["S4"] = ArmConfig("depleted", "generic", "fixed", transform_fraction=1.0, generic=strong)
    protocol.arms["T3"] = ArmConfig("depleted", "ecg_aug", "fixed", ecg_aug=EcgAugConfig(scale=[0.7, 1.43]))
    specs = plan_trials(
        protocol, Selection(arms=["A0", "A1", "S4", "T3"], seeds=[0]), index=synthetic_index, source=SOURCE
    )
    by_arm = {spec.arm: spec for spec in specs}
    assert by_arm["S4"].transform_fraction == 1.0 and by_arm["S4"].transform["wander_mv"] == [0.0, 7.0]
    assert by_arm["T3"].generator == "ecg_aug@1" and by_arm["T3"].transform["scale"] == [0.7, 1.43]
    # Arms without overrides keep the protocol's fraction and A1's envelope.
    assert (
        by_arm["A1"].transform_fraction == by_arm["T3"].transform_fraction == by_arm["A0"].transform_fraction
    )
    assert by_arm["A1"].transform == asdict(protocol.generic)
    assert len({spec.init_seed for spec in specs}) == 1  # paired: one initialization per seed


def test_the_augmentation_sweep_defines_42_paired_trials_with_one_budget():
    import syntide.generate.__main__  # noqa: F401  (registers the schema)
    from syntide.generate.plan import regime_arms

    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        cfg = compose("generate", overrides=["protocol=ptbxl_aug_sweep"])
    protocol = OmegaConf.to_object(cfg).protocol
    runs = {name: regime_arms(protocol, name) for name in protocol.regimes}
    assert sum(len(arms) for arms in runs.values()) * len(protocol.seeds) == 42
    assert runs["full100"] == ["C2"]
    assert {"A0", "A1", "A3"} <= set(runs["scarcity50"])  # the 50% experiment's references
    assert len({regime.max_steps for regime in protocol.regimes.values()}) == 1
    arms = protocol.arms
    # The dose ladder: S1 < S2 < A1 < S3 < S4 in (share of the batch) x (strength).
    dose = {
        name: (arms[name].transform_fraction or protocol.transform_fraction)
        * (arms[name].generic or protocol.generic).wander_mv[1]
        for name in ("S1", "S2", "A1", "S3", "S4")
    }
    assert list(dose) == sorted(dose, key=dose.get)
    # Technique arms augment exactly A1's slots, with one operation each.
    for name in ("T1", "T2", "T3", "T4", "T5"):
        assert arms[name].transform_fraction is None
        assert sum(value is not None for value in asdict(arms[name].ecg_aug).values()) == 1


def test_the_simvea_extension_plans_t6_on_the_sweeps_pool_budget_and_slots():
    """T6 pairs with the sweep's runs: same pool, budget, batch, cadence, seeds and share of slots."""
    import syntide.generate.__main__  # noqa: F401  (registers the schema)
    from syntide.generate.plan import regime_arms

    protocols = {}
    for name in ("ptbxl_aug_sweep", "ptbxl_aug_simvea"):
        with initialize_config_dir(config_dir=str(CONF), version_base=None):
            protocols[name] = OmegaConf.to_object(
                compose("generate", overrides=[f"protocol={name}"])
            ).protocol
    sweep, ext = protocols["ptbxl_aug_sweep"], protocols["ptbxl_aug_simvea"]
    assert {name: regime_arms(ext, name) for name in ext.regimes} == {"scarcity50": ["T6"]}
    assert not set(ext.arms) & set(sweep.arms)  # no arm of the sweep is trained again
    for key in ("batch_size", "val_every", "transform_fraction", "seeds", "dataset", "generic", "editor"):
        assert getattr(ext, key) == getattr(sweep, key), key
    for key in ("manifest", "max_steps"):
        assert getattr(ext.regimes["scarcity50"], key) == getattr(sweep.regimes["scarcity50"], key), key
    t6 = ext.arms["T6"]
    assert (t6.transform_family, t6.sampling_policy, t6.transform_fraction) == ("simvea", "fixed", None)
    assert t6.simvea == SimveaConfig()  # the paper's random-hyperparameter setting
