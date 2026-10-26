"""Convert an experiment protocol into validated training trial files.

A protocol describes experiment treatments, data conditions, random seeds, and
training budgets. Planning checks data separation and configuration consistency
before writing one ``trial.json`` for each selected combination. Trainers read
these saved files rather than the original protocol.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow as pa
from omegaconf import MISSING

from syntide.data.challenge import Manifest, load_manifest
from syntide.data.leakage import check_manifest
from syntide.generate import ecg_aug, editor, generic, simvea
from syntide.generate.ecg_aug import EcgAugConfig
from syntide.generate.editor import EditorConfig
from syntide.generate.generic import GenericAugConfig
from syntide.generate.policy import CalibrationError, load_calibration
from syntide.generate.simvea import SimveaConfig
from syntide.trial import TrialSpec, load_spec, make_spec, write_spec

# Roles an arm may train on. val, test, withheld and unused never feed a
# training stream.
TRAIN_ROLES = ("depleted", "natural_mix", "pool")
# Transform family -> (allowed sampling policies, generator id, blocking
# validators).
FAMILIES = {
    "none": (("none",), None, ()),
    "generic": (("fixed",), f"{generic.GENERATOR}@{generic.VERSION}", ()),
    "editor": (
        ("coverage_matched", "gap_targeted"),
        f"{editor.GENERATOR}@{editor.VERSION}",
        ("bounds@1", "alignment@1"),
    ),
    "ecg_aug": (("fixed",), f"{ecg_aug.GENERATOR}@{ecg_aug.VERSION}", ()),
    "simvea": (("fixed",), f"{simvea.GENERATOR}@{simvea.VERSION}", ()),
}


@dataclass
class ArmConfig:
    source_role: str = "depleted"
    transform_family: str = "none"
    sampling_policy: str = "none"
    # Dose arms (ptbxl_aug_sweep) set their own share of transformed slots and
    # their own envelope.
    # Unset, an arm takes the protocol's transform_fraction and generic
    # envelope.
    transform_fraction: float | None = None
    generic: GenericAugConfig | None = None  # generic arms only
    ecg_aug: EcgAugConfig | None = (
        None  # required by ecg_aug arms: the operations they switch on
    )
    simvea: SimveaConfig | None = (
        None  # required by simvea arms: SIMVEA_h's settings
    )


@dataclass
class RegimeConfig:
    manifest: str = MISSING
    calibration: str | None = None  # Phase 2B: required by editor arms
    generator_gate: str | None = (
        None  # Phase 2B: Gate 4 result; editor arms need "passed": true
    )
    max_steps: int | None = (
        None  # set from the pilot; planning refuses a regime without it
    )
    arms: list[str] | None = (
        None  # the arms this regime runs; None = every arm of the protocol
    )


@dataclass
class ProtocolConfig:
    """Experiment treatments and data conditions with matched budgets."""

    id: str = MISSING
    registry: str = "datasets/timenet-registry"
    dataset: str = "physionet/ptb-xl@1.0.3"
    batch_size: int = 64
    val_every: int = 250
    transform_fraction: float = 0.5
    generic: GenericAugConfig = field(default_factory=GenericAugConfig)
    editor: EditorConfig = field(default_factory=EditorConfig)
    arms: dict[str, ArmConfig] = field(default_factory=dict)
    regimes: dict[str, RegimeConfig] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)


@dataclass
class Selection:
    """Protocol subset to plan; an unspecified field selects all its values."""

    arms: list[str] | None = None
    regimes: list[str] | None = None
    seeds: list[int] | None = None


class PlanError(ValueError):
    """The requested experiment cannot be planned with these settings."""


def plan_trials(
    protocol: ProtocolConfig,
    select: Selection,
    *,
    index: pa.Table,
    source: dict[str, str],
) -> list[TrialSpec]:
    """Validate the selected experiment and build its trial specifications.

    Args:
        protocol: Complete experiment configuration.
        select: Treatments, data conditions, and seeds to include.
        index: Dataset index used to check record and patient separation.
        source: Dataset source metadata required by the manifest checks.

    Returns:
        Trial specifications ready to be written. This function writes no
        files.

    Raises:
        PlanError: The selection or experiment settings are inconsistent.
    """
    arms = _chosen("arm", select.arms, list(protocol.arms))
    regimes = _chosen("regime", select.regimes, list(protocol.regimes))
    seeds = _chosen("seed", select.seeds, list(protocol.seeds))
    if len(set(protocol.seeds)) != len(protocol.seeds):
        raise PlanError(f"protocol seeds {protocol.seeds} repeat a seed")
    _check_fraction(
        "protocol", protocol.transform_fraction, protocol.batch_size
    )
    for name in arms:
        _check_arm(name, protocol.arms[name])
        if protocol.arms[name].transform_fraction is not None:
            _check_fraction(
                f"arm {name}",
                protocol.arms[name].transform_fraction,
                protocol.batch_size,
            )
    _check_shared_envelope(protocol)
    # A regime runs the selected arms it lists (all of them when it lists none).
    runs = {
        name: [arm for arm in arms if arm in regime_arms(protocol, name)]
        for name in regimes
    }
    if not any(runs.values()):
        raise PlanError(
            f"no selected regime runs any of the selected arms {arms}"
        )

    specs = []
    for regime_name in regimes:
        if not runs[regime_name]:
            continue
        regime = protocol.regimes[regime_name]
        manifest = _checked_manifest(
            regime_name, regime, protocol, index, source
        )
        for arm_name in runs[regime_name]:
            arm = protocol.arms[arm_name]
            if arm.source_role not in manifest.roles:
                raise PlanError(
                    f"arm {arm_name}: {regime.manifest} has no role "
                    f"{arm.source_role!r}"
                )
            calibration = (
                _editor_calibration(
                    arm_name, regime_name, regime, manifest, protocol.editor
                )
                if arm.transform_family == "editor"
                else None
            )
            for seed in seeds:
                specs.append(
                    _spec(
                        protocol,
                        regime_name,
                        regime,
                        manifest,
                        arm_name,
                        arm,
                        seed,
                        calibration,
                    )
                )
    return specs


def regime_arms(protocol: ProtocolConfig, name: str) -> list[str]:
    """Return the experiment treatments enabled for the named data condition."""
    listed = protocol.regimes[name].arms
    if listed is None:
        return list(protocol.arms)
    unknown = sorted(set(listed) - set(protocol.arms))
    if unknown:
        raise PlanError(
            f"regime {name} lists unknown arms {unknown}; the "
            f"protocol has {list(protocol.arms)}"
        )
    return list(listed)


def write_plan(specs: list[TrialSpec], runs_dir: str | Path) -> list[Path]:
    """Write trial specifications and update the experiment index.

    Existing trial files are checked before any writes, so a conflict leaves
    the
    saved experiment unchanged.

    Args:
        specs: Validated trial specifications.
        runs_dir: Directory containing experiment runs.

    Returns:
        Run directories created for newly planned trials.
    """
    runs_dir = Path(runs_dir)
    run_dirs = [run_dir_for(runs_dir, spec) for spec in specs]
    conflicts = [
        (
            f"{run_dir}: holds {load_spec(run_dir).trial_id[:12]}, the "
            f"plan has {spec.trial_id[:12]}"
        )
        for run_dir, spec in zip(run_dirs, specs, strict=True)
        if (run_dir / "trial.json").exists()
        and load_spec(run_dir).trial_id != spec.trial_id
    ]
    if conflicts:
        raise PlanError(
            (
                "these run directories hold different trials; plan into "
                "a new runs_dir, or delete them on "
                "purpose:\n  "
            )
            + "\n  ".join(conflicts)
        )
    written = [
        run_dir
        for run_dir, spec in zip(run_dirs, specs, strict=True)
        if write_spec(run_dir, spec)
    ]

    by_protocol: dict[str, list[tuple[Path, TrialSpec]]] = {}
    for run_dir, spec in zip(run_dirs, specs, strict=True):
        by_protocol.setdefault(spec.protocol, []).append((run_dir, spec))
    for protocol, planned in by_protocol.items():
        index_path = runs_dir / protocol / "plan.json"
        entries = {}
        if index_path.exists():
            entries = {
                entry["run_dir"]: entry
                for entry in json.loads(index_path.read_text())["trials"]
            }
        for run_dir, spec in planned:
            key = str(run_dir.relative_to(runs_dir / protocol))
            entries[key] = {
                "run_dir": key,
                "trial_id": spec.trial_id,
                "regime": spec.regime,
                "arm": spec.arm,
                "seed": spec.replicate_seed,
            }
        payload = {
            "protocol": protocol,
            "trials": [entries[key] for key in sorted(entries)],
        }
        index_path.write_text(
            json.dumps(payload, indent=1) + "\n", encoding="utf-8"
        )
    return written


def run_dir_for(runs_dir: Path, spec: TrialSpec) -> Path:
    return (
        Path(runs_dir)
        / spec.protocol
        / spec.regime
        / spec.arm
        / f"seed{spec.replicate_seed}"
    )


def _chosen(kind: str, chosen: list | None, available: list) -> list:
    if not available:
        raise PlanError(f"the protocol defines no {kind}s")
    if chosen is None:
        return available
    unknown = [item for item in chosen if item not in available]
    if unknown:
        raise PlanError(
            f"unknown {kind}s {unknown}; the protocol has {available}"
        )
    return list(chosen)


def _check_fraction(where: str, fraction: float, batch_size: int) -> None:
    if not 0 <= fraction <= 1:
        raise PlanError(
            f"{where}: transform_fraction must be in [0, 1], got {fraction}"
        )
    if not (fraction * batch_size).is_integer():
        raise PlanError(
            f"{where}: transform_fraction {fraction} x batch_size "
            f"{batch_size} is not a whole number of slots"
        )


def _check_arm(name: str, arm: ArmConfig) -> None:
    if arm.source_role not in TRAIN_ROLES:
        raise PlanError(
            f"arm {name}: source_role {arm.source_role!r} is not one "
            f"of {TRAIN_ROLES}"
        )
    if arm.transform_family not in FAMILIES:
        raise PlanError(
            f"arm {name}: unknown transform_family {arm.transform_family!r}"
        )
    policies = FAMILIES[arm.transform_family][0]
    if arm.sampling_policy not in policies:
        raise PlanError(
            f"arm {name}: {arm.transform_family} arms take "
            f"sampling_policy in {policies}"
        )
    if arm.transform_family != "none" and arm.source_role != "depleted":
        raise PlanError(
            f"arm {name}: a {arm.transform_family} transform must "
            f"start from depleted parents"
        )
    if arm.transform_fraction is not None and arm.source_role != "depleted":
        raise PlanError(
            f"arm {name}: only depleted arms have transform slots to "
            f"set a fraction for"
        )
    if arm.generic is not None and arm.transform_family != "generic":
        raise PlanError(
            f"arm {name}: a generic envelope applies to generic arms only"
        )
    if (arm.ecg_aug is not None) != (arm.transform_family == "ecg_aug"):
        raise PlanError(
            f"arm {name}: ecg_aug arms, and only they, set the "
            f"ecg_aug operations"
        )
    if arm.ecg_aug is not None:
        try:
            ecg_aug.check(arm.ecg_aug)
        except ValueError as error:
            raise PlanError(f"arm {name}: {error}") from error
    if (arm.simvea is not None) != (arm.transform_family == "simvea"):
        raise PlanError(
            f"arm {name}: simvea arms, and only they, set the simvea settings"
        )
    if arm.simvea is not None:
        try:
            simvea.check(arm.simvea)
        except ValueError as error:
            raise PlanError(f"arm {name}: {error}") from error


def _check_shared_envelope(protocol: ProtocolConfig) -> None:
    """Require comparable severity limits for standard augmentation and editing.

    Treatments with an explicit augmentation range are exempt because their
    purpose is to compare different augmentation strengths.
    """
    envelope = protocol.editor.envelope
    pairs = (
        (
            "generic.wander_mv",
            protocol.generic.wander_mv,
            "baseline_drift",
            envelope.baseline_drift.amplitude_mv,
        ),
        (
            "generic.noise_rms_mv",
            protocol.generic.noise_rms_mv,
            "static_noise",
            envelope.static_noise.amplitude_mv,
        ),
    )
    for name, generic_range, kind, editor_range in pairs:
        if generic_range[1] != editor_range[1]:
            raise PlanError(
                f"{name} tops out at {generic_range[1]} mV but the "
                f"editor's {kind} at {editor_range[1]} mV; "
                "A1 and A2/A3 must share one severity envelope"
            )


def _checked_manifest(
    name: str,
    regime: RegimeConfig,
    protocol: ProtocolConfig,
    index: pa.Table,
    source: dict,
) -> Manifest:
    if regime.max_steps is None:
        raise PlanError(
            f"regime {name} has no max_steps; run the pilot and set "
            f"protocol.regimes.{name}.max_steps"
        )
    if regime.max_steps <= 0 or regime.max_steps % protocol.val_every:
        raise PlanError(
            f"regime {name}: max_steps {regime.max_steps} must be a "
            f"positive multiple of val_every "
            f"{protocol.val_every}"
        )
    manifest = load_manifest(Path(regime.manifest))
    check_manifest(manifest, index, source)  # Gate 1
    return manifest


def _editor_calibration(
    arm: str,
    regime_name: str,
    regime: RegimeConfig,
    manifest: Manifest,
    cfg: EditorConfig,
) -> tuple[str, str]:
    """Return the path and checksum of a validated calibration."""
    where = f"arm {arm} in {regime_name}"
    missing = [
        key
        for key in ("calibration", "generator_gate")
        if not getattr(regime, key) or not Path(getattr(regime, key)).exists()
    ]
    if missing:
        raise PlanError(
            f"{where} edits ECGs and needs the regime's "
            f"{' and '.join(missing)}; run the generator "
            "calibration (python -m syntide.generate mode=calibrate) first"
        )
    gate = json.loads(Path(regime.generator_gate).read_text(encoding="utf-8"))
    if gate.get("passed") is not True:
        raise PlanError(
            f"{where}: the generator gate {regime.generator_gate} did not pass"
        )
    digest = hashlib.sha256(Path(regime.calibration).read_bytes()).hexdigest()
    if gate.get("calibration_sha256") != digest:
        raise PlanError(
            f"{where}: {regime.generator_gate} was run on another "
            f"calibration; rerun the gate"
        )
    try:
        calibration = load_calibration(regime.calibration, digest)
    except CalibrationError as error:
        raise PlanError(f"{where}: {error}") from error
    if calibration["manifest_sha256"] != manifest.sha256:
        raise PlanError(
            f"{where}: {regime.calibration} was fit on another "
            f"manifest than {regime.manifest}"
        )
    settings = {
        "envelope": asdict(cfg.envelope),
        "type_smoothing": cfg.type_smoothing,
        "density_floor": cfg.density_floor,
    }
    if {key: calibration[key] for key in settings} != settings:
        raise PlanError(
            f"{where}: {regime.calibration} was fit with other editor "
            f"settings; recalibrate"
        )
    return regime.calibration, digest


def _spec(
    protocol: ProtocolConfig,
    regime_name: str,
    regime: RegimeConfig,
    manifest: Manifest,
    arm_name: str,
    arm: ArmConfig,
    seed: int,
    calibration: tuple[str, str] | None,
) -> TrialSpec:
    _, generator, validators = FAMILIES[arm.transform_family]
    transform = {
        "none": {},
        "generic": asdict(arm.generic or protocol.generic),
        "editor": asdict(protocol.editor),
        "ecg_aug": asdict(arm.ecg_aug) if arm.ecg_aug is not None else {},
        "simvea": asdict(arm.simvea) if arm.simvea is not None else {},
    }
    fraction = (
        protocol.transform_fraction
        if arm.transform_fraction is None
        else arm.transform_fraction
    )
    return make_spec(
        protocol=protocol.id,
        regime=regime_name,
        arm=arm_name,
        replicate_seed=seed,
        source_role=arm.source_role,
        transform_family=arm.transform_family,
        sampling_policy=arm.sampling_policy,
        generator=generator,
        validators=validators,
        transform=transform[arm.transform_family],
        dataset=protocol.dataset,
        source_sha256=manifest.source["sha256"],
        manifest_sha256=manifest.sha256,
        calibration_sha256=calibration[1] if calibration else None,
        batch_size=protocol.batch_size,
        max_steps=regime.max_steps,
        val_every=protocol.val_every,
        # Augmentation treatments share transformation positions; real-data
        # controls have none.
        transform_fraction=fraction if arm.source_role == "depleted" else 0.0,
        registry=protocol.registry,
        manifest=regime.manifest,
        calibration=calibration[0] if calibration else None,
    )
