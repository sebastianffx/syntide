"""The decision rules of one protocol, and the analysis they drive.

The rules live in ``conf/analysis/<protocol>.yaml`` and are frozen (``mode=freeze``) before the test
fold is scored. An arm is named within the rules' primary ``regime`` (``A3``), or as
``<arm>@<regime>`` for another regime (``C2@full100``).

Rules follow docs/SYNTIDE_MVE.md as repaired in docs/timenet_syntide.md §3:
- the primary contrast is read on the artifact slice; a **benefit** needs the artifact CI above 0 *and*
  clean-slice non-inferiority (clean CI low above ``noninferiority_margin``);
- the ``practical_target`` is reported separately as supported, plausible or ruled out;
- the fixed sequence (primary, then ``sequence``, in each of ``sequence_regimes``) stops at the first
  step without a benefit, and later steps are "not tested";
- "gap closed" is descriptive: numerator and denominator are reported with their CIs, and the ratio only
  when its denominator's CI is above 0 (and, for a ``gated`` entry, the gap gate passed on validation).
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from omegaconf import MISSING

from syntide.eval import contrasts
from syntide.eval.contrasts import SLICES, PairedScorer, PairingError, UtilityReport
from syntide.eval.runs import Run
from syntide.generate.plan import ProtocolConfig, regime_arms


class RulesError(ValueError):
    """The decision rules name an arm or regime the protocol does not run, or are malformed."""


@dataclass
class GapGateRule:
    """Gate 3: natural artifact coverage at matched N must help real-only training on validation."""

    contrast: list[str] = field(default_factory=lambda: ["C1", "A0"])
    min_effect: float = 0.02  # inclusive, on the validation artifact slice


@dataclass
class GapClosedRule:
    name: str = MISSING  # e.g. "natural mix"
    target: str = MISSING  # the arm whose gain over the baseline counts as 100%
    baseline: str = "A0"
    gated: bool = False  # also require the gap gate to have passed on validation


@dataclass
class AnalysisConfig:
    """Schema of ``conf/analysis/*.yaml``."""

    regime: str = MISSING  # the primary regime; unqualified arms are looked up here
    reference: str = "A1"  # results-table deltas are against this arm
    primary: list[str] = field(default_factory=lambda: ["A3", "A1"])
    sequence: list[list[str]] = field(default_factory=lambda: [["A3", "A2"], ["A2", "A1"]])
    sequence_regimes: list[str] = field(default_factory=list)  # empty = [regime]
    diagnostics: list[list[str]] = field(default_factory=list)  # reported, never evidence
    practical_target: float = 0.02
    noninferiority_margin: float = -0.01
    gap_gate: GapGateRule = field(default_factory=GapGateRule)
    gap_closed: list[GapClosedRule] = field(default_factory=list)
    n_boot: int = 2000
    ci_level: float = 0.95
    bootstrap_seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def chain_regimes(self) -> list[str]:
        return list(self.sequence_regimes) or [self.regime]


def arm_key(name: str, regime: str) -> tuple[str, str]:
    """``"A3"`` -> (regime, "A3"); ``"C2@full100"`` -> ("full100", "C2")."""
    arm, _, qualified = name.partition("@")
    return (qualified or regime, arm)


def arm_label(key: tuple[str, str], regime: str) -> str:
    """The inverse of ``arm_key``: unqualified within ``regime``."""
    return key[1] if key[0] == regime else f"{key[1]}@{key[0]}"


@dataclass(frozen=True)
class ContrastSpec:
    role: str  # primary | sequence | diagnostic | gap_gate | gap_closed | reference
    a: tuple[str, str]  # (regime, arm)
    b: tuple[str, str]

    def label(self, regime: str) -> str:
        return f"{arm_label(self.a, regime)} - {arm_label(self.b, regime)}"


def chain(rules: AnalysisConfig) -> list[ContrastSpec]:
    """The fixed sequence: the primary contrast, then ``sequence``, in each chain regime in turn."""
    steps = []
    for regime in rules.chain_regimes:
        for position, (a, b) in enumerate([rules.primary, *rules.sequence]):
            role = "primary" if position == 0 and regime == rules.regime else "sequence"
            steps.append(ContrastSpec(role, arm_key(a, regime), arm_key(b, regime)))
    return steps


def diagnostics(rules: AnalysisConfig) -> list[ContrastSpec]:
    return [
        ContrastSpec("diagnostic", arm_key(a, rules.regime), arm_key(b, rules.regime))
        for a, b in rules.diagnostics
    ]


def gap_gate(rules: AnalysisConfig) -> ContrastSpec:
    a, b = rules.gap_gate.contrast
    return ContrastSpec("gap_gate", arm_key(a, rules.regime), arm_key(b, rules.regime))


def check_rules(rules: AnalysisConfig, protocol: ProtocolConfig) -> None:
    """Refuse rules that name an arm a regime does not run, so a typo is caught before freezing."""
    if rules.regime not in protocol.regimes:
        raise RulesError(
            f"analysis regime {rules.regime!r} is not a regime of {protocol.id}: {list(protocol.regimes)}"
        )
    for regime in rules.chain_regimes:
        if regime not in protocol.regimes:
            raise RulesError(f"sequence regime {regime!r} is not a regime of {protocol.id}")
    pairs = [rules.primary, *rules.sequence, *rules.diagnostics, rules.gap_gate.contrast]
    if any(len(pair) != 2 for pair in pairs):
        raise RulesError(f"every contrast names exactly two arms; got {[p for p in pairs if len(p) != 2]}")
    if not 0 < rules.ci_level < 1 or rules.n_boot < 100:
        raise RulesError("ci_level must be in (0, 1) and n_boot at least 100")
    keys = [spec.a for spec in [*chain(rules), *diagnostics(rules), gap_gate(rules)]]
    keys += [spec.b for spec in [*chain(rules), *diagnostics(rules), gap_gate(rules)]]
    keys += [arm_key(rules.reference, rules.regime)]
    for entry in rules.gap_closed:
        keys += [arm_key(entry.target, rules.regime), arm_key(entry.baseline, rules.regime)]
    for regime, arm in keys:
        if regime not in protocol.regimes:
            raise RulesError(f"{arm}@{regime}: {protocol.id} has no regime {regime!r}")
        if arm not in regime_arms(protocol, regime):
            raise RulesError(f"{arm}@{regime}: regime {regime} does not run arm {arm}")


EXPOSURE_FIELDS = (
    "presentations",
    "transform_slots",
    "candidates",
    "accepted",
    "exhausted_slots",
    "distinct_recipes",
)


class Analysis:
    """Every contrast the rules ask for, on one split and checkpoint, with shared bootstrap draws."""

    def __init__(self, runs: Sequence[Run], rules: AnalysisConfig, *, checkpoint: str) -> None:
        if not runs:
            raise ValueError("no finished run to analyze")
        self.rules, self.checkpoint, self.split = rules, checkpoint, runs[0].split
        self.groups: dict[tuple[str, str], dict[int, Run]] = defaultdict(dict)
        for run in runs:
            self.groups[run.key][run.seed] = run
        self.scorer = PairedScorer(
            runs[0], checkpoint=checkpoint, n_boot=rules.n_boot, seed=rules.bootstrap_seed
        )
        self._cache: dict[tuple, UtilityReport | None] = {}

    def label(self, key: tuple[str, str]) -> str:
        return arm_label(key, self.rules.regime)

    def contrast(self, a: tuple[str, str], b: tuple[str, str], slice_name: str) -> UtilityReport | None:
        """``a - b`` over the seeds both arms finished; None if either arm has no finished run."""
        cache = (a, b, slice_name)
        if cache not in self._cache:
            seeds = sorted(set(self.groups.get(a, {})) & set(self.groups.get(b, {})))
            self._cache[cache] = (
                self.scorer.contrast(
                    [self.groups[a][seed] for seed in seeds],
                    [self.groups[b][seed] for seed in seeds],
                    slice_name,
                    ci_level=self.rules.ci_level,
                    labels=(self.label(a), self.label(b)),
                )
                if seeds
                else None
            )
        return self._cache[cache]

    def judged(self, spec: ContrastSpec) -> dict[str, Any]:
        """A contrast on every slice, with the findings flags; ``missing`` names arms without runs."""
        missing = [self.label(key) for key in (spec.a, spec.b) if not self.groups.get(key)]
        entry: dict[str, Any] = {
            "role": spec.role,
            "label": spec.label(self.rules.regime),
            "missing": missing,
        }
        if missing:
            return entry
        reports = {name: self.contrast(spec.a, spec.b, name) for name in SLICES}
        unpaired = sorted(set(self.groups[spec.a]) ^ set(self.groups[spec.b]))
        entry.update({name: report.to_dict() for name, report in reports.items()})
        entry["unpaired_seeds"] = unpaired
        entry["findings"] = contrasts.findings(
            reports["artifact"],
            reports["clean"],
            practical_target=self.rules.practical_target,
            noninferiority_margin=self.rules.noninferiority_margin,
        )
        return entry


def gap_gate_outcome(val_runs: Sequence[Run], rules: AnalysisConfig) -> dict[str, Any]:
    """Gate 3 on validation (best checkpoint): ``contrast`` estimate on the artifact slice >= min_effect."""
    spec = gap_gate(rules)
    outcome: dict[str, Any] = {"label": spec.label(rules.regime), "min_effect": rules.gap_gate.min_effect}
    analysis = Analysis(val_runs, rules, checkpoint="best") if val_runs else None
    report = analysis.contrast(spec.a, spec.b, "artifact") if analysis else None
    if report is None:
        return {**outcome, "passed": None, "status": "not run"}
    passed = report.estimate >= rules.gap_gate.min_effect
    return {
        **outcome,
        "passed": passed,
        "status": "passed" if passed else "failed",
        "artifact": report.to_dict(),
    }


def analyze(
    runs: Sequence[Run],
    protocol: ProtocolConfig,
    rules: AnalysisConfig,
    *,
    checkpoint: str,
    gate: dict[str, Any],
) -> dict[str, Any]:
    """The full comparison of one split: the one results table, the fixed sequence, diagnostics and gap
    closed. ``gate`` is the gap gate on validation (``gap_gate_outcome``), whatever the split."""
    analysis = Analysis(runs, rules, checkpoint=checkpoint)
    steps = [analysis.judged(spec) for spec in chain(rules)]
    statuses = contrasts.fixed_sequence([None if s["missing"] else s["findings"]["benefit"] for s in steps])
    for step, status in zip(steps, statuses, strict=True):
        step["status"] = status

    primary_arm = arm_key(rules.primary[0], rules.regime)
    gap_entries = []
    for entry in rules.gap_closed:
        baseline, target = arm_key(entry.baseline, rules.regime), arm_key(entry.target, rules.regime)
        numerator = analysis.contrast(primary_arm, baseline, "artifact")
        denominator = analysis.contrast(target, baseline, "artifact")
        outcome = {"name": entry.name, "gated": entry.gated}
        if numerator is None or denominator is None:
            outcome["withheld_because"] = "an arm has no finished run"
        else:
            outcome.update(
                contrasts.gap_closed(numerator, denominator, gated=entry.gated, gate_passed=gate["passed"])
            )
        gap_entries.append(outcome)

    return {
        "split": analysis.split,
        "checkpoint": checkpoint,
        "rules_sha256": rules.sha256,
        "gap_gate": gate,
        "chain": steps,
        "diagnostics": [analysis.judged(spec) for spec in diagnostics(rules)],
        "gap_closed": gap_entries,
        "table": _table(analysis, protocol, gate),
        "runs": [_run_summary(analysis, run) for group in analysis.groups.values() for run in group.values()],
    }


def _table(analysis: Analysis, protocol: ProtocolConfig, gate: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (regime, arm) with finished runs: scores over seeds, delta vs the reference, gap closed."""
    rules, rows = analysis.rules, []
    keys = [(regime, arm) for regime in protocol.regimes for arm in regime_arms(protocol, regime)]
    for key in keys:
        group = analysis.groups.get(key)
        if not group:
            continue
        seeds = sorted(group)
        row: dict[str, Any] = {"regime": key[0], "arm": key[1], "label": analysis.label(key), "seeds": seeds}
        for name in SLICES:
            values = [analysis.scorer.point(group[seed], name) for seed in seeds]
            row[name] = {"mean": float(np.mean(values)), "sd": _sd(values), "per_seed": values}
        row["best_step"] = (
            [group[seed].steps["best"] for seed in seeds] if analysis.checkpoint == "best" else None
        )
        row["exposure"] = {
            field: float(np.mean([group[seed].exposure[field] for seed in seeds]))
            for field in EXPOSURE_FIELDS
        }
        row["wall_time_s"] = float(np.mean([group[seed].result["wall_time_s"] for seed in seeds]))
        row["delta"] = _delta(analysis, key)
        row["gap_closed"] = {}
        for entry in rules.gap_closed:
            baseline, target = arm_key(entry.baseline, rules.regime), arm_key(entry.target, rules.regime)
            row["gap_closed"][entry.name] = _ratio(analysis, key, baseline, target, entry.gated, gate)
        rows.append(row)
    return rows


def _delta(analysis: Analysis, key: tuple[str, str]) -> dict[str, Any] | None:
    """A row's artifact and clean delta against the reference arm of its regime, else the primary one."""
    rules = analysis.rules
    reference = (key[0], rules.reference)
    if not analysis.groups.get(reference):
        reference = arm_key(rules.reference, rules.regime)
    if key == reference or not analysis.groups.get(reference):
        return None
    try:
        reports = {name: analysis.contrast(key, reference, name) for name in ("artifact", "clean")}
    except PairingError as error:
        return {"against": analysis.label(reference), "unpaired": str(error)}
    return {"against": analysis.label(reference), **{name: r.to_dict() for name, r in reports.items()}}


def _ratio(analysis, key, baseline, target, gated: bool, gate: dict[str, Any]) -> float | None:
    if gated and not gate["passed"]:
        return None
    try:
        numerator = analysis.contrast(key, baseline, "artifact")
        denominator = analysis.contrast(target, baseline, "artifact")
    except PairingError:
        return None
    if numerator is None or denominator is None or denominator.ci_low <= 0:
        return None
    return numerator.estimate / denominator.estimate


def _run_summary(analysis: Analysis, run: Run) -> dict[str, Any]:
    return {
        "run_dir": str(run.run_dir),
        "trial_id": run.spec.trial_id,
        "regime": run.spec.regime,
        "arm": run.spec.arm,
        "seed": run.seed,
        "step": run.steps[analysis.checkpoint],
        **{f"auroc_{name}": analysis.scorer.point(run, name) for name in SLICES},
        "wall_time_s": run.result["wall_time_s"],
    }


def _sd(values: Sequence[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None
