"""``python -m syntide.generate``: calibrate the generator (``mode=calibrate``) or plan trials.

``mode=calibrate`` (the default) runs per regime, before the editor arms are planned:

1. fit the calibration and gap report on the regime's ``retained_artifact`` records only;
2. run the generator gate (Gate 4) on 1,000 recipes per policy over ``depleted`` parents;
3. write the calibration and gate JSON (frozen once written), the gate's candidates
   (``<gate>/recipes.parquet``) and a review sample (``<gate>/review_sample.csv``, never overwritten:
   it may hold an expert's verdicts).

``mode=plan`` writes one frozen ``trial.json`` per selected (regime, arm, seed) under
``<runs_dir>/<protocol>/<regime>/<arm>/seed<n>/``. Narrow it with, e.g.,
``select.arms=[A0,C1] select.regimes=[scarcity25] select.seeds=[0,1,2]``, and set a budget from the
command line with ``protocol.regimes.scarcity25.max_steps=20000``. Train the planned trials with any
trainer, e.g. ``python -m syntide_trainers.torch_loop "runs/ptbxl-artifact-v2/*/*/seed*"``.
"""

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import hydra
import pyarrow.parquet as pq
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from syntide.data.challenge import Manifest, load_manifest
from syntide.data.leakage import check_manifest, check_recipe_parents
from syntide.data.schema import FS_HZ, Samples
from syntide.generate.editor import EditorConfig
from syntide.generate.gate import RECIPES_PER_POLICY, review_sample, run_gate, write_review_sample
from syntide.generate.plan import (
    PlanError,
    ProtocolConfig,
    RegimeConfig,
    Selection,
    plan_trials,
    regime_arms,
    write_plan,
)
from syntide.generate.policy import fit_calibration, save_calibration, save_frozen_json

log = logging.getLogger(__name__)
MODES = ("plan", "calibrate")


@dataclass
class GenerateConfig:
    mode: str = "calibrate"
    runs_dir: str = "runs"
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    select: Selection = field(default_factory=Selection)


ConfigStore.instance().store(name="generate_schema", node=GenerateConfig)


@hydra.main(version_base=None, config_path="../../../conf", config_name="generate")
def main(cfg: DictConfig) -> None:
    config: GenerateConfig = OmegaConf.to_object(cfg)
    if config.mode not in MODES:
        raise SystemExit(f"unknown mode {config.mode!r}; choose one of {', '.join(MODES)}")

    from syntide.data.adapter import PtbXlAdapter

    adapter = PtbXlAdapter(config.protocol.registry, config.protocol.dataset)
    if config.mode == "calibrate":
        index, source = adapter.index(), adapter.source()
        protocol = config.protocol
        # By default, calibrate only the regimes that run an editor arm; the others need no generator.
        regimes = config.select.regimes or [
            name
            for name in protocol.regimes
            if any(protocol.arms[arm].transform_family == "editor" for arm in regime_arms(protocol, name))
        ]
        for name in regimes:
            regime = config.protocol.regimes[name]
            manifest = load_manifest(Path(regime.manifest))
            check_manifest(manifest, index, source)  # Gate 1
            retained = adapter.load(manifest.roles["retained_artifact"])
            depleted = adapter.load(manifest.roles["depleted"])
            report = calibrate_regime(manifest, retained, depleted, config.protocol.editor, regime)
            _log_gate(name, report)
        return
    try:
        specs = plan_trials(config.protocol, config.select, index=adapter.index(), source=adapter.source())
        written = write_plan(specs, config.runs_dir)
    except PlanError as error:
        raise SystemExit(f"nothing was planned: {error}") from error
    log.info(
        "%d trials planned under %s/%s (%d new, %d already there)",
        len(specs),
        config.runs_dir,
        config.protocol.id,
        len(written),
        len(specs) - len(written),
    )
    for spec in specs:
        log.info("  %-28s max_steps=%d  trial %s", spec.run_name, spec.max_steps, spec.trial_id[:12])


def calibrate_regime(
    manifest: Manifest,
    retained: Samples,
    depleted: Samples,
    cfg: EditorConfig,
    regime: RegimeConfig,
    *,
    recipes: int = RECIPES_PER_POLICY,
) -> dict[str, Any]:
    """Calibration, Gate 4 and the review sample for one regime. Returns the gate report."""
    if not regime.calibration or not regime.generator_gate:
        raise SystemExit("the regime needs calibration and generator_gate paths in the protocol")
    check_recipe_parents(depleted.record_id, manifest)
    calibration = fit_calibration(manifest, retained, cfg, FS_HZ)
    calibration_sha256 = save_calibration(calibration, regime.calibration)
    report, candidates = run_gate(
        depleted, calibration, calibration_sha256, asdict(cfg), FS_HZ, recipes=recipes
    )
    save_frozen_json(regime.generator_gate, report, "generator gate result")
    gate_dir = Path(regime.generator_gate).with_suffix("")
    gate_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(candidates, gate_dir / "recipes.parquet", compression="zstd")
    if not write_review_sample(
        review_sample(candidates, calibration["envelope"]), gate_dir / "review_sample.csv"
    ):
        log.info("%s already exists and was kept (it may hold verdicts)", gate_dir / "review_sample.csv")
    log.info(
        "calibration %s (%s) -> %s", calibration["gap_report_id"], manifest.challenge_id, regime.calibration
    )
    for kind, fitted in calibration["types"].items():
        log.info(
            "  %-15s %3d retained records (%.0f%%)", kind, fitted["n_retained"], 100 * fitted["frequency"]
        )
    return report


def _log_gate(regime: str, report: dict[str, Any]) -> None:
    log.info("Gate 4 for %s: %s", regime, "passed" if report["passed"] else "FAILED")
    for policy, result in report["policies"].items():
        log.info(
            "  %-16s acceptance %.3f  types %s  rejections %s",
            policy,
            result["acceptance_rate"],
            result["artifact_types"],
            result["rejections"] or "none",
        )
        for control, cell in result["controls"].items():
            if cell.get("slope") is None:
                log.info("    %-28s n=%d  not judged (%s)", control, cell["n"], cell.get("note"))
                continue
            log.info(
                "    %-28s n=%4d  slope %.3f  intercept %+.3f  mae %.3f  corr %.3f  %s",
                control,
                cell["n"],
                cell["slope"],
                cell["intercept_of_range"],
                cell["mae_of_range"],
                cell["corr"],
                "ok" if cell["passed"] else "FAIL",
            )


if __name__ == "__main__":
    main()
