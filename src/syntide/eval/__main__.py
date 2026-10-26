"""Evaluate a protocol using rules fixed before test access.

Run with ``python -m syntide.eval`` and select the same protocol used for
planning. ``mode=audit`` inspects generated recordings. ``mode=freeze``
saves analysis rules before test evaluation. ``split=val`` reports available
validation results during training; ``checkpoint=final`` selects final model
parameters instead of the checkpoint chosen on validation.

``split=test`` verifies the frozen rules and evaluates completed trials
once. Reports go to ``runs/<protocol>/eval/<split>[-final]/`` and contain
comparisons, figures, a written summary, and dashboard records.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from syntide.config import as_dataclass
from syntide.eval.analysis import AnalysisConfig, RulesError, check_rules
from syntide.eval.freeze import FreezeError, freeze_rules, verify_frozen
from syntide.eval.tracking import TrackerConfig
from syntide.generate.plan import ProtocolConfig

log = logging.getLogger(__name__)
MODES = ("evaluate", "freeze", "audit")
SPLITS = ("val", "test")
CHECKPOINTS = ("best", "final")


@dataclass
class EvaluateConfig:
    mode: str = "evaluate"
    split: str = "val"
    checkpoint: str = "best"
    runs_dir: str = "runs"
    registry: str | None = None
    # where mode=audit runs A0's predictor; cpu leaves the GPU to training
    audit_device: str = "cpu"
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)


ConfigStore.instance().store(name="evaluate_schema", node=EvaluateConfig)


@hydra.main(
    version_base=None, config_path="../../../conf", config_name="evaluate"
)
def main(cfg: DictConfig) -> None:
    config = as_dataclass(EvaluateConfig, cfg)
    for name, value, choices in (
        ("mode", config.mode, MODES),
        ("split", config.split, SPLITS),
        ("checkpoint", config.checkpoint, CHECKPOINTS),
    ):
        if value not in choices:
            raise SystemExit(
                f"unknown {name} {value!r}; choose one of {', '.join(choices)}"
            )
    if config.split == "test" and config.checkpoint != "best":
        raise SystemExit(
            "the test fold is scored on the best checkpoint only; use "
            "checkpoint=final with split=val"
        )
    protocol_dir = Path(config.runs_dir) / config.protocol.id
    try:
        check_rules(config.analysis, config.protocol)
        if config.mode == "freeze":
            path, written = freeze_rules(protocol_dir, config.analysis)
            log.info(
                "%s %s (rules %s)",
                "froze" if written else "already frozen:",
                path,
                config.analysis.sha256[:12],
            )
            return
        if config.split == "test":
            verify_frozen(protocol_dir, config.analysis, claimed=False)
            from syntide.eval.holdout import HoldoutError, evaluate

            try:
                evaluate(protocol_dir, registry=config.registry)
            except HoldoutError as error:
                raise SystemExit(f"test fold not scored: {error}") from error
            verify_frozen(protocol_dir, config.analysis, claimed=True)
    except (RulesError, FreezeError) as error:
        raise SystemExit(str(error)) from error

    if config.mode == "audit":
        from syntide.eval.audit import write_audit

        out_dir = write_audit(
            protocol_dir,
            config.protocol,
            config.analysis,
            config.tracker,
            device=config.audit_device,
            registry=config.registry,
        )
        log.info("wrote %s", out_dir / "audit.md")
        return

    from syntide.eval.contrasts import PairingError
    from syntide.eval.report import write_report
    from syntide.eval.runs import RunError

    try:
        out_dir = write_report(
            protocol_dir,
            config.protocol,
            config.analysis,
            config.split,
            config.checkpoint,
            config.tracker,
        )
    except (RunError, PairingError) as error:
        raise SystemExit(f"no report: {error}") from error
    log.info("wrote %s", out_dir / "results.md")


if __name__ == "__main__":
    main()
