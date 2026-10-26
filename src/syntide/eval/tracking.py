"""Record experiment progress to files or a training dashboard.

The configuration in ``conf/tracker/`` selects the dashboard destination.
Predictions, ``result.json``, and the recipe ledger are saved with every
destination. Dashboard plots use optimizer updates as their horizontal axis.

Backends import their libraries lazily, so choosing ``files`` needs neither
torch nor wandb.
"""

import html
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from omegaconf import OmegaConf

from syntide.config import load_dataclass

BACKENDS = ("files", "tensorboard", "wandb")


@dataclass
class TrackerConfig:
    """Configuration for recording training and evaluation progress."""

    backend: str = "files"
    project: str = "syntide"  # W&B project
    entity: str | None = (
        None  # W&B entity (user or team); None uses the logged-in default
    )
    mode: str = "online"  # W&B mode: online | offline | disabled

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrackerConfig":
        """Load ``conf/tracker/<backend>.yaml`` without Hydra (for trainers)."""
        return load_dataclass(cls, OmegaConf.load(path))


class Tracker(Protocol):
    def log_scalars(self, values: Mapping[str, float], step: int) -> None: ...
    def log_figure(self, name: str, fig: Any, step: int) -> None: ...
    def log_text(self, name: str, markdown: str, step: int) -> None: ...
    def log_summary(
        self, hparams: Mapping[str, Any], metrics: Mapping[str, float]
    ) -> None: ...
    def close(self) -> None: ...


def make_tracker(
    cfg: TrackerConfig,
    log_dir: str | Path,
    run_name: str,
    group: str,
    config: Mapping[str, Any],
) -> Tracker:
    """Open a progress tracker for a named run and group."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if cfg.backend == "files":
        return FilesTracker(log_dir)
    if cfg.backend == "tensorboard":
        return TensorBoardTracker(log_dir)
    if cfg.backend == "wandb":
        return WandbTracker(cfg, log_dir, run_name, group, config)
    raise ValueError(
        f"unknown tracker backend {cfg.backend!r}; choose one of "
        f"{', '.join(BACKENDS)}"
    )


def _file_name(name: str) -> str:
    return name.replace("/", "-")


class FilesTracker:
    """Save metrics, figures, text, and a summary to local files.

    Each new run starts a fresh metrics.jsonl. Figures and text use separate
    subdirectories; summary.json holds the final run summary.
    """

    def __init__(self, log_dir: Path) -> None:
        self.dir = log_dir
        self._metrics = (log_dir / "metrics.jsonl").open("w", encoding="utf-8")

    def log_scalars(self, values: Mapping[str, float], step: int) -> None:
        line = {
            "step": int(step),
            **{key: float(value) for key, value in values.items()},
        }
        self._metrics.write(json.dumps(line) + "\n")
        self._metrics.flush()

    def log_figure(self, name: str, fig: Any, step: int) -> None:
        path = self.dir / "figures" / f"{_file_name(name)}_step{step}.png"
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    def log_text(self, name: str, markdown: str, step: int) -> None:
        path = self.dir / "text" / f"{_file_name(name)}_step{step}.md"
        path.parent.mkdir(exist_ok=True)
        path.write_text(markdown, encoding="utf-8")

    def log_summary(
        self, hparams: Mapping[str, Any], metrics: Mapping[str, float]
    ) -> None:
        summary = {"hparams": dict(hparams), "metrics": dict(metrics)}
        (self.dir / "summary.json").write_text(
            json.dumps(summary, indent=1, default=str) + "\n"
        )

    def close(self) -> None:
        self._metrics.close()


class TensorBoardTracker:
    """Write event files for viewing in TensorBoard."""

    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalars(self, values: Mapping[str, float], step: int) -> None:
        for key, value in values.items():
            self._writer.add_scalar(key, float(value), int(step))

    def log_figure(self, name: str, fig: Any, step: int) -> None:
        self._writer.add_figure(name, fig, int(step), close=False)

    def log_text(self, name: str, markdown: str, step: int) -> None:
        self._writer.add_text(name, markdown, int(step))

    def log_summary(
        self, hparams: Mapping[str, Any], metrics: Mapping[str, float]
    ) -> None:
        # TensorBoard's hparams table takes only scalars and strings.
        # run_name="." keeps the summary in
        # this run's directory instead of a new timestamped one.
        plain = {
            k: v if isinstance(v, int | float | str | bool) else str(v)
            for k, v in hparams.items()
        }
        self._writer.add_hparams(
            plain, {k: float(v) for k, v in metrics.items()}, run_name="."
        )

    def close(self) -> None:
        self._writer.close()


class WandbTracker:
    """Record one Weights & Biases run per trial or evaluation split.

    Logged step numbers must increase within each run.
    """

    def __init__(
        self,
        cfg: TrackerConfig,
        log_dir: Path,
        run_name: str,
        group: str,
        config: Mapping[str, Any],
    ) -> None:
        mode = cfg.mode
        if mode not in ("online", "offline", "disabled"):
            raise ValueError(
                f"unknown W&B mode {mode!r}; choose online, offline or disabled"
            )
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=cfg.project,
            entity=cfg.entity,
            group=group,
            name=run_name,
            dir=str(log_dir),
            config=dict(config),
            mode=mode,
        )

    def log_scalars(self, values: Mapping[str, float], step: int) -> None:
        self._run.log(
            {key: float(value) for key, value in values.items()}, step=int(step)
        )

    def log_figure(self, name: str, fig: Any, step: int) -> None:
        self._run.log({name: self._wandb.Image(fig)}, step=int(step))

    def log_text(self, name: str, markdown: str, step: int) -> None:
        self._run.log(
            {name: self._wandb.Html(f"<pre>{html.escape(markdown)}</pre>")},
            step=int(step),
        )

    def log_summary(
        self, hparams: Mapping[str, Any], metrics: Mapping[str, float]
    ) -> None:
        self._run.config.update(dict(hparams), allow_val_change=True)
        self._run.summary.update(dict(metrics))

    def close(self) -> None:
        self._run.finish()
