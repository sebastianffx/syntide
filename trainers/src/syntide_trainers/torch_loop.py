"""The plain PyTorch reference trainer for SYNTIDE trials.

Run ``python -m syntide_trainers.torch_loop`` with a quoted trial-directory
pattern. Optional arguments select the tracker configuration and override
model or device settings.

For each planned run directory it skips finished trials, then trains a 1-D
ResNet with AdamW and binary cross-entropy on exactly the batches SYNTIDE
serves. Every ``val_every`` steps it scores the validation set with
SYNTIDE's ``val_scores`` and keeps ``best.pt`` on the selection metric; it
saves ``final.pt`` at the last step and hands everything to
``Trial.finish``. Hydra is not used: the config is a YAML file plus
``key=value`` overrides.
"""

import argparse
import glob
import hashlib
import itertools
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from syntide.data.leakage import LeakageError
from syntide.eval.stats import SELECTION_METRIC, val_scores
from syntide.eval.tracking import TrackerConfig
from syntide.trial import (
    Checkpoint,
    TrainerInfo,
    Trial,
    TrialError,
    is_complete,
    prepare,
)
from syntide_trainers import __version__

log = logging.getLogger(__name__)

NAME = "syntide_trainers.torch_loop"
PREDICTOR = f"{NAME}:predict"
# fixed, so predict() reproduces the probabilities computed during training
PREDICT_BATCH = 512
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "conf" / "torch_loop.yaml"
)


@dataclass
class ModelConfig:
    width: int = 64
    kernel_size: int = 7


@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-2


@dataclass
class TorchLoopConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    device: str = "auto"
    log_every: int = 50
    tracker: str = "conf/tracker/tensorboard.yaml"
    registry: str | None = None


def load_config(
    path: str | Path | None = DEFAULT_CONFIG, overrides: Sequence[str] = ()
) -> TorchLoopConfig:
    """Apply a configuration file and explicit key=value overrides."""
    merged = OmegaConf.structured(TorchLoopConfig)
    if path is not None:
        merged = OmegaConf.merge(merged, OmegaConf.load(path))
    return OmegaConf.to_object(
        OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
    )


class _Block(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, stride: int, kernel_size: int
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(
            c_in, c_out, kernel_size, stride, pad, bias=False
        )
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.skip = (
            nn.Identity()
            if stride == 1 and c_in == c_out
            else nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm1d(c_out),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(h)) + self.skip(x))


class ResNet1d(nn.Module):
    """A one-dimensional residual network for 12-lead electrocardiograms.

    Four residual stages use widths w, w, 2w, and 3w. The final three halve
    the time axis before global average pooling. The default width of 64
    gives about 950,000 parameters.
    """

    def __init__(
        self,
        width: int = 64,
        kernel_size: int = 7,
        n_leads: int = 12,
        n_classes: int = 5,
    ) -> None:
        super().__init__()
        widths = [width, width, 2 * width, 2 * width, 3 * width]
        layers: list[nn.Module] = [
            nn.Conv1d(n_leads, widths[0], 15, 2, 7, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(),
        ]
        for stage, (c_in, c_out) in enumerate(itertools.pairwise(widths)):
            layers.append(
                _Block(c_in, c_out, 1 if stage == 0 else 2, kernel_size)
            )
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(widths[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).mean(dim=-1))


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def trainer_info(cfg: TorchLoopConfig) -> TrainerInfo:
    """Record all configuration settings that can affect the model."""
    return TrainerInfo(
        name=NAME,
        version=__version__,
        config={"model": asdict(cfg.model), "optim": asdict(cfg.optim)},
        device=resolve_device(cfg.device),
        predictor=PREDICTOR,
    )


def state_sha256(model: nn.Module) -> str:
    """Hash model parameters and buffers to check matched initialization."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _probs(model: nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(x), PREDICT_BATCH):
        batch = np.ascontiguousarray(
            x[start : start + PREDICT_BATCH], dtype=np.float32
        )
        chunks.append(
            torch.sigmoid(model(torch.from_numpy(batch).to(device)))
            .cpu()
            .numpy()
        )
    return np.concatenate(chunks).astype(np.float64)


def predict(
    checkpoint: Checkpoint | str | Path, x: np.ndarray, device: str
) -> np.ndarray:
    """Load a checkpoint and return five class probabilities per recording."""
    path = checkpoint.path if isinstance(checkpoint, Checkpoint) else checkpoint
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = ResNet1d(**state["model_config"])
    model.load_state_dict(state["model"])
    return _probs(model.to(device), x, device)


def _save(
    model: nn.Module, cfg: TorchLoopConfig, path: Path, step: int, trial_id: str
) -> Checkpoint:
    partial = path.with_name(path.name + ".partial")
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(cfg.model),
            "step": step,
            "trial_id": trial_id,
        },
        partial,
    )
    os.replace(partial, path)
    return Checkpoint(path=path, step=step)


def fit(trial: Trial, cfg: TorchLoopConfig, info: TrainerInfo) -> dict:
    """Train for the full budget and submit results to the trial."""
    spec, device, started = trial.spec, info.device, time.monotonic()
    torch.manual_seed(
        spec.init_seed
    )  # the same initial weights for every arm of a (regime, seed) block
    model = ResNet1d(**asdict(cfg.model))
    init_sha256 = state_sha256(model)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()

    history: list[dict] = []
    best: Checkpoint | None = None
    best_score, best_probs, probs = -np.inf, None, None
    window_loss, window_steps = torch.zeros((), device=device), 0
    model.train()
    for batch in trial.batches():
        x = torch.from_numpy(batch.x).to(device)
        y = torch.from_numpy(batch.y).to(device)
        loss = loss_fn(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        window_loss += loss.detach()
        window_steps += 1
        if batch.step % cfg.log_every == 0:
            trial.tracker.log_scalars(
                {"train/loss": (window_loss / window_steps).item()}, batch.step
            )
            window_loss, window_steps = torch.zeros((), device=device), 0
        if batch.step % spec.val_every == 0:
            probs = _probs(model, trial.val.x, device)
            scores = val_scores(trial.val, probs)
            history.append({"step": batch.step, **scores})
            trial.tracker.log_scalars(scores, batch.step)
            if scores[SELECTION_METRIC] > best_score:
                best_score, best_probs = scores[SELECTION_METRIC], probs
                best = _save(
                    model,
                    cfg,
                    trial.run_dir / "best.pt",
                    batch.step,
                    spec.trial_id,
                )
            model.train()

    final = _save(
        model, cfg, trial.run_dir / "final.pt", spec.max_steps, spec.trial_id
    )
    return trial.finish(
        info,
        best=best,
        final=final,
        val_history=history,
        val_probs={"best": best_probs, "final": probs},
        init_sha256=init_sha256,
        wall_time_s=time.monotonic() - started,
    )


def train(
    run_dir: str | Path, cfg: TorchLoopConfig, tracker: TrackerConfig
) -> dict | None:
    """Train a planned trial, or skip it when already completed."""
    info = trainer_info(cfg)
    if is_complete(run_dir, info):
        return None
    with prepare(run_dir, tracker=tracker, registry=cfg.registry) as trial:
        return fit(trial, cfg, info)


def expand_run_dirs(patterns: Sequence[str]) -> list[Path]:
    run_dirs: list[Path] = []
    for pattern in patterns:
        matches = (
            sorted(glob.glob(pattern))
            if any(c in pattern for c in "*?[")
            else [pattern]
        )
        found = [Path(m) for m in matches if (Path(m) / "trial.json").is_file()]
        if not found:
            raise SystemExit(
                f"{pattern} matches no run directory holding a trial.json"
            )
        run_dirs.extend(found)
    return list(dict.fromkeys(run_dirs))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m syntide_trainers.torch_loop",
        description=(
            "Train planned SYNTIDE trials with the plain PyTorch "
            "reference trainer."
        ),
    )
    parser.add_argument(
        "targets",
        nargs="+",
        metavar="RUN_DIR | KEY=VALUE",
        help=(
            "run directories or globs holding trial.json, plus overrides "
            "such as model.width=128"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="trainer YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--tracker",
        help=(
            "tracker YAML, e.g. conf/tracker/files.yaml (default: from "
            "--config)"
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )

    cfg = load_config(args.config, [t for t in args.targets if "=" in t])
    tracker = TrackerConfig.from_yaml(args.tracker or cfg.tracker)
    run_dirs = expand_run_dirs([t for t in args.targets if "=" not in t])
    info = trainer_info(cfg)
    log.info(
        "%d trials, device %s, trainer fingerprint %s",
        len(run_dirs),
        info.device,
        info.fingerprint[:12],
    )
    for n, run_dir in enumerate(run_dirs, 1):
        try:
            result = train(run_dir, cfg, tracker)
        except (TrialError, LeakageError) as error:
            raise SystemExit(
                f"[{n}/{len(run_dirs)}] {run_dir}: refused: {error}"
            ) from error
        if result is None:
            log.info(
                "[%d/%d] %s: already complete, skipped",
                n,
                len(run_dirs),
                run_dir,
            )
        else:
            log.info(
                "[%d/%d] %s: best %s %.4f at step %d, %.0f s",
                n,
                len(run_dirs),
                run_dir,
                SELECTION_METRIC,
                result["best_val_auroc"],
                result["best_step"],
                result["wall_time_s"],
            )


if __name__ == "__main__":
    main()
