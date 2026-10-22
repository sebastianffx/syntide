"""Original DiffuSETS inference with pinned external architecture and local conditioning.

Architecture stays in an explicitly downloaded upstream checkout/cache; no training code,
external embedding service, or arbitrary remote code is loaded implicitly.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from syntide.generate.full.api import GeneratedSignal, GenerationRequest, SignalSpec
from syntide.generate.full.assets import SOURCE_HASHES, SOURCE_REVISION

CHANNELS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
NATIVE_SPEC = SignalSpec(modality="ecg", channels=CHANNELS, sample_rate=102.4, num_samples=1024, units="mV")
EMBEDDING_MODEL = "text-embedding-ada-002"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def format_report(report: str) -> str:
    """Reproduce upstream report prompting before obtaining ada-002 embeddings."""
    reports = report.split("|")
    if reports and reports[-1] == "":
        reports.pop()
    result = []
    for index, part in enumerate(reports, 1):
        ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(index, f"{index}th")
        prefix = "Most importantly" if index == 1 else "As a supplementary condition"
        result.append(f"{prefix}, the {ordinal} diagnosis is {{{part}}}.")
    return "".join(result)


def _device(requested: str):
    import torch

    if requested == "auto":
        requested = (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if requested not in {"cpu", "mps", "cuda"}:
        raise ValueError("DiffuSETS device must be auto, cpu, mps or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS was requested but is unavailable")
    return torch.device(requested)


class DiffuSETSBackend:
    """Text/demographic conditioned, ten-second 12-lead ECG synthesis.

    The configuration is deliberately independent of the dataset or training adapter.
    Other modalities belong to other registered generators, not to DiffuSETS.
    """

    def __init__(self, config: dict):
        allowed = {
            "source_dir",
            "unet_checkpoint",
            "vae_checkpoint",
            "unet_sha256",
            "vae_sha256",
            "device",
            "num_inference_steps",
            "resample",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"Unknown DiffuSETS configuration keys: {sorted(unknown)}")
        for name in ("source_dir", "unet_checkpoint", "vae_checkpoint"):
            if not isinstance(config.get(name), str) or not config[name]:
                raise ValueError(f"DiffuSETS requires {name}")
        self.config = dict(config)
        self.source_dir = Path(config["source_dir"]).expanduser().resolve()
        self.checkpoints = {
            name: Path(config[f"{name}_checkpoint"]).expanduser().resolve() for name in ("unet", "vae")
        }
        self.steps = config.get("num_inference_steps", 1000)
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or not 1 <= self.steps <= 1000:
            raise ValueError("num_inference_steps must be an integer in [1, 1000]")
        self.resample = config.get("resample", False)
        if not isinstance(self.resample, bool):
            raise ValueError("resample must be a boolean")
        self.requested_device = config.get("device", "auto")
        if not isinstance(self.requested_device, str) or self.requested_device not in {
            "auto",
            "cpu",
            "mps",
            "cuda",
        }:
            raise ValueError("DiffuSETS device must be auto, cpu, mps or cuda")
        self._assets = None
        self._conditions: dict[str, dict] = {}
        self._unet = self._decoder = self._scheduler = None
        self._resolved_device = None

    def capabilities(self) -> dict:
        return {
            "modalities": ["ecg"],
            "channels": list(CHANNELS),
            "native_sample_rate": 102.4,
            "native_num_samples": 1024,
            "duration_seconds": 10,
            "units": "mV",
            "text_embedding_model": EMBEDDING_MODEL,
            "text_embedding_dimension": 1536,
            "conditioning": ["report", "age", "sex", "heart_rate", "text_embedding_path"],
            "devices": ["cpu", "mps", "cuda"],
            "explicit_resampling": True,
        }

    def validate_request(self, request: GenerationRequest) -> None:
        spec = request.signal
        if spec.modality != "ecg" or tuple(spec.channels) != CHANNELS or spec.units != "mV":
            raise ValueError("DiffuSETS requires ecg, the standard 12-lead order, and mV units")
        if not math.isclose(spec.num_samples / spec.sample_rate, 10, rel_tol=0, abs_tol=1e-9):
            raise ValueError("DiffuSETS generates exactly ten seconds")
        if (spec.sample_rate != 102.4 or spec.num_samples != 1024) and not self.resample:
            raise ValueError("DiffuSETS is native 1024 samples at 102.4 Hz; set resample=true explicitly")
        if spec.sample_rate <= 0 or spec.num_samples <= 0:
            raise ValueError("Signal rate and length must be positive")
        if (
            isinstance(request.seed, bool)
            or not isinstance(request.seed, int)
            or not 0 <= request.seed < 2**32
        ):
            raise ValueError("seed must be an integer in [0, 2**32)")
        self._condition(request.conditioning)
        self._check_assets()

    def _condition(self, condition: dict) -> dict:
        allowed = {
            "report",
            "age",
            "sex",
            "heart_rate",
            "text_embedding_path",
            "text_embedding_model",
            "embedding_input",
            "embedding_source",
            "text_embedding_sha256",
        }
        if not isinstance(condition, dict) or set(condition) - allowed:
            raise ValueError("Unsupported DiffuSETS conditioning fields")
        if not isinstance(condition.get("report"), str) or not condition["report"].strip():
            raise ValueError("DiffuSETS requires an explicit clinical report")
        if not isinstance(condition.get("sex"), str) or condition["sex"] not in {"M", "F"}:
            raise ValueError("DiffuSETS sex conditioning must be M or F, as encoded by its checkpoint")
        for name, upper in (("age", 110), ("heart_rate", float("inf"))):
            value = condition.get(name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if not 0 < value < upper:
                raise ValueError(f"{name} is outside the DiffuSETS conditioning range")
        if condition.get("text_embedding_model") != EMBEDDING_MODEL:
            raise ValueError("Original DiffuSETS requires text-embedding-ada-002 (1536 dimensions)")
        embedding_input = condition.get("embedding_input")
        if embedding_input is not None and embedding_input != format_report(condition["report"]):
            raise ValueError(
                "embedding_input must equal format_report(report) for the original DiffuSETS prompt"
            )
        if embedding_input is None and not condition.get("embedding_source"):
            raise ValueError(
                "Provide embedding_input or an explicit embedding_source for a published legacy vector"
            )
        if condition.get("embedding_source") is not None and not isinstance(
            condition["embedding_source"], str
        ):
            raise ValueError("embedding_source must be a string")
        if not isinstance(condition.get("text_embedding_path"), str):
            raise ValueError("text_embedding_path must point to a local .npy vector")
        path = Path(condition["text_embedding_path"]).expanduser().resolve()
        if path.suffix != ".npy":
            raise ValueError("text_embedding_path must point to a local .npy vector")
        digest = _sha256(path)
        if condition.get("text_embedding_sha256") not in (None, digest):
            raise ValueError("Text embedding checksum does not match")
        key = json.dumps(condition, sort_keys=True, allow_nan=False)
        if key in self._conditions:
            result = self._conditions[key]
            if result["sha256"] != digest:
                raise ValueError("Text embedding changed after validation")
            return result
        vector = np.load(path, allow_pickle=False)
        if not isinstance(vector, np.ndarray) or vector.shape != (1536,):
            raise ValueError("DiffuSETS text embedding must be a .npy vector with shape (1536,)")
        if (
            not np.issubdtype(vector.dtype, np.number)
            or np.iscomplexobj(vector)
            or not np.isfinite(vector).all()
        ):
            raise ValueError("Text embedding must contain finite real numbers")
        if np.max(np.abs(vector.astype(np.float64))) > np.finfo(np.float32).max:
            raise ValueError("Text embedding exceeds float32 range")
        if not np.any(vector):
            raise ValueError("An all-zero vector is not valid DiffuSETS text conditioning")
        result = {
            "path": str(path),
            "sha256": digest,
            "model": EMBEDDING_MODEL,
            "report": condition["report"],
            "embedding_input": embedding_input,
            "embedding_source": condition.get("embedding_source"),
            "prompt_provenance": "declared_matching"
            if embedding_input is not None
            else "unknown_legacy_input",
            "vector": vector.astype(np.float32),
        }
        self._conditions[key] = result
        return result

    def _check_assets(self) -> dict:
        # Check again before actual loading, including files changed after preflight.
        sources = {name: _sha256(self.source_dir / name) for name in SOURCE_HASHES}
        if sources != SOURCE_HASHES:
            raise ValueError(f"Architecture must match pinned DiffuSETS source {SOURCE_REVISION}")
        weights = {name: _sha256(path) for name, path in self.checkpoints.items()}
        for name, digest in weights.items():
            if self.config.get(f"{name}_sha256") not in (None, digest):
                raise ValueError(f"{name} checkpoint checksum does not match")
        assets = {"source_sha256": sources, "checkpoint_sha256": weights}
        if self._assets is not None and assets != self._assets:
            raise ValueError("DiffuSETS assets changed after validation")
        self._assets = assets
        return assets

    def _load(self) -> None:
        if self._unet is not None:
            return
        import torch

        try:
            from diffusers import DDPMScheduler
        except ImportError as error:
            raise ImportError("Install DiffuSETS dependencies with uv sync --extra diffusets") from error
        self._check_assets()
        self._resolved_device = _device(self.requested_device)
        modules = []
        for name, digest in SOURCE_HASHES.items():
            spec = importlib.util.spec_from_file_location(
                f"_syntide_diffusets_{digest}", self.source_dir / name
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
        # Constructor initialization must not consume the caller's global RNG stream.
        with torch.random.fork_rng(devices=[]):
            unet = modules[0].ECGconditional(
                number_of_diffusions=1000, kernel_size=7, num_levels=7, n_channels=4, text_embed_dim=1536
            )
            decoder = modules[1].VAE_Decoder()
        unet.load_state_dict(
            torch.load(self.checkpoints["unet"], map_location="cpu", weights_only=True), strict=True
        )
        vae = torch.load(self.checkpoints["vae"], map_location="cpu", weights_only=True)
        decoder.load_state_dict(vae["decoder"], strict=True)
        self._unet = unet.to(device=self._resolved_device, dtype=torch.float32).eval()
        self._decoder = decoder.to(device=self._resolved_device, dtype=torch.float32).eval()
        self._scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.0120,
            beta_schedule="linear",
            prediction_type="epsilon",
            clip_sample=True,
            variance_type="fixed_small",
            timestep_spacing="leading",
        )

    def provenance(self) -> dict:
        import torch

        self._check_assets()
        device = self._resolved_device or _device(self.requested_device)
        return {
            "generator": "diffusets@1",
            "variant": "original_ada002",
            "source_revision": SOURCE_REVISION,
            "source_repository": "https://github.com/Raiiyf/DiffuSETS_Exp",
            **self._assets,
            "source_dir": str(self.source_dir),
            "checkpoints": {name: str(path) for name, path in self.checkpoints.items()},
            "device": str(device),
            "dtype": "float32",
            "torch_version": torch.__version__,
            "diffusers_version": importlib.metadata.version("diffusers"),
            "num_inference_steps": self.steps,
            "training_steps": 1000,
            "scheduler": {
                "name": "DDPM",
                "beta_start": 0.00085,
                "beta_end": 0.012,
                "beta_schedule": "linear",
                "prediction_type": "epsilon",
                "clip_sample": True,
                "variance_type": "fixed_small",
                "timestep_spacing": "leading",
            },
            "randomness": "per-sample seeded CPU torch.Generator; device-specific arithmetic",
            "capabilities": self.capabilities(),
            "resample": self.resample,
            "conditioning_embeddings": [
                {k: v for k, v in c.items() if k != "vector"} for c in self._conditions.values()
            ],
        }

    def generate(self, request: GenerationRequest) -> GeneratedSignal:
        import torch

        self.validate_request(request)
        self._load()
        condition = request.conditioning
        prepared = self._condition(condition)
        device = self._resolved_device
        rng = torch.Generator(device="cpu").manual_seed(request.seed)
        latent = torch.randn((1, 4, 128), generator=rng, dtype=torch.float32).to(device)
        text = torch.from_numpy(prepared["vector"]).reshape(1, 1, 1536).to(device)
        # Ordering is part of the trained model: gender, age, heart rate.
        demographic = {
            "gender": 1.0 if condition["sex"] == "M" else 0.0,
            "age": float(condition["age"]),
            "heart rate": float(condition["heart_rate"]),
        }
        demographic = {
            key: torch.tensor([[[value]]], dtype=torch.float32, device=device)
            for key, value in demographic.items()
        }
        self._scheduler.set_timesteps(self.steps)
        with torch.inference_mode():
            for step in self._scheduler.timesteps:
                timestep = torch.full((1,), int(step), dtype=torch.long, device=device)
                residual = self._unet(latent, timestep, text, demographic)
                latent = self._scheduler.step(residual, step, latent, generator=rng).prev_sample
            signal = self._decoder(latent)[0].transpose(0, 1).cpu().numpy().copy()
        if signal.shape != (12, 1024) or not np.isfinite(signal).all():
            raise ValueError("DiffuSETS returned a malformed or non-finite waveform")
        if request.signal.sample_rate != 102.4:
            from scipy.signal import resample_poly

            ratio = Fraction(request.signal.num_samples, 1024)
            signal = resample_poly(signal, ratio.numerator, ratio.denominator, axis=1).astype(np.float32)
        return GeneratedSignal(
            signal=signal,
            spec=request.signal,
            metadata={
                "native_sample_rate": 102.4,
                "native_num_samples": 1024,
                "resampling": "polyphase" if request.signal.sample_rate != 102.4 else "none",
                "text_embedding_sha256": prepared["sha256"],
                "prompt_provenance": prepared["prompt_provenance"],
                "clinical_validation": "pending",
                "num_inference_steps": self.steps,
            },
        )
