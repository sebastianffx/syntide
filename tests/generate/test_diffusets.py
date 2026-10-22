"""DiffuSETS preflight and inference contracts without downloading model weights."""

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from syntide.generate.full import diffusets
from syntide.generate.full.api import GenerationRequest
from syntide.generate.full.diffusets import NATIVE_SPEC, DiffuSETSBackend, format_report


@pytest.fixture
def assets(tmp_path, monkeypatch):
    source = tmp_path / "source"
    hashes = {}
    for name in diffusets.SOURCE_HASHES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Test architecture fixture: {name}\n")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(diffusets, "SOURCE_HASHES", hashes)
    unet = tmp_path / "unet.pt"
    unet.write_bytes(b"test unet weights")
    vae = tmp_path / "vae.pt"
    vae.write_bytes(b"test vae weights")
    embedding = tmp_path / "clinical-text.npy"
    np.save(embedding, np.linspace(-1, 1, 1536, dtype=np.float32))
    report = "Sinus rhythm|Normal ECG|"
    config = {
        "source_dir": str(source),
        "unet_checkpoint": str(unet),
        "vae_checkpoint": str(vae),
        "device": "cpu",
        "num_inference_steps": 4,
    }
    conditioning = {
        "report": report,
        "age": 42,
        "sex": "M",
        "heart_rate": 65,
        "text_embedding_path": str(embedding),
        "text_embedding_model": "text-embedding-ada-002",
        "embedding_input": format_report(report),
    }
    return config, GenerationRequest(seed=123, signal=NATIVE_SPEC, conditioning=conditioning)


def test_preflight_checks_assets_without_loading_weights(assets, monkeypatch):
    config, request = assets
    backend = DiffuSETSBackend(config)
    monkeypatch.setattr(backend, "_load", lambda: pytest.fail("Preflight must not load weights"))
    backend.validate_request(request)
    assert backend._unet is None and backend._decoder is None
    assert set(backend._assets["checkpoint_sha256"]) == {"unet", "vae"}
    assert backend.capabilities()["modalities"] == ["ecg"]


@pytest.mark.parametrize(
    ("signal", "message"),
    [
        (replace(NATIVE_SPEC, modality="ppg"), "requires ecg"),
        (replace(NATIVE_SPEC, channels=NATIVE_SPEC.channels[::-1]), "12-lead order"),
        (replace(NATIVE_SPEC, units="V"), "mV units"),
        (replace(NATIVE_SPEC, sample_rate=100), "exactly ten seconds"),
        (replace(NATIVE_SPEC, sample_rate=100, num_samples=1000), "resample=true"),
    ],
)
def test_preflight_rejects_unsupported_signal_specs(assets, signal, message):
    config, request = assets
    with pytest.raises(ValueError, match=message):
        DiffuSETSBackend(config).validate_request(replace(request, signal=signal))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"text_embedding_model": "all-MiniLM-L6-v2"}, "ada-002"),
        ({"report": " "}, "clinical report"),
        ({"embedding_input": "unformatted report"}, "format_report"),
        ({"sex": "unknown"}, "M or F"),
        ({"sex": []}, "M or F"),
        ({"age": 0}, "conditioning range"),
        ({"age": 110}, "conditioning range"),
        ({"age": True}, "finite number"),
        ({"heart_rate": 0}, "conditioning range"),
        ({"unrecognized": "value"}, "Unsupported"),
        ({"text_embedding_sha256": "0" * 64}, "checksum"),
    ],
)
def test_preflight_rejects_invalid_conditioning(assets, changes, message):
    config, request = assets
    request = replace(request, conditioning={**request.conditioning, **changes})
    with pytest.raises(ValueError, match=message):
        DiffuSETSBackend(config).validate_request(request)


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        (np.ones(384), "shape"),
        (np.ones((1, 1536)), "shape"),
        (np.full(1536, np.nan), "finite real"),
        (np.full(1536, np.inf), "finite real"),
        (np.ones(1536, dtype=np.complex64), "finite real"),
        (np.full(1536, "one"), "finite real"),
        (np.full(1536, np.finfo(np.float64).max), "float32 range"),
        (np.zeros(1536), "all-zero"),
    ],
)
def test_preflight_rejects_unusable_embedding_vectors(assets, vector, message):
    config, request = assets
    np.save(request.conditioning["text_embedding_path"], vector)
    with pytest.raises(ValueError, match=message):
        DiffuSETSBackend(config).validate_request(request)


def test_report_prompt_and_legacy_embedding_provenance(assets):
    assert format_report("Sinus rhythm|Normal ECG|No ST change|Fourth|") == (
        "Most importantly, the 1st diagnosis is {Sinus rhythm}."
        "As a supplementary condition, the 2nd diagnosis is {Normal ECG}."
        "As a supplementary condition, the 3rd diagnosis is {No ST change}."
        "As a supplementary condition, the 4th diagnosis is {Fourth}."
    )
    assert format_report("Normal ECG") == "Most importantly, the 1st diagnosis is {Normal ECG}."
    config, request = assets
    condition = {key: value for key, value in request.conditioning.items() if key != "embedding_input"}
    backend = DiffuSETSBackend(config)
    with pytest.raises(ValueError, match="embedding_input or an explicit embedding_source"):
        backend.validate_request(replace(request, conditioning=condition))
    condition["embedding_source"] = "published DiffuSETS test condition"
    backend.validate_request(replace(request, conditioning=condition))
    prepared = backend._condition(condition)
    assert prepared["prompt_provenance"] == "unknown_legacy_input"
    assert prepared["embedding_input"] is None


@pytest.mark.parametrize("changed_asset", ["source", "unet", "vae", "embedding"])
def test_generation_rejects_assets_changed_since_preflight(assets, monkeypatch, changed_asset):
    config, request = assets
    backend = DiffuSETSBackend(config)
    backend.validate_request(request)
    monkeypatch.setattr(backend, "_load", lambda: pytest.fail("Changed assets must fail before loading"))
    if changed_asset == "source":
        path = backend.source_dir / next(iter(diffusets.SOURCE_HASHES))
        path.write_text("# modified architecture\n")
        message = "pinned DiffuSETS source"
    elif changed_asset == "embedding":
        np.save(request.conditioning["text_embedding_path"], np.ones(1536, dtype=np.float32))
        message = "embedding changed after validation"
    else:
        backend.checkpoints[changed_asset].write_bytes(b"modified checkpoint")
        message = "assets changed after validation"
    with pytest.raises(ValueError, match=message):
        backend.generate(request)


@pytest.mark.parametrize(
    "changes",
    [
        {"guidance_scale": 2},
        {"num_inference_steps": 0},
        {"num_inference_steps": 1001},
        {"num_inference_steps": True},
        {"resample": "yes"},
        {"device": "tpu"},
        {"device": []},
    ],
)
def test_unknown_or_invalid_backend_configuration_is_rejected(assets, changes):
    config, _ = assets
    with pytest.raises(ValueError):
        DiffuSETSBackend({**config, **changes})


@pytest.mark.parametrize("checkpoint", ["unet", "vae"])
def test_explicit_checkpoint_checksums_are_enforced(assets, checkpoint):
    config, request = assets
    backend = DiffuSETSBackend({**config, f"{checkpoint}_sha256": "0" * 64})
    with pytest.raises(ValueError, match=f"{checkpoint} checkpoint checksum"):
        backend.validate_request(request)


class ToyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, latent, timestep, text, demographic):
        self.calls.append(
            {
                "demographic": {key: value.detach().cpu().clone() for key, value in demographic.items()},
                "text": text.detach().cpu().clone(),
                "timestep": timestep.detach().cpu().clone(),
            }
        )
        return latent * 0.05 + text.mean() * 0.01


class ToyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.latents = []

    def forward(self, latent):
        self.latents.append(latent.detach().cpu().clone())
        return latent.mean(dim=1).repeat_interleave(8, dim=1).unsqueeze(-1).repeat(1, 1, 12)


class IdentityScheduler:
    """Keep the latent unchanged so an extra scaling factor cannot hide in a model."""

    def set_timesteps(self, steps):
        self.timesteps = range(steps - 1, -1, -1)

    def step(self, residual, timestep, latent, *, generator):
        assert generator.device.type == "cpu"
        return SimpleNamespace(prev_sample=latent)


def install_toy_models(backend, monkeypatch, scheduler):
    unet, decoder = ToyUNet(), ToyDecoder()

    def load():
        backend._resolved_device = torch.device("cpu")
        backend._unet, backend._decoder, backend._scheduler = unet, decoder, scheduler

    monkeypatch.setattr(backend, "_load", load)
    return unet, decoder


@pytest.mark.parametrize(("sex", "gender"), [("M", 1.0), ("F", 0.0)])
def test_inference_preserves_condition_order_seed_and_native_latent_scale(assets, monkeypatch, sex, gender):
    config, request = assets
    request = replace(request, conditioning={**request.conditioning, "sex": sex})
    backend = DiffuSETSBackend(config)
    unet, decoder = install_toy_models(backend, monkeypatch, IdentityScheduler())
    rng_before = torch.random.get_rng_state().clone()
    output = backend.generate(request)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert output.signal.shape == (12, 1024)
    assert output.spec == NATIVE_SPEC
    assert output.metadata["resampling"] == "none"
    assert np.isfinite(output.signal).all()
    expected = torch.randn((1, 4, 128), generator=torch.Generator().manual_seed(request.seed))
    torch.testing.assert_close(decoder.latents[0], expected, rtol=0, atol=0)
    assert len(unet.calls) == config["num_inference_steps"]
    demographic = unet.calls[0]["demographic"]
    assert list(demographic) == ["gender", "age", "heart rate"]
    assert [value.item() for value in demographic.values()] == [gender, 42, 65]
    assert all(value.shape == (1, 1, 1) for value in demographic.values())
    assert unet.calls[0]["text"].shape == (1, 1, 1536)
    np.testing.assert_array_equal(output.signal, backend.generate(request).signal)
    assert not np.array_equal(output.signal, backend.generate(replace(request, seed=124)).signal)


def test_native_waveform_resamples_to_ptbxl_100hz_length(assets, monkeypatch):
    from scipy.signal import resample_poly

    config, request = assets
    backend = DiffuSETSBackend({**config, "resample": True})
    install_toy_models(backend, monkeypatch, IdentityScheduler())
    native = backend.generate(request)
    requested = replace(NATIVE_SPEC, sample_rate=100, num_samples=1000)
    resampled = backend.generate(replace(request, signal=requested))
    assert resampled.signal.shape == (12, 1000)
    assert resampled.spec == requested
    assert resampled.metadata["resampling"] == "polyphase"
    expected = resample_poly(native.signal, 125, 128, axis=1).astype(np.float32)
    np.testing.assert_allclose(resampled.signal, expected, rtol=0, atol=0)


def test_real_ddpm_scheduler_uses_each_requests_seed(assets, monkeypatch):
    diffusers = pytest.importorskip("diffusers")
    config, request = assets
    backend = DiffuSETSBackend(config)
    scheduler = diffusers.DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=True,
    )
    install_toy_models(backend, monkeypatch, scheduler)
    first = backend.generate(request).signal
    different_seed = backend.generate(replace(request, seed=456)).signal
    repeated = backend.generate(request).signal
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different_seed)
