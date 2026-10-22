"""Pretrained signal and text embeddings used by the exploration command.

One recording is one vector: time-pooled lead vectors are concatenated in a
fixed, named channel order. Long recordings are covered by non-overlapping
chunks, never silently truncated to a foundation model's context length.

The OpenTSLM encoder below adapts TransformerCNNEncoder from OpenTSLM commit
2968f4b (https://github.com/OpenTSLM/OpenTSLM). Only the signal encoder is loaded;
the language model, projector, prompts, and labels never enter signal features.

MIT License for the adapted OpenTSLM encoder:
Copyright (c) 2025 Stanford University, ETH Zurich, and the project authors
(see https://github.com/OpenTSLM/OpenTSLM/blob/2968f4b/CONTRIBUTORS.md).
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to
do so, subject to the following conditions: The above copyright notice and
this permission notice shall be included in all copies or substantial portions
of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

CHRONOS_MODEL = "amazon/chronos-t5-tiny"
CHRONOS_REVISION = "29d808298f1a62493e7b9a5e08529d0d930fa189"
OPENTSLM_MODEL = "OpenTSLM/llama-3.2-1b-tsqa-sp"
OPENTSLM_REVISION = "1904441f0d87c458d6e9376de04ada5f4c9ab5b7"
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _dependency(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(
            f"Exploration needs {name!r}; install the optional dependencies with `uv sync --extra explore`."
        ) from exc


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_device(device: str = "auto") -> str:
    """Resolve CPU, Apple MPS, or CUDA, failing when an explicitly requested device is absent."""
    torch = _dependency("torch")
    if device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("device must be one of auto, cpu, mps, cuda")
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if device == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if device == "cuda" and not cuda:
        raise ValueError("CUDA was requested but no CUDA device is available; use --device cpu or mps.")
    if device == "mps" and not mps:
        raise ValueError("MPS was requested but Apple Metal is unavailable; use --device cpu or cuda.")
    return device


def _batch_size(config: dict) -> int:
    size = config.get("batch_size", 32)
    if isinstance(size, bool) or int(size) != size or size < 1:
        raise ValueError("batch_size must be a positive integer")
    return int(size)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _model_revision(config: dict, default_model: str, default_revision: str) -> tuple[str, str]:
    model = str(config.get("model") or default_model)
    revision = str(config.get("revision") or (default_revision if model == default_model else "main"))
    return model, revision


def _provenance(backend: str, device: str, config: dict) -> dict:
    return {
        "backend": backend,
        "device": device,
        "dtype": "float32",
        "seed": int(config.get("seed", 0)),
        "batch_size": _batch_size(config),
        "torch_version": _version("torch"),
        "representation": "signal_encoder_before_text_fusion",
        "aggregation": "sample_weighted_chunk_mean_then_named_channel_concatenation",
        "cross_channel_attention": False,
        "chunk_overlap": 0,
        "training": False,
    }


def _prepare_signals(
    signals: list[np.ndarray], sample_rates: list[float], channels: list[list[str]], config: dict
) -> tuple[list[np.ndarray], list[str], dict]:
    if not signals or len(signals) != len(sample_rates) or len(signals) != len(channels):
        raise ValueError("signals, sample_rates, and channels must have the same nonzero length")
    channel_order = list(config.get("channel_order") or channels[0])
    if not channel_order or len(channel_order) != len(set(channel_order)):
        raise ValueError("Channel names must be nonempty and unique within each recording")
    if any(not isinstance(name, str) or not name.strip() for name in channel_order):
        raise ValueError("Every channel must have a nonempty name")
    rates = np.asarray(sample_rates, dtype=float)
    if not np.isfinite(rates).all() or np.any(rates <= 0):
        raise ValueError("sample_rates must be finite and positive")
    target_rate = config.get("target_sample_rate")
    if target_rate is None and not np.allclose(rates, rates[0], rtol=1e-7, atol=0):
        raise ValueError(
            "Mixed sample rates require target_sample_rate so equal time intervals stay comparable"
        )
    if target_rate is not None and (not math.isfinite(float(target_rate)) or float(target_rate) <= 0):
        raise ValueError("target_sample_rate must be finite and positive")
    prepared = []
    for i, (signal, names, rate) in enumerate(zip(signals, channels, rates, strict=True)):
        arr = np.asarray(signal, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(names) or arr.shape[1] < 1:
            raise ValueError(f"Recording {i}: expected [channels, time] with at least one sample")
        if len(names) != len(channel_order) or set(names) != set(channel_order):
            raise ValueError(f"Recording {i}: channel names differ; select a channel-consistent cohort")
        if not np.isfinite(arr).all():
            raise ValueError(f"Recording {i}: nonfinite signal values must be cleaned before embedding")
        arr = arr[[names.index(name) for name in channel_order]]
        if target_rate is not None and not math.isclose(float(target_rate), rate, rel_tol=1e-7):
            from scipy.signal import resample_poly

            ratio = Fraction(float(target_rate) / float(rate)).limit_denominator(100_000)
            arr = resample_poly(arr, ratio.numerator, ratio.denominator, axis=-1).astype(np.float32)
        prepared.append(arr)
    return (
        prepared,
        channel_order,
        {
            "channel_order": channel_order,
            "input_sample_rates_hz": sorted(set(rates.tolist())),
            "effective_sample_rate_hz": float(target_rate) if target_rate is not None else float(rates[0]),
            "resampling": "scipy.signal.resample_poly" if target_rate is not None else "none",
            "model_sample_rate_assumption": "sample-index model; no fixed training sample rate",
            "recording_count": len(signals),
            "sample_lengths": [int(x.shape[1]) for x in prepared],
        },
    )


def _chunks(signals: list[np.ndarray], context_length: int):
    for record, signal in enumerate(signals):
        for channel, lead in enumerate(signal):
            for start in range(0, len(lead), context_length):
                yield record, channel, lead[start : start + context_length]


def _finish_vectors(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    vectors = (sums / counts[..., None]).reshape(sums.shape[0], -1).astype(np.float32)
    if not np.isfinite(vectors).all():
        raise RuntimeError("Embedding model produced nonfinite features")
    return vectors


def _load_chronos(config: dict, device: str):
    torch = _dependency("torch")
    chronos = _dependency("chronos")
    model, revision = _model_revision(config, CHRONOS_MODEL, CHRONOS_REVISION)
    checkpoint = config.get("checkpoint")
    source = str(checkpoint or model)
    kwargs = {"torch_dtype": torch.float32}
    if not checkpoint:
        kwargs["revision"] = revision
    # Loading on CPU then moving avoids an unnecessary accelerate dependency.
    pipeline = chronos.ChronosPipeline.from_pretrained(source, **kwargs)
    pipeline.model.to(device)
    pipeline.model.eval()
    if pipeline.model.config.model_type != "seq2seq":
        raise ValueError("Chronos embeddings require an original Chronos T5 encoder-decoder checkpoint")
    hf_config = pipeline.model.model.config
    resolved = getattr(hf_config, "_commit_hash", None)
    details = {
        "model": model,
        "revision_requested": revision,
        "revision": resolved or revision,
        "chronos_version": _version("chronos-forecasting"),
        "normalization": "Chronos tokenizer mean absolute scale per chunk",
        "pooling": "mean of observation tokens; padding and EOS excluded",
        "training_data": "Chronos pretrained forecasting corpus; see model card",
    }
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        details["checkpoint"] = str(path)
        details["checkpoint_provenance"] = config.get("checkpoint_provenance")
        details["checkpoint_files_sha256"] = {
            str(p.relative_to(path)): _sha256(p)
            for p in sorted(path.rglob("*"))
            if p.is_file() and p.suffix in {".json", ".safetensors", ".bin"}
        }
    return pipeline, details


def _chronos(signals: list[np.ndarray], config: dict, device: str) -> tuple[np.ndarray, dict]:
    torch = _dependency("torch")
    pipeline, details = _load_chronos(config, device)
    native_context = int(pipeline.model.config.context_length)
    context = int(config.get("context_length") or native_context)
    if not 1 <= context <= native_context:
        raise ValueError(f"context_length must be between 1 and the model limit ({native_context})")
    batch_size = _batch_size(config)
    sums = None
    counts = np.zeros((len(signals), len(signals[0])), dtype=np.float64)
    batch = []

    def consume(jobs):
        nonlocal sums
        width = max(len(job[2]) for job in jobs)
        values = torch.full((len(jobs), width), float("nan"), dtype=torch.float32)
        for row, (_, _, lead) in enumerate(jobs):
            values[row, -len(lead) :] = torch.from_numpy(np.array(lead, copy=True))
        with torch.inference_mode():
            hidden, _ = pipeline.embed(values)
        hidden = hidden.detach().float().cpu().numpy()
        # Original Chronos adds EOS after the observation tokens. Pool only
        # real positions; a final short chunk is not weighted like a full one.
        if hidden.ndim != 3 or hidden.shape[1] not in {width, width + 1}:
            raise RuntimeError(
                "Unexpected Chronos embedding shape; expected observation tokens plus optional EOS"
            )
        if sums is None:
            sums = np.zeros((*counts.shape, hidden.shape[-1]), dtype=np.float64)
        for row, (record, channel, lead) in enumerate(jobs):
            sums[record, channel] += hidden[row, width - len(lead) : width].sum(axis=0)
            counts[record, channel] += len(lead)

    for job in _chunks(signals, context):
        batch.append(job)
        if len(batch) == batch_size:
            consume(batch)
            batch = []
    if batch:
        consume(batch)
    details.update(context_length=context, native_context_length=native_context)
    return _finish_vectors(sums, counts), details


def _make_opentslm_encoder(state: dict, config: dict):
    """Construct the upstream CNN+Transformer signal encoder, without an LLM.

    Architecture and parameter names match upstream TransformerCNNEncoder.
    The strict load below prevents a partial or random encoder being used.
    """
    torch = _dependency("torch")
    nn = torch.nn
    try:
        dimension, in_channels, patch_size = state["patch_embed.weight"].shape
        _, max_patches, position_dimension = state["pos_embed"].shape
        layer_ids = {int(key.split(".")[2]) for key in state if key.startswith("encoder.layers.")}
        ff_dim = state["encoder.layers.0.linear1.weight"].shape[0]
    except (KeyError, ValueError, AttributeError) as exc:
        raise ValueError("Checkpoint does not contain an OpenTSLM-SP TransformerCNNEncoder state") from exc
    if in_channels != 1 or dimension != position_dimension or layer_ids != set(range(len(layer_ids))):
        raise ValueError("Unsupported OpenTSLM encoder architecture in checkpoint")
    num_heads = int(config.get("encoder_num_heads", 8))
    if num_heads < 1 or dimension % num_heads:
        raise ValueError("encoder_num_heads must be positive and divide the encoder dimension")

    class SignalEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_size = patch_size
            self.patch_embed = nn.Conv1d(1, dimension, patch_size, stride=patch_size, bias=False)
            self.pos_embed = nn.Parameter(torch.empty(1, max_patches, dimension))
            self.input_norm = nn.LayerNorm(dimension)
            self.input_dropout = nn.Dropout(0.0)
            layer = nn.TransformerEncoderLayer(
                d_model=dimension,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=len(layer_ids))

        def forward(self, values):
            features = self.patch_embed(values.unsqueeze(1)).transpose(1, 2)
            features = self.input_dropout(self.input_norm(features + self.pos_embed[:, : features.shape[1]]))
            return self.encoder(features)

    encoder = SignalEncoder()
    try:
        encoder.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"OpenTSLM checkpoint is incomplete or incompatible: {exc}") from exc
    return encoder, {
        "patch_size": int(patch_size),
        "max_patches": int(max_patches),
        "encoder_dimension": int(dimension),
        "encoder_layers": len(layer_ids),
        "encoder_num_heads": num_heads,
        "encoder_source_commit": "2968f4b",
    }


def _load_opentslm(config: dict, device: str):
    torch = _dependency("torch")
    model, revision = _model_revision(config, OPENTSLM_MODEL, OPENTSLM_REVISION)
    checkpoint = config.get("checkpoint")
    declaration = config.get("checkpoint_provenance")
    if checkpoint or model != OPENTSLM_MODEL:
        if not isinstance(declaration, dict) or not declaration.get("training_data"):
            raise ValueError("Custom OpenTSLM checkpoints require checkpoint_provenance.training_data")
        if declaration.get("contains_ecgqa") is not False:
            raise ValueError(
                "Declare checkpoint_provenance.contains_ecgqa: false to exclude ECG-QA contamination"
            )
    else:
        declaration = {
            "training_data": "ChengsenWang/TSQA (OpenTSLM curriculum stage 1)",
            "contains_ecgqa": False,
            "source": "https://github.com/OpenTSLM/OpenTSLM/blob/2968f4b/README.md#training-stages",
        }
    if not checkpoint and "-ecg-" in model.lower():
        raise ValueError(
            "OpenTSLM ECG checkpoints are excluded because they overlap the ECG-QA test population"
        )
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"OpenTSLM checkpoint must be a .pt checkpoint file: {path}")
        resolved_revision = None
    else:
        hub = _dependency("huggingface_hub")
        path = Path(
            hub.hf_hub_download(
                repo_id=model,
                filename="model_checkpoint.pt",
                revision=revision,
                local_files_only=bool(config.get("local_files_only", False)),
            )
        )
        resolved_revision = path.parent.name if path.parent.parent.name == "snapshots" else revision
    # Never deserialize arbitrary checkpoint Python objects or load an LLM.
    checkpoint_data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_data, dict) or "encoder_state" not in checkpoint_data:
        raise ValueError("OpenTSLM checkpoint must contain an encoder_state dictionary")
    encoder, architecture = _make_opentslm_encoder(checkpoint_data["encoder_state"], config)
    encoder.to(device=device, dtype=torch.float32).eval()
    return encoder, {
        "model": model,
        "revision_requested": revision,
        "revision": resolved_revision,
        "checkpoint_sha256": _sha256(path),
        "checkpoint_provenance": declaration,
        "normalization": "whole-record per-lead z-score, sample std (ddof=1), epsilon=1e-8",
        "pooling": "valid-sample-weighted patch mean; final patch zero-padded on right",
        "text_fusion": False,
        "llm_loaded": False,
        **architecture,
    }


def _opentslm(signals: list[np.ndarray], config: dict, device: str) -> tuple[np.ndarray, dict]:
    torch = _dependency("torch")
    encoder, details = _load_opentslm(config, device)
    patch = details["patch_size"]
    native_context = patch * details["max_patches"]
    context = int(config.get("context_length") or native_context)
    if context < patch or context > native_context or context % patch:
        raise ValueError(
            f"context_length must be a multiple of {patch}, between {patch} and {native_context}"
        )
    normalized = []
    for signal in signals:
        centered = signal - signal.mean(axis=-1, keepdims=True, dtype=np.float64)
        std = signal.std(axis=-1, keepdims=True, ddof=1 if signal.shape[-1] > 1 else 0, dtype=np.float64)
        normalized.append((centered / (std + 1e-8)).astype(np.float32))
    counts = np.zeros((len(signals), len(signals[0])), dtype=np.float64)
    sums = np.zeros((*counts.shape, details["encoder_dimension"]), dtype=np.float64)
    # Group by padded length so padding never becomes artificial attention
    # context for a shorter sequence in a mixed-length batch.
    batches = defaultdict(list)
    batch_size = _batch_size(config)

    def consume(jobs, width):
        values = np.zeros((len(jobs), width), dtype=np.float32)
        for row, (_, _, lead) in enumerate(jobs):
            values[row, : len(lead)] = lead
        with torch.inference_mode():
            hidden = encoder(torch.from_numpy(values).to(device)).detach().float().cpu().numpy()
        if hidden.shape != (len(jobs), width // patch, details["encoder_dimension"]):
            raise RuntimeError("Unexpected OpenTSLM encoder shape")
        for row, (record, channel, lead) in enumerate(jobs):
            weights = np.minimum(patch, len(lead) - np.arange(width // patch) * patch)
            sums[record, channel] += (hidden[row] * weights[:, None]).sum(axis=0)
            counts[record, channel] += len(lead)

    for job in _chunks(normalized, context):
        width = math.ceil(len(job[2]) / patch) * patch
        batches[width].append(job)
        if len(batches[width]) == batch_size:
            consume(batches.pop(width), width)
    for width, jobs in batches.items():
        consume(jobs, width)
    details.update(context_length=context, native_context_length=native_context)
    return _finish_vectors(sums, counts), details


def embed_recordings(
    signals: list[np.ndarray], sample_rates: list[float], channels: list[list[str]], config: dict
) -> tuple[np.ndarray, dict]:
    """Embed complete multichannel records with pretrained Chronos or OpenTSLM-SP.

    Configuration supports backend, model, revision, checkpoint, device,
    batch_size, seed, context_length, target_sample_rate, and channel_order.
    Custom OpenTSLM checkpoints additionally require checkpoint_provenance with
    training_data and contains_ecgqa=false. No training or random fallback occurs.
    """
    backend = config.get("backend", "chronos")
    if backend not in {"chronos", "opentslm"}:
        raise ValueError("backend must be chronos or opentslm")
    prepared, _, details = _prepare_signals(signals, sample_rates, channels, config)
    device = resolve_device(config.get("device", "auto"))
    common = _provenance(backend, device, config)
    function = _chronos if backend == "chronos" else _opentslm
    vectors, model_details = function(prepared, config, device)
    return vectors, {**common, **details, **model_details, "embedding_dimension": vectors.shape[1]}


def _load_text_model(config: dict, device: str):
    st = _dependency("sentence_transformers")
    model, revision = _model_revision(config, TEXT_MODEL, TEXT_REVISION)
    encoder = st.SentenceTransformer(model, revision=revision, device=device, trust_remote_code=False)
    encoder.eval()
    transformer = encoder[0].auto_model
    resolved = getattr(transformer.config, "_commit_hash", None)
    return encoder, {
        "model": model,
        "revision_requested": revision,
        "revision": resolved or revision,
        "sentence_transformers_version": _version("sentence-transformers"),
    }


def embed_texts(texts: list[str], config: dict) -> tuple[np.ndarray, dict]:
    """Sentence embeddings, covering long text with weighted token chunks.

    Empty text receives a zero vector and is marked missing in provenance;
    callers must exclude those rows from semantic-distance calculations.
    """
    if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
        raise ValueError("texts must be a list of strings")
    device = resolve_device(config.get("device", "auto"))
    batch_size = _batch_size(config)
    encoder, details = _load_text_model(config, device)
    tokenizer = encoder.tokenizer
    capacity = int(encoder.max_seq_length) - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity < 1:
        raise ValueError("Sentence encoder has no usable token context")
    chunks, owners, weights = [], [], []
    missing = []
    for row, text in enumerate(texts):
        if not text.strip():
            missing.append(row)
            continue
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        if not ids:
            missing.append(row)
        start = 0
        while start < len(ids):
            end = min(start + capacity, len(ids))
            # Decoding a chunk that begins inside a word can change its token
            # count when SentenceTransformer tokenizes it again. Check the
            # round trip explicitly so encode() cannot silently truncate it.
            while True:
                chunk_text = tokenizer.decode(ids[start:end], skip_special_tokens=True)
                round_trip = tokenizer(chunk_text, add_special_tokens=False, truncation=False)["input_ids"]
                if len(round_trip) <= capacity:
                    break
                end -= max(1, len(round_trip) - capacity)
                if end <= start:
                    raise ValueError("A text token cannot fit the sentence model context after decoding")
            chunks.append(chunk_text)
            owners.append(row)
            weights.append(end - start)
            start = end
    get_dimension = getattr(encoder, "get_embedding_dimension", None)
    if get_dimension is None:
        get_dimension = encoder.get_sentence_embedding_dimension
    dimension = int(get_dimension())
    sums = np.zeros((len(texts), dimension), dtype=np.float64)
    if chunks:
        encoded = np.asarray(
            encoder.encode(
                chunks,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if encoded.shape != (len(chunks), dimension) or not np.isfinite(encoded).all():
            raise RuntimeError("Sentence encoder produced invalid embeddings")
        for row, weight, vector in zip(owners, weights, encoded, strict=True):
            sums[row] += weight * vector
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    vectors = np.divide(sums, norms, out=np.zeros_like(sums), where=norms > 0).astype(np.float32)
    return vectors, {
        **details,
        "backend": "sentence-transformers",
        "device": device,
        "seed": int(config.get("seed", 0)),
        "batch_size": batch_size,
        "embedding_dimension": dimension,
        "aggregation": "token_count_weighted_chunk_mean_then_l2_normalization",
        "max_chunk_tokens": capacity,
        "chunk_count": len(chunks),
        "missing_text_indices": missing,
    }
