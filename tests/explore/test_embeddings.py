"""Signal embeddings satisfy model-independent numerical checks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from syntide.explore import embeddings as emb


def test_device_selection_validates_requested_hardware(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert emb.resolve_device() == "mps"
    assert emb.resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="CUDA was requested"):
        emb.resolve_device("cuda")
    with pytest.raises(ValueError, match="one of"):
        emb.resolve_device("gpu")


class FakeChronos:
    model = SimpleNamespace(config=SimpleNamespace(context_length=4))

    def __init__(self):
        self.seen = []

    def embed(self, values):
        self.seen.extend([row[torch.isfinite(row)].tolist() for row in values])
        # Padded values and EOS carry deliberately huge activations: they must
        # not affect a short chunk or the sample-weighted recording mean.
        hidden = torch.nan_to_num(values, nan=1e6).unsqueeze(-1)
        return torch.cat(
            [hidden, torch.full((len(values), 1, 1), 1e6)], dim=1
        ), None


def test_chronos_covers_first_and_last_samples_and_excludes_padding_and_eos(
    monkeypatch,
):
    fake = FakeChronos()
    monkeypatch.setattr(emb, "_load_chronos", lambda *_: (fake, {}))
    signal = np.array(
        [[1, 2, 3, 4, 50], [10, 20, 30, 40, 500]], dtype=np.float32
    )
    vectors, metadata = emb.embed_recordings(
        [signal, signal[::-1]],
        [100, 100],
        [["I", "II"], ["II", "I"]],
        {"backend": "chronos", "device": "cpu", "batch_size": 3},
    )
    np.testing.assert_allclose(vectors, [[12, 120], [12, 120]])
    assert sum(map(len, fake.seen)) == 20
    assert fake.seen[0] == [1, 2, 3, 4]
    assert metadata["channel_order"] == ["I", "II"]
    assert metadata["context_length"] == 4


def test_channel_identity_affects_vector_not_input_order(monkeypatch):
    monkeypatch.setattr(emb, "_load_chronos", lambda *_: (FakeChronos(), {}))
    signal = np.array([[1, 2], [10, 20]], dtype=np.float32)
    vectors, _ = emb.embed_recordings(
        [signal, signal[::-1]],
        [100, 100],
        [["I", "II"], ["I", "II"]],
        {"device": "cpu"},
    )
    assert not np.allclose(vectors[0], vectors[1])
    np.testing.assert_allclose(vectors[0], vectors[1][::-1])


@pytest.mark.parametrize(
    ("signals", "rates", "channels", "message"),
    [
        ([], [], [], "nonzero length"),
        ([np.ones((2, 8))], [100], [["I", "I"]], "unique"),
        (
            [np.ones((2, 8)), np.ones((1, 8))],
            [100, 100],
            [["I", "II"], ["I"]],
            "differ",
        ),
        (
            [np.ones((1, 8)), np.ones((1, 8))],
            [100, 500],
            [["I"], ["I"]],
            "Mixed sample rates",
        ),
        ([np.array([[np.nan]])], [100], [["I"]], "nonfinite"),
        ([np.ones((1, 8))], [-100], [["I"]], "positive"),
    ],
)
def test_invalid_signal_cohorts_fail_before_model_loading(
    signals, rates, channels, message
):
    with pytest.raises(ValueError, match=message):
        emb.embed_recordings(signals, rates, channels, {})


def test_resampling_records_effective_rate_and_retains_duration():
    signals, channels, metadata = emb._prepare_signals(
        [np.ones((1, 100)), np.ones((1, 500))],
        [100, 500],
        [["I"], ["I"]],
        {"target_sample_rate": 100},
    )
    assert [x.shape for x in signals] == [(1, 100), (1, 100)]
    assert channels == ["I"]
    assert metadata["effective_sample_rate_hz"] == 100
    assert metadata["input_sample_rates_hz"] == [100, 500]


def tiny_encoder_state():
    # A real, small torch Transformer with the upstream checkpoint key layout.
    torch.manual_seed(4)
    layer = torch.nn.TransformerEncoderLayer(
        8, 2, dim_feedforward=12, batch_first=True, activation="gelu"
    )
    transformer = torch.nn.TransformerEncoder(layer, num_layers=1)
    state = {
        f"encoder.{key}": value
        for key, value in transformer.state_dict().items()
    }
    state.update(
        {
            "pos_embed": torch.randn(1, 3, 8),
            "patch_embed.weight": torch.randn(8, 1, 4),
            "input_norm.weight": torch.ones(8),
            "input_norm.bias": torch.zeros(8),
        }
    )
    return state


def test_opentslm_strictly_loads_complete_encoder():
    state = tiny_encoder_state()
    encoder, details = emb._make_opentslm_encoder(
        state, {"encoder_num_heads": 2}
    )
    encoder.eval()
    with torch.inference_mode():
        output = encoder(torch.randn(2, 12))
    assert output.shape == (2, 3, 8)
    assert details["encoder_layers"] == 1
    del state["input_norm.weight"]
    with pytest.raises(ValueError, match="incomplete"):
        emb._make_opentslm_encoder(state, {"encoder_num_heads": 2})


def test_opentslm_requires_training_provenance_before_local_load(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"encoder_state": tiny_encoder_state()}, path)
    with pytest.raises(ValueError, match="training_data"):
        emb._load_opentslm({"checkpoint": str(path)}, "cpu")
    with pytest.raises(ValueError, match="contains_ecgqa"):
        emb._load_opentslm(
            {
                "checkpoint": str(path),
                "checkpoint_provenance": {"training_data": "unknown"},
            },
            "cpu",
        )


def test_opentslm_real_tiny_checkpoint_is_deterministic_and_fingerprinted(
    tmp_path,
):
    path = tmp_path / "checkpoint.pt"
    torch.save({"encoder_state": tiny_encoder_state()}, path)
    config = {
        "backend": "opentslm",
        "checkpoint": str(path),
        "device": "cpu",
        "batch_size": 2,
        "encoder_num_heads": 2,
        "checkpoint_provenance": {
            "training_data": "test-only miniature encoder",
            "contains_ecgqa": False,
        },
    }
    signal = np.array(
        [[1, 0, 2, 3, 1, 2, 9, 4, 0, 3, 5, 2, 4]], dtype=np.float32
    )
    first, provenance = emb.embed_recordings([signal], [100], [["I"]], config)
    second, _ = emb.embed_recordings([signal], [100], [["I"]], config)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1, 8)
    assert provenance["checkpoint_sha256"] == emb._sha256(path)
    assert provenance["context_length"] == 12
    assert not provenance["llm_loaded"]
    assert not provenance["text_fusion"]


def test_opentslm_covers_partial_patch_and_normalizes_before_chunks(
    monkeypatch,
):
    seen = []

    class PatchMeans:
        def __call__(self, values):
            seen.append(values.cpu().numpy().copy())
            return values.reshape(len(values), -1, 4).mean(dim=-1, keepdim=True)

    monkeypatch.setattr(
        emb,
        "_load_opentslm",
        lambda *_: (
            PatchMeans(),
            {"patch_size": 4, "max_patches": 2, "encoder_dimension": 1},
        ),
    )
    signal = np.arange(1, 12, dtype=np.float32)[None, :]
    vector, _ = emb.embed_recordings(
        [signal], [100], [["I"]], {"backend": "opentslm", "device": "cpu"}
    )
    normalized = (
        (signal - signal.mean()) / (signal.std(ddof=1) + 1e-8)
    ).reshape(-1)
    np.testing.assert_allclose(seen[0][0], normalized[:8], atol=1e-6)
    np.testing.assert_allclose(seen[1][0], np.r_[normalized[8:], 0], atol=1e-6)
    expected = (normalized[:8].sum() + normalized[8:].mean() * 0.75 * 3) / 11
    np.testing.assert_allclose(vector, [[expected]], atol=1e-6)


class FakeTokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, **kwargs):
        return {"input_ids": [int(token) for token in text.split()]}

    def decode(self, ids, **kwargs):
        return " ".join(str(x) for x in ids)


class FakeSentenceEncoder:
    tokenizer = FakeTokenizer()
    max_seq_length = 5

    def __init__(self):
        self.seen = []

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, chunks, **kwargs):
        self.seen.extend(chunks)
        output = np.array(
            [
                [sum(map(int, chunk.split())), len(chunk.split())]
                for chunk in chunks
            ],
            dtype=float,
        )
        return output / np.linalg.norm(output, axis=1, keepdims=True)


def test_sentence_embeddings_cover_long_text_and_mark_missing(monkeypatch):
    encoder = FakeSentenceEncoder()
    monkeypatch.setattr(emb, "_load_text_model", lambda *_: (encoder, {}))
    vectors, metadata = emb.embed_texts(
        ["1 2 3 4 5 6 7", " ", "9"], {"device": "cpu"}
    )
    assert encoder.seen == ["1 2 3", "4 5 6", "7", "9"]
    assert metadata["missing_text_indices"] == [1]
    assert vectors.shape == (3, 2)
    np.testing.assert_allclose(
        np.linalg.norm(vectors, axis=1), [1, 0, 1], atol=1e-6
    )


def test_missing_optional_dependency_gives_install_instruction(monkeypatch):
    def missing(_):
        raise ImportError("module absent")

    monkeypatch.setattr(emb.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="uv sync --extra explore"):
        emb._dependency("chronos")


def test_sentence_chunk_round_trip_never_truncates_or_drops_source_tokens(
    monkeypatch,
):
    class ExpandingTokenizer(FakeTokenizer):
        def __call__(self, text, **kwargs):
            if text.startswith("prefix "):
                return {
                    "input_ids": [
                        0,
                        *map(int, text.removeprefix("prefix ").split()),
                    ]
                }
            return super().__call__(text, **kwargs)

        def decode(self, ids, **kwargs):
            return "prefix " + super().decode(ids, **kwargs)

    class Encoder(FakeSentenceEncoder):
        tokenizer = ExpandingTokenizer()

        def encode(self, chunks, **kwargs):
            self.seen.extend(chunks)
            return np.tile([1.0, 0.0], (len(chunks), 1))

    encoder = Encoder()
    monkeypatch.setattr(emb, "_load_text_model", lambda *_: (encoder, {}))
    _, metadata = emb.embed_texts(["1 2 3 4 5"], {"device": "cpu"})
    assert encoder.seen == ["prefix 1 2", "prefix 3 4", "prefix 5"]
    assert metadata["chunk_count"] == 3
