import builtins
import json

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from syntide.explore.analysis import analyze


def catalog(n, labels=None):
    return [
        {
            "record_id": f"r{i}",
            "subject_uid": f"p{i // 2}",
            "labels": labels[i] if labels else ["common"],
            "metadata": {"fold": 1},
            "texts": [],
        }
        for i in range(n)
    ]


def signals(rows, samples=32):
    return [
        {
            "record_id": row["record_id"],
            "signal": np.stack([np.arange(samples), -np.arange(samples)]),
            "sample_rate": 4,
            "channels": ["I", "II"],
            "units": "mV",
        }
        for row in rows
    ]


def run(x, rows=None, population=None, config=None, **kwargs):
    rows = rows if rows is not None else catalog(len(x))
    return analyze(
        np.asarray(x),
        rows,
        population if population is not None else rows,
        signals(rows),
        {
            "projection": "pca",
            "n_clusters": 2,
            "metric_neighbors": 1,
            **(config or {}),
        },
        **kwargs,
    )


def test_clusters_and_distances_use_normalized_embeddings_without_labels():
    x = np.array([[1, 0], [1, 0.05], [-1, 0], [-1, 0.05]])
    rows = catalog(4, [["a", "b"], ["a", "b"], ["b"], ["b"]])
    result = run(x, rows)
    rescaled = run(
        x * np.arange(1, 5)[:, None], catalog(4, [["q"], [], ["r"], ["s"]])
    )
    assert [r["cluster"] for r in result["records"]] == [0, 0, 1, 1]
    assert [r["cluster"] for r in rescaled["records"]] == [0, 0, 1, 1]
    normalized = x / np.linalg.norm(x, axis=1, keepdims=True)
    expected = np.linalg.norm(
        normalized[:2].mean(axis=0) - normalized[2:].mean(axis=0)
    )
    assert result["clusters"][0]["nearest_cluster_distance"] == pytest.approx(
        expected
    )
    assert result["clusters"][0]["label_purity"] == 1
    assert (
        result["clusters"][0]["label_entropy"] == 0
    )  # Consistent co-occurrence is pure.
    assert result["summary"]["silhouette"]["mean"] > 0.95
    for first, second in zip(
        result["records"], rescaled["records"], strict=True
    ):
        assert first["signal_novelty"] == pytest.approx(
            second["signal_novelty"]
        )


def test_multilabel_binary_entropy_and_labelset_purity():
    result = run([[1, 0], [1, 0]], catalog(2, [["a", "b"], ["a"]]))
    assert result["clusters"][0]["label_entropy"] == pytest.approx(0.5)
    assert result["clusters"][0]["label_purity"] == 0.5
    assert result["clusters"][0]["label_fractions"] == {"a": 1, "b": 0.5}


def test_population_counts_use_distinct_recordings_and_patients():
    population = catalog(
        6,
        [["common"], ["common"], ["common"], ["rare"], ["common"], ["common"]],
    )
    population += [
        dict(population[3])
    ]  # Duplicate QA task must not inflate counts.
    rows = [*population[:2], population[3]]
    result = run(
        [[1, 0], [1, 1], [0, 1]],
        rows,
        population,
        {
            "target_count": {"rare": 4, "absent": 2},
            "target_fraction": {"rare": 0.75},
        },
    )
    summary = {row["label"]: row for row in result["label_summary"]}
    assert result["summary"]["population_records"] == 6
    assert result["summary"]["population_patients"] == 3
    assert summary["common"]["population_count"] == 5
    assert summary["common"]["sample_count"] == 2
    assert summary["common"]["sample_patients"] == 1
    assert summary["rare"]["relative_imbalance"] == pytest.approx(0.8)
    assert (
        summary["rare"]["target_fraction_count"] == 5
    )  # Full population denominator.
    assert summary["rare"]["effective_target_count"] == 5
    assert summary["rare"]["deficit"] == 4
    assert summary["absent"]["population_count"] == 0
    assert summary["absent"]["deficit"] == 2


def test_density_excludes_self_even_with_duplicate_vectors():
    result = run([[1, 0], [1, 0], [0, 1]])
    assert [r["signal_novelty"] for r in result["records"]] == pytest.approx(
        [0, 0, np.sqrt(2)]
    )
    cluster = result["clusters"][0]
    assert cluster["intra_pairwise_rms_distance"] == 0


def test_pairwise_rms_is_exact_without_pairwise_matrix():
    x = np.array([[1, 0], [0, 1], [-1, 0]])
    result = run(x, config={"n_clusters": 1})
    d = pairwise_distances(x)
    expected = np.sqrt(np.square(d[np.triu_indices(3, k=1)]).mean())
    assert result["clusters"][0][
        "intra_pairwise_rms_distance"
    ] == pytest.approx(expected)


def test_sentence_novelty_collapses_duplicates_but_retains_qa_provenance():
    rows = catalog(3)
    text_rows = [
        {
            "record_id": "r0",
            "text": "normal rhythm",
            "task_id": "a",
            "question": "Rhythm?",
            "answer": "normal",
        },
        {
            "record_id": "r1",
            "text": "normal rhythm",
            "task_id": "b",
            "flags": ["boilerplate"],
        },
        {"record_id": "r1", "text": "sinus rhythm", "task_id": "c"},
        {"record_id": "r2", "text": "extreme tachycardia", "task_id": "d"},
    ]
    rows[0]["texts"] = [
        dict(text_rows[0]),
        {"text": "unembedded sentence", "flags": ["excluded"]},
    ]
    vectors = np.array([[1, 0], [1, 0], [1, 0.05], [-1, 0]])
    result = run(np.eye(3), rows, text_embeddings=vectors, text_rows=text_rows)
    first, second, rare = result["records"]
    assert len(first["texts"]) == 2
    assert first["texts"][0]["task_id"] == "a"
    assert first["texts"][0]["answer"] == "normal"
    assert first["texts"][0]["duplicate_record_count"] == 2
    assert first["texts"][1]["novelty_status"] == "not_embedded"
    assert first["texts"][1]["flags"] == ["excluded"]
    assert second["texts"][0]["flags"] == ["boilerplate"]
    assert (
        first["text_novelty"] > 0
    )  # Duplicate text is not its nearest distinct neighbour.
    assert rare["text_novelty"] > 1.9
    assert first["text_novelty"] < 0.01
    assert result["summary"]["text"]["unique_sentences"] == 3
    assert result["summary"]["text"]["duplicates_collapsed_for_metrics"] == 1


def test_only_repeated_text_and_zero_embeddings_have_no_novelty():
    text_rows = [
        {"record_id": "r0", "text": "same"},
        {"record_id": "r1", "text": "same"},
        {"record_id": "r1", "text": "empty vector"},
    ]
    result = run(
        [[1, 0], [0, 1]],
        text_rows=text_rows,
        text_embeddings=np.array([[1, 0], [1, 0], [0, 0]]),
    )
    assert all(record["text_novelty"] is None for record in result["records"])
    assert (
        result["records"][1]["texts"][1]["novelty_status"]
        == "zero_norm_embedding"
    )


def test_qa_combined_embedding_fields_keep_one_tuple_per_text_id():
    from syntide.explore.pipeline import text_rows_for

    rows = catalog(2)
    rows[0]["texts"] = [
        {
            "text_id": "q1",
            "text": "",
            "question": "Rhythm?",
            "answer": "sinus",
            "rationale": "regular",
        },
        {
            "text_id": "q2",
            "text": "",
            "question": "Diagnosis?",
            "answer": "rare",
            "flags": ["review"],
        },
    ]
    rows[1]["texts"] = [
        {
            "text_id": "q3",
            "text": "recording note",
            "question": "Rhythm?",
            "answer": "sinus",
        },
    ]
    embedded_rows = text_rows_for(rows, ["question", "answer", "text"])
    vectors = np.array([[1, 0], [-1, 0], [1, 0.01]])
    result = run(
        [[1, 0], [0, 1]], rows, text_rows=embedded_rows, text_embeddings=vectors
    )
    first = result["records"][0]
    assert [text["text_id"] for text in first["texts"]] == ["q1", "q2"]
    assert first["texts"][0]["text"] == ""
    assert first["texts"][0]["source_text"] == ""
    assert first["texts"][0]["embedding_text"] == "Rhythm?\nsinus"
    assert first["texts"][0]["rationale"] == "regular"
    assert first["texts"][1]["flags"] == ["review"]
    assert all(text["novelty_status"] == "available" for text in first["texts"])
    assert first["text_novelty_mean"] == pytest.approx(
        np.mean([t["novelty"] for t in first["texts"]])
    )
    second = result["records"][1]
    assert len(second["texts"]) == 1
    assert second["texts"][0]["text"] == "recording note"
    assert (
        second["texts"][0]["embedding_text"] == "Rhythm?\nsinus\nrecording note"
    )


def test_waveform_statistics_use_full_signal_and_json_is_finite():
    rows = catalog(1)
    signal = np.stack([np.arange(2500, dtype=float), np.full(2500, np.nan)])
    signal[0, 10] = np.inf
    result = analyze(
        np.ones((1, 3)),
        rows,
        rows,
        [
            {
                "record_id": "r0",
                "signal": signal,
                "sample_rate": 100,
                "channels": ["I", "II"],
                "units": ["mV", "mV"],
            }
        ],
        {"projection": "pca"},
    )
    record = result["records"][0]
    valid = signal[0][np.isfinite(signal[0])]
    first, second = record["stats"]["channels"]
    assert first["mean"] == pytest.approx(valid.mean())
    assert first["rms"] == pytest.approx(np.sqrt(np.mean(valid**2)))
    assert first["missing_fraction"] == pytest.approx(1 / 2500)
    assert second["mean"] is None and second["missing_fraction"] == 1
    assert record["stats"]["duration_seconds"] == 25
    assert len(record["waveform"]["times"]) == 1000
    assert record["waveform"]["times"][-1] == 24.99
    assert record["waveform"]["values"][1] == [None] * 1000
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("n", [0, 1, 2, 7])
def test_empty_tiny_and_identical_inputs_are_deterministic(n):
    result = run(np.ones((n, 4)))
    assert result == run(np.ones((n, 4)))
    assert result["summary"]["clusters"] == min(n, 1)
    assert result["summary"]["silhouette"]["mean"] is None
    assert all(
        record["coordinates"]["pca"] == [0, 0] for record in result["records"]
    )
    json.dumps(result, allow_nan=False)


def test_silhouette_subsample_is_disclosed_and_reproducible():
    rng = np.random.default_rng(3)
    x = np.concatenate(
        [
            rng.normal([1, 0], 0.05, size=(50, 2)),
            rng.normal([-1, 0], 0.05, size=(50, 2)),
        ]
    )
    config = {"silhouette_max_samples": 20}
    result = run(x, config=config)
    assert result["summary"]["silhouette"]["sampled"] is True
    assert result["summary"]["silhouette"]["evaluated_records"] == 20
    assert sum(r["silhouette"] is not None for r in result["records"]) == 20
    assert result == run(x, config=config)


def test_umap_dependency_is_explicit_and_no_silent_fallback(monkeypatch):
    original_import = builtins.__import__

    def no_umap(name, *args, **kwargs):
        if name == "umap":
            raise ImportError("test dependency unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_umap)
    with pytest.raises(ImportError, match="No projection fallback"):
        run(np.eye(3), config={"projection": "umap"})


def test_umap_handles_small_and_duplicate_data_if_installed():
    pytest.importorskip("umap")
    tiny = run([[1, 0], [0, 1]], config={"projection": "umap"})
    assert tiny["summary"]["projection"]["umap_status"].startswith(
        "unavailable_"
    )
    assert tiny["records"][0]["coordinates"]["umap"] is None
    result = run(
        [[1, 0], [0, 1], [-1, 0], [1, 0]], config={"projection": "umap"}
    )
    assert result["summary"]["projection"]["umap_status"] == "available"
    assert (
        result["records"][0]["coordinates"]["umap"]
        == result["records"][3]["coordinates"]["umap"]
    )
    assert result == run(
        [[1, 0], [0, 1], [-1, 0], [1, 0]], config={"projection": "umap"}
    )


@pytest.mark.parametrize(
    "config, match",
    [
        ({"n_clusters": 0}, "positive integer"),
        ({"metric_neighbors": 1.5}, "positive integer"),
        ({"target_fraction": 1.1}, "between 0 and 1"),
        ({"target_count": -1}, "non-negative"),
        ({"target_count": 1.5}, "integer"),
        ({"projection": "secret_fallback"}, "projection must"),
    ],
)
def test_invalid_configuration_rejected(config, match):
    with pytest.raises(ValueError, match=match):
        run([[1, 0]], config=config)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="NaN"):
        run([[np.nan, 1]])
    with pytest.raises(ValueError, match="distinct recording"):
        run([[1, 0], [0, 1]], [catalog(1)[0]] * 2)
    with pytest.raises(ValueError, match="selected population"):
        run([[1, 0]], population=[])
    with pytest.raises(ValueError, match="matching text_embeddings"):
        run([[1, 0]], text_rows=[{"record_id": "r0", "text": "example"}])
