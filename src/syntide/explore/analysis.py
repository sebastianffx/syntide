"""Explore recording embeddings without using labels to fit the clusters.

All signal distances and clustering use L2-normalized *original* embeddings. PCA
and UMAP are views only. Text novelty is a separate cosine-distance calculation
over distinct, non-empty sentence strings; repeated QA boilerplate is not a set
of independent neighbours. The returned object can be written as strict JSON.
"""

import math
from collections import Counter, defaultdict
from collections.abc import Mapping

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def analyze(
    embeddings: np.ndarray,
    rows: list[dict],
    population_rows: list[dict],
    signals: list[dict],
    config: dict,
    text_embeddings: np.ndarray | None = None,
    text_rows: list[dict] | None = None,
) -> dict:
    """Build cluster, label, waveform and sentence summaries for one sample.

    ``rows`` and ``embeddings`` have matching order, one row per recording.
    Signals are matched by ``record_id``. Text rows have ``record_id`` and
    ``text`` plus any task/question/answer provenance. All original text entries
    are retained, even when their embedding or novelty cannot be computed.

    ``target_fraction`` refers to the fixed selected population, not the sample
    and not a hypothetical population after generation. Reported deficits are
    independent per-label targets, so multi-label deficits must not be summed.
    """
    cfg = _configuration(config)
    x = _matrix(embeddings, len(rows), "signal embeddings")
    ids = [str(row["record_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Sample rows must contain one row per distinct recording")
    population = _population(population_rows)
    if set(ids) - set(population):
        raise ValueError("Every sampled recording must belong to the selected population")
    signal_of = {str(signal["record_id"]): signal for signal in signals}
    if len(signal_of) != len(signals):
        raise ValueError("Signals must contain one entry per distinct recording")
    if set(ids) - set(signal_of):
        raise ValueError("Every sampled recording requires a signal for statistics and preview")

    x = normalize(x) if len(x) else x
    pca, umap, projection_info = _projections(x, cfg)
    labels, effective_clusters = _clusters(x, cfg)
    novelty, neighbour_count = _knn(x, cfg["metric_neighbors"], "euclidean")
    silhouette, silhouette_info = _silhouettes(x, labels, cfg)
    text_of, text_summary = _texts(rows, text_embeddings, text_rows, cfg)
    clusters, centroid_distances = _cluster_summaries(x, labels, rows, novelty, silhouette)
    records = []
    for index, row in enumerate(rows):
        record_id = ids[index]
        stats, waveform = _signal_summary(signal_of[record_id], cfg["waveform_points"])
        texts = text_of[record_id]
        scores = list(
            {
                _sentence(text.get("embedding_text", text.get("text"))): text["novelty"]
                for text in texts
                if text.get("novelty") is not None
            }.values()
        )
        cluster = int(labels[index])
        records.append(
            {
                "id": record_id,
                "record_id": record_id,
                "subject_uid": row.get("subject_uid"),
                "labels": sorted(set(row.get("labels") or [])),
                "metadata": row.get("metadata") or {},
                "texts": texts,
                "coordinates": {"pca": pca[index], "umap": umap[index] if umap is not None else None},
                "cluster": cluster,
                "stats": stats,
                "waveform": waveform,
                "signal_novelty": novelty[index],
                "density_radius": novelty[index],
                "text_novelty": max(scores) if scores else None,
                "text_novelty_mean": float(np.mean(scores)) if scores else None,
                "distance_to_centroid": centroid_distances[index],
                "silhouette": silhouette[index],
            }
        )
    label_summary = _label_summary(rows, list(population.values()), cfg)
    summary = {
        "sample_records": len(rows),
        "population_records": len(population),
        "sample_patients": len(_patients(rows)),
        "population_patients": len(_patients(population.values())),
        "sample_missing_patient_ids": sum(not _patients([row]) for row in rows),
        "population_missing_patient_ids": sum(not _patients([row]) for row in population.values()),
        "embedding_dimensions": x.shape[1],
        "zero_norm_embeddings": int(np.count_nonzero(~np.any(x, axis=1))),
        "clusters": effective_clusters,
        "metric_neighbors": neighbour_count,
        "signal_distance": "euclidean on L2-normalized original embeddings",
        "signal_novelty_definition": "mean distance to k other sampled recording embeddings",
        "density_definition": "mean k-neighbour distance (larger radius means lower local density)",
        "text_novelty_definition": "maximum sentence novelty per recording; distinct-text cosine neighbours",
        "label_purity_definition": "fraction of records having the most common complete label set",
        "label_entropy_definition": "mean binary label-presence entropy across labels present in the sample",
        "target_fraction_denominator": "selected population records before generation",
        "cluster_metrics_scope": "sampled recordings; population label counts are reported separately",
        "projection": projection_info,
        "silhouette": silhouette_info,
        "text": text_summary,
    }
    return _json_value(
        {
            "records": records,
            "clusters": clusters,
            "label_summary": label_summary,
            "summary": summary,
            "config": cfg,
        }
    )


def _configuration(config: dict) -> dict:
    cfg = {
        "projection": "umap",
        "n_clusters": 8,
        "seed": 0,
        "n_neighbors": 15,
        "umap_min_dist": 0.1,
        "metric_neighbors": 10,
        "waveform_points": 1000,
        "silhouette_max_samples": 2000,
        "target_count": None,
        "target_fraction": None,
        **config,
    }
    if "clusters" in config and "n_clusters" not in config:
        cfg["n_clusters"] = config["clusters"]
    if cfg["projection"] not in {"umap", "pca", "both"}:
        raise ValueError("projection must be 'umap', 'both', or 'pca'")
    for name in (
        "n_clusters",
        "n_neighbors",
        "metric_neighbors",
        "waveform_points",
        "silhouette_max_samples",
    ):
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if cfg["n_neighbors"] < 2:
        raise ValueError("n_neighbors must be at least 2 for UMAP")
    cfg["waveform_points"] = min(1000, cfg["waveform_points"])
    if not math.isfinite(float(cfg["umap_min_dist"])) or not 0 <= cfg["umap_min_dist"] <= 1:
        raise ValueError("umap_min_dist must be between 0 and 1")
    for name in ("target_count", "target_fraction"):
        target = cfg[name]
        targets = target.values() if isinstance(target, Mapping) else [target]
        for value in targets:
            if value is None:
                continue
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            if name == "target_count" and int(value) != value:
                raise ValueError("target_count must be an integer")
            if name == "target_fraction" and value > 1:
                raise ValueError("target_fraction must be between 0 and 1")
    return cfg


def _matrix(values: np.ndarray, count: int, name: str) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != count or x.shape[1] < 1:
        raise ValueError(f"{name} must be a [rows, dimensions] matrix matching its catalog rows")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contain NaN or infinite values")
    return x


def _population(rows: list[dict]) -> dict[str, dict]:
    """Count a recording once even if a task-oriented catalog repeats it."""
    result = {}
    for row in rows:
        record_id = str(row["record_id"])
        if record_id not in result:
            result[record_id] = {**row, "labels": set(row.get("labels") or [])}
        else:
            previous = result[record_id]
            if previous.get("subject_uid") != row.get("subject_uid"):
                raise ValueError(f"Conflicting patient ids for recording {record_id!r}")
            previous["labels"].update(row.get("labels") or [])
    return result


def _projections(x: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray | None, dict]:
    pca = np.zeros((len(x), 2))
    variance = []
    if len(x) > 1 and np.any(x != x[0]):
        components = min(2, *x.shape)
        model = PCA(n_components=components, svd_solver="full")
        pca[:, :components] = model.fit_transform(x)
        variance = model.explained_variance_ratio_.tolist()
    info = {
        "requested": cfg["projection"],
        "pca_explained_variance_ratio": variance,
        "umap_status": "not_requested",
        "metrics_use_projection": False,
    }
    if cfg["projection"] == "pca":
        return pca, None, info
    try:
        from umap import UMAP
    except ImportError as error:
        raise ImportError(
            "UMAP requires umap-learn. Install the explorer dependencies or select --projection pca. "
            "No projection fallback was applied."
        ) from error
    # UMAP cannot form a useful neighbour graph with fewer than three distinct
    # vectors. Keep that view unavailable, with an explicit status, rather than
    # disguising PCA coordinates as UMAP or inventing separated duplicate points.
    unique, inverse = np.unique(x, axis=0, return_inverse=True)
    if len(unique) < 3:
        info["umap_status"] = "unavailable_fewer_than_three_distinct_embeddings"
        return pca, None, info
    neighbours = min(cfg["n_neighbors"], len(unique) - 1)
    coordinates = UMAP(
        n_components=2,
        n_neighbors=neighbours,
        min_dist=cfg["umap_min_dist"],
        metric="euclidean",
        random_state=cfg["seed"],
        transform_seed=cfg["seed"],
        n_jobs=1,
        init="random",
    ).fit_transform(unique)
    if not np.isfinite(coordinates).all():
        raise ValueError("UMAP returned non-finite coordinates; choose PCA or adjust UMAP parameters")
    info.update(umap_status="available", umap_neighbors=neighbours, umap_unique_vectors=len(unique))
    return pca, coordinates[inverse], info


def _clusters(x: np.ndarray, cfg: dict) -> tuple[np.ndarray, int]:
    if not len(x):
        return np.array([], dtype=int), 0
    count = min(cfg["n_clusters"], len(np.unique(x, axis=0)))
    if count == 1:
        return np.zeros(len(x), dtype=int), 1
    labels = KMeans(n_clusters=count, random_state=cfg["seed"], n_init=10).fit_predict(x)
    # Stable, readable ids: the cluster containing the first catalog row is 0.
    mapping = {label: index for index, label in enumerate(dict.fromkeys(labels))}
    return np.array([mapping[label] for label in labels]), len(mapping)


def _knn(x: np.ndarray, neighbours: int, metric: str) -> tuple[list[float | None], int]:
    if len(x) < 2:
        return [None] * len(x), 0
    k = min(neighbours, len(x) - 1)
    model = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(x)
    distances, indices = model.kneighbors(x)
    # Explicit index exclusion handles ties between duplicate vectors. Dropping
    # column zero would occasionally discard a duplicate while retaining self.
    scores = [
        float(np.mean(distance[index != i][:k]))
        for i, (distance, index) in enumerate(zip(distances, indices, strict=True))
    ]
    return scores, k


def _silhouettes(x: np.ndarray, labels: np.ndarray, cfg: dict) -> tuple[list, dict]:
    scores = [None] * len(x)
    info = {
        "mean": None,
        "evaluated_records": 0,
        "sampled": False,
        "status": "unavailable_requires_2_to_n_minus_1_clusters",
    }
    if not 1 < len(set(labels)) < len(x):
        return scores, info
    indices = np.arange(len(x))
    if len(x) > cfg["silhouette_max_samples"]:
        indices = np.sort(
            np.random.default_rng(cfg["seed"]).choice(len(x), cfg["silhouette_max_samples"], replace=False)
        )
        info["sampled"] = True
    if not 1 < len(set(labels[indices])) < len(indices):
        info["status"] = "unavailable_insufficient_clusters_in_metric_subsample"
        return scores, info
    values = silhouette_samples(x[indices], labels[indices], metric="euclidean")
    for index, value in zip(indices, values, strict=True):
        scores[index] = float(value)
    info.update(mean=float(values.mean()), evaluated_records=len(indices), status="available")
    return scores, info


def _cluster_summaries(x, assignments, rows, novelty, silhouette):
    all_labels = sorted({label for row in rows for label in row.get("labels") or []})
    cluster_ids = sorted(set(assignments))
    centroids = np.array([x[assignments == cluster].mean(axis=0) for cluster in cluster_ids])
    distances = pairwise_distances(centroids) if len(centroids) else np.empty((0, 0))
    result, record_distances = [], np.zeros(len(x))
    for cluster in cluster_ids:
        indices = np.flatnonzero(assignments == cluster)
        members = [rows[index] for index in indices]
        count = len(indices)
        counts = Counter(label for row in members for label in set(row.get("labels") or []))
        sets = Counter(tuple(sorted(set(row.get("labels") or []))) for row in members)
        entropy = []
        for label in all_labels:
            p = counts[label] / count
            entropy.append(-p * math.log2(p) - (1 - p) * math.log2(1 - p) if 0 < p < 1 else 0.0)
        intra = np.linalg.norm(x[indices] - centroids[cluster], axis=1)
        record_distances[indices] = intra
        other = [(int(c), float(distances[cluster, c])) for c in cluster_ids if c != cluster]
        other.sort(key=lambda item: (item[1], item[0]))
        scores = [silhouette[i] for i in indices if silhouette[i] is not None]
        novelties = [novelty[i] for i in indices if novelty[i] is not None]
        result.append(
            {
                "id": int(cluster),
                "cluster": int(cluster),
                "size": count,
                "patients": len(_patients(members)),
                "centroid": centroids[cluster],
                "label_counts": dict(sorted(counts.items())),
                "label_fractions": {label: counts[label] / count for label in sorted(counts)},
                "label_purity": max(sets.values()) / count if counts else None,
                "label_entropy": float(np.mean(entropy)) if entropy else None,
                "label_sets": [
                    {"labels": list(labels), "count": n}
                    for labels, n in sorted(sets.items(), key=lambda item: (-item[1], item[0]))
                ],
                "intra_distance": float(intra.mean()),
                "intra_distance_mean": float(intra.mean()),
                "intra_distance_std": float(intra.std()),
                "intra_distance_max": float(intra.max()),
                "intra_pairwise_rms_distance": float(np.sqrt(2 * np.sum(intra**2) / (count - 1)))
                if count > 1
                else None,
                "nearest_cluster": other[0][0] if other else None,
                "nearest_cluster_id": other[0][0] if other else None,
                "nearest_cluster_distance": other[0][1] if other else None,
                "centroid_distances": [{"cluster": c, "distance": d} for c, d in other],
                "silhouette": float(np.mean(scores)) if scores else None,
                "silhouette_evaluated_records": len(scores),
                "signal_novelty_mean": float(np.mean(novelties)) if novelties else None,
            }
        )
    return result, record_distances


def _patients(rows) -> set:
    return {
        str(row["subject_uid"])
        for row in rows
        if row.get("subject_uid") is not None and str(row["subject_uid"]).strip()
    }


def _label_summary(sample, population, cfg):
    labels = {label for row in population for label in row.get("labels") or []}
    for key in ("target_count", "target_fraction"):
        if isinstance(cfg[key], Mapping):
            labels.update(cfg[key])
    population_counts = Counter(label for row in population for label in set(row.get("labels") or []))
    largest = max(population_counts.values(), default=0)
    result = []
    for label in sorted(labels):
        pop_members = [row for row in population if label in (row.get("labels") or [])]
        sample_members = [row for row in sample if label in (row.get("labels") or [])]
        count, sample_count = len(pop_members), len(sample_members)
        target_count = _target(cfg["target_count"], label)
        target_fraction = _target(cfg["target_fraction"], label)
        fraction_count = math.ceil(target_fraction * len(population)) if target_fraction is not None else None
        targets = [n for n in (target_count, fraction_count) if n is not None]
        target = max(targets) if targets else None
        result.append(
            {
                "label": label,
                "population_count": count,
                "sample_count": sample_count,
                "population_patients": len(_patients(pop_members)),
                "sample_patients": len(_patients(sample_members)),
                "population_fraction": count / len(population) if population else 0.0,
                "sample_fraction": sample_count / len(sample) if sample else 0.0,
                "relative_count": count / largest if largest else None,
                "relative_imbalance": 1 - count / largest if largest else None,
                "relative_deficit": largest - count,
                "target_count": target_count,
                "target_fraction": target_fraction,
                "target_fraction_count": fraction_count,
                "effective_target_count": target,
                "deficit": max(0, target - count) if target is not None else None,
            }
        )
    return result


def _target(value, label):
    return value.get(label) if isinstance(value, Mapping) else value


def _signal_summary(record, maximum):
    signal = np.asarray(record["signal"], dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] < 1 or signal.shape[1] < 1:
        raise ValueError("Signals must have non-empty [channels, time] shape")
    sample_rate = float(record["sample_rate"])
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Signal sample_rate must be positive and finite")
    channels = record.get("channels") or [f"channel_{i}" for i in range(len(signal))]
    if len(channels) != len(signal):
        raise ValueError("Signal channel names must match the channel dimension")
    units = record.get("units")
    if isinstance(units, str) or units is None:
        units = [units] * len(signal)
    if len(units) != len(signal):
        raise ValueError("Signal units must match the channel dimension")
    channel_stats = []
    for name, unit, values in zip(channels, units, signal, strict=True):
        finite = values[np.isfinite(values)]
        stats = {
            "name": name,
            "unit": unit,
            "missing_fraction": 1 - len(finite) / len(values),
            "finite_samples": len(finite),
        }
        stats.update(
            {
                "mean": float(finite.mean()),
                "std": float(finite.std()),
                "rms": float(np.sqrt(np.mean(finite**2))),
                "min": float(finite.min()),
                "max": float(finite.max()),
                "ptp": float(np.ptp(finite)),
            }
            if len(finite)
            else {key: None for key in ("mean", "std", "rms", "min", "max", "ptp")}
        )
        channel_stats.append(stats)
    indices = np.linspace(0, signal.shape[1] - 1, min(maximum, signal.shape[1]), dtype=int)
    stats = {
        "channels": channel_stats,
        "channel_count": len(signal),
        "samples": signal.shape[1],
        "sample_rate": sample_rate,
        "duration_seconds": signal.shape[1] / sample_rate,
        "missing_fraction": float((~np.isfinite(signal)).mean()),
    }
    waveform = {
        "times": indices / sample_rate,
        "channels": channels,
        "units": units,
        "values": signal[:, indices],
        "sample_rate": sample_rate,
        "original_samples": signal.shape[1],
        "preview_samples": len(indices),
        "method": "uniform_sample",
        "decimated": len(indices) < signal.shape[1],
    }
    return stats, waveform


def _sentence(value):
    return " ".join(str(value or "").split())


def _texts(rows, embeddings, text_rows, cfg):
    source = text_rows or []
    if embeddings is not None:
        x = _matrix(embeddings, len(source), "text embeddings")
    elif source:
        raise ValueError("text_rows require matching text_embeddings")
    else:
        x = np.empty((0, 1))
    # Distinct sentences, with duplicates grouped before nearest-neighbour search.
    groups = defaultdict(list)
    for index, text in enumerate(source):
        sentence = _sentence(text.get("text"))
        if sentence:
            groups[sentence].append(index)
    unique_sentences, vectors = [], []
    details = {}
    for sentence, indices in groups.items():
        vector = x[indices[0]]
        details[sentence] = {
            "duplicate_count": len(indices),
            "duplicate_record_count": len({str(source[i]["record_id"]) for i in indices}),
            "novelty": None,
            "novelty_status": "zero_norm_embedding" if not np.any(vector) else "insufficient_distinct_texts",
        }
        if np.any(vector):
            unique_sentences.append(sentence)
            vectors.append(vector)
    neighbour_count = 0
    if vectors:
        distances, neighbour_count = _knn(normalize(np.array(vectors)), cfg["metric_neighbors"], "cosine")
        for sentence, distance in zip(unique_sentences, distances, strict=True):
            details[sentence].update(
                novelty=distance,
                novelty_status="available" if distance is not None else "insufficient_distinct_texts",
            )
    text_of = {}
    by_record = defaultdict(list)
    for entry in source:
        by_record[str(entry["record_id"])].append(entry)
    for row in rows:
        record_id = str(row["record_id"])
        original = row.get("texts") or []
        # Preserve all source entries. If rows already carry the same texts,
        # enrich those rather than duplicating them a second time.
        entries = [dict(item) if isinstance(item, dict) else {"text": str(item)} for item in original]
        represented = defaultdict(list)
        ids = defaultdict(list)
        for index, entry in enumerate(entries):
            represented[_sentence(entry.get("text"))].append(index)
            if entry.get("text_id"):
                ids[entry["text_id"]].append(index)
        matched = set()
        for entry in by_record[record_id]:
            # QA embeddings may combine fields while the original text is
            # empty. Match tuple identity before sentence contents so a QA
            # tuple remains one tuple. Keep both text representations explicit.
            source_text = entry.get("source_text", entry.get("text", ""))
            embedded = {**entry, "text": source_text, "embedding_text": entry.get("text", "")}
            candidates = ids[entry["text_id"]] if entry.get("text_id") else []
            candidates = [index for index in candidates if index not in matched]
            if not candidates:
                candidates = [
                    index
                    for index in represented[_sentence(source_text)]
                    if index not in matched
                    and (not entry.get("text_id") or not entries[index].get("text_id"))
                ]
            if candidates:
                index = candidates[0]
                matched.add(index)
                entries[index] = {**embedded, **entries[index], "embedding_text": embedded["embedding_text"]}
            else:
                entries.append(embedded)
        for entry in entries:
            sentence = _sentence(entry.get("embedding_text", entry.get("text")))
            entry.update(details.get(sentence, {"novelty": None, "novelty_status": "not_embedded"}))
        text_of[record_id] = entries
    return text_of, {
        "embedded_text_rows": len(source),
        "unique_sentences": len(groups),
        "valid_unique_sentences": len(vectors),
        "metric_neighbors": neighbour_count,
        "scope": "provided text rows",
        "distance": "cosine over distinct non-empty sentence strings",
        "duplicates_collapsed_for_metrics": sum(len(indices) - 1 for indices in groups.values()),
    }


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return value
