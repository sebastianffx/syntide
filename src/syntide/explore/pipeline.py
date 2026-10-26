"""Run reproducible sampling, vector encoding, and dataset analysis."""

import contextlib
import json
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from syntide.explore.plans import (
    content_id,
    seal_analysis,
    verify_analysis,
    write_frozen,
)

log = logging.getLogger(__name__)


def sample_rows(
    rows: list[dict], size: int | None, fraction: float | None, seed: int
) -> list[dict]:
    if not rows:
        raise ValueError("The selected training population is empty")
    if size is not None and (isinstance(size, bool) or size <= 0):
        raise ValueError("sample must be positive")
    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1]")
    if size is not None and fraction is not None:
        raise ValueError("Specify a sample count or fraction, not both")
    count = min(len(rows), size) if size is not None else len(rows)
    if fraction is not None:
        count = max(1, int(np.ceil(len(rows) * fraction)))
    ordered = sorted(rows, key=lambda row: row["record_id"])
    indices = sorted(
        np.random.default_rng(seed)
        .choice(len(rows), size=count, replace=False)
        .tolist()
    )
    return [ordered[i] for i in indices]


def text_rows_for(rows: list[dict], fields: list[str]) -> list[dict]:
    result = []
    for row in rows:
        for item in row.get("texts", []):
            values = []
            for field in fields:
                value = item.get(field, item.get("metadata", {}).get(field, ""))
                if value and str(value).strip() not in values:
                    values.append(str(value).strip())
            if values:
                result.append(
                    {
                        **item,
                        "record_id": row["record_id"],
                        "source_text": item.get("text", ""),
                        "text": "\n".join(values),
                        "embedding_fields": fields,
                    }
                )
    return result


def run_analysis(config: dict, output: Path, *, reuse: bool = False) -> dict:
    from syntide.explore.analysis import analyze
    from syntide.explore.data import open_source
    from syntide.explore.embeddings import embed_recordings, embed_texts
    from syntide.explore.viewer import write_viewer

    output = Path(output)
    source = open_source(config["source"])
    sampling = config["sampling"]
    rows = sample_rows(
        source.rows,
        sampling.get("size"),
        sampling.get("fraction"),
        sampling["seed"],
    )
    signature = content_id(
        {
            "config": config,
            "source": source.provenance,
            "ids": [r["record_id"] for r in rows],
        }
    )
    analysis_path = output / "analysis.json"
    if analysis_path.exists():
        existing = json.loads(analysis_path.read_text())
        verify_analysis(existing)
        if existing["provenance"]["request_id"] != signature:
            raise FileExistsError(
                f"{output} holds a different analysis; choose a new output "
                "directory"
            )
        if not reuse:
            raise FileExistsError(
                f"{output} already exists; pass --reuse to use its frozen "
                "embeddings"
            )
        write_viewer(existing, output / "index.html")
        log.info(
            (
                "Reusing frozen analysis %s (including its resolved model "
                "versions)"
            ),
            existing["analysis_id"],
        )
        return existing
    log.info("Loading %d of %d recording(s)", len(rows), len(source.rows))
    signals = source.load([row["record_id"] for row in rows])
    if [item["record_id"] for item in signals] != [
        row["record_id"] for row in rows
    ]:
        raise ValueError("Source returned signals out of catalog order")
    log.info("Embedding signals with %s", config["embedding"]["backend"])
    embeddings, embedding_provenance = embed_recordings(
        [item["signal"] for item in signals],
        [item["sample_rate"] for item in signals],
        [item["channels"] for item in signals],
        config["embedding"],
    )
    texts = text_rows_for(rows, config["text"]["fields"])
    text_embeddings, text_provenance = None, None
    if texts and config["text"].get("enabled", True):
        # Embed duplicate sentences once; analysis still counts each linked
        # task.
        unique = list(dict.fromkeys(item["text"] for item in texts))
        log.info(
            "Embedding %d distinct texts (%d linked tuples)",
            len(unique),
            len(texts),
        )
        vectors, text_provenance = embed_texts(unique, config["text"])
        index = {text: i for i, text in enumerate(unique)}
        text_embeddings = np.stack(
            [vectors[index[item["text"]]] for item in texts]
        )
        text_provenance = {
            **text_provenance,
            "fields": config["text"]["fields"],
            "distinct_texts": len(unique),
            "tuple_count": len(texts),
        }
    log.info("Clustering, measuring coverage and preparing the linked explorer")
    result = analyze(
        embeddings,
        rows,
        source.rows,
        signals,
        config["analysis"],
        text_embeddings,
        texts if text_embeddings is not None else None,
    )
    packages = {}
    for name in (
        "syntide",
        "numpy",
        "scikit-learn",
        "torch",
        "chronos-forecasting",
        "umap-learn",
        "sentence-transformers",
    ):
        with contextlib.suppress(PackageNotFoundError):
            packages[name] = version(name)
    result.update(
        {
            "schema_version": "syntide.coverage-analysis@1",
            "provenance": {
                "request_id": signature,
                "source": source.provenance,
                "embedding": embedding_provenance,
                "text_embedding": text_provenance,
                "sampling": {
                    **sampling,
                    "method": "uniform_without_replacement_by_recording",
                    "population_count": len(source.rows),
                    "sample_count": len(rows),
                    "record_ids": [row["record_id"] for row in rows],
                },
                "packages": packages,
            },
            "request_config": config,
        }
    )
    result = seal_analysis(result)
    output.mkdir(parents=True, exist_ok=True)
    # Store explicit row IDs, including only the tuples actually embedded; no
    # pickle.
    arrays = {
        "signals": embeddings,
        "record_ids": np.asarray([row["record_id"] for row in rows], dtype=str),
    }
    if text_embeddings is not None:
        arrays["texts"] = text_embeddings
        arrays["text_ids"] = np.asarray(
            [row["text_id"] for row in texts], dtype=str
        )
        arrays["text_record_ids"] = np.asarray(
            [row["record_id"] for row in texts], dtype=str
        )
    np.savez_compressed(output / "embeddings.npz", **arrays)
    write_frozen(analysis_path, result)
    write_viewer(result, output / "index.html")
    log.info("Saved analysis %s to %s", result["analysis_id"], output)
    return result
