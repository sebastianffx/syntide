"""Extract training-report vectors from the authors' published embedding table.

The table revision and the training candidate plan must match their saved
hashes.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CSV_SHA = "aee14e1aade59f025192dacc62b0d1f756f345e2ff01575aa16766f063d7c49d"
REVISION = "80b04013aef594c7c3012a8b932e5e09524e581a"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract(plan_path, csv_path, output):
    plan_path, csv_path, output = map(Path, (plan_path, csv_path, output))
    if sha(csv_path) != CSV_SHA:
        raise ValueError("Published source CSV checksum differs")
    plan = json.loads(plan_path.read_text())
    metadata_sources = {
        name: expected
        for name, expected in plan["provenance"]["sources_sha256"].items()
        if Path(name).name == "ptbxl_database.csv"
    }
    if len(metadata_sources) != 1:
        raise ValueError(
            "Frozen plan must identify one original PTB-XL metadata CSV"
        )
    metadata_path, metadata_sha = next(iter(metadata_sources.items()))
    if sha(metadata_path) != metadata_sha:
        raise ValueError("Original planning metadata CSV checksum differs")
    selected = {}
    for candidate in plan["candidates"]:
        record = candidate["source_record_id"]
        fields = {
            k: candidate[k] for k in ("report", "age", "sex", "subject_uid")
        }
        if record in selected and selected[record] != fields:
            raise ValueError(f"Inconsistent parent metadata: {record}")
        selected[record] = fields
    output.mkdir(parents=True, exist_ok=True)
    (output / "vectors").mkdir(exist_ok=True)
    entries = {}
    with csv_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            record = f"ptbxl-{int(row['ecg_id'])}"
            if record not in selected:
                continue
            expected = selected[record]
            actual = {
                "report": row["report"],
                "age": float(row["age"]),
                "sex": {"0": "M", "1": "F"}[row["sex"]],
                "subject_uid": f"ptb-xl:{int(float(row['patient_id']))}",
            }
            if actual != expected or record in entries:
                raise ValueError(
                    f"Published metadata differs or is duplicated: {record}"
                )
            vector = np.asarray(
                ast.literal_eval(row["text_embed"]), dtype=np.float32
            )
            if (
                vector.shape != (1536,)
                or not np.isfinite(vector).all()
                or not np.any(vector)
            ):
                raise ValueError(f"Invalid published vector: {record}")
            path = output / "vectors" / f"{record}.npy"
            if path.exists():
                if not np.array_equal(
                    np.load(path, allow_pickle=False), vector
                ):
                    raise ValueError(f"Existing vector differs: {record}")
            else:
                with path.open("xb") as destination:
                    np.save(destination, vector, allow_pickle=False)
            entries[record] = {
                **actual,
                "text_embedding_path": str(path.relative_to(output)),
                "text_embedding_model": "text-embedding-ada-002",
                "text_embedding_sha256": sha(path),
                "embedding_source": (
                    f"Laiyf/DiffuSETS@{REVISION}:prerequsites_for_inference.zip/"
                    f"prerequisites/ptbxl_text_embed.csv#ecg_id={row['ecg_id']}"
                ),
                "prompt_provenance": "unknown_legacy_input",
            }
    if set(entries) != set(selected):
        raise ValueError(
            f"Missing published vectors: "
            f"{sorted(set(selected) - set(entries))[:10]}"
        )
    index = {
        "schema_version": 1,
        "source_csv_sha256": metadata_sha,
        "source_csv_kind": (
            "Original ptbxl_database.csv used by frozen allocation plan"
        ),
        "embedding_csv_sha256": CSV_SHA,
        "candidate_plan_file_sha256": sha(plan_path),
        "source_revision": REVISION,
        "original_api_input": (
            "Not supplied by authors; exact input text cannot be verified."
        ),
        "selection_scope": (
            "Only TRAIN parents listed in the frozen candidate plan."
        ),
        "entries": entries,
    }
    path = output / "embedding-index.json"
    if path.exists():
        if json.loads(path.read_text()) != index:
            raise ValueError("Existing embedding index differs")
    else:
        with path.open("x") as stream:
            json.dump(index, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return {"records": len(entries), "index_sha256": sha(path), "api_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.plan, args.csv, args.output), sort_keys=True))
