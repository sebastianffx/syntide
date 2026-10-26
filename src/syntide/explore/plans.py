"""Validate and save generation plans identified by a hash of their contents."""

import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any

METHODS = {
    "augmentation": "per_lead_aug@1",
    "editing": "ecg.electrode_artifact@1.0.0",
    "diffusion": "diffusets@1",
}
GENERATOR_ID = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]*@[A-Za-z0-9][A-Za-z0-9_.+-]*"
)


def _string_order(value: str) -> bytes:
    return value.encode("utf-16-be")


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "Non-finite numbers cannot appear in a reproducible artifact"
            )
        return int(value) if value.is_integer() else value
    return value


def canonical_json(value: Any) -> str:
    """Encode values identically for Python and JavaScript content hashes.

    Numbers use their exact 64-bit floating-point representation. Type
    markers keep objects and arrays distinct. Integers must fit JavaScript's
    exact integer range.
    """

    def encode(item: Any) -> list:
        if item is None:
            return ["null"]
        if isinstance(item, bool):
            return ["boolean", item]
        if isinstance(item, str):
            return ["string", item]
        if isinstance(item, int | float):
            if not math.isfinite(item) or (
                float(item).is_integer() and abs(item) > 2**53 - 1
            ):
                raise ValueError(
                    "Artifact numbers must be finite and integers "
                    "must be JavaScript-safe"
                )
            return [
                "number",
                struct.pack(">d", float(item) if item else 0.0).hex(),
            ]
        if isinstance(item, dict):
            keys = sorted(item, key=lambda key: key.encode("utf-16-be"))
            return ["object", [[key, encode(item[key])] for key in keys]]
        if isinstance(item, list | tuple):
            return ["array", [encode(child) for child in item]]
        raise ValueError(f"Unsupported artifact value: {type(item).__name__}")

    return json.dumps(
        encode(value),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_id(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def seal_analysis(analysis: dict) -> dict:
    body = {k: v for k, v in analysis.items() if k != "analysis_id"}
    return {**body, "analysis_id": content_id(body)}


def verify_analysis(analysis: dict) -> None:
    if seal_analysis(analysis)["analysis_id"] != analysis.get("analysis_id"):
        raise ValueError(
            "Analysis checksum does not match; rebuild the analysis "
            "instead of editing its JSON"
        )


def verify_plan(plan: dict) -> None:
    """Validate the exported plan before checking generator settings."""
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != "syntide.generation-plan@1"
    ):
        raise ValueError(
            "Unsupported generation plan schema; expected "
            "syntide.generation-plan@1"
        )
    body = {key: value for key, value in plan.items() if key != "plan_id"}
    if content_id(body) != plan.get("plan_id"):
        raise ValueError(
            "Plan checksum does not match; export a new plan instead "
            "of editing its JSON"
        )


def make_plan(analysis: dict, selection: dict) -> dict:
    """Save the inputs, seeds, and counts requested for generation.

    These requests describe intended generation. The selected generator must
    validate its own settings before execution, and generated labels still
    require review. Diffusion requests default to DiffuSETS for
    electrocardiograms but may name another versioned model and signal type.
    """
    verify_analysis(analysis)
    records = {r["record_id"]: r for r in analysis["records"]}
    ids = sorted(set(selection.get("record_ids", [])), key=_string_order)
    if not ids:
        raise ValueError("Select at least one recording")
    unknown = set(ids) - records.keys()
    if unknown:
        raise ValueError(f"Unknown recording IDs: {sorted(unknown)[:5]}")
    seed = selection.get("seed", 42)
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**32
    ):
        raise ValueError("seed must be an integer in [0, 2**32)")
    requests = selection.get("requests", [])
    if not isinstance(requests, list) or not requests:
        raise ValueError("Choose at least one generation method")
    normalized = []
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("Every method request must be a JSON object")
        method, count = request.get("method"), request.get("count")
        if not isinstance(method, str) or method not in METHODS:
            raise ValueError(
                f"Unknown method {method!r}; choose {', '.join(METHODS)}"
            )
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("Every method needs a positive integer count")
        parameters = request.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("Method parameters must be a JSON object")
        normalized_request = {
            "method": method,
            "count": count,
            "generator": METHODS[method],
            "parameters": parameters,
            "configuration_status": "requires_plugin_configuration"
            if method == "diffusion"
            else "requires_recipe_calibration_and_validation",
        }
        if method == "diffusion":
            generator = request.get("generator", METHODS[method])
            if not isinstance(generator, str) or not GENERATOR_ID.fullmatch(
                generator
            ):
                raise ValueError(
                    "Diffusion generator must be a versioned ID such "
                    "as diffusets@1"
                )
            modality = request.get("modality", "ecg")
            if not isinstance(modality, str) or not modality.strip():
                raise ValueError("Diffusion modality must be a nonempty string")
            normalized_request.update(
                generator=generator, modality=modality.strip()
            )
        elif "generator" in request or "modality" in request:
            raise ValueError(
                "Generator and modality overrides are supported only "
                "for diffusion requests"
            )
        normalized.append(normalized_request)
    text_ids = selection.get("text_ids")
    available_texts = {
        t["text_id"]
        for record_id in ids
        for t in records[record_id].get("texts", [])
        if t.get("text_id")
    }
    if text_ids is None:
        text_ids = sorted(available_texts, key=_string_order)
    elif set(text_ids) - available_texts:
        raise ValueError(
            "Selected texts must belong to the selected recordings"
        )
    body = {
        "schema_version": "syntide.generation-plan@1",
        "analysis_id": analysis["analysis_id"],
        "source": analysis["provenance"]["source"],
        "embedding": analysis["provenance"].get("embedding"),
        "text_embedding": analysis["provenance"].get("text_embedding"),
        "sampling": analysis["provenance"].get("sampling"),
        "seed": seed,
        "targets": {
            "record_ids": ids,
            "text_ids": sorted(set(text_ids), key=_string_order),
            "labels": sorted(
                {
                    label
                    for record_id in ids
                    for label in records[record_id].get("labels", [])
                },
                key=_string_order,
            ),
            "clusters": sorted(
                {records[record_id]["cluster"] for record_id in ids}
            ),
            "criteria": selection.get("criteria", {}),
        },
        "requests": normalized,
        "total_requested": sum(r["count"] for r in normalized),
        ("status"): (
            "intent_only_requires_generator_configuration_and_label_review"
        ),
        "requirements": [
            (
                "Resolve and pin every generator, checkpoint and control "
                "configuration before execution."
            ),
            (
                "Validate generated signals; record accepted/rejected "
                "recipes and exact seeds in a ledger."
            ),
            (
                "Review label effects per method; augmentation/editing "
                "cannot imply a new diagnosis."
            ),
            (
                "Review text, answer and rationale validity after every "
                "signal change; never inherit them blindly."
            ),
            (
                "Keep selected source roles and held-out patients "
                "separate under the experiment protocol."
            ),
        ],
    }
    return {**body, "plan_id": content_id(body)}


def write_frozen(path: Path, value: dict) -> None:
    payload = (
        json.dumps(
            _normalize(value), indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != _normalize(value):
            raise FileExistsError(
                f"Refusing to replace a different artifact: {path}"
            )
        return
    with path.open("x") as stream:
        stream.write(payload)
