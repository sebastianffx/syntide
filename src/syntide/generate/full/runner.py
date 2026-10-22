"""Execute one explicit full-generation request into a new immutable dataset bundle."""

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from syntide.explore.plans import content_id, verify_plan
from syntide.generate.full.api import GeneratedSignal, GenerationRequest, SignalSpec, json_object
from syntide.generate.full.registry import create_generator


def _merge(bound: dict, supplied: dict, prefix: str = "config") -> dict:
    """Explicit execution settings may fill gaps, but never change an exported intention."""
    merged = copy.deepcopy(bound)
    for key, value in supplied.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge(merged[key], value, f"{prefix}.{key}")
        elif key in merged and merged[key] != value:
            raise ValueError(f"{prefix}.{key} conflicts with the selected plan request")
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _ids(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
        raise ValueError(f"{field} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} contains duplicate IDs")
    return value


def _select_request(plan: dict, request_index: int | None) -> tuple[int, dict]:
    requests = plan.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("Plan needs at least one generation request")
    if request_index is None:
        if len(requests) != 1:
            raise ValueError("Mixed or multiple requests require an explicit --request-index (zero-based)")
        request_index = 0
    if (
        isinstance(request_index, bool)
        or not isinstance(request_index, int)
        or not 0 <= request_index < len(requests)
    ):
        raise ValueError("request-index is outside the plan's requests")
    request = requests[request_index]
    if not isinstance(request, dict) or request.get("method") != "diffusion":
        raise ValueError("The selected request must use method=diffusion; other requests remain unexecuted")
    count = request.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("The selected request count must be a positive integer")
    return request_index, request


def _resolve(plan: dict, supplied: dict, selected: dict) -> tuple[str, SignalSpec, list[dict], dict, dict]:
    json_object(supplied, "config")
    parameters = json_object(selected.get("parameters", {}), "plan parameters")
    config = _merge(parameters, supplied)
    generator = selected.get("generator") or config.get("generator")
    if not generator:
        raise ValueError("Choose a versioned generator in the plan or execution config")
    if config.get("generator", generator) != generator:
        raise ValueError("config.generator conflicts with the selected plan request")
    config["generator"] = generator
    signal = SignalSpec.from_dict(config.get("signal"))
    for modality in (selected.get("modality"), config.get("modality")):
        if modality is not None and modality != signal.modality:
            raise ValueError("signal.modality conflicts with the selected plan request")
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("config.conditions must contain at least one explicit conditioning object")
    targets = json_object(plan.get("targets"), "plan.targets")
    record_ids = set(_ids(targets.get("record_ids", []), "targets.record_ids"))
    text_ids = set(_ids(targets.get("text_ids", []), "targets.text_ids"))
    normalized = []
    for index, condition in enumerate(conditions):
        json_object(condition, f"conditions[{index}]")
        unknown = set(condition) - {"conditioning", "source_record_ids", "source_text_ids"}
        if unknown:
            raise ValueError(f"conditions[{index}] has unknown fields: {sorted(unknown)}")
        conditioning = json_object(condition.get("conditioning"), f"conditions[{index}].conditioning")
        records = _ids(condition.get("source_record_ids", []), "source_record_ids")
        texts = _ids(condition.get("source_text_ids", []), "source_text_ids")
        if set(records) - record_ids or set(texts) - text_ids:
            raise ValueError(f"conditions[{index}] source references must be a subset of the plan targets")
        normalized.append(
            {"conditioning": conditioning, "source_record_ids": records, "source_text_ids": texts}
        )
    config["signal"], config["conditions"] = signal.to_dict(), normalized
    backend_config = {
        key: value
        for key, value in config.items()
        if key not in {"generator", "modality", "signal", "conditions"}
    }
    return generator, signal, normalized, backend_config, config


def _seed(seed: int, request_index: int, sample_index: int) -> int:
    digest = hashlib.sha256(f"syntide.full@1:{seed}:{request_index}:{sample_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _check_output(result: GeneratedSignal, expected: SignalSpec) -> np.ndarray:
    if not isinstance(result, GeneratedSignal) or result.spec != expected:
        raise ValueError("Generator returned a different signal specification than requested")
    values = np.asarray(result.signal)
    if values.shape != (len(expected.channels), expected.num_samples):
        raise ValueError(f"Generator returned shape {values.shape}, incompatible with the requested signal")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("Generated signals must contain real numeric values")
    if not np.isfinite(values).all() or np.max(np.abs(values.astype(np.float64))) > np.finfo(np.float32).max:
        raise ValueError("Generated signals must be finite and representable as float32")
    json_object(result.metadata, "generated metadata")
    return values.astype(np.float32, copy=False)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_generation(plan: dict, config: dict, output: str | Path, *, request_index: int | None = None) -> dict:
    """Generate exactly one requested count, publishing only a completely successful bundle.

    Conditions cycle in their declared order. Each sample gets its own seed, independent of
    model batching. Requested labels and QA are evidence to review, never accepted supervision.
    """
    verify_plan(plan)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing output: {output}")
    index, selected = _select_request(plan, request_index)
    generator, signal, conditions, backend_config, resolved = _resolve(plan, config, selected)
    seed = plan.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Plan seed must be an integer in [0, 2**32)")
    backend = create_generator(generator, copy.deepcopy(backend_config))
    # All conditions are checked before the first call that can load model weights.
    for condition in conditions:
        backend.validate_request(GenerationRequest(seed, signal, copy.deepcopy(condition["conditioning"])))
    provenance = copy.deepcopy(json_object(backend.provenance(), "backend provenance"))
    identity = {
        "schema_version": "syntide.full-generation-run@1",
        "plan_id": plan["plan_id"],
        "request_index": index,
        "generator": generator,
        "config": resolved,
        "config_id": content_id(resolved),
        "backend": provenance,
        "count": selected["count"],
        "seed_derivation": "sha256(syntide.full@1:plan_seed:request_index:sample_index)[:4] big-endian",
    }
    run_id = content_id(identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".syntide-generation-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "signals").mkdir()
        _write_json(stage / "plan.json", plan)
        _write_json(stage / "config.json", resolved)
        with (
            (stage / "ledger.jsonl").open("w", encoding="utf-8") as ledger,
            (stage / "catalog.jsonl").open("w", encoding="utf-8") as catalog,
        ):
            for sample_index in range(selected["count"]):
                condition_index = sample_index % len(conditions)
                condition = conditions[condition_index]
                sample_seed = _seed(seed, index, sample_index)
                request = GenerationRequest(sample_seed, signal, copy.deepcopy(condition["conditioning"]))
                result = backend.generate(request)
                values = _check_output(result, signal)
                record_id = f"synthetic:{run_id}:{sample_index:08d}"
                relative = f"signals/{sample_index:08d}.npy"
                np.save(stage / relative, values, allow_pickle=False)
                row = {
                    "record_id": record_id,
                    "run_id": run_id,
                    "plan_id": plan["plan_id"],
                    "request_index": index,
                    "sample_index": sample_index,
                    "condition_index": condition_index,
                    "seed": sample_seed,
                    "generator": generator,
                    "conditioning": request.conditioning,
                    "conditioning_id": content_id(request.conditioning),
                    "source_record_ids": condition["source_record_ids"],
                    "source_text_ids": condition["source_text_ids"],
                    "signal": signal.to_dict(),
                    "signal_path": relative,
                    "signal_sha256": _file_hash(stage / relative),
                    "validation": {"shape": True, "finite_float32": True},
                    "status": "generated_pending_domain_and_label_review",
                    "metadata": result.metadata,
                }
                ledger.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                entry = {
                    "record_id": record_id,
                    "subject_uid": record_id,
                    "split": "train",
                    "signal_path": relative,
                    "sample_rate": signal.sample_rate,
                    "channels": list(signal.channels),
                    "units": signal.units,
                    "labels": [],
                    "metadata": {
                        "synthetic": True,
                        "modality": signal.modality,
                        "generation": row,
                        "requested_labels": plan["targets"].get("labels", []),
                        "label_status": "pending_review",
                        "qa_status": "pending_review_not_inherited",
                    },
                }
                catalog.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
        final_provenance = json_object(backend.provenance(), "final backend provenance")
        if final_provenance != provenance:
            raise ValueError("Backend provenance changed during generation; the run cannot be frozen")
        run = {
            **identity,
            "run_id": run_id,
            "status": "complete_pending_domain_and_label_review",
            "unexecuted_request_indices": [i for i in range(len(plan["requests"])) if i != index],
            "artifacts": {
                name: _file_hash(stage / name)
                for name in ("plan.json", "config.json", "ledger.jsonl", "catalog.jsonl")
            },
        }
        _write_json(stage / "run.json", run)
        # Reserve the destination exclusively, then replace only our own empty directory.
        output.mkdir()
        try:
            os.replace(stage, output)
        except BaseException:
            output.rmdir()
            raise
    return run
