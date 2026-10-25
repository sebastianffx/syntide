"""Execute a frozen 5,000-slot DiffuSETS plan, then prepare its unchanged-trainer arm.

Each slot publishes an immutable bundle independently. Retries reuse completed slots and
repeat only an interrupted slot's identical seed/condition; rejected slots are never topped up.
No API calls are made: conditioning vectors must already exist with explicit provenance.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np

from syntide.data.challenge import load_manifest
from syntide.data.qa import SignalInfo
from syntide.data.schema import LEADS
from syntide.generate.full.api import GenerationRequest, SignalSpec
from syntide.generate.full.assets import CHECKPOINT_HASHES
from syntide.generate.full.diffusets import CHANNELS, EMBEDDING_MODEL, DiffuSETSBackend, format_report
from syntide.generate.full.runner import _check_output
from syntide.generate.qa.evidence import EvidenceExtractor, ExtractorConfig
from syntide.generate.qa.generated import GeneratedSample, to_input, verify_generated

COUNT = 5000
POOL_SHA = "6c46d78a53838895c076f66c5a2006753c50dcfbbfed081f19dc7dd4f42726c4"
REQUIRED = ["heart_rate_bpm", "beat_count", "rr_regularity"]
SPEC = SignalSpec("ecg", CHANNELS, 100, 1000, "mV")
QA_POLICY = {
    "templates": ["rhythm.heart_rate@1", "rhythm.beat_count@1", "rhythm.regularity@1", "rhythm.rate_band@1"],
    "max_seed_tasks": 4,
    "evolutions": 0,
    "heart_rate_pairs": False,
    "required": REQUIRED,
    "min_confidence": 0.8,
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)  # exclusive publication; never replace an existing artifact
    except FileExistsError:
        if read(path) != value:
            raise ValueError(f"Concurrent frozen artifact differs: {path}") from None
    finally:
        temporary.unlink()


@contextmanager
def lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another generation process owns {output}") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_plan(plan, pool, *, expected_count=COUNT):
    if (
        plan.get("plan_seed") != 0
        or plan.get("manifest_content_sha256", plan.get("manifest_sha256")) != POOL_SHA
    ):
        raise ValueError("Expected seed-0 plan and original ecgqa-artifact-50 pool manifest")
    rows = plan.get("candidates", [])
    if len(rows) != expected_count or [r.get("slot_id") for r in rows] != list(range(expected_count)):
        raise ValueError(f"Plan must contain exactly {expected_count} ordered slots")
    seeds = set()
    for row in rows:
        if row.get("source_record_id") not in pool:
            raise ValueError(f"Slot {row['slot_id']} is outside the training pool")
        seed = row.get("seed")
        if type(seed) is not int or not 0 <= seed < 2**32 or seed in seeds:
            raise ValueError("Each slot needs a unique uint32 seed")
        seeds.add(seed)
        if not isinstance(row.get("report"), str) or not row["report"].strip():
            raise ValueError("Every slot needs an explicit source report")
        if row.get("sex") not in {"M", "F"}:
            raise ValueError("Sex conditioning must be M or F")
        for field, upper in (("age", 110), ("heart_rate", float("inf"))):
            value = row.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < upper:
                raise ValueError(f"Invalid {field} conditioning")
    return rows


def load_conditions(rows, index_path):
    index_path = Path(index_path).resolve()
    index = read(index_path)
    entries = index["entries"]
    conditions, subjects, fingerprints = [], {}, {}
    for row in rows:
        record = row["source_record_id"]
        item = entries[record]
        if item["report"] != row["report"]:
            raise ValueError(f"Embedding report differs from frozen slot: {record}")
        if float(item["age"]) != row["age"] or item["sex"] != row["sex"]:
            raise ValueError(f"Embedding-source demographics differ: {record}")
        subject = item.get("subject_uid")
        if not isinstance(subject, str) or not subject.startswith("ptb-xl:"):
            raise ValueError(f"Missing original patient identity: {record}")
        if row.get("subject_uid", subject) != subject:
            raise ValueError(f"Embedding-source patient identity differs: {record}")
        subjects[record] = subject
        vector_path = (index_path.parent / item["text_embedding_path"]).resolve()
        vector_sha = sha(vector_path)
        if vector_sha != item["text_embedding_sha256"] or item["text_embedding_model"] != EMBEDDING_MODEL:
            raise ValueError(f"Embedding hash/model mismatch: {record}")
        if record not in fingerprints:
            vector = np.load(vector_path, allow_pickle=False)
            if (
                vector.shape != (1536,)
                or not np.issubdtype(vector.dtype, np.number)
                or np.iscomplexobj(vector)
            ):
                raise ValueError(f"Embedding must be a real 1536-vector: {record}")
            if (
                not np.isfinite(vector).all()
                or not np.any(vector)
                or np.max(np.abs(vector.astype(np.float64))) > np.finfo(np.float32).max
            ):
                raise ValueError(f"Embedding must be finite, nonzero and float32-safe: {record}")
            fingerprints[record] = {k: v for k, v in item.items() if k != "text_embedding_path"}
        condition = {
            "report": row["report"],
            "age": row["age"],
            "sex": row["sex"],
            "heart_rate": row["heart_rate"],
            "text_embedding_path": str(vector_path),
            "text_embedding_model": EMBEDDING_MODEL,
            "text_embedding_sha256": vector_sha,
        }
        if item.get("embedding_input") is not None:
            if item["embedding_input"] != format_report(row["report"]):
                raise ValueError(f"Embedding input differs from required prompt: {record}")
            condition["embedding_input"] = item["embedding_input"]
        else:
            if not isinstance(item.get("embedding_source"), str) or not item["embedding_source"]:
                raise ValueError(f"Legacy vector needs an explicit published source: {record}")
            if item.get("prompt_provenance") != "unknown_legacy_input":
                raise ValueError(f"Legacy vector must declare unknown original input: {record}")
            condition["embedding_source"] = item["embedding_source"]
        conditions.append(condition)
    return (
        conditions,
        subjects,
        {"source_csv_sha256": index.get("source_csv_sha256"), "entries": fingerprints},
    )


def make_backend(assets, device):
    assets = Path(assets).resolve()
    return DiffuSETSBackend(
        {
            "source_dir": str(assets / "source"),
            "unet_checkpoint": str(assets / "checkpoints/unet_all.pth"),
            "vae_checkpoint": str(assets / "checkpoints/vae_model.pth"),
            "unet_sha256": CHECKPOINT_HASHES["unet_all.pth"],
            "vae_sha256": CHECKPOINT_HASHES["vae_model.pth"],
            "device": device,
            "num_inference_steps": 1000,
            "resample": True,
        }
    )


def implementation_fingerprint():
    import syntide.generate.descriptors as descriptors
    import syntide.generate.full.diffusets as diffusets
    import syntide.generate.qa.build as build
    import syntide.generate.qa.evidence as evidence
    import syntide.generate.qa.generated as generated
    import syntide.generate.qa.templates as templates

    return {
        "source_sha256": {
            module.__name__: sha(module.__file__)
            for module in (descriptors, diffusets, evidence, generated, build, templates)
        },
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy")},
    }


def common_provenance(backend):
    # Cache locations can change on transfer; hashes, model, runtime and device cannot.
    excluded = {"conditioning_embeddings", "source_dir", "checkpoints"}
    return {k: v for k, v in backend.provenance().items() if k not in excluded}


def check_published(path, generation_id, row):
    result = read(path / "result.json")
    if result["generation_id"] != generation_id or result["candidate_sha256"] != digest(row):
        raise ValueError(f"Completed slot identity differs: {path}")
    for relative, expected in result["artifacts"].items():
        if sha(path / relative) != expected:
            raise ValueError(f"Completed slot artifact changed: {path / relative}")
    return result


def review_signal(stage, record_id, row, condition, subjects, extractor):
    sample = GeneratedSample(
        record_id=record_id,
        run_id=record_id.split(":")[1],
        generator="diffusets@1",
        seed=row["seed"],
        conditioning=condition,
        source_record_ids=(row["source_record_id"],),
        channels=CHANNELS,
        sample_rate=100,
        units="mV",
        signal_path=stage / "signals/00000000.npy",
        signal_sha256=sha(stage / "signals/00000000.npy"),
        status="generated_pending_domain_and_label_review",
        requested_labels=(),
    )
    ref, signal = to_input(sample, SignalInfo(LEADS, 100, "mV", 1000), subjects, allow_resample=False)
    _, evidence, review = verify_generated(ref, signal, extractor, REQUIRED, min_confidence=0.8)
    return {**review, "evidence": [e.to_json() for e in evidence]}


def publish_slot(
    output, generation_id, row, condition, subjects, backend, provenance, extractor, *, reviewer=review_signal
):
    slot = row["slot_id"]
    destination = output / "candidates" / f"{slot:05d}"
    if destination.exists():
        return check_published(destination, generation_id, row)
    freeze(
        output / "claims" / f"{slot:05d}.json",
        {
            "generation_id": generation_id,
            "candidate_sha256": digest(row),
            "slot_id": slot,
            "seed": row["seed"],
        },
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f".slot-{slot:05d}-", dir=output) as temporary:
        stage = Path(temporary)
        (stage / "signals").mkdir()
        request = GenerationRequest(row["seed"], SPEC, condition)
        generated = backend.generate(request)
        try:
            values = _check_output(generated, SPEC)
        except ValueError as error:
            # A produced but malformed candidate consumes its slot; it is not replaced.
            result = {"status": "rejected_technical", "reason": str(error), "accepted": False}
        else:
            np.save(stage / "signals/00000000.npy", values, allow_pickle=False)
            record_id = f"synthetic:{generation_id}:{slot:08d}"
            review = reviewer(stage, record_id, row, condition, subjects, extractor)
            write(stage / "acceptance.json", review)
            portable_condition = {
                **condition,
                "text_embedding_path": "external-cache:" + condition["text_embedding_sha256"],
            }
            ledger = {
                "record_id": record_id,
                "run_id": generation_id,
                "plan_id": generation_id,
                "request_index": slot,
                "sample_index": 0,
                "seed": row["seed"],
                "generator": "diffusets@1",
                "conditioning": portable_condition,
                "source_record_ids": [row["source_record_id"]],
                "source_text_ids": [],
                "signal": SPEC.to_dict(),
                "signal_path": "signals/00000000.npy",
                "signal_sha256": sha(stage / "signals/00000000.npy"),
                "status": "generated_pending_domain_and_label_review",
                "metadata": generated.metadata,
            }
            catalog = {
                "record_id": record_id,
                "subject_uid": record_id,
                "split": "train",
                "signal_path": ledger["signal_path"],
                "sample_rate": 100,
                "channels": list(CHANNELS),
                "units": "mV",
                "labels": [],
                "metadata": {
                    "synthetic": True,
                    "generation": ledger,
                    "requested_labels": [],
                    "label_status": "pending_review",
                },
            }
            write(stage / "candidate.json", row)
            (stage / "ledger.jsonl").write_text(json.dumps(ledger, allow_nan=False) + "\n")
            (stage / "catalog.jsonl").write_text(json.dumps(catalog, allow_nan=False) + "\n")
            write(
                stage / "run.json",
                {
                    "schema_version": "syntide.full-generation-run@1",
                    "run_id": generation_id,
                    "count": 1,
                    "generator": "diffusets@1",
                    "backend": provenance,
                    "status": "complete_pending_domain_and_label_review",
                    "artifacts": {
                        name: sha(stage / name)
                        for name in ("candidate.json", "ledger.jsonl", "catalog.jsonl", "acceptance.json")
                    },
                },
            )
            result = {
                "status": review["status"],
                "accepted": review["status"] == "verified",
                "reasons": review["reasons"],
                "record_id": record_id,
            }
        if common_provenance(backend) != provenance:
            raise ValueError("Backend provenance changed during generation")
        result.update(
            {
                "generation_id": generation_id,
                "slot_id": slot,
                "seed": row["seed"],
                "candidate_sha256": digest(row),
                "generation_seconds": time.monotonic() - started,
                "artifacts": {str(p.relative_to(stage)): sha(p) for p in stage.rglob("*") if p.is_file()},
            }
        )
        write(stage / "result.json", result)
        # The whole output is locked. Rename directly: no empty reservation can survive a kill.
        os.rename(stage, destination)
    return result


def execute(
    plan,
    conditions,
    subjects,
    embedding_fingerprint,
    output,
    backend,
    *,
    expected_count=COUNT,
    reviewer=review_signal,
):
    output = Path(output).resolve()
    rows = plan["candidates"]
    if len(rows) != expected_count or len(conditions) != expected_count:
        raise ValueError("Candidate/condition count differs from frozen budget")
    extractor = EvidenceExtractor(ExtractorConfig())
    with lock(output):
        # Validate all inputs before loading weights; deduplicate identical report/demographic conditions.
        validated = set()
        for row, condition in zip(rows, conditions, strict=True):
            key = digest(condition)
            if key not in validated:
                backend.validate_request(GenerationRequest(row["seed"], SPEC, condition))
                validated.add(key)
                if len(validated) % 100 == 0:
                    print(
                        json.dumps({"phase": "validate_conditions", "validated": len(validated)}), flush=True
                    )
        provenance = common_provenance(backend)
        identity = {
            "schema_version": "syntide.diffusets-fixed-slots@1",
            "plan_sha256": digest(plan),
            "backend": provenance,
            "embedding_fingerprint": digest(embedding_fingerprint),
            "count": expected_count,
            "signal": SPEC.to_dict(),
            "extractor": asdict(ExtractorConfig()),
            "required": REQUIRED,
            "min_confidence": 0.8,
            "amplitude_mv": [0.2, 10.0],
            "driver_sha256": sha(__file__),
            "implementation": implementation_fingerprint(),
            "no_top_up": True,
            "qa_policy": QA_POLICY,
            "qa_policy_sha256": digest(QA_POLICY),
        }
        generation_id = digest(identity)
        freeze(output / "identity.json", {**identity, "generation_id": generation_id})
        freeze(output / "candidate-plan.json", plan)
        counts, seconds = Counter(), 0.0
        for row, condition in zip(rows, conditions, strict=True):
            result = publish_slot(
                output,
                generation_id,
                row,
                condition,
                subjects,
                backend,
                provenance,
                extractor,
                reviewer=reviewer,
            )
            counts[result["status"]] += 1
            seconds += result["generation_seconds"]
            print(
                json.dumps(
                    {
                        "slot": row["slot_id"],
                        "completed": sum(counts.values()),
                        "status": result["status"],
                        "counts": dict(counts),
                    }
                ),
                flush=True,
            )
        summary = {
            "schema_version": 1,
            "generation_id": generation_id,
            "candidate_count": expected_count,
            "accepted": counts["verified"],
            "rejected": expected_count - counts["verified"],
            "status_counts": dict(counts),
            "observed_generation_seconds": seconds,
            "slots": [
                {
                    "slot_id": r["slot_id"],
                    "result_sha256": sha(output / "candidates" / f"{r['slot_id']:05d}" / "result.json"),
                }
                for r in rows
            ],
        }
        measurements = measurement_summary(output, rows)
        freeze(output / "measurement-summary.json", measurements)
        summary["measurement_summary_sha256"] = sha(output / "measurement-summary.json")
        freeze(output / "summary.json", summary)
    return summary


def measurement_summary(output, rows):
    values = {
        group: {
            name: []
            for name in (
                "requested_hr",
                "measured_hr",
                "hr_difference",
                "max_wander_mv",
                "max_hf_rms_mv",
                "max_burst_rms_mv",
            )
        }
        for group in ("all_measured", "accepted")
    }
    regularity = {group: Counter() for group in values}
    strata, rejections = {}, Counter()
    for row in rows:
        path = Path(output) / "candidates" / f"{row['slot_id']:05d}"
        result = read(path / "result.json")
        stratum = row.get("stratum", "unspecified")
        strata.setdefault(stratum, Counter())[result["status"]] += 1
        rejections.update(
            result.get("reasons", [result.get("reason")]) if result["status"] != "verified" else []
        )
        if not (path / "acceptance.json").exists():
            continue
        evidence = read(path / "acceptance.json")["evidence"]
        present = [e for e in evidence if e["status"] == "present"]
        by_name = {e["attribute"]: e for e in present}
        hr = by_name.get("heart_rate_bpm", {}).get("value")
        for group in ["all_measured", "accepted"] if result["accepted"] else ["all_measured"]:
            values[group]["requested_hr"].append(row["heart_rate"])
            if isinstance(hr, (int, float)):
                values[group]["measured_hr"].append(hr)
                values[group]["hr_difference"].append(hr - row["heart_rate"])
            regularity[group][str(by_name.get("rr_regularity", {}).get("value", "unmeasured"))] += 1
            for attribute in ("wander_mv", "hf_rms_mv", "burst_rms_mv"):
                numbers = [
                    e["value"]
                    for e in present
                    if e["attribute"] == attribute and isinstance(e["value"], (int, float))
                ]
                if numbers:
                    values[group]["max_" + attribute].append(max(numbers))

    def stats(numbers):
        if not numbers:
            return {"count": 0}
        return dict(
            zip(
                ("min", "p10", "median", "p90", "max"),
                map(float, np.quantile(numbers, [0, 0.1, 0.5, 0.9, 1])),
                strict=True,
            )
        ) | {"count": len(numbers)}

    return {
        "schema_version": 1,
        "note": (
            "Measured recording descriptors include physiology; "
            "they do not validate a noise or diagnostic label. "
            "Parent strata are conditioning provenance only."
        ),
        "distributions": {
            group: {name: stats(numbers) for name, numbers in attrs.items()}
            for group, attrs in values.items()
        },
        "regularity": {group: dict(counts) for group, counts in regularity.items()},
        "requested_strata_outcomes": {name: dict(counts) for name, counts in strata.items()},
        "rejection_reasons": dict(rejections),
    }


def accepted_bundles(output, plan, *, expected_count=COUNT):
    output = Path(output).resolve()
    summary, identity = read(output / "summary.json"), read(output / "identity.json")
    if identity["plan_sha256"] != digest(plan) or summary["generation_id"] != identity["generation_id"]:
        raise ValueError("Summary/plan identity mismatch")
    if summary["candidate_count"] != expected_count or len(summary["slots"]) != expected_count:
        raise ValueError("Generation did not complete the exact candidate budget")
    if identity.get("qa_policy") != QA_POLICY or identity.get("qa_policy_sha256") != digest(QA_POLICY):
        raise ValueError("Frozen QA policy differs from this preparation implementation")
    if sha(output / "measurement-summary.json") != summary["measurement_summary_sha256"]:
        raise ValueError("Measurement summary changed after completion")
    bundles = []
    for row, slot in zip(plan["candidates"], summary["slots"], strict=True):
        path = output / "candidates" / f"{row['slot_id']:05d}"
        if slot["slot_id"] != row["slot_id"] or sha(path / "result.json") != slot["result_sha256"]:
            raise ValueError("Candidate result differs from completion manifest")
        result = check_published(path, summary["generation_id"], row)
        if result["accepted"]:
            bundles.append(path)
    if len(bundles) != summary["accepted"]:
        raise ValueError("Accepted count mismatch")
    return bundles


def qa_config(bundles):
    from syntide.generate.qa.build import BuildConfig, GeneratedConfig
    from syntide.generate.qa.evol import EvolConfig

    return BuildConfig(
        name="ecgqa-cot-generated-5000-seed0",
        components=["generated"],
        max_records=0,  # source remains available for ancestry; no real-signal work units are needed
        evol=EvolConfig(max_seed_tasks=QA_POLICY["max_seed_tasks"], evolutions=QA_POLICY["evolutions"]),
        generated=GeneratedConfig(
            bundles=[str(p) for p in bundles],
            allow_resample=False,
            heart_rate_pairs=QA_POLICY["heart_rate_pairs"],
            templates=QA_POLICY["templates"],
            required=QA_POLICY["required"],
            min_confidence=QA_POLICY["min_confidence"],
        ),
    )


def prepare(
    plan, output, *, registry, manifest, qa_output, baseline_prepared, grounded, acquisition, prepared_output
):
    from syntide.generate.qa.build import build_corpus, identity
    from syntide.generate.qa.corpus import verify_corpus_files
    from syntide.generate.qa.export import export_prepared
    from syntide.generate.qa.sources import open_source

    frozen_generation = read(Path(output) / "identity.json")
    if frozen_generation["implementation"] != implementation_fingerprint():
        raise ValueError("Preparation must use the frozen generation measurement code and package versions")
    if frozen_generation["driver_sha256"] != sha(__file__):
        raise ValueError("Preparation driver differs from the frozen generation driver")
    bundles = accepted_bundles(output, plan)
    if not bundles:
        raise ValueError("No candidates passed acceptance; no generated arm can be built")
    source = open_source(registry, manifest, "pool", manifest_sha256=POOL_SHA)
    config = qa_config(bundles)
    expected_identity = identity(config, source, None, None)
    freeze(
        Path(output) / "preparation.json",
        {
            "qa_config": asdict(config),
            "qa_config_sha256": digest(asdict(config)),
            "qa_identity_sha256": digest(expected_identity),
            "generation_summary_sha256": sha(Path(output) / "summary.json"),
            "baseline_manifest_file_sha256": sha(Path(baseline_prepared) / "manifest.json"),
            "grounded_corpus_file_sha256": sha(Path(grounded) / "corpus.json"),
            "acquisition_corpus_file_sha256": sha(Path(acquisition) / "corpus.json"),
        },
    )
    qa_output = Path(qa_output)
    if (qa_output / "corpus.json").exists():
        info = verify_corpus_files(qa_output)
        if digest(info["identity"]) != digest(expected_identity):
            raise ValueError("Published generated corpus belongs to different QA config or source")
    else:
        info = build_corpus(config, source, qa_output)
    if info["counts"]["generated_waveforms"] != len(bundles):
        raise ValueError("Local QA remeasurement disagrees with candidate acceptance; review before export")
    families = info["counts"]["by_family"]
    expected_families = {t.split("@")[0] for t in QA_POLICY["templates"]}
    # Rate-band questions intentionally abstain within 3 bpm of a band boundary.
    if (
        not set(families) <= expected_families
        or any(families.get(name, 0) != len(bundles) for name in expected_families - {"rhythm.rate_band"})
        or not 3 * len(bundles) <= info["counts"]["qa_examples"] <= 4 * len(bundles)
    ):
        raise ValueError("Generated QA differs from the four frozen rhythm templates")
    export_prepared([grounded, acquisition, qa_output], baseline_prepared, prepared_output)
    return {
        "corpus_id": info["corpus_id"],
        "accepted_waveforms": len(bundles),
        "prepared": str(Path(prepared_output).resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "generate", "prepare"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embedding-index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--device", default="cuda", choices=["cpu", "mps", "cuda"])
    for name in ("registry", "qa-output", "baseline-prepared", "grounded", "acquisition", "prepared-output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if sha(args.plan) != args.plan_sha256:
        raise ValueError("Candidate-plan file checksum mismatch")
    manifest = load_manifest(args.manifest)
    if manifest.sha256 != POOL_SHA:
        raise ValueError("Training manifest differs from baseline")
    plan = read(args.plan)
    if "manifest_content_sha256" in plan and plan["manifest_sha256"] != sha(args.manifest):
        raise ValueError("Manifest file bytes differ from the frozen plan")
    rows = validate_plan(plan, set(manifest.roles["pool"]))
    if args.command == "prepare":
        required = (
            "registry",
            "qa_output",
            "baseline_prepared",
            "grounded",
            "acquisition",
            "prepared_output",
        )
        if any(getattr(args, key) is None for key in required):
            parser.error(
                "prepare requires registry, qa-output, baseline-prepared, grounded, "
                "acquisition and prepared-output"
            )
        result = prepare(
            plan, args.output, manifest=args.manifest, **{key: getattr(args, key) for key in required}
        )
    else:
        if args.embedding_index is None:
            parser.error("validate/generate require --embedding-index")
        conditions, subjects, fingerprint = load_conditions(rows, args.embedding_index)
        source_hashes = plan.get("provenance", {}).get("sources_sha256", {})
        expected_csv_hashes = {
            value for name, value in source_hashes.items() if Path(name).name == "ptbxl_database.csv"
        }
        if expected_csv_hashes and expected_csv_hashes != {fingerprint["source_csv_sha256"]}:
            raise ValueError("Embedding index source metadata differs from frozen planning CSV")
        if args.command == "validate":
            result = {
                "status": "passed",
                "candidates": len(rows),
                "source_records": len(subjects),
                "embedding_fingerprint": digest(fingerprint),
            }
        else:
            if args.assets is None:
                parser.error("generate requires --assets")
            result = execute(
                plan, conditions, subjects, fingerprint, args.output, make_backend(args.assets, args.device)
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
