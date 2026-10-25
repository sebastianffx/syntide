#!/usr/bin/env python3
"""CPU-only, immutable baseline + frozen DiffuSETS prepared-data subset.

Only training parquet is parsed. Held-out tables are copied and hashed as opaque
bytes. This script never loads a model, regenerates QA, or remeasures an ECG.
The production CLI has fixed source identities/counts; Contract is injectable only
for small local tests. Sampling and optimizer policy belong to the trainer.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FILES = ("train.parquet", "selection.parquet", "validation.parquet", "test.parquet", "records.json", "signals.npy")
HELDOUT = ("selection.parquet", "validation.parquet", "test.parquet")
VERSION = "syntide.qa-exposure-subset@1"
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@dataclass(frozen=True)
class Contract:
    baseline_rows: int = 82829
    generated_rows: int = 11672
    accepted: int = 3030
    attempts: int = 3380
    baseline_signals: int = 10075
    source_rows: int = 221394
    source_signals: int = 17925
    baseline_file_sha256: str = "c77ab178b55a652282fbdfbaf729cc4a1ccb19ec75f67482cf7ee761900fe513"
    baseline_content_sha256: str = "b47801817e9661919b71ed96049d6fb473c2d63430de69434e657a7fd6c00e8f"
    prepared_file_sha256: str = "355f67dfd7ae2f9a264b14a9a3ee4e9a12f462115233fa86b2d88972048517fc"
    prepared_content_sha256: str = "d0d919378f3aa659ecb872b48a92297c635081b45537d48d5ac159a0ebcc0d6d"
    snapshot_file_sha256: str = "2e670dc7dc793c3adddffba89e73f825d0be46fb19f98f3b5e35cc7bf1c33bfd"
    corpus_file_sha256: str = "372c82f822edad87d161b90ffd0063b9451f2b00cfe124d88d90262ad257de22"
    corpus_id: str = "corpus-0172ff51a3ea48d72957b917"
    generation_id: str = "bb68a35b6430053d2c9d2993a0c5ab45c2f6b775a78c53f773674fd98f1b522e"
    family_counts: tuple = (("rhythm.beat_count", 3030), ("rhythm.heart_rate", 3030), ("rhythm.rate_band", 2582), ("rhythm.regularity", 3030))


PRODUCTION = Contract()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def task_ids_sha256(ids: list[str]) -> str:
    """Trainer contract: default json.dumps separators and ensure_ascii=True."""
    return hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), f"not a regular source file: {path}")
    return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}


def read_pinned(path: Path, expected: str) -> dict:
    require(file_info(path)["sha256"] == expected, f"source file pin mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_prepared(path: Path, file_pin: str | None = None, content_pin: str | None = None) -> dict:
    manifest = read_pinned(path / "manifest.json", file_pin) if file_pin else json.loads((path / "manifest.json").read_text())
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    require(canonical_sha256(body) == manifest.get("sha256"), "prepared manifest checksum mismatch")
    require(not content_pin or manifest["sha256"] == content_pin, "prepared content pin mismatch")
    require(manifest.get("schema_version") == 1 and set(manifest.get("files", {})) == set(FILES), "unsupported prepared schema")
    for name in FILES:
        require(file_info(path / name) == manifest["files"][name], f"prepared file checksum mismatch: {path / name}")
    binding = manifest.get("exposure_source_manifest")
    if binding:
        require(binding["file"] == "source-manifest.json", "unexpected source manifest path")
        require(file_info(path / binding["file"]) == {k: binding[k] for k in ("bytes", "sha256")}, "source manifest binding mismatch")
    return manifest


def waveform_sha256(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x, dtype=np.float32)
    digest = hashlib.sha256(str(x.shape).encode())
    digest.update(x.tobytes())
    return digest.hexdigest()


def exact_arrays(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def preflight_groups(table: pa.Table, groups: dict, contract: Contract = PRODUCTION) -> dict:
    """Reusable CPU preflight; checks every ID and the contiguous source groups."""
    require(groups.get("schema_version") == 1, "unsupported source manifest schema")
    ids = table["task_id"].to_pylist()
    require(len(ids) == contract.baseline_rows + contract.generated_rows and len(set(ids)) == len(ids), "training task IDs are duplicated or count differs")
    for name, start, count in (("original", 0, contract.baseline_rows), ("generated", contract.baseline_rows, contract.generated_rows)):
        group = groups[name]
        require(group["start"] == start and group["count"] == count, f"incorrect {name} boundaries")
        require(group["task_ids_sha256"] == task_ids_sha256(ids[start:start + count]), f"incorrect {name} task order")
    require(groups["train_task_ids_sha256"] == task_ids_sha256(ids), "incorrect combined task order")
    return {name: groups[name] for name in ("original", "generated")}


def counts(table: pa.Table) -> dict:
    rows = table.to_pylist()
    records = {r["record_id"]: r["is_artifact"] for r in rows}
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
        digest.update(b"\n")
    return {"questions": len(rows), "records": len(records), "patients": len({r["subject_uid"] for r in rows}),
            "noisy_records": sum(bool(v) for v in records.values()), "noisy_questions": sum(bool(r["is_artifact"]) for r in rows),
            "templates": len({r["template_id"] for r in rows}),
            "template_noise_counts": dict(sorted(Counter(f"{r['template_id']}:{'noisy' if r['is_artifact'] else 'clean'}" for r in rows).items())),
            "task_ids_sha256": canonical_sha256([r["task_id"] for r in rows]), "tasks_sha256": digest.hexdigest()}


def validate_snapshot(snapshot: dict, contract: Contract) -> dict[str, dict]:
    require(snapshot.get("schema") == "syntide.diffusets-qa-snapshot@1" and snapshot.get("local_and_remote_hashes_verified") is True, "unverified snapshot")
    inv = snapshot["remote_inventory"]
    digest = hashlib.sha256(json.dumps(inv, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(digest == snapshot["remote_inventory_sha256"], "snapshot inventory checksum mismatch")
    require(inv["generation_id"] == contract.generation_id, "incorrect generation")
    require(inv["candidate_count"] == contract.attempts and inv["accepted_count"] == contract.accepted and inv["rejected_count"] == contract.attempts - contract.accepted, "incorrect frozen prefix counts")
    require(inv["prefix_slot_ids"] == list(range(contract.attempts)) and [s["slot_id"] for s in inv["slots"]] == inv["prefix_slot_ids"], "incorrect candidate prefix")
    accepted = [s for s in inv["slots"] if s["accepted"]]
    require([s["slot_id"] for s in accepted] == inv["accepted_slot_ids"], "accepted slots mismatch")
    require(len(accepted) == contract.accepted and all(s["status"] == "verified" for s in accepted), "unverified accepted candidate")
    by_id = {s["record_id"]: s for s in accepted}
    require(len(by_id) == contract.accepted and all(s["record_id"] == f"synthetic:{contract.generation_id}:{s['slot_id']:08d}" for s in accepted), "accepted record identity mismatch")
    return by_id


def build(baseline: Path, snapshot_prepared: Path, generated_corpus: Path, snapshot_manifest: Path, output: Path, contract: Contract = PRODUCTION) -> dict:
    """Build only into a new directory. All sources are treated as immutable."""
    baseline, snapshot_prepared, generated_corpus, snapshot_manifest, output = map(lambda p: Path(p).absolute(), (baseline, snapshot_prepared, generated_corpus, snapshot_manifest, output))
    require(not output.exists() and not output.is_symlink(), f"refusing existing output: {output}")
    for source in (baseline, snapshot_prepared, generated_corpus, snapshot_manifest.parent):
        require(not output.resolve().is_relative_to(source.resolve()), "output may not be inside a source tree")
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not output.exists(), "output appeared before preparation lock")
        print("Verifying pinned source files", flush=True)
        base = verify_prepared(baseline, contract.baseline_file_sha256, contract.baseline_content_sha256)
        source = verify_prepared(snapshot_prepared, contract.prepared_file_sha256, contract.prepared_content_sha256)
        frozen = read_pinned(snapshot_manifest, contract.snapshot_file_sha256)
        corpus = read_pinned(generated_corpus / "corpus.json", contract.corpus_file_sha256)
        accepted = validate_snapshot(frozen, contract)
        require(not base.get("qa_export"), "baseline must be the original QA export")
        require(source["qa_export"]["identity"]["baseline_manifest_sha256"] == base["sha256"], "snapshot prepared has a different baseline")
        require(corpus["schema_version"] == "syntide.qa-corpus@1" and corpus["status"] == "published" and corpus["corpus_id"] == contract.corpus_id, "incorrect generated corpus")
        corpus_source = corpus["identity"]["source"]
        require(corpus["identity"]["purpose"] == corpus_source["purpose"] == "training" and corpus_source["role"] == base["preparation"]["train_role"], "corpus is not baseline training data")
        require(corpus_source["manifest_sha256"] == base["source_manifest"]["sha256"] and corpus_source["source"] == base["source"], "corpus training source mismatch")
        require(corpus["identity"]["context_policy"] == "ecgqa-cot-baseline@1", "student context differs")
        require(corpus["counts"]["qa_examples"] == corpus["counts"]["accepted"] == contract.generated_rows and corpus["counts"]["generated_waveforms"] == contract.accepted, "corpus count mismatch")
        require(corpus["counts"]["by_family"] == dict(contract.family_counts), "corpus family coverage mismatch")
        ledger_path = generated_corpus / "ledger.json"
        require(file_info(ledger_path)["sha256"] == corpus["files"]["ledger.json"], "generated ledger hash mismatch")
        ledger = json.loads(ledger_path.read_text())
        require(len(ledger) == contract.generated_rows, "generated ledger row count mismatch")
        by_task = {r["task_id"]: r for r in ledger}
        require(len(by_task) == len(ledger), "duplicate generated task IDs")
        require(Counter(r["family"] for r in ledger) == dict(contract.family_counts), "ledger family coverage mismatch")
        require(all(r["decision"] == "accepted" and r["origins"] == ["generated"] and r["answerability"] == "answerable" and len(r["inputs"]) == 1 and r["inputs"][0] in accepted for r in ledger), "ledger includes a non-frozen/non-generated task")
        require({r["inputs"][0] for r in ledger} == set(accepted), "ledger does not cover every accepted waveform")
        per_wave_family = Counter((r["inputs"][0], r["family"]) for r in ledger)
        require(all(count == 1 for count in per_wave_family.values()), "repeated generated family for one waveform")
        for family, count in contract.family_counts:
            if count == contract.accepted:
                require(all(per_wave_family[(wid, family)] == 1 for wid in accepted), "required family missing from an accepted waveform")
        print("Selecting existing generated training rows", flush=True)
        original = pq.read_table(baseline / "train.parquet")
        old_train = pq.read_table(snapshot_prepared / "train.parquet")
        require(len(original) == contract.baseline_rows and len(old_train) == contract.source_rows, "source training row count mismatch")
        require(original.schema.equals(old_train.schema, check_metadata=True) and original.equals(old_train.slice(0, len(original)), check_metadata=True), "source baseline row prefix changed")
        original_rows = original.to_pylist()
        original_ids = original["task_id"].to_pylist()
        require(len(set(original_ids)) == len(original_ids) and not set(original_ids) & set(by_task), "baseline/generated task overlap or duplicates")
        selected_indices = [i for i, tid in enumerate(old_train["task_id"].to_pylist()) if tid in by_task]
        generated = old_train.take(pa.array(selected_indices, type=pa.int64()))
        generated_ids = generated["task_id"].to_pylist()
        require(len(generated) == contract.generated_rows and set(generated_ids) == set(by_task), "prepared generated tasks differ from the exact corpus")
        generated_rows = generated.to_pylist()
        templates = source["qa_export"]["synthetic_template_ids"]
        for row in generated_rows:
            task = by_task[row["task_id"]]
            require(row["record_id"] == task["inputs"][0] and row["template_id"] == templates[task["template"]] and row["question_type"] == task["family"], "prepared task does not match ledger provenance")
        base_records = json.loads((baseline / "records.json").read_text())
        old_records = json.loads((snapshot_prepared / "records.json").read_text())
        require(len(base_records) == contract.baseline_signals and len(set(base_records)) == len(base_records), "baseline signal record count/uniqueness mismatch")
        require(len(old_records) == contract.source_signals and old_records[:len(base_records)] == base_records and len(set(old_records)) == len(old_records), "snapshot record prefix/uniqueness mismatch")
        for row in original_rows:
            require(0 <= row["signal_index"] < len(base_records) and base_records[row["signal_index"]] == row["record_id"], "baseline signal index changed")
        roots = {r["record_id"]: r for r in original_rows}
        appended = sorted((r for r in source["qa_export"]["appended_waveforms"] if r["input"]["origin"] == "generated"), key=lambda r: r["signal_index"])
        require(len(appended) == contract.accepted and {r["input"]["waveform_id"] for r in appended} == set(accepted), "prepared generated waveform set differs")
        require(len({r["signal_index"] for r in appended}) == len(appended), "duplicate generated signal index")
        remap, ref_by_id = {}, {}
        for offset, entry in enumerate(appended):
            ref, index = entry["input"], entry["signal_index"]
            wid = ref["waveform_id"]
            require(len(base_records) <= index < len(old_records) and old_records[index] == wid and wid not in base_records, "generated signal index is not appended")
            root = accepted[wid]["source_record_id"]
            require(root in roots and ref["root_record_ids"] == [root] and ref["root_subject_uids"] == [roots[root]["subject_uid"]], "generated parent is outside baseline training roots")
            require(ref["lineage"].get("review_status") == "verified" and ref["lineage"].get("run_id") == contract.generation_id, "generated lineage is unverified or from another run")
            require(ref["signal"] == {"channels": LEADS, "sample_rate": 100, "num_samples": 1000, "units": "mV"}, "generated signal specification differs")
            remap[index], ref_by_id[wid] = len(base_records) + offset, ref
        for row in generated_rows:
            require(row["signal_index"] in remap and old_records[row["signal_index"]] == row["record_id"] and row["subject_uid"] == ref_by_id[row["record_id"]]["root_subject_uids"][0], "generated row signal/patient lineage mismatch")
        remapped = pa.array([remap[r["signal_index"]] for r in generated_rows], type=generated.schema.field("signal_index").type)
        generated_new = generated.set_column(generated.schema.get_field_index("signal_index"), generated.schema.field("signal_index"), remapped)
        require(generated_new.drop(["signal_index"]).equals(generated.drop(["signal_index"]), check_metadata=True), "generated fields changed beyond signal index")
        combined = pa.concat_tables([original, generated_new])
        base_x = np.load(baseline / "signals.npy", mmap_mode="r", allow_pickle=False)
        old_x = np.load(snapshot_prepared / "signals.npy", mmap_mode="r", allow_pickle=False)
        require(base_x.shape == (contract.baseline_signals, 12, 1000) and old_x.shape == (contract.source_signals, 12, 1000) and base_x.dtype == old_x.dtype == np.dtype("<f4"), "source signal shape/dtype mismatch")
        for start in range(0, len(base_x), 256):
            require(exact_arrays(base_x[start:start+256], old_x[start:min(start+256, len(base_x))]), "source baseline signal bytes changed")
        for entry in appended:
            signal = old_x[entry["signal_index"]]
            require(np.isfinite(signal).all() and waveform_sha256(signal) == entry["input"]["waveform_sha256"], "generated signal differs from its measured content hash")
        for name in HELDOUT:
            require(source["files"][name] == base["files"][name], "source held-out bytes differ from baseline")
        groups = {"schema_version": 1, "version": VERSION,
                  "original": {"start": 0, "count": len(original), "task_ids_sha256": task_ids_sha256(original_ids)},
                  "generated": {"start": len(original), "count": len(generated), "task_ids_sha256": task_ids_sha256(generated_ids)},
                  "train_task_ids_sha256": task_ids_sha256(original_ids + generated_ids),
                  "task_id_hash_encoding": "sha256(json.dumps(ids).encode('utf-8')); Python defaults",
                  "original_train_parquet_sha256": base["files"]["train.parquet"]["sha256"],
                  "original_rows_sha256": counts(original)["tasks_sha256"],
                  "generated_rows_before_signal_remap_sha256": counts(generated)["tasks_sha256"],
                  "baseline_manifest": {"file_sha256": contract.baseline_file_sha256, "content_sha256": base["sha256"]},
                  "snapshot_generated_binding": {"prepared_manifest_file_sha256": contract.prepared_file_sha256, "prepared_manifest_content_sha256": source["sha256"],
                      "snapshot_manifest_file_sha256": contract.snapshot_file_sha256, "snapshot_inventory_sha256": frozen["remote_inventory_sha256"],
                      "generation_id": contract.generation_id, "accepted_waveforms": contract.accepted, "accepted_waveform_ids_sha256": canonical_sha256(sorted(accepted)),
                      "corpus_id": corpus["corpus_id"], "corpus_manifest_file_sha256": contract.corpus_file_sha256, "ledger_sha256": corpus["files"]["ledger.json"],
                      "prepared_train_parquet_sha256": source["files"]["train.parquet"]["sha256"], "family_counts": dict(contract.family_counts)},
                  "builder_sha256": file_sha256(Path(__file__)),
                  "policy": "original baseline rows in original order, then existing generated rows in source prepared order; no grounded or acquisition rows"}
        preflight_groups(combined, groups, contract)
        print("Writing and verifying immutable prepared output", flush=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        try:
            pq.write_table(combined, staging / "train.parquet", compression="zstd")
            for name in HELDOUT:
                shutil.copyfile(baseline / name, staging / name)
                require(file_info(staging / name) == base["files"][name], "held-out copy differs")
            records = base_records + [r["input"]["waveform_id"] for r in appended]
            (staging / "records.json").write_text(json.dumps(records, separators=(",", ":")) + "\n")
            new_x = np.lib.format.open_memmap(staging / "signals.npy", mode="w+", dtype="<f4", shape=(len(records), 12, 1000))
            for start in range(0, len(base_x), 256):
                new_x[start:min(start+256, len(base_x))] = base_x[start:start+256]
            for entry in appended:
                new_x[remap[entry["signal_index"]]] = old_x[entry["signal_index"]]
            new_x.flush()
            for start in range(0, len(base_x), 256):
                require(exact_arrays(new_x[start:min(start+256, len(base_x))], base_x[start:start+256]), "output baseline signal prefix differs")
            for entry in appended:
                require(exact_arrays(new_x[remap[entry["signal_index"]]], old_x[entry["signal_index"]]), "output generated signal differs")
            del new_x
            roundtrip = pq.read_table(staging / "train.parquet")
            require(roundtrip.equals(combined, check_metadata=True), "output training parquet changed row values or schema")
            preflight_groups(roundtrip, groups, contract)
            write_json(staging / "source-manifest.json", groups)
            content = copy.deepcopy({k: v for k, v in base.items() if k not in ("sha256", "files")})
            content["signal"]["shape"] = [len(records), 12, 1000]
            content["splits"]["train"] = counts(combined)
            content["qa_export"] = {"identity": {"version": VERSION, "baseline_manifest_sha256": base["sha256"], "corpus_ids": [corpus["corpus_id"]], "source_prepared_manifest_sha256": source["sha256"], "context_policy": "ecgqa-cot-baseline@1"},
                "baseline_rows_preserved": len(original), "added_training_rows": len(generated), "baseline_signal_prefix": len(base_records),
                "appended_waveforms": [{"signal_index": remap[e["signal_index"]], "input": e["input"]} for e in appended],
                "synthetic_template_ids": {name: templates[name] for name in sorted({r["template"] for r in ledger})},
                "corpora": [r for r in source["qa_export"]["corpora"] if r["corpus_id"] == contract.corpus_id],
                "evaluation_policy": "selection/validation/test copied byte-for-byte; labels not parsed",
                "training_slice_metadata": source["qa_export"].get("training_slice_metadata"),
                "lineage_check": "all generated roots and patients match accepted snapshot and original baseline training rows"}
            content["exposure_source_manifest"] = {"file": "source-manifest.json", **file_info(staging / "source-manifest.json")}
            content["files"] = {name: file_info(staging / name) for name in FILES}
            payload = {**content, "sha256": canonical_sha256(content)}
            write_json(staging / "manifest.json", payload)
            verify_prepared(staging)
            # Recheck source files after use so concurrent source mutation cannot
            # silently become part of the published immutable subset.
            verify_prepared(baseline, contract.baseline_file_sha256, contract.baseline_content_sha256)
            verify_prepared(snapshot_prepared, contract.prepared_file_sha256, contract.prepared_content_sha256)
            for path, pin in ((baseline / "manifest.json", contract.baseline_file_sha256), (snapshot_prepared / "manifest.json", contract.prepared_file_sha256), (generated_corpus / "corpus.json", contract.corpus_file_sha256), (snapshot_manifest, contract.snapshot_file_sha256), (ledger_path, corpus["files"]["ledger.json"])):
                require(file_sha256(path) == pin, "source identity changed during preparation")
            write_json(staging / "builder-audit.json", {"schema_version": 1, "status": "verified", "builder_sha256": groups["builder_sha256"], "prepared_manifest": file_info(staging / "manifest.json"),
                "source_manifest": file_info(staging / "source-manifest.json"), "original_rows": len(original), "generated_rows": len(generated), "total_rows": len(combined), "signals": len(records),
                "baseline_row_order_and_values_preserved": True, "baseline_signal_prefix_bitwise_identical": True, "generated_fields_unchanged_except_signal_index": True,
                "heldout_copied_byte_exact_without_parsing": True, "accepted_generated_signals_content_hash_verified": len(appended), "no_regeneration_or_remeasurement": True})
            require(not output.exists(), "output appeared during preparation")
            os.rename(staging, output)
            print(f"Published {len(combined)} rows and {len(records)} signals: {output}", flush=True)
            return payload
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("baseline", "snapshot-prepared", "generated-corpus", "snapshot-manifest", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    build(args.baseline, args.snapshot_prepared, args.generated_corpus, args.snapshot_manifest, args.output)


if __name__ == "__main__":
    main()
