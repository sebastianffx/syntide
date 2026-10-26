#!/usr/bin/env python3
"""Frozen offline DiffuSETS classification experiment; never runs without an explicit label policy.

Run with the project's environment:
  uv run python runs/ptbxl-diffusets-1530/run_experiment.py plan
  uv run python runs/ptbxl-diffusets-1530/run_experiment.py train
  uv run python runs/ptbxl-diffusets-1530/run_experiment.py verify
  uv run python runs/ptbxl-diffusets-1530/run_experiment.py evaluate

The adjacent protocol.json must explicitly authorize provisional conditioning-parent diagnostic
labels and bind data/manifest.json by SHA256. This script does not infer authorization from a
generated catalog, acceptance status, or other external metadata. The existing generator's
rhythm/quality acceptance is not diagnostic-label validation.

Data manifest schema: {"schema_version":"ptbxl-offline-training@1", "arms":{
  "G1":{"path":"data/G1.npz","sha256":"..."},
  "R1":{"path":"data/R1.npz","sha256":"..."}}}.
Paths above resolve against this experiment directory. Each NPZ has x(float32[N,12,1000]),
labels(bool[N,5]), artifacts(bool[N,4]), record_id/subject_uid/root_record_id/root_subject_uid
(Unicode[N]), and is_synthetic(bool[N]). N=10109. First 8579 rows are the frozen pool in its
manifest order; final 1530 rows are aligned generated/real-parent pairs, preserving multiplicity.
R1 repeated real parents use unique record aliases; root_record_id retains their actual identity.
G1 artifact flags are all-false placeholders for unknown synthetic quality, never used as labels.

The generic prepare()/trainer CLI cannot load this offline arm. Custom source roles deliberately
make that incorrect route fail; this driver uses open_trial() and the unchanged torch_loop.fit().
Only this experiment directory is written. Original baselines are referenced and hash-checked.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
ARMS = ("G1", "R1")
SEEDS = (0, 1, 2)
N_REAL, N_EXTRA, N_TOTAL = 8579, 1530, 10109
STEPS, BATCH, CADENCE = 6500, 64, 100
POLICY = "provisional_conditioning_parent_superclasses"
ANALYSIS = {
    "primary": ["G1", "C2"], "key_control": ["G1", "R1"],
    "diagnostic": ["R1", "C2"], "contrasts": [["G1", "C2"], ["G1", "R1"], ["R1", "C2"]],
    "slices": ["overall", "clean", "artifact"], "n_boot": 2000,
    "ci_level": 0.95, "bootstrap_seed": 0,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(value, base=ROOT):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def write_new(path, payload):
    """Create-only scientific records; equal repeated calls are no-ops."""
    path = Path(path)
    require(path.resolve().is_relative_to(ROOT), f"Output escapes experiment directory: {path}")
    if path.exists():
        require(read(path) == payload, f"Frozen output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def load_protocol():
    protocol = read(ROOT / "protocol.json")
    require(protocol.get("schema_version") == "ptbxl-diffusets-classification@1", "Wrong protocol schema")
    require(protocol.get("id") == ROOT.name, "Protocol ID must equal this experiment directory name")
    require(protocol.get("regime") == "scarcity50", "Expected scarcity50 regime")
    require(protocol.get("label_policy") == POLICY, "No supported explicit diagnostic label policy")
    require(protocol.get("label_policy_authorized") is True, "Provisional diagnostic labels are not authorized")
    require(isinstance(protocol.get("label_policy_authorization"), str)
            and bool(protocol["label_policy_authorization"].strip()), "Missing explicit user authorization record")
    for key, expected in (("seeds", list(SEEDS)), ("batch_size", BATCH),
                          ("max_steps", STEPS), ("val_every", CADENCE)):
        require(protocol.get(key) == expected, f"Protocol {key} must be {expected}")
    require(protocol.get("analysis") == ANALYSIS, "Analysis must explicitly freeze the planned contrasts and bootstrap")
    require(protocol.get("lineage_allowed_roles") in (["pool"], ["pool", "unused"]),
            "Explicit conditioning lineage roles must be pool, optionally with unused training patients")
    require(protocol.get("dataset") == "physionet/ptb-xl@1.0.3", "Wrong evaluation dataset")
    return protocol


def baseline_state(protocol):
    """Pin original outputs without writing or rescoring any baseline."""
    from syntide.trial import init_seed, load_spec
    from syntide_trainers.torch_loop import ModelConfig, OptimConfig, TorchLoopConfig, trainer_info

    original = local_path(protocol["original_run"], PROJECT)
    require(original != ROOT and not original.is_relative_to(ROOT) and not ROOT.is_relative_to(original),
            "Original run and experiment must be separate")
    paths = [original / "plan.json", original / "test_claim.json"]
    baselines = {}
    for seed in SEEDS:
        directory = original / "scarcity50" / "C2" / f"seed{seed}"
        spec, result = load_spec(directory), read(directory / "result.json")
        require(result["status"] == "complete" and result["trial_id"] == spec.trial_id,
                f"Incomplete or mismatched baseline seed {seed}")
        require((spec.batch_size, spec.max_steps, spec.val_every, spec.init_seed) ==
                (BATCH, STEPS, CADENCE, init_seed(seed)), f"Baseline settings differ for seed {seed}")
        require(spec.source_role == "pool" and spec.transform_family == "none" and spec.transform_fraction == 0,
                f"Baseline seed {seed} is not the unaugmented full 50% pool")
        require(result["exposure"]["presentations"] == STEPS * BATCH, "Baseline exposure budget differs")
        for name in ("trial.json", "result.json", "test_scores.json", "best.pt", "final.pt",
                     "predictions_val.parquet", "predictions_test.parquet"):
            paths.append(directory / name)
        for name in ("best", "final"):
            checkpoint = result["provenance"]["checkpoints"][name]
            require(sha(directory / checkpoint["path"]) == checkpoint["sha256"],
                    f"Original {seed}/{name} checkpoint changed")
        baselines[seed] = {"directory": directory, "spec": spec, "result": result}
    devices = {b["result"]["trainer"]["device"] for b in baselines.values()}
    require(len(devices) == 1, "Baseline devices differ")
    cfg = TorchLoopConfig(model=ModelConfig(width=64, kernel_size=7),
                          optim=OptimConfig(lr=0.001, weight_decay=0.01), device=next(iter(devices)))
    info = trainer_info(cfg)
    require(all(b["result"]["trainer"]["fingerprint"] == info.fingerprint for b in baselines.values()),
            "Current reference trainer/config/device fingerprint differs from original C2")
    hashes = {str(p.relative_to(PROJECT)) if p.is_relative_to(PROJECT) else str(p): sha(p) for p in paths}
    return baselines, cfg, info, hashes


def load_inputs(protocol, baselines):
    """Read training/validation only; test waveform loading remains inside holdout.evaluate()."""
    from syntide.data.adapter import PtbXlAdapter
    from syntide.data.challenge import load_manifest
    from syntide.data.leakage import check_lineage, check_manifest
    from syntide.data.schema import LEADS, Samples, SUPERCLASSES

    manifest_path = local_path(protocol["manifest"], PROJECT)
    registry = local_path(protocol["registry"], PROJECT)
    manifest = load_manifest(manifest_path)
    adapter = PtbXlAdapter(registry, protocol["dataset"])
    index, source = adapter.index(), adapter.source()
    check_manifest(manifest, index, source)
    require(len(manifest.roles["pool"]) == N_REAL and len(manifest.roles["val"]) == 2146
            and len(manifest.roles["test"]) == 2158, "Original split counts changed")
    for baseline in baselines.values():
        spec = baseline["spec"]
        require(spec.source_sha256 == source["sha256"] and spec.manifest_sha256 == manifest.sha256,
                "Input dataset or manifest differs from original C2")
    real = adapter.load(manifest.roles["pool"])
    val = adapter.load(manifest.roles["val"])
    rows = {row["record_id"]: row for row in index.to_pylist()}
    ref = protocol["data_manifest"]
    data_path = local_path(ref["path"])
    require(sha(data_path) == ref["sha256"], "Data manifest does not match frozen protocol")
    data = read(data_path)
    require(data.get("schema_version") == "ptbxl-offline-training@1" and set(data.get("arms", {})) == set(ARMS),
            "Data manifest must name exactly G1 and R1")
    loaded, hashes = {}, {str(data_path): sha(data_path)}
    metadata = {}
    for name in ("conditioning_labels", "provenance"):
        reference = data[name]
        path = local_path(reference["path"])
        digest = sha(path)
        require(digest == reference["sha256"], f"Data {name} metadata hash differs")
        hashes[str(path)] = digest
        metadata[name] = read(path)
    provenance = metadata["provenance"]
    require(data.get("label_policy") == POLICY and provenance.get("label_policy") == POLICY,
            "Input provenance does not describe the authorized provisional label policy")
    require(provenance.get("class_order") == list(SUPERCLASSES)
            and provenance.get("lead_order") == list(LEADS)
            and provenance.get("sample_rate_hz") == 100 and provenance.get("units") == "mV",
            "Input provenance signal units, lead order, sample rate, or class order differs")
    require(provenance.get("source_sha256") == source["sha256"]
            and provenance.get("classification_manifest_sha256") == manifest.sha256,
            "Input provenance is tied to another real dataset or classification split")
    conditioning = metadata["conditioning_labels"]
    require(isinstance(conditioning, list) and len(conditioning) == N_EXTRA,
            "Expected exactly 1530 conditioning-label provenance rows")
    required = {"x", "labels", "artifacts", "record_id", "subject_uid", "root_record_id",
                "root_subject_uid", "is_synthetic"}
    for arm in ARMS:
        entry = data["arms"][arm]
        path = local_path(entry["path"])
        digest = sha(path)
        require(digest == entry["sha256"], f"{arm} NPZ hash differs")
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == required, f"{arm} NPZ fields differ: {set(archive.files) ^ required}")
            a = {key: archive[key] for key in required}
        require(a["x"].shape == (N_TOTAL, 12, 1000) and a["x"].dtype == np.float32
                and np.isfinite(a["x"]).all(), f"{arm} needs finite float32 mV ECGs [10109,12,1000]")
        for field, shape in (("labels", (N_TOTAL, 5)), ("artifacts", (N_TOTAL, 4)),
                             ("is_synthetic", (N_TOTAL,))):
            require(a[field].shape == shape and a[field].dtype == np.bool_, f"{arm}/{field} needs boolean {shape}")
        for field in ("record_id", "subject_uid", "root_record_id", "root_subject_uid"):
            require(a[field].shape == (N_TOTAL,) and a[field].dtype.kind == "U"
                    and np.all(np.char.str_len(a[field]) > 0), f"{arm}/{field} needs nonempty Unicode IDs")
        require(len(set(a["record_id"])) == N_TOTAL, f"{arm} needs unique record aliases, including repeated parents")
        for field in ("x", "labels", "artifacts", "record_id", "subject_uid"):
            require(np.array_equal(a[field][:N_REAL], getattr(real, field)), f"{arm}'s original pool {field} changed")
        require(np.array_equal(a["root_record_id"][:N_REAL], real.record_id)
                and np.array_equal(a["root_subject_uid"][:N_REAL], real.subject_uid), "Real-row roots differ")
        expected_synthetic = np.zeros(N_TOTAL, dtype=bool)
        expected_synthetic[N_REAL:] = arm == "G1"
        require(np.array_equal(a["is_synthetic"], expected_synthetic), f"{arm} synthetic-row mask differs")
        if arm == "G1":
            require(not a["artifacts"][N_REAL:].any(), "Synthetic quality flags must be unknown/all-false placeholders")
        ancestry = [(str(a["record_id"][i]), [str(a["root_record_id"][i])],
                     [str(a["root_subject_uid"][i])]) for i in range(N_REAL, N_TOTAL)]
        check_lineage(ancestry, manifest, index, protocol["lineage_allowed_roles"])
        labels = np.asarray([[label in rows[str(root)]["labels"] for label in SUPERCLASSES]
                             for root in a["root_record_id"]], dtype=bool)
        require(np.array_equal(a["labels"], labels), f"{arm} labels are not the explicit conditioning-parent superclasses")
        loaded[arm] = a
        hashes[str(path)] = digest
    for field in ("root_record_id", "root_subject_uid", "labels"):
        require(np.array_equal(loaded["G1"][field], loaded["R1"][field]), f"Generated/real-parent pair alignment differs: {field}")
    for offset, declared in enumerate(conditioning):
        i = N_REAL + offset
        root = str(loaded["G1"]["root_record_id"][i])
        require(declared.get("record_id") == loaded["G1"]["record_id"][i]
                and declared.get("conditioning_record_id") == root
                and declared.get("conditioning_subject_uid") == loaded["G1"]["root_subject_uid"][i]
                and sorted(declared.get("conditioning_labels", [])) == sorted(rows[root]["labels"])
                and declared.get("label_policy") == POLICY,
                f"Conditioning metadata does not match appended NPZ row {i}")
    parents = sorted(set(map(str, loaded["R1"]["root_record_id"][N_REAL:])))
    actual = adapter.load(parents)
    position = {r: i for i, r in enumerate(actual.record_id)}
    for i in range(N_REAL, N_TOTAL):
        root = loaded["R1"]["root_record_id"][i]
        j = position[root]
        require(np.array_equal(loaded["R1"]["x"][i], actual.x[j])
                and np.array_equal(loaded["R1"]["artifacts"][i], actual.artifacts[j])
                and loaded["R1"]["subject_uid"][i] == actual.subject_uid[j],
                f"R1 appended row {i} is not the unchanged corresponding real parent")
    samples = {arm: Samples(**{field: values[field] for field in
               ("record_id", "subject_uid", "labels", "artifacts", "x")}) for arm, values in loaded.items()}
    extra_roots = loaded["G1"]["root_record_id"][N_REAL:]
    real_ids = set(real.record_id)
    outside = [str(r) for r in extra_roots if r not in real_ids]
    audit = {"real_records": N_REAL, "extra_records": N_EXTRA, "total_training_rows": N_TOTAL,
             "extra_distinct_parent_records": len(set(extra_roots)),
             "extra_rows_with_parent_outside_pool": len(outside),
             "extra_distinct_parents_outside_pool": len(set(outside)),
             "conditioning_lineage_roles": protocol["lineage_allowed_roles"],
             "label_policy": POLICY, "synthetic_artifact_flags": "unknown; all-false training placeholders only",
             "diagnostic_labels_verified_on_generated_signals": False,
             "sample_rate_hz": 100, "duration_seconds": 10, "units": "mV",
             "classes": list(SUPERCLASSES), "validation_ecgs": len(val), "test_ecgs": len(manifest.roles["test"])}
    return samples, val, manifest, source, hashes, audit, str(manifest_path), str(registry)


def setup():
    protocol = load_protocol()
    baselines, cfg, info, original_hashes = baseline_state(protocol)
    inputs = load_inputs(protocol, baselines)
    return protocol, baselines, cfg, info, original_hashes, inputs


def planned_specs(protocol, inputs):
    from syntide.trial import make_spec
    _, _, manifest, source, hashes, _, manifest_path, registry = inputs
    specs = []
    for arm in ARMS:
        metadata = {"offline_loader": "run_experiment.py@1", "offline_arm": arm,
                    "protocol_sha256": sha(ROOT / "protocol.json"), "data_files_sha256": hashes,
                    "label_policy": POLICY, "real_count": N_REAL, "extra_count": N_EXTRA,
                    "sampling": "one seeded permutation of all rows per pass; no online transform"}
        for seed in SEEDS:
            specs.append(make_spec(protocol=protocol["id"], regime=protocol["regime"], arm=arm,
                replicate_seed=seed, source_role="pool_plus_offline_" + arm.lower(),
                transform_family="none", sampling_policy="none", generator="diffusets@1" if arm == "G1" else None,
                validators=[], transform=metadata, dataset=protocol["dataset"], source_sha256=source["sha256"],
                manifest_sha256=manifest.sha256, calibration_sha256=None, batch_size=BATCH, max_steps=STEPS,
                val_every=CADENCE, transform_fraction=0.0, registry=registry, manifest=manifest_path, calibration=None))
    return specs


def frozen_payload(protocol, original_hashes, inputs):
    return {"schema_version": "ptbxl-classification-freeze@1", "protocol_sha256": sha(ROOT / "protocol.json"),
            "runner_sha256": sha(Path(__file__)), "plan_sha256": sha(ROOT / "plan.json"),
            "input_files_sha256": inputs[4], "original_files_sha256": original_hashes,
            "analysis": protocol["analysis"], "label_policy": protocol["label_policy"],
            "test_interpretation": "exploratory extension; original PTB-XL test results already viewed"}


def verify_freeze(protocol, original_hashes, inputs):
    frozen = read(ROOT / "experiment_freeze.json")
    expected = frozen_payload(protocol, original_hashes, inputs)
    require({k: v for k, v in frozen.items() if k != "frozen_at"} == expected,
            "Experiment protocol, code, plan, data, analysis, or original files changed after freezing")
    for value, expected_sha in frozen["original_files_sha256"].items():
        require(sha(local_path(value, PROJECT)) == expected_sha, f"Original output changed: {value}")
    for value, expected_sha in frozen["input_files_sha256"].items():
        require(sha(Path(value)) == expected_sha, f"Offline input changed: {value}")
    if (ROOT / "test_claim.json").exists():
        claim = read(ROOT / "test_claim.json")
        require(datetime.fromisoformat(frozen["frozen_at"]) <= datetime.fromisoformat(claim["claimed_at"]),
                "Test was claimed before the experiment was frozen")
    return frozen


def plan(context):
    from syntide.generate.plan import write_plan
    protocol, _, _, _, original_hashes, inputs = context
    require(not (ROOT / "test_claim.json").exists(), "Do not plan after the test claim")
    specs = planned_specs(protocol, inputs)
    write_plan(specs, ROOT.parent)
    payload = frozen_payload(protocol, original_hashes, inputs)
    path = ROOT / "experiment_freeze.json"
    if path.exists():
        verify_freeze(protocol, original_hashes, inputs)
    else:
        require(not any(ROOT.glob("*/*/seed*/result.json")), "Freeze before any training result is written")
        write_new(path, {**payload, "frozen_at": datetime.now(UTC).isoformat(timespec="seconds")})
    write_new(ROOT / "input_audit.json", inputs[5])
    print(f"Frozen {len(specs)} trials; inputs verified. No training or test evaluation performed.", flush=True)


def verify_training(context):
    from syntide.eval.runs import load_run
    from syntide.trial import load_spec
    protocol, baselines, _, info, original_hashes, inputs = context
    verify_freeze(protocol, original_hashes, inputs)
    expected = {s.run_name: s for s in planned_specs(protocol, inputs)}
    entries = read(ROOT / "plan.json")["trials"]
    require(len(entries) == 6 and {e["run_dir"] for e in entries} == set(expected), "Expected exactly six planned trials")
    checked = []
    for entry in entries:
        directory = ROOT / entry["run_dir"]
        run = load_run(directory, "val")
        spec, result = load_spec(directory), run.result
        require(spec.to_dict() == expected[entry["run_dir"]].to_dict() and spec.trial_id == entry["trial_id"],
                f"Trial spec differs from frozen offline inputs: {directory}")
        baseline = baselines[spec.replicate_seed]["result"]
        require(result["trainer"]["fingerprint"] == info.fingerprint
                and result["trainer"]["init_sha256"] == baseline["trainer"]["init_sha256"],
                f"Trainer or seed initialization differs: {directory}")
        history = result["val_history"]
        require([h["step"] for h in history] == list(range(CADENCE, STEPS + 1, CADENCE)), "Validation grid must have 65 rows")
        best = max(history, key=lambda h: h["val/auroc_overall"])
        require(result["best_step"] == best["step"] and result["best_val_auroc"] == best["val/auroc_overall"],
                "Checkpoint is not the first maximum of validation overall AUROC")
        require(result["final_step"] == STEPS and result["exposure"]["presentations"] == STEPS * BATCH
                and result["exposure"]["transform_slots"] == 0, "Training exposure differs")
        checked.append({"run_dir": entry["run_dir"], "trial_id": spec.trial_id, "best_step": result["best_step"],
                        "best_val_auroc": result["best_val_auroc"], "initial_weights_match_original": True,
                        "trainer_matches_original": True, "validation_rows": len(history)})
    write_new(ROOT / "training_audit.json", {"passed": True, "trials": checked})
    return checked


def train(context):
    import torch
    from syntide.eval.tracking import TrackerConfig, make_tracker
    from syntide.trial import is_complete, load_spec, open_trial
    from syntide_trainers.torch_loop import ResNet1d, fit, state_sha256

    protocol, baselines, cfg, info, original_hashes, inputs = context
    verify_freeze(protocol, original_hashes, inputs)
    require(not (ROOT / "test_claim.json").exists(), "Do not train after a test claim")
    if cfg.device == "mps":
        require(torch.backends.mps.is_available(), "Original MPS device is unavailable; do not silently change device")
    elif cfg.device == "cuda":
        require(torch.cuda.is_available(), "Original CUDA device is unavailable; do not silently change device")
    samples, val = inputs[:2]
    for number, spec in enumerate(planned_specs(protocol, inputs), 1):
        directory = ROOT / spec.run_name
        require(load_spec(directory).to_dict() == spec.to_dict(), f"Planned spec differs: {directory}")
        torch.manual_seed(spec.init_seed)
        probe = ResNet1d(**asdict(cfg.model))
        require(state_sha256(probe) == baselines[spec.replicate_seed]["result"]["trainer"]["init_sha256"],
                f"Seed {spec.replicate_seed} initialization differs before training")
        del probe
        if is_complete(directory, info):
            print(f"[{number}/6] {spec.run_name}: already complete", flush=True)
            continue
        print(f"[{number}/6] Training {spec.run_name}: {len(samples[spec.arm])} rows", flush=True)
        tracker = make_tracker(TrackerConfig(backend="files"), directory, spec.run_name, spec.protocol, spec.to_dict())
        with open_trial(spec, samples[spec.arm], val, tracker, directory) as trial:
            result = fit(trial, cfg, info)
        print(f"[{number}/6] {spec.run_name}: best {result['best_val_auroc']:.6f} at {result['best_step']}", flush=True)
    verify_training(context)


def evaluate(context):
    from syntide.eval.holdout import evaluate as score_holdout
    from syntide.eval.runs import load_runs

    protocol, _, _, _, original_hashes, inputs = context
    verify_training(context)
    verify_freeze(protocol, original_hashes, inputs)
    scores = score_holdout(ROOT)
    verify_freeze(protocol, original_hashes, inputs)
    runs, pending = load_runs(ROOT, "test")
    require(not pending and len(runs) == 6 and len(scores) == 6, "Test scoring is incomplete")
    require(all(s["n_test"] == 2158 for s in scores), "Test record count changed")
    write_new(ROOT / "evaluation_audit.json", {"passed": True, "test_runs": 6, "n_test": 2158,
              "frozen_before_test_claim": True, "original_outputs_unchanged": True,
              "primary": protocol["analysis"]["primary"], "key_control": protocol["analysis"]["key_control"]})
    print("Six held-out evaluations verified. Compare saved predictions with original C2 using PairedScorer.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "train", "verify", "evaluate"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    context = setup()
    {"plan": plan, "train": train, "verify": verify_training, "evaluate": evaluate}[args.command](context)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"Experiment refused: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
