"""Small CPU fixtures; held-out files deliberately are not readable parquet."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prepare_exposure_data", HERE / "prepare_exposure_data.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.base, self.source, self.corpus, self.output = (self.root / x for x in ("base", "source", "corpus", "output"))
        for p in (self.base, self.source, self.corpus):
            p.mkdir()
        (self.root / "snapshot").mkdir()
        self.snapshot = self.root / "snapshot" / "snapshot.json"
        self.gen = "test-generation"
        self.wids = [f"synthetic:{self.gen}:{i:08d}" for i in range(2)]
        self.rows = [self.row("base-a", "root-a", 0), self.row("base-b", "root-b", 1)]
        self.base_x = np.arange(3 * 12 * 1000, dtype=np.float32).reshape(3, 12, 1000)
        self.base_x[0, 0, 0] = -0.0
        self.source_x = np.concatenate([self.base_x, np.full((1, 12, 1000), 999, dtype=np.float32), np.full((1, 12, 1000), 42, dtype=np.float32), np.full((1, 12, 1000), 43, dtype=np.float32)])
        source_rows = self.rows + [self.row("grounded", "root-a", 0), self.row("acquisition", "acq-a", 3)]
        self.ledger = []
        # Generated rows deliberately have a different order from signal slots.
        for i in (1, 0):
            for family in ("rhythm.beat_count", "rhythm.heart_rate"):
                tid = f"gen-{i}-{family}"
                row = self.row(tid, self.wids[i], 4+i, family=family)
                row["subject_uid"] = f"patient-{i}"
                row["template_id"] = 1011 if family.endswith("beat_count") else 1012
                row["answer_options"] = []
                source_rows.append(row)
                self.ledger.append({"task_id": tid, "inputs": [self.wids[i]], "origins": ["generated"], "answerability": "answerable", "decision": "accepted", "family": family, "template": family + "@1"})
        self.table = pa.Table.from_pylist(self.rows)
        self.source_table = pa.Table.from_pylist(source_rows, schema=self.table.schema)
        self.base_manifest = self.prepared(self.base, self.table, self.base_x, ["root-a", "root-b", "heldout-c"])
        refs = []
        for i in range(2):
            refs.append({"signal_index": 4+i, "input": {"origin": "generated", "waveform_id": self.wids[i], "waveform_sha256": m.waveform_sha256(self.source_x[4+i]), "root_record_ids": ["root-a" if i == 0 else "root-b"], "root_subject_uids": [f"patient-{i}"], "lineage": {"review_status": "verified", "run_id": self.gen}, "signal": {"channels": m.LEADS, "sample_rate": 100, "num_samples": 1000, "units": "mV"}}})
        self.source_manifest = self.prepared(self.source, self.source_table, self.source_x, ["root-a", "root-b", "heldout-c", "acq-a"] + self.wids)
        self.source_manifest["qa_export"] = {"identity": {"baseline_manifest_sha256": self.base_manifest["sha256"]}, "appended_waveforms": [{"signal_index": 3, "input": {"origin": "transformed"}}] + refs,
            "synthetic_template_ids": {"rhythm.beat_count@1": 1011, "rhythm.heart_rate@1": 1012},
            "corpora": [{"corpus_id": "other"}, {"corpus_id": "test-corpus", "counts": {"exported_tasks": 4}}], "training_slice_metadata": "inherited"}
        self.save_manifest(self.source, self.source_manifest)
        slots = [{"accepted": True, "slot_id": i, "status": "verified", "record_id": wid, "source_record_id": "root-a" if i == 0 else "root-b"} for i, wid in enumerate(self.wids)]
        inv = {"generation_id": self.gen, "candidate_count": 2, "accepted_count": 2, "rejected_count": 0, "prefix_slot_ids": [0, 1], "accepted_slot_ids": [0, 1], "slots": slots}
        snapshot = {"schema": "syntide.diffusets-qa-snapshot@1", "local_and_remote_hashes_verified": True, "remote_inventory": inv,
                    "remote_inventory_sha256": m.hashlib.sha256(json.dumps(inv, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
        m.write_json(self.snapshot, snapshot)
        m.write_json(self.corpus / "ledger.json", self.ledger)
        self.corpus_info = {"schema_version": "syntide.qa-corpus@1", "status": "published", "corpus_id": "test-corpus", "identity": {"purpose": "training", "context_policy": "ecgqa-cot-baseline@1", "source": {"purpose": "training", "role": "pool", "manifest_sha256": "split-sha", "source": {"dataset": "test"}}},
            "counts": {"qa_examples": 4, "accepted": 4, "generated_waveforms": 2, "by_family": {"rhythm.beat_count": 2, "rhythm.heart_rate": 2}}, "files": {"ledger.json": m.file_sha256(self.corpus / "ledger.json")}}
        m.write_json(self.corpus / "corpus.json", self.corpus_info)
        self.contract = m.Contract(baseline_rows=2, generated_rows=4, accepted=2, attempts=2, baseline_signals=3, source_rows=8, source_signals=6,
            baseline_file_sha256=m.file_sha256(self.base / "manifest.json"), baseline_content_sha256=self.base_manifest["sha256"],
            prepared_file_sha256=m.file_sha256(self.source / "manifest.json"), prepared_content_sha256=self.source_manifest["sha256"],
            snapshot_file_sha256=m.file_sha256(self.snapshot), corpus_file_sha256=m.file_sha256(self.corpus / "corpus.json"), corpus_id="test-corpus", generation_id=self.gen,
            family_counts=(("rhythm.beat_count", 2), ("rhythm.heart_rate", 2)))

    def row(self, tid, rid, index, family="original"):
        return {"task_id": tid, "record_id": rid, "signal_index": index, "subject_uid": "patient-0" if rid != "root-b" else "patient-1",
                "question": f"question for {tid}", "answer": "42", "answer_options": ["41", "42"], "rationale": "Exact rationale.\nKeep punctuation.", "has_rationale": True,
                "clinical_context": "Clinical context", "template_id": 12, "question_type": family, "cot_split": "train", "strat_fold": 1, "is_artifact": False, "artifact_types": ["none"]}

    def prepared(self, path, table, signals, records):
        pq.write_table(table, path / "train.parquet")
        for name in m.HELDOUT:
            (path / name).write_bytes(b"NOT PARQUET: held-out data must stay opaque " + name.encode())
        np.save(path / "signals.npy", signals, allow_pickle=False)
        m.write_json(path / "records.json", records)
        manifest = {"schema_version": 1, "preparation": {"train_role": "pool"}, "source_manifest": {"sha256": "split-sha"}, "source": {"dataset": "test"},
                    "signal": {"dtype": "float32", "shape": list(signals.shape)}, "splits": {"train": m.counts(table), "selection": {"questions": 1}, "validation": {"questions": 1}, "test": {"questions": 1}}}
        self.save_manifest(path, manifest)
        return manifest

    def save_manifest(self, path, manifest):
        manifest["files"] = {name: m.file_info(path / name) for name in m.FILES}
        manifest["sha256"] = m.canonical_sha256({k: v for k, v in manifest.items() if k != "sha256"})
        m.write_json(path / "manifest.json", manifest)

    def repin_source(self):
        self.save_manifest(self.source, self.source_manifest)
        self.contract = replace(self.contract, prepared_file_sha256=m.file_sha256(self.source / "manifest.json"), prepared_content_sha256=self.source_manifest["sha256"])

    def build(self):
        return m.build(self.base, self.source, self.corpus, self.snapshot, self.output, self.contract)


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(self.tmp.name)

    def test_exact_subset_preserves_rows_signals_and_opaque_heldout(self):
        actual_read = pq.read_table
        def only_train(path, *args, **kwargs):
            self.assertEqual(Path(path).name, "train.parquet", "held-out parquet must never be parsed")
            return actual_read(path, *args, **kwargs)
        with mock.patch.object(m.pq, "read_table", side_effect=only_train):
            manifest = self.f.build()
        out = actual_read(self.f.output / "train.parquet")
        self.assertTrue(out.slice(0, 2).equals(actual_read(self.f.base / "train.parquet"), check_metadata=True))
        self.assertEqual(out["task_id"].to_pylist()[2:], self.f.source_table["task_id"].to_pylist()[4:])
        self.assertTrue(out.slice(2).drop(["signal_index"]).equals(actual_read(self.f.source / "train.parquet").slice(4).drop(["signal_index"]), check_metadata=True))
        self.assertEqual(out["signal_index"].to_pylist(), [0, 1, 4, 4, 3, 3])
        x = np.load(self.f.output / "signals.npy")
        self.assertTrue(m.exact_arrays(x[:3], self.f.base_x))
        self.assertTrue(m.exact_arrays(x[3:], self.f.source_x[4:]))
        self.assertTrue(np.signbit(x[0, 0, 0]))
        for name in m.HELDOUT:
            self.assertEqual((self.f.output / name).read_bytes(), (self.f.base / name).read_bytes())
        self.assertEqual(set(manifest["files"]), set(m.FILES))
        groups = json.loads((self.f.output / "source-manifest.json").read_text())
        m.preflight_groups(out, groups, self.f.contract)
        self.assertEqual(manifest["qa_export"]["identity"]["corpus_ids"], ["test-corpus"])
        self.assertEqual(json.loads((self.f.output / "builder-audit.json").read_text())["status"], "verified")
        m.verify_prepared(self.f.output)

    def test_original_prefix_change_rejected_even_with_updated_source_manifest(self):
        rows = self.f.source_table.to_pylist()
        rows[0]["answer"] = "changed"
        pq.write_table(pa.Table.from_pylist(rows, schema=self.f.table.schema), self.f.source / "train.parquet")
        self.f.repin_source()
        with self.assertRaisesRegex(ValueError, "baseline row prefix changed"):
            self.f.build()
        self.assertFalse(self.f.output.exists())

    def test_generated_signal_content_mismatch_rejected(self):
        self.f.source_x[4, 0, 0] = 1
        np.save(self.f.source / "signals.npy", self.f.source_x)
        self.f.repin_source()
        with self.assertRaisesRegex(ValueError, "measured content hash"):
            self.f.build()

    def test_baseline_signal_negative_zero_change_rejected(self):
        self.f.source_x[0, 0, 0] = 0.0
        np.save(self.f.source / "signals.npy", self.f.source_x)
        self.f.repin_source()
        with self.assertRaisesRegex(ValueError, "baseline signal bytes changed"):
            self.f.build()

    def test_nonfrozen_record_in_ledger_rejected(self):
        self.f.ledger[0]["inputs"] = ["other-waveform"]
        m.write_json(self.f.corpus / "ledger.json", self.f.ledger)
        self.f.corpus_info["files"]["ledger.json"] = m.file_sha256(self.f.corpus / "ledger.json")
        m.write_json(self.f.corpus / "corpus.json", self.f.corpus_info)
        self.f.contract = replace(self.f.contract, corpus_file_sha256=m.file_sha256(self.f.corpus / "corpus.json"))
        with self.assertRaisesRegex(ValueError, "non-frozen/non-generated"):
            self.f.build()

    def test_altered_heldout_source_bytes_rejected_without_parsing(self):
        (self.f.base / "test.parquet").write_bytes(b"altered")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.f.build()

    def test_prepared_task_ledger_mismatch_rejected(self):
        rows = self.f.source_table.to_pylist()
        rows[4]["record_id"] = self.f.wids[0]
        pq.write_table(pa.Table.from_pylist(rows, schema=self.f.table.schema), self.f.source / "train.parquet")
        self.f.repin_source()
        with self.assertRaisesRegex(ValueError, "ledger provenance"):
            self.f.build()

    def test_existing_output_refused_and_source_manifest_tampering_detected(self):
        self.f.build()
        with self.assertRaisesRegex(ValueError, "refusing existing output"):
            self.f.build()
        groups = json.loads((self.f.output / "source-manifest.json").read_text())
        groups["generated"]["start"] = 0
        with self.assertRaisesRegex(ValueError, "generated boundaries"):
            m.preflight_groups(pq.read_table(self.f.output / "train.parquet"), groups, self.f.contract)
        m.write_json(self.f.output / "source-manifest.json", groups)
        with self.assertRaisesRegex(ValueError, "source manifest binding"):
            m.verify_prepared(self.f.output)

    def test_unchanged_baseline_manifest_loader_accepts_output(self):
        self.f.build()
        candidates = [Path("/workspace/baseline-runtime/src/syntide/baseline/data.py")]
        candidates.extend(parent / "synthetic_data_engine-runpod-baseline/src/syntide/baseline/data.py" for parent in HERE.parents)
        baseline_module = next((path for path in candidates if path.is_file()), None)
        if baseline_module is None:
            self.skipTest("sibling baseline source unavailable")
        sys.path.insert(0, str(baseline_module.parents[2]))
        self.addCleanup(lambda: sys.path.pop(0))
        spec = importlib.util.spec_from_file_location("frozen_baseline_data", baseline_module)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        checked = module.load_export_manifest(self.f.output)
        self.assertEqual(checked["sha256"], m.verify_prepared(self.f.output)["sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
