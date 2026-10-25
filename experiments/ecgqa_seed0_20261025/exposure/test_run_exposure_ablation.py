"""CPU-only runner checks: no model load, subprocess, remote, or held-out read."""
from contextlib import ExitStack
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m = load("runner_under_review", HERE / "run_exposure_ablation.py")
h = load("exposure_helper_for_runner_tests", HERE / "trainer/exposure_protocol.py")
PRIOR = Path("/workspace/diffusets-study/snapshot3030/training/preflight.json")
if not PRIOR.is_file():
    PRIOR = HERE.parent / "diffusets-3030/remote/training/preflight.json"


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"work_deadline_utc": m.WORK_DEADLINE}
        self.deadline = m.stamp(m.WORK_DEADLINE)

    def test_admission_includes_all_four_complete_passes(self):
        rates = {"selection": .1, "selection_zero": .2}
        expected = (27264 * 2 + 26841 * 2) * .2 * 1.25 + 180 * 4 + 1800
        value = m.evaluation_budget(self.protocol, [], rates, now=self.deadline - expected)
        self.assertEqual(value["status"], "admitted")
        self.assertEqual(value["required_seconds_estimate"], expected)
        self.assertEqual(set(value["pending_pass_questions"]), set(m.PASSES))
        self.assertEqual(m.evaluation_budget(self.protocol, [], rates, now=self.deadline - expected + .01)["status"], "needs_time_budget_review")

    def test_pending_budget_uses_slowest_observed_rate_and_retains_zero_test(self):
        rates = {"selection": .1, "selection_zero": .2, "validation": .5}
        value = m.evaluation_budget(self.protocol, ["selection", "selection_zero", "validation"], rates, now=0)
        self.assertEqual(value["pending_pass_questions"], {"validation_zero": 27264, "test": 26841, "test_zero": 26841})
        self.assertEqual(value["required_seconds_estimate"], (27264+26841*2)*.5*1.25+180*3+1800)

    def test_both_calibrations_and_finite_positive_timings_required(self):
        with self.assertRaisesRegex(RuntimeError, "Both selection"):
            m.evaluation_budget(self.protocol, [], {"selection": .1})
        for bad in (0, -1, float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError, "Invalid generation timing"):
                m.evaluation_budget(self.protocol, [], {"selection": .1, "selection_zero": bad})


class AuditFixture:
    def __init__(self, path):
        self.root = Path(path)
        self.run = self.root / "seed0"
        self.run.mkdir()
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        self.baseline = m.read(PRIOR)["arms"]["real_qa"]["expected_config"]
        m.write(self.root / "baseline.json", self.baseline)
        self.rows = [{"task_id": f"original-{i}"} for i in range(h.ORIGINAL_ROWS)] + [{"task_id": f"generated-{i}"} for i in range(h.GENERATED_ROWS)]
        self.groups = {"schema_version": 1, "original_train_parquet_sha256": h.BASELINE_TRAIN_SHA256,
            "original": {"start": 0, "count": h.ORIGINAL_ROWS, "task_ids_sha256": h.task_ids_sha256(self.rows[:h.ORIGINAL_ROWS])},
            "generated": {"start": h.ORIGINAL_ROWS, "count": h.GENERATED_ROWS, "task_ids_sha256": h.task_ids_sha256(self.rows[h.ORIGINAL_ROWS:])}}
        m.write(self.prepared / "source-manifest.json", self.groups)
        self.config = copy.deepcopy(self.baseline)
        self.config.update(trainer_source_sha256=m.sha(HERE / "trainer/opentslm_exposure.py"), exposure_helper_sha256=m.sha(HERE / "trainer/exposure_protocol.py"),
            source_manifest_sha256=m.sha(self.prepared / "source-manifest.json"), max_steps=12000, learning_rate_schedule_steps=15531,
            warmup_steps=1554, checkpoint_policy="fixed_original_exposure")
        self.history = [{"step": step, "accuracy": .5} for step in range(1000, 12001, 1000)]
        self.payload = {"config": self.config, "step": 12000, "selection_accuracy": .5}
        (self.run / "primary.pt").write_bytes(b"fake checkpoint: torch.load is stubbed")
        exposure = {}
        for name, rows, batch in (("original", self.rows[:h.ORIGINAL_ROWS], 16), ("generated", self.rows[h.ORIGINAL_ROWS:], 2)):
            stream = h.PermutationStream(rows, 0)
            for _ in range(12000):
                stream.take(batch)
            exposure[name] = stream.audit()
        self.result = {"status": "trained", "seed": 0, "distinct_truncated_rationales": 0, "distinct_seen_training_questions": 94501,
            "primary_checkpoint_sha256": m.sha(self.run / "primary.pt"), "max_steps": 12000, "primary_step": 12000, "source_exposure": exposure}
        self.protocol = {"bindings": {}, "baseline_config": str(self.root / "baseline.json"), "trainer_path": str(HERE / "trainer/opentslm_exposure.py"),
            "prepared": str(self.prepared), "protocol_path": str(self.root / "protocol.json"), "intentional_differences": ["generated exposure"]}
        m.write(self.root / "protocol.json", self.protocol)
        self.save()

    def save(self):
        for name, value in (("config.json", self.config), ("train_result.json", self.result), ("selection_history.json", self.history)):
            m.write(self.run / name, value)

    def audit(self):
        import pyarrow.parquet as pq
        with mock.patch.dict(sys.modules, {"torch": SimpleNamespace(load=lambda *a, **kw: self.payload)}), mock.patch.object(pq, "read_table", return_value=SimpleNamespace(to_pylist=lambda: self.rows)):
            return m.audit_training(self.protocol, self.root, self.run)


class TrainingAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = AuditFixture(self.tmp.name)

    def test_exact_checkpoint_and_full_ordered_draws_freeze(self):
        self.f.audit()
        freeze = m.read(self.f.root / "freeze.json")
        self.assertEqual(freeze["primary_step"], 12000)
        self.assertEqual(set(Path(p).name for p in freeze["inventory"]), {"config.json", "train_result.json", "selection_history.json", "primary.pt"})
        self.assertTrue(m.read(self.f.root / "run_audit.json")["draw_order_verified"])

    def test_same_draw_counts_with_wrong_order_are_rejected(self):
        self.f.result["source_exposure"]["original"]["draw_task_ids_sha256"] = "wrong-order"
        self.f.save()
        with self.assertRaisesRegex(RuntimeError, "Actual original exposure/order differs"):
            self.f.audit()
        self.assertFalse((self.f.root / "freeze.json").exists())

    def test_earlier_selection_peak_cannot_replace_primary(self):
        self.f.payload["step"] = 8000
        with self.assertRaisesRegex(ValueError, "Early selection peak"):
            self.f.audit()

    def test_incomplete_selection_history_is_rejected(self):
        self.f.history.pop(3)
        self.f.save()
        with self.assertRaisesRegex(ValueError, "history is incomplete"):
            self.f.audit()

    def test_truncation_or_missing_source_blocks_freeze(self):
        for field, value in (("distinct_truncated_rationales", 1), ("distinct_seen_training_questions", 94500)):
            with self.subTest(field=field):
                previous = self.f.result[field]
                self.f.result[field] = value
                self.f.save()
                with self.assertRaisesRegex(RuntimeError, "Training incomplete"):
                    self.f.audit()
                self.f.result[field] = previous


class EvaluationArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.config = {"manifest_sha256": "prepared", "trainer_source_sha256": "trainer", "metrics_source_sha256": "metrics", "exposure_helper_sha256": "helper"}
        self.result = {"primary_checkpoint_sha256": "checkpoint"}
        m.write(self.run / "config.json", self.config)
        self.label = "test_zero"
        self.identity = {"checkpoint_sha256": "checkpoint", "checkpoint_step": 12000, "manifest_sha256": "prepared", "questions": 26841,
            "seed": 0, "split": "test", "zero_signal": True, "config_sha256": m.sha(self.run / "config.json"), "trainer_source_sha256": "trainer",
            "metrics_source_sha256": "metrics", "exposure_helper_sha256": "helper", "max_new_tokens": 384, "max_length": 1536}
        m.write(self.run / f"{self.label}_claim.json", {"identity": self.identity, "batch_size": 256})
        (self.run / f"{self.label}_predictions.jsonl").write_bytes(b"{}\n" * 26841)
        self.metrics = {**self.identity, "prediction_sha256": m.sha(self.run / f"{self.label}_predictions.jsonl"),
            "claim_sha256": m.sha(self.run / f"{self.label}_claim.json"), "metrics_recovered_from_predictions": False}
        m.write(self.run / f"{self.label}_metrics.json", self.metrics)

    def test_complete_zero_test_artifacts_pass_without_parsing_labels(self):
        self.assertEqual(m.verify_evaluation(self.run, self.label, self.config, self.result)["questions"], 26841)

    def test_identity_mismatch_and_inference_recovery_rejected(self):
        for key, value in (("checkpoint_step", 8000), ("zero_signal", False), ("metrics_recovered_from_predictions", True)):
            with self.subTest(key=key):
                altered = {**self.metrics, key: value}
                m.write(self.run / f"{self.label}_metrics.json", altered)
                with self.assertRaises(RuntimeError):
                    m.verify_evaluation(self.run, self.label, self.config, self.result)

    def test_incomplete_predictions_rejected_even_if_hash_updated(self):
        prediction = self.run / f"{self.label}_predictions.jsonl"
        prediction.write_bytes(b"{}\n")
        self.metrics["prediction_sha256"] = m.sha(prediction)
        m.write(self.run / f"{self.label}_metrics.json", self.metrics)
        with self.assertRaisesRegex(RuntimeError, "Incomplete published predictions"):
            m.verify_evaluation(self.run, self.label, self.config, self.result)


class OneShotRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.root = self.home / "training"
        self.prepared = self.home / "prepared"
        self.prepared.mkdir()
        m.write(self.prepared / "manifest.json", {"sha256": "prepared"})
        prior = m.read(PRIOR)
        m.write(self.home / "baseline.json", prior["arms"]["real_qa"]["expected_config"])
        prior["baseline_config_sha256"] = m.sha(self.home / "baseline.json")
        m.write(self.prepared / "source-manifest.json", {"snapshot_generated_binding": {"prepared_manifest_content_sha256": prior["arms"]["grounded_acq_gen"]["expected_config"]["manifest_sha256"]}})
        self.prior = self.home / "prior.json"
        m.write(self.prior, prior)
        self.audit = self.home / "builder-audit.json"
        m.write(self.audit, {"schema_version": 1, "status": "verified", "original_rows": 82829, "generated_rows": 11672, "total_rows": 94501, "signals": 13105,
            "baseline_row_order_and_values_preserved": True, "baseline_signal_prefix_bitwise_identical": True, "generated_fields_unchanged_except_signal_index": True,
            "heldout_copied_byte_exact_without_parsing": True, "accepted_generated_signals_content_hash_verified": 3030, "no_regeneration_or_remeasurement": True,
            "prepared_manifest": {"sha256": m.sha(self.prepared / "manifest.json")}, "source_manifest": {"sha256": m.sha(self.prepared / "source-manifest.json")}})
        self.protocol = {"schema": "syntide.exposure3030-run@1", "max_spend_usd": 40, "work_deadline_utc": m.WORK_DEADLINE, "bindings": {},
            "training_root": str(self.root), "protocol_path": str(self.home / "protocol.json"), "prepared": str(self.prepared), "prepared_manifest_content_sha256": "prepared",
            "prior_snapshot_preflight": str(self.prior), "preparation_audit": str(self.audit), "trainer_path": str(HERE / "trainer/opentslm_exposure.py"),
            "baseline_config": str(self.home / "baseline.json"), "train_arguments": {}}
        m.write(self.home / "protocol.json", self.protocol)
        self.calls = []
        self.fail_test = False

    def fake_audit(self, protocol, root, run):
        m.write(root / "run_audit.json", {"status": "passed"})
        freeze = {"inventory": {}}
        m.write(root / "freeze.json", freeze)
        return {}, {}, freeze

    def fake_execute(self, command, log, hold_fd, deadline):
        label = Path(log).stem
        self.calls.append(label)
        if self.fail_test and label == "test":
            raise RuntimeError("simulated child failure")

    def invoke(self, budget=None):
        original_stat = Path.stat
        def safe_stat(path, *args, **kwargs):
            if str(path) == "/workspace/diffusets-study/generation/.writer.lock":
                return SimpleNamespace(st_dev=1, st_ino=2)
            return original_stat(path, *args, **kwargs)
        def verify(run, label, config, result):
            return {"elapsed_s": 1, "questions": 2000 if label.startswith("selection") else 27264 if label.startswith("validation") else 26841}
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, {"ABLATION_HOLD_FD": "123"}))
            stack.enter_context(mock.patch.object(m.os, "fstat", return_value=SimpleNamespace(st_dev=1, st_ino=2)))
            stack.enter_context(mock.patch.object(Path, "stat", safe_stat))
            stack.enter_context(mock.patch.object(m, "execute", side_effect=self.fake_execute))
            stack.enter_context(mock.patch.object(m, "audit_training", side_effect=self.fake_audit))
            stack.enter_context(mock.patch.object(m, "verify_evaluation", side_effect=verify))
            stack.enter_context(mock.patch.object(m.time, "time", return_value=m.stamp(m.WORK_DEADLINE)-100000))
            if budget:
                stack.enter_context(mock.patch.object(m, "evaluation_budget", return_value=budget))
            m.run_ablation(self.protocol)

    def test_all_passes_run_once_in_fixed_order(self):
        self.invoke()
        self.assertEqual(self.calls, ["train", "selection", "selection_zero", *m.PASSES])
        self.assertEqual(m.read(self.root / "result.json")["completed_passes"], list(m.PASSES))
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(self.calls.count("train"), 1)

    def test_partial_test_failure_cannot_replay(self):
        self.fail_test = True
        with self.assertRaisesRegex(RuntimeError, "simulated child failure"):
            self.invoke()
        self.assertTrue((self.root / "test-attempt.json").exists())
        self.assertFalse((self.root / "test_zero-attempt.json").exists())
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(self.calls.count("test"), 1)

    def test_insufficient_full_budget_leaves_all_heldout_closed(self):
        with self.assertRaisesRegex(RuntimeError, "Insufficient time for all remaining"):
            self.invoke({"status": "needs_time_budget_review"})
        self.assertEqual(self.calls, ["train", "selection", "selection_zero"])
        self.assertFalse((self.root / "holdout-claim.json").exists())
        for label in m.PASSES:
            self.assertFalse((self.root / f"{label}-attempt.json").exists())

    def test_inherited_scan_requires_exact_snapshot_and_complete_zero_truncation(self):
        prepared = m.read(self.prepared / "manifest.json")
        self.assertEqual(m.inherited_preflight(self.protocol, prepared)["status"], "passed")
        prior = m.read(self.prior)
        for path, value in ((["arms", "grounded_acq_gen", "full_training_token_scan", "questions"], 221393),
                            (["arms", "grounded_acq_gen", "full_training_token_scan", "truncated_rationales"], 1),
                            (["arms", "grounded_acq_gen", "expected_config", "manifest_sha256"], "different-snapshot"),
                            (["arms", "grounded_acq_gen", "expected_config", "prompt_version"], "different-prompt")):
            with self.subTest(path=path):
                altered = copy.deepcopy(prior)
                cursor = altered
                for key in path[:-1]:
                    cursor = cursor[key]
                cursor[path[-1]] = value
                m.write(self.prior, altered)
                with self.assertRaises(RuntimeError):
                    m.inherited_preflight(self.protocol, prepared)
        m.write(self.prior, prior)

    def test_inherited_scan_rejects_missing_preservation_or_changed_output_binding(self):
        prepared = m.read(self.prepared / "manifest.json")
        audit = m.read(self.audit)
        for key, value in (("generated_rows", 11671), ("baseline_row_order_and_values_preserved", False),
                           ("source_manifest", {"sha256": "different-output"})):
            with self.subTest(key=key):
                m.write(self.audit, {**audit, key: value})
                with self.assertRaises(RuntimeError):
                    m.inherited_preflight(self.protocol, prepared)


if __name__ == "__main__":
    unittest.main(verbosity=2)
