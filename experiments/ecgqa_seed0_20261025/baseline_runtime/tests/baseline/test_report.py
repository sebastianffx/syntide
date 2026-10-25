"""End-to-end report fixtures contain only synthetic rows, never models."""

import json
import tempfile
import unittest
from pathlib import Path

from syntide.baseline.metrics import score_predictions
from syntide.baseline.report import _file_hash, _json_hash, build_report, write_report


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [
            {
                "task_id": "q1",
                "template_id": 1,
                "answer": "yes",
                "answer_options": ["yes", "no"],
                "is_artifact": False,
                "output_text": "Answer: yes",
            }
        ]
        counts = {"questions": 1, "task_ids_sha256": _json_hash(["q1"])}
        self.manifest = {"sha256": "dataset-hash", "splits": {"validation": counts, "test": counts}}
        for seed in (0, 1, 2):
            directory = self.root / f"seed{seed}"
            directory.mkdir()
            config = {
                "seed": seed,
                "init_sha256": f"init-{seed}",
                "manifest_sha256": "dataset-hash",
                "max_steps": 100,
            }
            (directory / "config.json").write_text(json.dumps(config))
            result = {
                "status": "trained",
                "seed": seed,
                "manifest_sha256": "dataset-hash",
                "max_steps": 100,
                "best_step": 20,
                "best_checkpoint_sha256": f"checkpoint-{seed}",
                "smoke_only": False,
            }
            (directory / "train_result.json").write_text(json.dumps(result))
            for split in ("validation", "test"):
                for zero in (False, True):
                    scenario = f"{split}{'_zero' if zero else ''}"
                    rows = [
                        {**self.rows[0], "output_text": "Answer: no" if zero or seed == 1 else "Answer: yes"}
                    ]
                    path = directory / f"{scenario}_predictions.jsonl"
                    path.write_text(json.dumps(rows[0]) + "\n")
                    metrics = {
                        "split": split,
                        "zero_signal": zero,
                        "seed": seed,
                        "checkpoint_step": 20,
                        "checkpoint_sha256": f"checkpoint-{seed}",
                        "manifest_sha256": "dataset-hash",
                        "prediction_sha256": _file_hash(path),
                        "questions": 1,
                        "strict": score_predictions(rows),
                        "upstream_parser": score_predictions(rows, parser="upstream"),
                    }
                    (directory / f"{scenario}_metrics.json").write_text(json.dumps(metrics))

    def test_complete_report_recomputes_and_aggregates_seed_variation(self):
        report = build_report(self.root, self.manifest, {"1": "yes"})
        self.assertEqual(report["status"], "complete")
        stats = report["aggregate"]["test"]["all"]["accuracy"]
        self.assertAlmostEqual(stats["mean"], 2 / 3)
        self.assertAlmostEqual(stats["sd"], (1 / 3) ** 0.5)
        self.assertEqual(report["majority_reference"]["test"]["groups"]["all"]["accuracy"], 1)
        self.assertAlmostEqual(report["signal_ablation_gap"]["test"]["all"]["accuracy"]["mean"], 2 / 3)
        self.assertIsNone(report["aggregate"]["test"]["noisy"]["accuracy"]["mean"])
        write_report(report, self.root / "report")
        text = (self.root / "report" / "report.md").read_text()
        self.assertIn("not directly comparable", text)
        self.assertIn("unavailable", text)
        self.assertTrue((self.root / "report" / "per_template.csv").is_file())

    def test_missing_evaluation_cannot_be_reported_as_complete(self):
        (self.root / "seed2" / "test_predictions.jsonl").unlink()
        with self.assertRaises(ValueError):
            build_report(self.root, self.manifest, {"1": "yes"})
        report = build_report(self.root, self.manifest, {"1": "yes"}, allow_partial=True)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["aggregate"]["test"]["all"]["accuracy"]["n_seeds"], 2)

    def test_tampered_predictions_are_rejected_even_for_partial_report(self):
        path = self.root / "seed0" / "test_predictions.jsonl"
        path.write_text(path.read_text().replace("Answer: yes", "Answer: no"))
        with self.assertRaises(ValueError):
            build_report(self.root, self.manifest, {"1": "yes"}, allow_partial=True)

    def test_mixed_training_budgets_are_never_averaged(self):
        path = self.root / "seed2" / "config.json"
        config = json.loads(path.read_text())
        path.write_text(json.dumps({**config, "max_steps": 200}))
        with self.assertRaises(ValueError):
            build_report(self.root, self.manifest, {"1": "yes"}, allow_partial=True)


if __name__ == "__main__":
    unittest.main()
