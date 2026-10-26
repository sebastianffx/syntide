"""Check archive integrity and the summary of saved classification scores."""

import importlib.util
import pathlib
import shutil
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "classification_reproduce", ROOT / "reproduce.py"
)
REPRODUCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPRODUCE)


class ArchiveTest(unittest.TestCase):
    """Check package integrity without changing the original evidence."""

    def test_complete_archive_and_table(self):
        result = REPRODUCE.verify(ROOT)
        self.assertEqual(result["completed_trials"], 85)
        self.assertEqual(result["historical_code_pins"], 17)
        table = REPRODUCE.table(ROOT)
        self.assertIn(
            "ptbxl-diffusets-4515-mix75-2000 | pool50 | mix75 | 1 | 0.922215",
            table,
        )

    def test_changed_numerical_source_fails_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = pathlib.Path(temporary) / "archive"
            shutil.copytree(
                ROOT, copied, ignore=shutil.ignore_patterns("__pycache__")
            )
            source = copied / "runtime/src/syntide/trial.py"
            source.write_bytes(source.read_bytes() + b"\n# Changed source.\n")
            with self.assertRaisesRegex(ValueError, "File content differs"):
                REPRODUCE.verify(copied)

    def test_missing_evidence_fails_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = pathlib.Path(temporary) / "archive"
            shutil.copytree(
                ROOT, copied, ignore=shutil.ignore_patterns("__pycache__")
            )
            (copied / "runtime/runs/ptbxl-scaling/plan.json").unlink()
            with self.assertRaisesRegex(ValueError, "File inventory differs"):
                REPRODUCE.verify(copied)


if __name__ == "__main__":
    unittest.main()
