#!/usr/bin/env python3
"""Run the archived exposure tests with the bundled preflight fixture.

Only the tests' fixture paths are redirected. The scientific source and
test files retain their archived bytes. No model or real dataset is loaded.
"""

import importlib.util
from pathlib import Path
import sys
import unittest


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here / "trainer"))
    modules = [
        load("archived_exposure_sampler_tests", here / "trainer/test_exposure_protocol.py"),
        load("archived_exposure_builder_tests", here / "test_prepare_exposure_data.py"),
        load("archived_exposure_runner_tests", here / "test_run_exposure_ablation.py"),
    ]
    # The original builder test searched for its baseline source on the worker
    # or in a sibling checkout. Resolve that one lookup to this frozen package.
    archived_baseline = "/workspace/baseline-runtime/src/syntide/baseline/data.py"
    bundled_baseline = here.parent / "baseline_runtime/src/syntide/baseline/data.py"
    modules[1].Path = lambda value: bundled_baseline if str(value) == archived_baseline else Path(value)
    modules[-1].PRIOR = here / "fixtures/snapshot-preflight.json"
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.loadTestsFromModule(module) for module in modules)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
