"""Full preflight must catch unseen truncation and reject unsafe task deduplication."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def preflight():
    path = Path(__file__).parents[1] / "scripts/runpod/qa_preflight.py"
    spec = importlib.util.spec_from_file_location("qa_preflight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCollator:
    def __init__(self, signal=b"same signal"):
        self.prepared = SimpleNamespace(signal=lambda row: SimpleNamespace(tobytes=lambda: signal))
        self.prompt_cache = {}
        self.signal_cache = {}
        self.training_token_cache = {}
        self.truncated_tasks = set()
        self.calls = []
        self.maximum_cached = 0

    def encode(self, row, *, training):
        assert training
        task = row["task_id"]
        self.calls.append(task)
        self.prompt_cache[task] = "prompt"
        self.signal_cache[row["record_id"]] = "normalized"
        self.training_token_cache[task] = [1, 2, 3]
        self.maximum_cached = max(self.maximum_cached, len(self.training_token_cache))
        if row["rationale"] == "too long":
            self.truncated_tasks.add(task)
        return [1, 2, 3], [-100, 2, 3], b"normalized"


def row(number):
    return {
        "task_id": str(number),
        "record_id": "record",
        "subject_uid": "patient",
        "question": "Question?",
        "clinical_context": "Context",
        "answer_options": ["yes", "no"],
        "rationale": "Rationale",
        "answer": "yes",
        "template_id": 1,
        "is_artifact": False,
    }


def test_every_training_row_is_encoded_with_bounded_caches(preflight):
    rows = [row(number) for number in range(2050)]
    rows[-1]["rationale"] = "too long"
    collator, seen, progress = FakeCollator(), {}, []
    result = preflight.scan_rows(collator, rows, seen, on_progress=progress.append, progress_every=1000)
    assert len(collator.calls) == 2050
    assert collator.maximum_cached == 128
    assert result["truncated_rationales"] == 1
    assert result["truncated_task_ids"] == ["2049"]
    assert [item["scanned"] for item in progress] == [1000, 2000, 2050]
    assert not collator.prompt_cache and not collator.training_token_cache


def test_shared_tasks_reuse_only_verified_equivalent_encoding(preflight):
    first, second, seen = FakeCollator(), FakeCollator(), {}
    preflight.scan_rows(first, [row(1), row(2)], seen)
    result = preflight.scan_rows(second, [row(1), row(2), row(3)], seen)
    assert second.calls == ["3"]
    assert result["reused_verified_tasks"] == 2
    assert result["newly_encoded"] == 1
    assert result["records_verified"] == 1


@pytest.mark.parametrize("change", ["signal", "question", "answer", "patient"])
def test_shared_task_id_cannot_hide_changed_training_input(preflight, change):
    seen = {}
    preflight.scan_rows(FakeCollator(), [row(1)], seen)
    changed = row(1)
    if change == "signal":
        collator = FakeCollator(b"changed signal")
    else:
        collator = FakeCollator()
        changed["subject_uid" if change == "patient" else change] = "changed"
    with pytest.raises(ValueError, match="Shared task ID has changed"):
        preflight.scan_rows(collator, [changed], seen)
    assert collator.calls == []


def test_duplicate_tasks_inside_one_corpus_fail(preflight):
    with pytest.raises(ValueError, match="Duplicate task ID"):
        preflight.scan_rows(FakeCollator(), [row(1), row(1)], {})


def test_runtime_identity_allows_device_change_but_no_package_or_weight_change(preflight):
    baseline = {"gpu": "H100", "packages": {"torch": "pinned"}, "init_sha256": "same"}
    actual = {**baseline, "gpu": "A100"}
    preflight.check_runtime(baseline, actual)
    actual["init_sha256"] = "changed"
    with pytest.raises(ValueError, match="init_sha256"):
        preflight.check_runtime(baseline, actual)
    actual = {**baseline, "packages": {"torch": "new version"}}
    with pytest.raises(ValueError, match="packages"):
        preflight.check_runtime(baseline, actual)
