"""Task inputs and their source records obey patient isolation rules."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pytest

from syntide.data.challenge import ChallengeSpec, build_manifest
from syntide.data.leakage import (
    LeakageError,
    check_disjoint_subjects,
    check_lineage,
)
from syntide.data.schema import LEADS
from syntide.generate.qa.sources import open_external_source, source_tasks


@pytest.fixture(scope="module")
def manifest(synthetic_index):
    return build_manifest(
        synthetic_index,
        ChallengeSpec(pool_fractions=[0.25]),
        0.25,
        {"sha256": "x"},
    )


def _subject(index, record_id):
    rows = index.select(["record_id", "subject_uid"]).to_pydict()
    return dict(zip(rows["record_id"], rows["subject_uid"], strict=True))[
        record_id
    ]


def test_inputs_must_descend_from_the_allowed_role_and_its_patients(
    manifest, synthetic_index
):
    depleted = manifest.roles["depleted"][0]
    ok = (depleted, (depleted,), (_subject(synthetic_index, depleted),))
    check_lineage([ok], manifest, synthetic_index, ["depleted"])
    for role in ("val", "test", "withheld", "unused"):
        root = manifest.roles[role][0]
        # A synthetic descendant with a fresh id still carries its real root.
        child = (
            "synthetic:run:00000001",
            (root,),
            (_subject(synthetic_index, root),),
        )
        with pytest.raises(LeakageError):
            check_lineage([ok, child], manifest, synthetic_index, ["depleted"])
    with pytest.raises(LeakageError, match="another patient"):
        check_lineage(
            [(depleted, (depleted,), ("ptb-xl:someone-else",))],
            manifest,
            synthetic_index,
            ["depleted"],
        )
    with pytest.raises(LeakageError, match="no resolvable"):
        check_lineage(
            [("synthetic:orphan", (), ())],
            manifest,
            synthetic_index,
            ["depleted"],
        )
    with pytest.raises(LeakageError, match="outside the index"):
        check_lineage(
            [("x", ("ptbxl-unknown",), ("ptb-xl:0",))],
            manifest,
            synthetic_index,
            ["depleted"],
        )
    pair = [
        ok,
        (
            manifest.roles["val"][0],
            (manifest.roles["val"][0],),
            (_subject(synthetic_index, manifest.roles["val"][0]),),
        ),
    ]
    with pytest.raises(LeakageError):  # every input of a paired task is checked
        check_lineage(pair, manifest, synthetic_index, ["depleted"])


def test_a_partial_index_is_refused_rather_than_weakening_the_patient_rule(
    manifest, synthetic_index
):
    allowed = set(manifest.roles["depleted"])
    partial = synthetic_index.filter(
        pa.array(
            [r in allowed for r in synthetic_index["record_id"].to_pylist()]
        )
    )
    depleted = manifest.roles["depleted"][0]
    with pytest.raises(LeakageError, match="missing from the index"):
        check_lineage(
            [(depleted, (depleted,), (_subject(synthetic_index, depleted),))],
            manifest,
            partial,
            ["depleted"],
        )


def test_held_out_populations_never_share_patients():
    check_disjoint_subjects(
        {"train": {"a", "b"}, "selection": {"c"}, "test": {"d"}}
    )
    with pytest.raises(LeakageError, match="both"):
        check_disjoint_subjects({"train": {"a", "b"}, "test": {"b"}})


def test_external_catalogs_are_evaluation_only_and_namespaced(tmp_path):
    x = np.zeros((12, 1000), dtype=np.float32)
    np.save(tmp_path / "r1.npy", x)
    row = {
        "record_id": "r1",
        "subject_uid": "p1",
        "split": "test",
        "signal_path": "r1.npy",
        "sample_rate": 100,
        "channels": ["I", "II", "III", "aVR", "aVL", "aVF", *LEADS[6:]],
        "units": "mV",
        "texts": [
            {
                "text_id": "q1",
                "question": "Does this ECG show atrial fibrillation?",
                "answer": "Y",
                "answer_options": ["Y", "N"],
            }
        ],
    }
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(json.dumps(row) + "\n")
    source = open_external_source(
        catalog, "mimic-ecgqa", label_mapping={"Y": "yes", "N": "no"}
    )
    assert (
        source.purpose == "evaluation"
        and source.rows[0].subject_uid == "external:mimic-ecgqa:p1"
    )
    ((ref, loaded),) = source.load([source.rows[0].record_id])
    (task,) = source_tasks(source, ref)
    assert (task.answer, task.options) == (
        "yes",
        ("yes", "no"),
    ) and loaded.shape == (12, 1000)
    catalog.write_text(json.dumps({**row, "split": "train"}) + "\n")
    with pytest.raises(PermissionError, match="evaluation rows only"):
        open_external_source(catalog, "mimic-ecgqa")
