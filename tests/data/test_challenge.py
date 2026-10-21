"""How scarcity manifests are drawn, and that they cannot be silently edited."""

import json
import math

import pytest

from syntide.data.challenge import (
    ChallengeSpec,
    ManifestIntegrityError,
    build_manifest,
    load_manifest,
    save_manifest,
)
from syntide.data.leakage import check_manifest

SOURCE = {"dataset": "physionet/ptb-xl", "version": "1.0.3", "sha256": "test"}


def _columns(index):
    return (
        dict(zip(index["record_id"].to_pylist(), index["subject_uid"].to_pylist(), strict=True)),
        dict(zip(index["record_id"].to_pylist(), index["is_artifact"].to_pylist(), strict=True)),
        dict(zip(index["record_id"].to_pylist(), index["strat_fold"].to_pylist(), strict=True)),
    )


def test_pools_take_whole_subjects_nest_and_deplete_artifacts_by_the_ceil_rule(synthetic_index):
    spec = ChallengeSpec()
    m10 = build_manifest(synthetic_index, spec, 0.10, SOURCE)
    m25 = build_manifest(synthetic_index, spec, 0.25, SOURCE)
    subject_of, is_artifact, fold = _columns(synthetic_index)

    for manifest in (m10, m25):
        check_manifest(manifest, synthetic_index)
    # Whole subjects: nobody has records on both sides of the pool boundary.
    pool_subjects = {subject_of[r] for r in m25.roles["pool"]}
    assert not pool_subjects & {subject_of[r] for r in m25.roles["unused"]}
    train_subjects = {s for r, s in subject_of.items() if fold[r] <= 8}
    assert len(pool_subjects) == math.ceil(0.25 * len(train_subjects))
    # The 10% world is a sub-world of the 25% one.
    assert set(m10.roles["pool"]) <= set(m25.roles["pool"])

    pool_artifacts = [r for r in m25.roles["pool"] if is_artifact[r]]
    assert len(m25.roles["retained_artifact"]) == math.ceil(0.10 * len(pool_artifacts))
    # C1 drops the same number of records at random, so it keeps artifacts at roughly natural prevalence.
    depleted_artifacts = sum(is_artifact[r] for r in m25.roles["depleted"])
    natural_artifacts = sum(is_artifact[r] for r in m25.roles["natural_mix"])
    assert len(m25.roles["natural_mix"]) == len(m25.roles["depleted"])
    assert natural_artifacts > 3 * depleted_artifacts


def test_manifest_is_deterministic_and_hand_edits_are_rejected(synthetic_index, tmp_path):
    spec = ChallengeSpec()
    first = build_manifest(synthetic_index, spec, 0.25, SOURCE)
    again = build_manifest(synthetic_index, spec, 0.25, SOURCE)
    other_seed = build_manifest(synthetic_index, ChallengeSpec(seed=1), 0.25, SOURCE)

    assert first == again
    assert first.roles["pool"] != other_seed.roles["pool"]

    path = tmp_path / "pool25.json"
    save_manifest(first, path)
    assert load_manifest(path) == first

    # Smuggle one withheld record back into the A0-A3 parents without rebuilding.
    payload = json.loads(path.read_text())
    payload["roles"]["depleted"].append(payload["roles"]["withheld"][0])
    path.write_text(json.dumps(payload))
    with pytest.raises(ManifestIntegrityError):
        load_manifest(path)
