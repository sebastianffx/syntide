"""Gate 1: every isolation rule the experiment relies on is enforced, and the offender is named."""

from collections import Counter
from dataclasses import replace

import pyarrow as pa
import pytest

from syntide.data.challenge import ChallengeSpec, build_manifest
from syntide.data.leakage import LeakageError, check_calibration_ids, check_manifest, check_recipe_parents

SOURCE = {"dataset": "physionet/ptb-xl", "version": "1.0.3", "sha256": "test"}


@pytest.fixture(scope="module")
def manifest(synthetic_index):
    return build_manifest(synthetic_index, ChallengeSpec(), 0.25, SOURCE)


def _with_roles(manifest, **changes):
    return replace(manifest, roles={**manifest.roles, **changes})


def _columns(index):
    ids = index["record_id"].to_pylist()
    return (
        dict(zip(ids, index["subject_uid"].to_pylist(), strict=True)),
        dict(zip(ids, index["is_artifact"].to_pylist(), strict=True)),
    )


def _built_from_other_data(m, index):
    return replace(m, source={**SOURCE, "sha256": "rebuilt"}), "rebuilt", "not from the local data"


def _add_to_pool(m, index):
    record = m.roles["val"][0]
    return _with_roles(m, pool=(*m.roles["pool"], record)), record, "in both pool and val"


def _test_record_into_val(m, index):
    record = m.roles["test"][0]
    corrupted = _with_roles(m, val=(*m.roles["val"], record), test=m.roles["test"][1:])
    return corrupted, record, r"val == folds \[9\]"


def _unbalance_withheld(m, index):
    _, is_artifact = _columns(index)
    record = next(r for r in m.roles["withheld"] if is_artifact[r])
    withheld = tuple(r for r in m.roles["withheld"] if r != record)
    return _with_roles(m, withheld=withheld), record, "retained \\+ withheld"


def _withheld_into_depleted(m, index):
    record = m.roles["withheld"][0]
    return _with_roles(m, depleted=(*m.roles["depleted"], record)), record, "withheld records are in depleted"


def _withheld_patient_keeps_a_clean_ecg_in_depleted(m, index):
    subject_of, is_artifact = _columns(index)
    record = next(r for r in m.roles["withheld"] if not is_artifact[r])
    withheld = tuple(r for r in m.roles["withheld"] if r != record)
    corrupted = _with_roles(m, withheld=withheld, depleted=(*m.roles["depleted"], record))
    return corrupted, subject_of[record], "subjects are in both withheld and depleted"


def _natural_mix_outside_pool(m, index):
    record = m.roles["unused"][0]
    mixed = (*m.roles["natural_mix"][1:], record)
    return _with_roles(m, natural_mix=mixed), record, "natural_mix records are outside the pool"


def _natural_mix_splits_a_patient(m, index):
    subject_of, _ = _columns(index)
    kept = Counter(subject_of[r] for r in m.roles["natural_mix"])
    record = next(r for r in m.roles["natural_mix"] if kept[subject_of[r]] > 1)
    dropped = min(set(m.roles["pool"]) - set(m.roles["natural_mix"]))
    mixed = (*(r for r in m.roles["natural_mix"] if r != record), dropped)
    return _with_roles(m, natural_mix=mixed), subject_of[record], "subjects are in both natural_mix"


def _natural_mix_too_small(m, index):
    subject_of, _ = _columns(index)
    kept = Counter(subject_of[r] for r in m.roles["natural_mix"])
    record = next(r for r in m.roles["natural_mix"] if kept[subject_of[r]] == 1)
    mixed = tuple(r for r in m.roles["natural_mix"] if r != record)
    return _with_roles(m, natural_mix=mixed), "natural_mix", "matched-N control"


@pytest.mark.parametrize(
    "corrupt",
    [
        _built_from_other_data,
        _add_to_pool,
        _test_record_into_val,
        _unbalance_withheld,
        _withheld_into_depleted,
        _withheld_patient_keeps_a_clean_ecg_in_depleted,
        _natural_mix_outside_pool,
        _natural_mix_splits_a_patient,
        _natural_mix_too_small,
    ],
)
def test_each_broken_role_relation_is_rejected_and_named(manifest, synthetic_index, corrupt):
    corrupted, offender, message = corrupt(manifest, synthetic_index)
    with pytest.raises(LeakageError, match=message) as error:
        check_manifest(corrupted, synthetic_index, SOURCE)
    assert offender in str(error.value)


@pytest.mark.parametrize(("role", "pair"), [("test", "train and test"), ("unused", "pool and unused")])
def test_a_patient_split_across_partitions_is_rejected(manifest, synthetic_index, role, pair):
    # Same ECG ids, but one ECG of `role` now belongs to a patient who is in the training pool.
    subject_of, _ = _columns(synthetic_index)
    pool_subject = subject_of[manifest.roles["pool"][0]]
    moved = manifest.roles[role][0]
    subjects = [pool_subject if r == moved else s for r, s in subject_of.items()]
    leaky_index = synthetic_index.set_column(1, "subject_uid", pa.array(subjects))

    with pytest.raises(LeakageError, match=f"subjects are in both {pair}") as error:
        check_manifest(manifest, leaky_index, SOURCE)
    assert pool_subject in str(error.value)


@pytest.mark.parametrize(
    ("check", "allowed", "message"),
    [
        (check_calibration_ids, "retained_artifact", "calibration records are outside retained_artifact"),
        (check_recipe_parents, "depleted", "recipe parents are outside depleted"),
    ],
)
def test_generation_may_not_read_withheld_records(manifest, check, allowed, message):
    check(manifest.roles[allowed], manifest)
    intruder = manifest.roles["withheld"][0]
    with pytest.raises(LeakageError, match=message) as error:
        check([*manifest.roles[allowed], intruder], manifest)
    assert intruder in str(error.value)
