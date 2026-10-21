"""Gate 1: every isolation rule the experiment relies on is enforced, and the offender is named."""

from dataclasses import replace

import pyarrow as pa
import pytest

from syntide.data.challenge import ChallengeSpec, build_manifest
from syntide.data.leakage import LeakageError, check_calibration_ids, check_manifest

SOURCE = {"dataset": "physionet/ptb-xl", "version": "1.0.3", "sha256": "test"}


@pytest.fixture(scope="module")
def manifest(synthetic_index):
    return build_manifest(synthetic_index, ChallengeSpec(), 0.25, SOURCE)


def _with_roles(manifest, **changes):
    return replace(manifest, roles={**manifest.roles, **changes})


def _add_to_pool(m):
    record = m.roles["val"][0]
    return _with_roles(m, pool=(*m.roles["pool"], record)), record, "in both pool and val"


def _drop_from_unused(m):
    record = m.roles["unused"][0]
    return _with_roles(m, unused=m.roles["unused"][1:]), record, "in no partition"


def _unbalance_withheld(m):
    record = m.roles["withheld"][0]
    return _with_roles(m, withheld=m.roles["withheld"][1:]), record, "retained \\+ withheld"


def _withheld_into_depleted(m):
    record = m.roles["withheld"][0]
    return _with_roles(m, depleted=(*m.roles["depleted"], record)), record, "withheld records are in depleted"


def _natural_mix_outside_pool(m):
    record = m.roles["unused"][0]
    mixed = (*m.roles["natural_mix"][1:], record)
    return _with_roles(m, natural_mix=mixed), record, "outside the pool"


def _natural_mix_too_small(m):
    return _with_roles(m, natural_mix=m.roles["natural_mix"][1:]), "natural_mix", "matched-N control"


@pytest.mark.parametrize(
    "corrupt",
    [
        _add_to_pool,
        _drop_from_unused,
        _unbalance_withheld,
        _withheld_into_depleted,
        _natural_mix_outside_pool,
        _natural_mix_too_small,
    ],
)
def test_each_broken_role_relation_is_rejected_and_named(manifest, synthetic_index, corrupt):
    corrupted, offender, message = corrupt(manifest)
    with pytest.raises(LeakageError, match=message) as error:
        check_manifest(corrupted, synthetic_index)
    assert offender in str(error.value)


def test_a_patient_in_both_train_and_test_is_rejected(manifest, synthetic_index):
    # Same ECG ids, but one test ECG now belongs to a patient who is in the training pool.
    pool_subject = dict(
        zip(synthetic_index["record_id"].to_pylist(), synthetic_index["subject_uid"].to_pylist(), strict=True)
    )[manifest.roles["pool"][0]]
    test_record = manifest.roles["test"][0]
    subjects = [
        pool_subject if r == test_record else s
        for r, s in zip(
            synthetic_index["record_id"].to_pylist(), synthetic_index["subject_uid"].to_pylist(), strict=True
        )
    ]
    leaky_index = synthetic_index.set_column(1, "subject_uid", pa.array(subjects))

    with pytest.raises(LeakageError, match="subjects are in both train and test") as error:
        check_manifest(manifest, leaky_index)
    assert pool_subject in str(error.value)


def test_generator_calibration_may_not_read_withheld_records(manifest):
    check_calibration_ids(manifest.roles["retained_artifact"], manifest)
    with pytest.raises(LeakageError, match="outside retained_artifact") as error:
        check_calibration_ids([*manifest.roles["retained_artifact"], manifest.roles["withheld"][0]], manifest)
    assert manifest.roles["withheld"][0] in str(error.value)
