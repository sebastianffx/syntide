"""Gate 1: the leakage guard.

Arms deliberately overlap (A0-A3 share parents; natural_mix and depleted share most records), so the
guard does not demand that every role be disjoint. It enforces the relations the experiment relies on
(MVE Gate 1; training_loop doc, "narrow the leakage assertion"):

- the manifest was built from this exact data (same source fingerprint);
- the outer partitions (pool, unused, val, test) are disjoint and are exactly the spec's folds;
- no patient is in more than one of train, val and test, nor in both the pool and unused;
- retained_artifact and withheld cover the pool's artifact ECGs, depleted is the pool minus withheld,
  and no withheld patient keeps an ECG in depleted;
- natural_mix is the pool minus whole patients, with the same size as depleted;
- generator calibration reads only retained_artifact records, and every recipe parent is in depleted;
- for question-answer data, no question of a validation or test patient reaches a training role.
"""

from collections import Counter
from collections.abc import Iterable, Mapping

import pyarrow as pa

from syntide.data.challenge import Manifest

_OUTER = ("pool", "unused", "val", "test")


class LeakageError(Exception):
    """A manifest or generation input breaks an isolation rule; the run must stop."""


def check_manifest(manifest: Manifest, index: pa.Table, source: Mapping[str, str]) -> None:
    """Raise LeakageError, naming the offending records or subjects, if any invariant fails."""
    if dict(manifest.source) != dict(source):
        raise LeakageError(
            f"{manifest.challenge_id} was built from {manifest.source}, not from the local data "
            f"{dict(source)}; rebuild the manifest or the dataset"
        )
    roles = {role: set(ids) for role, ids in manifest.roles.items()}
    columns = index.select(["record_id", "subject_uid", "strat_fold", "is_artifact"]).to_pydict()
    subject_of = dict(zip(columns["record_id"], columns["subject_uid"], strict=True))
    fold_of = dict(zip(columns["record_id"], columns["strat_fold"], strict=True))
    artifact = {r for r, flag in zip(columns["record_id"], columns["is_artifact"], strict=True) if flag}

    _fail_if(set().union(*roles.values()) - subject_of.keys(), "records are not labelled PTB-XL records")

    for i, a in enumerate(_OUTER):
        for b in _OUTER[i + 1 :]:
            _fail_if(roles[a] & roles[b], f"records are in both {a} and {b}")
    train = roles["pool"] | roles["unused"]
    for name, ids in (("train", train), ("val", roles["val"]), ("test", roles["test"])):
        folds = manifest.spec[f"{name}_folds"]
        expected = {r for r, fold in fold_of.items() if fold in folds}
        _fail_if(ids ^ expected, f"records break {name} == folds {folds}")

    for role in ("retained_artifact", "withheld", "depleted", "natural_mix"):
        _fail_if(roles[role] - roles["pool"], f"{role} records are outside the pool")
    _fail_if(roles["retained_artifact"] & roles["withheld"], "records are both retained and withheld")
    _fail_if(
        (roles["pool"] & artifact) ^ (roles["retained_artifact"] | (roles["withheld"] & artifact)),
        "records break retained + withheld == the pool's artifact records",
    )
    _fail_if(roles["depleted"] & roles["withheld"], "withheld records are in depleted")
    _fail_if(
        roles["depleted"] ^ (roles["pool"] - roles["withheld"]), "records break depleted == pool - withheld"
    )

    def subjects(ids: set[str]) -> set[str]:
        return {subject_of[r] for r in ids}

    for a, a_ids, b, b_ids in (
        ("pool", roles["pool"], "unused", roles["unused"]),
        ("train", train, "val", roles["val"]),
        ("train", train, "test", roles["test"]),
        ("val", roles["val"], "test", roles["test"]),
        ("withheld", roles["withheld"], "depleted", roles["depleted"]),
        ("natural_mix", roles["natural_mix"], "the records C1 drops", roles["pool"] - roles["natural_mix"]),
    ):
        _fail_if(subjects(a_ids) & subjects(b_ids), f"subjects are in both {a} and {b}")

    if len(roles["natural_mix"]) != len(roles["depleted"]):
        raise LeakageError(
            f"{manifest.challenge_id}: natural_mix has {len(roles['natural_mix'])} records "
            f"but depleted has {len(roles['depleted'])}; the matched-N control needs equal sizes"
        )


def check_tasks(tasks: pa.Table, manifest: Manifest, index: pa.Table) -> None:
    """Gate 1 for question-answer data: no question of a validation or test patient reaches training.

    ``tasks`` has one row per question with its ``task_id``, ``record_id`` and ``subject_uid``. A run's
    questions are the questions on its role's records, so this checks the join itself: every question
    sits on a record of ``index`` with that record's patient, task ids are unique, and the patients
    of the questions in training roles never ask a question in val or test.
    """
    columns = tasks.select(["task_id", "record_id", "subject_uid"]).to_pydict()
    subject_of = dict(zip(index["record_id"].to_pylist(), index["subject_uid"].to_pylist(), strict=True))
    _fail_if({t for t, n in Counter(columns["task_id"]).items() if n > 1}, "task ids are repeated")
    _fail_if(set(columns["record_id"]) - subject_of.keys(), "questions sit on records outside the index")
    _fail_if(
        {
            task_id
            for task_id, record_id, subject in zip(*columns.values(), strict=True)
            if subject_of.get(record_id, subject) != subject
        },
        "questions name another patient than their record",
    )
    role_of_record: dict[str, set[str]] = {}
    for role, ids in manifest.roles.items():
        for record_id in ids:
            role_of_record.setdefault(record_id, set()).add(role)
    held_out = {"val", "test"}
    train_subjects, held_out_subjects = set(), set()
    for record_id, subject in zip(columns["record_id"], columns["subject_uid"], strict=True):
        roles = role_of_record.get(record_id, set())
        if roles & held_out:
            held_out_subjects.add(subject)
        if roles - held_out:
            train_subjects.add(subject)
    _fail_if(train_subjects & held_out_subjects, "patients ask questions in both training roles and val/test")


def check_calibration_ids(record_ids: Iterable[str], manifest: Manifest) -> None:
    """Generator calibration may only see retained artifact records (never withheld, val or test)."""
    _fail_if(
        set(record_ids) - set(manifest.roles["retained_artifact"]),
        "calibration records are outside retained_artifact",
    )


def check_recipe_parents(parent_ids: Iterable[str], manifest: Manifest) -> None:
    """Every recipe (artifact edit or standard augmentation) must start from a depleted-pool record."""
    _fail_if(set(parent_ids) - set(manifest.roles["depleted"]), "recipe parents are outside depleted")


def _fail_if(offenders: set[str], message: str) -> None:
    if offenders:
        examples = ", ".join(sorted(offenders)[:5])
        raise LeakageError(f"{len(offenders)} {message}: {examples}")
