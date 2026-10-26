"""Augmentation figures compare each recipe with unchanged training data."""

from types import SimpleNamespace

import numpy as np
import pytest

from syntide.data.schema import SUPERCLASSES
from syntide.eval.aug_figure import (
    COLUMNS,
    classes_figure,
    curves_figure,
    dose,
    dose_figure,
    paired,
    shown,
    summary,
)
from syntide.generate.generic import GenericAugConfig

# (arm, family, policy, share, strength, change over A0 on every column)
ARMS = [
    ("A0", "none", "none", 0.5, 0, 0.0),
    ("S1", "generic", "fixed", 0.125, 0.25, 0.001),
    ("S2", "generic", "fixed", 0.25, 0.5, 0.003),
    ("A1", "generic", "fixed", 0.5, 1, 0.005),
    ("S3", "generic", "fixed", 1.0, 1, 0.006),
    ("S4", "generic", "fixed", 1.0, 2, 0.002),
    ("A3", "editor", "gap_targeted", 0.5, 1, 0.006),
    ("E1", "editor", "gap_targeted", 1.0, 1, 0.007),
    *(
        (f"T{i}", "ecg_aug", "fixed", 0.5, 1, 0.001 * i - 0.002)
        for i in range(1, 6)
    ),
    ("T6", "simvea", "fixed", 0.5, 1, 0.002),
    ("T6W", "simvea", "fixed", 1.0, 1, 0.003),
]
# T6 and T6W were planned in protocols of their own, after the sweep's test
# claim.
EXTENSIONS = {
    "T6": "runs/ptbxl-aug-simvea",
    "T6W": "runs/ptbxl-aug-simvea-whole",
}


def _rows():
    rows = []
    for arm, family, policy, share, strength, change in ARMS:
        for seed in range(3):
            base = 0.90 + 0.001 * seed
            scores = {
                column: base + change + (0.0005 * seed if arm == "T1" else 0)
                for column in COLUMNS
            }
            if arm == "T3":
                scores["HYP"] = (
                    base - 0.004
                )  # amplitude scaling hurts hypertrophy
            rows.append(
                {
                    "regime": "scarcity50",
                    "arm": arm,
                    "seed": seed,
                    "family": family,
                    "source": EXTENSIONS.get(arm, "runs/ptbxl-aug-sweep"),
                    "series": (family, policy),
                    "share": share,
                    "dose": share * strength,
                    "best_step": int(1000 + 4000 * share * strength),
                    "max_steps": 10000,
                    "history": [
                        (s, 0.88 + change * s / 10000)
                        for s in range(100, 10001, 100)
                    ],
                    "scores": scores,
                }
            )
    for seed in range(3):
        scores = {column: 0.917 + 0.001 * seed for column in COLUMNS}
        rows.append(
            {
                "regime": "full100",
                "arm": "C2",
                "seed": seed,
                "family": "none",
                "series": ("none", "none"),
                "source": "runs/ptbxl-aug-sweep",
                "share": 0.0,
                "dose": 0.0,
                "best_step": 3000,
                "max_steps": 10000,
                "history": [(s, 0.92) for s in range(100, 10001, 100)],
                "scores": scores,
            }
        )
    return rows


def test_dose_is_share_times_strength_of_a1s_envelope():
    def spec(family, share, transform):
        return SimpleNamespace(
            transform_family=family,
            transform_fraction=share,
            transform=transform,
        )

    a1 = GenericAugConfig().__dict__
    strong = {
        "gain": [0.8, 1.2],
        "max_shift_s": 1.0,
        "noise_rms_mv": [0, 0.2],
        "wander_mv": [0, 7.0],
    }
    assert dose(spec("generic", 0.5, a1)) == pytest.approx(0.5)
    assert dose(spec("generic", 1.0, strong)) == pytest.approx(2.0)
    assert dose(spec("editor", 1.0, {})) == 1.0
    assert dose(spec("none", 0.5, {})) == 0.0


def test_changes_are_paired_seed_by_seed_and_the_table_reports_them():
    rows = _rows()
    deltas = paired(rows, ("scarcity50", "T1"), ("scarcity50", "A0"))
    assert deltas["overall"] == pytest.approx([-0.001, -0.0005, 0.0])
    assert paired(rows, ("scarcity50", "T3"), ("scarcity50", "A0"))[
        "HYP"
    ] == pytest.approx([-0.004] * 3)

    text = summary(rows, "scarcity50", "A0", ("full100", "C2"), "test")
    # The target gains 0.017 over A0, so A1's +0.005 closes 29% of the gap.
    assert "| A1 | standard, 1/2 of batch | 0.5 | +0.0050 (3/3) |" in text
    assert "| 29% |" in text
    assert (
        "| C2@full100 | 100% of patients (target) | - | +0.0170 (3/3) |" in text
    )
    # S4 peaks at step 9,000 of 10,000, so the budget may have cut it short.
    assert (
        "Picked past 80% of the budget, so they may not have peaked: S4."
        in text
    )
    assert all(name in text for name in SUPERCLASSES)


def test_both_figures_are_written(tmp_path):
    rows = _rows()
    reference = ("full100", "C2")
    paths = [
        dose_figure(
            rows, "scarcity50", "A0", reference, "test", tmp_path / "dose.png"
        ),
        classes_figure(
            rows,
            "scarcity50",
            "A0",
            reference,
            "test",
            tmp_path / "classes.png",
        ),
        curves_figure(
            rows, "scarcity50", "A0", reference, tmp_path / "curves.png"
        ),
    ]
    assert all(path.stat().st_size > 20_000 for path in paths)
    np.testing.assert_array_equal(
        [p.name for p in paths], ["dose.png", "classes.png", "curves.png"]
    )


def test_hidden_arms_leave_the_figures_but_the_summary_keeps_every_arm(
    tmp_path,
):
    import matplotlib.image

    rows = _rows()
    reference = ("full100", "C2")
    drawn = shown(rows, ["S4"], "scarcity50", "A0", reference)
    assert "S4" not in {r["arm"] for r in drawn} and {"T6", "T6W"} <= {
        r["arm"] for r in drawn
    }
    text = summary(rows, "scarcity50", "A0", reference, "val")
    assert (
        "| S4 |" in text
        and "| T6 | SIMVEA_h chest-lead mixing, 1/2 of batch | 0.5 |" in text
    )
    assert "| T6W | SIMVEA_h chest-lead mixing, whole batch | 1 |" in text
    assert (
        f"Runs from runs/ptbxl-aug-sweep, {', '.join(EXTENSIONS.values())}"
        in text
    )

    # SIMVEA_h gets a fourth curves panel, so the panels go two by two; without
    # it they stay in one row.
    four = curves_figure(
        drawn, "scarcity50", "A0", reference, tmp_path / "four.png"
    )
    only_whole = [r for r in drawn if r["arm"] != "T6"]
    also_four = curves_figure(
        only_whole, "scarcity50", "A0", reference, tmp_path / "also_four.png"
    )
    without = [r for r in drawn if r["arm"] not in EXTENSIONS]
    three = curves_figure(
        without, "scarcity50", "A0", reference, tmp_path / "three.png"
    )
    for path in (four, also_four):
        height, width = matplotlib.image.imread(path).shape[:2]
        assert width / height < 1.6
    height, width = matplotlib.image.imread(three).shape[:2]
    assert width / height > 3
    dose_figure(
        drawn, "scarcity50", "A0", reference, "val", tmp_path / "dose.png"
    )
    classes_figure(
        drawn, "scarcity50", "A0", reference, "val", tmp_path / "classes.png"
    )
