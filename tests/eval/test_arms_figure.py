"""The arms summary: gaps to the 100% target and seed-paired changes over the baseline arm."""

from syntide.eval.arms_figure import figures, summary


def _row(regime, arm, seed, auroc, group="augmented", best_step=1000):
    scores = {"overall": auroc, "artifact": auroc - 0.02, "clean": auroc + 0.01}
    return {
        "regime": regime,
        "arm": arm,
        "seed": seed,
        "group": group,
        "best_step": best_step,
        "max_steps": 4000,
        "val": dict(scores),
        "test": dict(scores),
    }


ROWS = [
    *(_row("half", "A0", seed, 0.90 + 0.001 * seed, group="clean") for seed in range(3)),
    *(_row("half", "A3", seed, 0.91 + 0.002 * seed) for seed in range(3)),
    *(_row("full", "C2", seed, 0.93, group="real") for seed in range(3)),
]


def test_summary_reports_gaps_to_the_target_and_paired_changes_seed_by_seed():
    text = summary(ROWS, "half", ("full", "C2"), "A0")
    # A0 averages 0.901 and the target 0.93; the gap is the same on the noisy slice (both shift by 0.02).
    assert "| A0 clean only | 0.9010 ± 0.0010 |" in text
    assert "-0.0290 | -0.0290 |" in text
    # A3 - A0 per seed: +0.010, +0.011, +0.012, so every seed is better.
    assert "| A3 | +0.0110 | +0.0100, +0.0110, +0.0120 |" in text
    assert "3/3 / 3/3" in text
    assert "Every run peaked before 80% of its budget" in text


def test_a_late_best_step_is_flagged():
    rows = [*ROWS[:-1], _row("full", "C2", 2, 0.93, group="real", best_step=3750)]
    assert "1 runs picked a checkpoint past 80% of the budget" in summary(rows, "half", ("full", "C2"), "A0")


def test_figures_are_written(tmp_path):
    paths = figures(ROWS, "half", ("full", "C2"), tmp_path)
    assert [p.name for p in paths] == ["arms_overall.png", "arms_slices.png"]
    assert all(p.stat().st_size > 10_000 for p in paths)
