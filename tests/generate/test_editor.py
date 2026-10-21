"""The editor works in electrode space and regenerates every edit from its recipe."""

import numpy as np
import pytest

from syntide.data.schema import LEADS
from syntide.generate import editor
from syntide.generate.editor import Recipe, edit

FS = 100
PARENT = np.zeros((12, 1000), dtype=np.float32)


def _recipe(electrode: str, artifact_type: str = "baseline_drift", **controls) -> Recipe:
    defaults = {
        "baseline_drift": {"amplitude_mv": 0.5, "freq_hz": 0.3},
        "static_noise": {"amplitude_mv": 0.05},
        "burst_noise": {"amplitude_mv": 0.2, "onset_s": 2.0, "duration_s": 1.5},
    }[artifact_type]
    return Recipe(
        "rec-0",
        editor.GENERATOR,
        editor.VERSION,
        {"artifact_type": artifact_type, "electrode": electrode, **defaults, **controls},
        seed_key=7,
        gap_report_id="gap-test",
    )


def _leads_touched(electrode: str) -> set[str]:
    residual = edit(PARENT, _recipe(electrode), FS).residual
    return {lead for lead, row in zip(LEADS, residual, strict=True) if np.abs(row).max() > 0}


def test_electrode_artifacts_project_through_einthoven_goldberger_and_wilson():
    every = set(LEADS)
    # Lead II = LL - RA is the only lead without the left arm; it stays exactly zero.
    assert _leads_touched("LA") == every - {"II"}
    assert np.all(edit(PARENT, _recipe("LA"), FS).residual[LEADS.index("II")] == 0)
    assert _leads_touched("RA") == every - {"III"}  # III = LL - LA
    assert _leads_touched("V3") == {"V3"}  # precordial electrodes are not part of the Wilson terminal
    # The strongest lead carries the requested amplitude, whatever the electrode.
    for electrode in editor.ELECTRODES:
        assert np.abs(editor.LEAD_FROM_ELECTRODE[:, editor.ELECTRODES.index(electrode)]).max() == 1


@pytest.mark.parametrize("artifact_type", editor.ARTIFACTS)
def test_an_edit_regenerates_bit_exactly_and_only_from_its_recipe(artifact_type):
    parent = np.random.default_rng(0).standard_normal((12, 1000)).astype(np.float32)
    recipe = _recipe("LL", artifact_type)
    first, again = edit(parent, recipe, FS), edit(parent, recipe, FS)
    assert np.array_equal(first.x, again.x)
    assert np.array_equal(first.x, (parent.astype(np.float64) + first.residual).astype(np.float32))
    other = Recipe(**{**recipe.__dict__, "seed_key": 8})
    assert other.recipe_id != recipe.recipe_id
    if artifact_type != "baseline_drift":  # drift's only randomness is its phase
        assert not np.array_equal(edit(parent, other, FS).x, first.x)
    if artifact_type == "burst_noise":
        quiet = np.ones(1000, dtype=bool)
        quiet[200:350] = False
        assert np.all(first.residual[:, quiet] == 0) and first.interval == (2.0, 3.5)
