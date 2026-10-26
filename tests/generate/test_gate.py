"""Generator checks measure control accuracy on matched editing samples."""

import json
from dataclasses import asdict

import numpy as np

from syntide.generate.editor import EditorConfig
from syntide.generate.gate import control_fidelity, review_sample, run_gate


def test_the_gate_fails_a_biased_generator_that_correlation_alone_would_pass():
    requested = np.linspace(0.1, 3.5, 200)
    biased = control_fidelity(requested, 2 * requested + 0.2, 0.1, 3.5)
    assert biased["corr"] == 1.0
    assert biased["passed"] is False
    faithful = control_fidelity(requested, requested * 1.02 + 0.01, 0.1, 3.5)
    assert faithful["passed"] is True
    assert (
        control_fidelity(requested[:5], requested[:5], 0.1, 3.5)["passed"]
        is None
    )  # too few to judge


def test_gate_passes_the_editor_on_paired_draws_and_samples_severity_extremes(
    make_samples, editor_calibration
):
    parents = make_samples(30, n_samples=1000)
    report, recipes = run_gate(
        parents,
        editor_calibration,
        "3" * 64,
        asdict(EditorConfig()),
        100,
        recipes=300,
    )

    assert report["passed"] is True
    assert report["calibration_sha256"] == "3" * 64
    for result in report["policies"].values():
        assert result["all_finite"] and result["acceptance_rate"] > 0.9
        assert all(
            cell["passed"] is not False for cell in result["controls"].values()
        )
    # Recipe i uses the same parent and draw under both policies; only the
    # controls differ.
    rows = recipes.to_pylist()
    a2 = {r["step"]: r for r in rows if r["policy"] == "coverage_matched"}
    a3 = {r["step"]: r for r in rows if r["policy"] == "gap_targeted"}
    assert all(
        a2[i]["parent_record_id"] == a3[i]["parent_record_id"] for i in a2
    )
    assert all(a2[i]["seed_key"] == a3[i]["seed_key"] for i in a2)
    assert any(
        a2[i]["controls_requested"] != a3[i]["controls_requested"] for i in a2
    )

    sample = review_sample(recipes, editor_calibration["envelope"], n=12)
    assert len(sample) == 12 and len({row["recipe_id"] for row in sample}) == 12
    assert all(row["verdict"] == "" for row in sample)
    for kind in ("baseline_drift", "static_noise", "burst_noise"):
        chosen = [
            row["severity_position"]
            for row in sample
            if row["artifact_type"] == kind
        ]
        accepted = [
            json.loads(r["controls_requested"])
            for r in rows
            if r["decision"] == "accepted"
            and json.loads(r["controls_requested"])["artifact_type"] == kind
        ]
        low, high = editor_calibration["envelope"][kind]["amplitude_mv"]
        positions = sorted(
            (c["amplitude_mv"] - low) / (high - low) for c in accepted
        )
        assert min(chosen) == round(positions[0], 4) and max(chosen) == round(
            positions[-1], 4
        )
