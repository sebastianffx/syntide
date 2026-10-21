"""The two sampling policies: same draws, A2 copies the retained data, A3 fills what it lacks."""

import json
from dataclasses import replace

import numpy as np
import pytest

from syntide.data.challenge import ChallengeSpec, build_manifest
from syntide.data.leakage import LeakageError
from syntide.generate.editor import ARTIFACTS, CONTROLS, ELECTRODES, EditorConfig, control_range
from syntide.generate.policy import (
    POLICIES,
    U_FIELDS,
    CalibrationError,
    controls_for,
    fit_calibration,
    load_calibration,
    save_calibration,
)

FS, N = 100, 1000
SOURCE = {"dataset": "physionet/ptb-xl", "version": "1.0.3", "sha256": "test"}


def _draws(n: int) -> np.ndarray:
    return np.random.default_rng(0).random((n, len(U_FIELDS)))


def test_both_policies_map_the_same_draws_inside_the_envelope(editor_calibration):
    envelope = editor_calibration["envelope"]
    for u in _draws(500):
        a2, a3 = (controls_for(policy, u, editor_calibration, N, FS) for policy in POLICIES)
        for controls in (a2, a3):
            kind = controls["artifact_type"]
            assert controls["electrode"] in ELECTRODES
            assert set(controls) == {"artifact_type", "electrode", *CONTROLS[kind]}
            for name in CONTROLS[kind]:
                low, high = control_range(envelope, kind, name, N / FS)
                assert low <= controls[name] <= high
            if kind == "burst_noise":
                assert controls["onset_s"] + controls["duration_s"] <= N / FS
        # Nuisance controls come from the same draws, so they match whenever the types do.
        if a2["artifact_type"] == a3["artifact_type"]:
            assert a2["electrode"] == a3["electrode"]


def test_coverage_matched_reproduces_the_retained_quantiles(editor_calibration):
    retained = editor_calibration["types"]["baseline_drift"]["severities"]
    for q in (0.0, 0.1, 0.5, 0.9, 1.0):
        u = np.array([0.0, q, 0.5, 0.5, 0.5, 0.5])  # u_type 0 -> the first type, baseline_drift
        assert controls_for("coverage_matched", u, editor_calibration, N, FS)[
            "amplitude_mv"
        ] == pytest.approx(np.quantile(retained, q))


def test_gap_targeted_moves_mass_to_rare_types_and_sparse_severities(editor_calibration):
    planned = {
        policy: [controls_for(policy, u, editor_calibration, N, FS) for u in _draws(4000)]
        for policy in POLICIES
    }

    def share(policy, kind):
        return np.mean([c["artifact_type"] == kind for c in planned[policy]])

    # Bursts are 2 of 24 retained records: A2 keeps them rare, A3 up-weights them.
    assert share("coverage_matched", "burst_noise") == pytest.approx(2 / 24, abs=0.02)
    assert share("gap_targeted", "burst_noise") > 2 * share("coverage_matched", "burst_noise")

    # Retained drift tops out at 1.2 mV in an envelope up to 3.5 mV: A2 never goes past it, A3 mostly does.
    top = max(editor_calibration["types"]["baseline_drift"]["severities"])

    def beyond_retained(policy):
        drift = [c["amplitude_mv"] for c in planned[policy] if c["artifact_type"] == "baseline_drift"]
        return np.mean(np.array(drift) > top)

    assert beyond_retained("coverage_matched") == 0.0
    assert beyond_retained("gap_targeted") > 0.5


def test_calibration_reads_only_retained_records_and_is_frozen(synthetic_index, make_samples, tmp_path):
    manifest = build_manifest(synthetic_index, ChallengeSpec(), 0.25, SOURCE)
    retained = make_samples(6, n_samples=N)
    retained = replace(retained, record_id=np.array(manifest.roles["retained_artifact"][:6]))
    calibration = fit_calibration(manifest, retained, EditorConfig(), FS)
    assert calibration["types"]["baseline_drift"]["n_retained"] == 3  # make_samples flags every other record
    assert calibration["manifest_sha256"] == manifest.sha256

    leaked = replace(retained, record_id=np.array(manifest.roles["withheld"][:6]))
    with pytest.raises(LeakageError, match="outside retained_artifact"):
        fit_calibration(manifest, leaked, EditorConfig(), FS)

    path = tmp_path / "calibration.json"
    sha = save_calibration(calibration, path)
    assert load_calibration(path, sha) == calibration
    assert save_calibration(calibration, path) == sha  # the same calibration again is fine
    with pytest.raises(FileExistsError, match="frozen"):
        save_calibration({**calibration, "density_floor": 2.0}, path)
    edited = json.loads(path.read_text())
    edited["types"]["baseline_drift"]["severities"][0] = 9.0
    path.write_text(json.dumps(edited))
    with pytest.raises(CalibrationError):
        load_calibration(path)


@pytest.mark.parametrize("kind", ARTIFACTS)
def test_every_type_has_a_severity_map_for_each_policy(editor_calibration, kind):
    fitted = editor_calibration["types"][kind]
    assert fitted["targeted_cdf"][0] == 0.0 and fitted["targeted_cdf"][-1] == pytest.approx(1.0)
    assert np.all(np.diff(fitted["targeted_cdf"]) >= 0)
