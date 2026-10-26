"""The browser compares real and generated signals with accurate labels."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from syntide.explore import viewer


def _record(
    record_id,
    values,
    *,
    origin="real",
    times=(0, 1),
    units="mV",
    channels=("I",),
):
    return {
        "record_id": record_id,
        "metadata": {"origin": origin},
        "labels": [],
        "waveform": {
            "channels": list(channels),
            "values": values,
            "times": list(times),
            "units": units,
            "sample_rate": 100,
            "original_samples": 1000,
            "preview_samples": len(times),
        },
        "stats": {"duration_seconds": 10},
    }


def _javascript(payload, expression, *, render=False):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to validate browser comparison geometry")
    template = Path(viewer.__file__).with_name("viewer.html").read_text()
    helpers = template[
        template.index("function recordOrigin(r)") : template.index(
            "const clusters = new Map"
        )
    ]
    script = (
        "const p=JSON.parse(require('fs').readFileSync(0,'utf8'));\n" + helpers
    )
    if render:
        script += """
        const R = p.records, active = R[0].record_id;
        const nodes = {
          waveform: {children: [], replaceChildren(){this.children=[]},
            setAttribute(){}, append(node){this.children.push(node)}},
          channel: {value: "I"}, "compare-record": {value: R[1].record_id},
          "wave-legend": {}, "comparison-note": {}, "wave-caption": {}
        };
        const $ = id => nodes[id], fmt = String, esc = String;
        const text = (node, value) => {node.textContent=value};
        const svg = (tag, attrs) => ({tag, attrs});
        """
        script += template[
            template.index("function wave() {") : template.index(
                "function selectionChanged()"
            )
        ]
        script += "wave();\n"
    script += "process.stdout.write(JSON.stringify(" + expression + "));"
    result = subprocess.run(
        [node, "-e", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_overlay_preserves_amplitudes_and_actual_seconds():
    real = _record("real-1", [[0, 2]], times=(0, 2))
    synthetic = _record(
        "generated-1", [[0, 1]], origin="synthetic", times=(0, 1)
    )
    result = _javascript(
        {"records": [synthetic, real]},
        'nodes.waveform.children.filter(node => node.tag === "path")',
        render=True,
    )
    paths = {row["attrs"]["data-record-id"]: row["attrs"] for row in result}
    # The same 0 mV has the same Y value. A 1 mV generated trace reaches only
    # half of the real trace's 2 mV excursion, and 1 s occupies half of 2 s.
    assert paths["real-1"]["d"] == "M60.00,210.00L780.00,34.00"
    assert paths["generated-1"]["d"] == "M60.00,210.00L420.00,122.00"
    assert paths["real-1"]["stroke-dasharray"] == "5 3"
    assert paths["generated-1"]["stroke-dasharray"] == "none"
    assert paths["real-1"]["stroke"] != paths["generated-1"]["stroke"]


def test_channels_align_by_name_and_incompatible_units_are_not_overlaid():
    real = _record(
        "real-1", [[-1, 1], [-5, 5]], channels=("I", "II"), units=["mV", "mV"]
    )
    synthetic = _record(
        "generated-1",
        [[-3, 3], [-7, 7]],
        channels=("II", "I"),
        units=["mV", "V"],
    )
    result = _javascript(
        {"records": [real, synthetic]},
        'waveformComparison(p.records[0], p.records[1], ["I", "II"])',
    )
    assert len(result["rows"][0]["series"]) == 1
    assert result["rows"][0]["hi"] == 1
    assert result["rows"][1]["series"][1]["values"] == [-3, 3]
    assert result["rows"][1]["lo"] == -5
    assert result["rows"][1]["hi"] == 5
    assert result["notices"] == [
        "I: comparison omitted because channel units differ."
    ]


def test_missing_channels_and_constant_missing_values_have_finite_axes():
    real = _record("real-1", [[None, None], [3, 3]], channels=("I", "II"))
    synthetic = _record("generated-1", [[None, None]])
    result = _javascript(
        {"records": [real, synthetic]},
        'waveformComparison(p.records[0], p.records[1], ["I", "II"])',
    )
    assert (result["rows"][0]["lo"], result["rows"][0]["hi"]) == (-1, 1)
    assert (result["rows"][1]["lo"], result["rows"][1]["hi"]) == (2.5, 3.5)
    assert result["notices"] == ["generated-1: channel II is unavailable."]


def test_requested_labels_remain_distinct_from_dataset_labels():
    synthetic = _record("generated-1", [[1, 2]], origin="synthetic")
    synthetic["metadata"]["requested_labels"] = ["AFIB"]
    result = _javascript(
        {"record": synthetic},
        (
            "({origin: recordOrigin(p.record), requested: "
            "requestedLabels(p.record), labels: "
            "p.record.labels})"
        ),
    )
    assert result == {
        "origin": "synthetic",
        "requested": ["AFIB"],
        "labels": [],
    }


def test_generated_catalog_preserves_origin_and_conditioning():
    real = _record("real-1", [[1, 2]])
    synthetic = _record("generated-1", [[3, 4]])
    # This is the metadata contract emitted by generate/full/runner.py.
    synthetic["metadata"] = {
        "synthetic": True,
        "generation": {
            "source_record_ids": ["not-in-analysis", "generated-1", "real-1"]
        },
        "requested_labels": ["AFIB"],
        "label_status": "pending_review",
    }
    result = _javascript(
        {"records": [synthetic, real]},
        (
            "({origin: recordOrigin(p.records[0]), paired:"
            " pairedRecordId(p.records[0], p.records), "
            "requested: requestedLabels(p.records[0]), "
            "labels: p.records[0].labels})"
        ),
    )
    assert result == {
        "origin": "synthetic",
        "paired": "real-1",
        "requested": ["AFIB"],
        "labels": [],
    }


def test_explicit_origin_and_conditioning_reference_take_precedence():
    first = _record("real-1", [[1, 2]])
    second = _record("real-2", [[3, 4]])
    record = _record("generated-1", [[5, 6]])
    record["metadata"].update(
        synthetic=True,
        source_record_id="real-2",
        generation={"source_record_ids": ["real-1"]},
    )
    result = _javascript(
        {"records": [record, first, second]},
        (
            "({origin: recordOrigin(p.records[0]), paired:"
            " pairedRecordId(p.records[0], p.records)})"
        ),
    )
    assert result == {"origin": "real", "paired": "real-2"}


def test_display_names_shorten_presentation_without_changing_identifiers():
    record_id = "synthetic:" + "a" * 64 + ":00000001"
    record = _record(record_id, [[1, 2]], origin="synthetic")
    result = _javascript(
        {"record": record},
        "({label: recordLabel(p.record), id: p.record.record_id})",
    )
    assert len(result["label"]) <= 42
    assert result["label"].startswith("synthetic:")
    assert result["label"].endswith(":00000001")
    assert result["id"] == record_id
    record["metadata"]["display_name"] = "DiffuSETS 01 · ref ptbxl-5998"
    assert (
        _javascript({"record": record}, "recordLabel(p.record)")
        == record["metadata"]["display_name"]
    )
