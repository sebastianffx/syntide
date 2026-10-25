"""Integration checks for sampling, cache identity, portable plans and the HTTP seam."""

import json
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from syntide.explore import embeddings
from syntide.explore.pipeline import run_analysis, sample_rows
from syntide.explore.plans import (
    content_id,
    make_plan,
    seal_analysis,
    verify_analysis,
    verify_plan,
    write_frozen,
)
from syntide.explore.server import make_server
from syntide.explore.viewer import write_viewer


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    def fake_signals(signals, rates, channels, config):
        return np.array([[x.mean(), x.std(), x.max(), 1] for x in signals]), {"backend": "test"}

    def fake_text(texts, config):
        return np.array([[len(t), i + 1, 1] for i, t in enumerate(texts)]), {"backend": "test"}

    monkeypatch.setattr(embeddings, "embed_recordings", fake_signals)
    monkeypatch.setattr(embeddings, "embed_texts", fake_text)
    rows = []
    for i in range(6):
        path = tmp_path / f"signal-{i}.npy"
        np.save(path, np.stack([np.linspace(i, i + 2, 30), np.sin(np.arange(30)) * (i + 1)]))
        for j in range(2):
            rows.append(
                {
                    "record_id": f"r{i}",
                    "subject_uid": f"p{i}",
                    "split": "train",
                    "signal_path": path.name,
                    "sample_rate": 10,
                    "channels": ["I", "II"],
                    "labels": ["rare" if i == 0 else "common"],
                    "question": f"Question {j}?",
                    "answer": f"Answer {i}",
                    "text_id": f"q{i}-{j}",
                }
            )
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("\n".join(json.dumps(row) for row in rows))
    config = {
        "source": {"dataset": "jsonl", "input": str(catalog)},
        "sampling": {"size": 4, "fraction": None, "seed": 42},
        "embedding": {"backend": "chronos", "seed": 42},
        "text": {"fields": ["question", "answer", "text"], "enabled": True},
        "analysis": {"projection": "pca", "n_clusters": 2, "seed": 42, "target_count": 3},
    }
    output = tmp_path / "analysis"
    return config, output


def test_record_sampling_is_order_independent_and_bounds_checked():
    rows = [{"record_id": str(i)} for i in range(12)]
    assert sample_rows(rows, 5, None, 2) == sample_rows(rows[::-1], 5, None, 2)
    assert len(sample_rows(rows, None, 0.2, 2)) == 3
    for size, fraction in [(0, None), (1, 0.5), (None, 2)]:
        with pytest.raises(ValueError):
            sample_rows(rows, size, fraction, 2)


def test_pipeline_keeps_record_and_question_counts_and_freezes_cache(bundle):
    config, output = bundle
    analysis = run_analysis(config, output)
    assert len(analysis["records"]) == 4
    assert all(len(record["texts"]) == 2 for record in analysis["records"])
    assert all(record["text_novelty"] is not None for record in analysis["records"])
    assert analysis["provenance"]["sampling"]["population_count"] == 6
    assert (output / "embeddings.npz").is_file()
    with np.load(output / "embeddings.npz", allow_pickle=False) as arrays:
        assert len(arrays["record_ids"]) == arrays["signals"].shape[0] == 4
        assert len(arrays["text_ids"]) == len(arrays["text_record_ids"]) == arrays["texts"].shape[0] == 8
    verify_analysis(analysis)
    assert run_analysis(config, output, reuse=True) == json.loads((output / "analysis.json").read_text())
    with pytest.raises(FileExistsError):
        run_analysis(config, output)
    config["sampling"]["seed"] += 1
    with pytest.raises(FileExistsError, match="different analysis"):
        run_analysis(config, output, reuse=True)


def test_no_text_preserves_tuples_without_requiring_text_model(bundle, monkeypatch):
    config, output = bundle
    config["text"]["enabled"] = False
    monkeypatch.setattr(embeddings, "embed_texts", lambda *_: pytest.fail("Text model must not load"))
    result = run_analysis(config, output)
    assert all(len(row["texts"]) == 2 and row["text_novelty"] is None for row in result["records"])


def test_plan_validates_membership_counts_and_integrity(bundle):
    config, output = bundle
    analysis = run_analysis(config, output)
    selection = {
        "record_ids": [analysis["records"][0]["record_id"]],
        "seed": 7,
        "requests": [{"method": "editing", "count": 50}, {"method": "diffusion", "count": 20}],
    }
    plan = make_plan(analysis, selection)
    assert plan["total_requested"] == 70 and len(plan["targets"]["text_ids"]) == 2
    assert "intent_only" in plan["status"]
    assert plan["requests"][1]["generator"] == "diffusets@1"
    assert plan["requests"][1]["modality"] == "ecg"
    assert "modality" not in plan["requests"][0]
    verify_plan(plan)
    with pytest.raises(ValueError, match="checksum"):
        verify_plan({**plan, "seed": 100})
    with pytest.raises(ValueError, match="schema"):
        verify_plan({**plan, "schema_version": "syntide.generation-plan@2"})
    write_frozen(output / "plan.json", plan)
    write_frozen(output / "plan.json", plan)
    with pytest.raises(FileExistsError):
        write_frozen(output / "plan.json", {**plan, "seed": 100})
    with pytest.raises(ValueError, match="Unknown recording"):
        make_plan(analysis, {**selection, "record_ids": ["not-in-sample"]})
    with pytest.raises(ValueError, match="positive integer"):
        make_plan(analysis, {**selection, "requests": [{"method": "editing", "count": -1}]})
    with pytest.raises(ValueError, match="belong"):
        make_plan(analysis, {**selection, "text_ids": ["unknown"]})
    analysis["records"][0]["labels"].append("tampered")
    with pytest.raises(ValueError, match="checksum"):
        make_plan(analysis, selection)


@pytest.mark.parametrize(
    "backend",
    [{}, {"generator": "research.ppg@2.0.0", "modality": " ppg "}],
)
def test_portable_browser_plan_matches_python_even_with_exponents(bundle, backend):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is optional; needed to compare browser/Python plan serialization")
    config, output = bundle
    analysis = run_analysis(config, output)
    selection = {
        "record_ids": [analysis["records"][0]["record_id"]],
        "seed": 3,
        "criteria": {"label": "é ⚡"},
        "requests": [
            {
                "method": "diffusion",
                "count": 8,
                "parameters": {"epsilon": 1e-7, "threshold": 1e-6, "scale": -0.0},
                **backend,
            }
        ],
    }
    template = Path(embeddings.__file__).with_name("viewer.html").read_text()
    export_handler = re.search(r"""\$\(["']export-plan["']\)\.onclick""", template)
    functions = template[template.index("function canonical(value)") : export_handler.start()]
    script = (
        "const p=JSON.parse(require('fs').readFileSync(0,'utf8'));"
        "const A=p.analysis,cleanAnalysis=A;"
        + functions
        + "\nportablePlan(p.selection).then(x=>process.stdout.write(JSON.stringify(x)));"
    )
    result = subprocess.run(
        [node, "-e", script],
        input=json.dumps({"analysis": analysis, "selection": selection}),
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == make_plan(analysis, selection)
    assert content_id({"x": 1.0}) == content_id({"x": 1})
    assert content_id({"x": -0.0}) == content_id({"x": 0})
    with pytest.raises(ValueError, match="JavaScript-safe"):
        content_id({"x": 1e20})


def test_malformed_generation_requests_are_rejected_in_python_and_browser():
    analysis = seal_analysis({"records": [{"record_id": "r1", "cluster": 0}], "provenance": {"source": {}}})
    invalid = [
        None,
        [],
        {"method": [], "count": 1},
        {"method": "unknown", "count": 1},
        {"method": "diffusion", "count": True},
        {"method": "diffusion", "count": 0},
        {"method": "diffusion", "count": 1, "parameters": []},
        {"method": "diffusion", "count": 1, "generator": None},
        {"method": "diffusion", "count": 1, "generator": ""},
        {"method": "diffusion", "count": 1, "generator": "diffusets"},
        {"method": "diffusion", "count": 1, "generator": "diffusets@"},
        {"method": "diffusion", "count": 1, "generator": " diffusets@1"},
        {"method": "diffusion", "count": 1, "generator": "diffusets@1\n"},
        {"method": "diffusion", "count": 1, "modality": " "},
        {"method": "diffusion", "count": 1, "modality": None},
        {"method": "editing", "count": 1, "generator": "diffusets@1"},
        {"method": "augmentation", "count": 1, "modality": "ecg"},
    ]
    for request in invalid:
        with pytest.raises(ValueError):
            make_plan(analysis, {"record_ids": ["r1"], "requests": [request]})
    node = shutil.which("node")
    if node:
        template = Path(embeddings.__file__).with_name("viewer.html").read_text()
        start = template.index("function normalizeRequest(request)")
        function = template[start : template.index("async function portablePlan(s)")]
        script = (
            "const requests=JSON.parse(require('fs').readFileSync(0,'utf8'));"
            + function
            + "process.stdout.write(JSON.stringify(requests.map(r=>{"
            "try {normalizeRequest(r);return false;} catch {return true;}})));"
        )
        result = subprocess.run(
            [node, "-e", script], input=json.dumps(invalid), text=True, capture_output=True, check=True
        )
        assert json.loads(result.stdout) == [True] * len(invalid)


def test_old_plan_schema_remains_verifiable_without_new_diffusion_fields():
    body = {
        "schema_version": "syntide.generation-plan@1",
        "requests": [{"method": "diffusion", "generator": None, "count": 10}],
    }
    verify_plan({**body, "plan_id": content_id(body)})


def test_viewer_embedded_json_and_local_server_plan(bundle):
    config, output = bundle
    analysis = run_analysis(config, output)
    server = make_server(output, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(url) as response:
            assert b"Decide what to generate" in response.read()
        selection = {
            "record_ids": [analysis["records"][0]["record_id"]],
            "requests": [{"method": "augmentation", "count": 5}],
        }
        request = urllib.request.Request(
            url + "/api/plan", json.dumps(selection).encode(), {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["total_requested"] == 5
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url + "/../catalog.jsonl")
        assert error.value.code == 404
        for headers in ({"Host": "untrusted.example"}, {"Origin": "https://untrusted.example"}):
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(url + "/api/analysis", headers=headers))
            assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_dataset_text_cannot_end_embedded_json_script(tmp_path):
    malicious_text = "</script><script>alert('dataset')</script>"
    output = tmp_path / "index.html"
    write_viewer({"text": malicious_text}, output)
    assert malicious_text not in output.read_text()
    assert "\\u003c/script>" in output.read_text()
