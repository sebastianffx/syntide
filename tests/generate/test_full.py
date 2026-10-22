"""Full synthesis is modality-neutral, reproducible, explicit and transactional."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from syntide.explore.data import open_source
from syntide.explore.plans import content_id
from syntide.generate.full import registry
from syntide.generate.full.api import GeneratedSignal, GenerationRequest, SignalSpec
from syntide.generate.full.runner import run_generation


class FakeGenerator:
    def __init__(self, config):
        unknown = set(config) - {"scale", "fail_after", "bad_shape", "bad_values", "changing_provenance"}
        if unknown:
            raise ValueError(f"Unknown backend fields: {sorted(unknown)}")
        self.config, self.validated, self.generated = config, [], []

    def validate_request(self, request):
        if request.signal.modality != "ppg":
            raise ValueError("This generator supports ppg only")
        if set(request.conditioning) != {"rate"} or request.conditioning["rate"] <= 0:
            raise ValueError("Conditioning needs a positive rate")
        self.validated.append(request)

    def generate(self, request):
        self.generated.append(request)
        if len(self.generated) > self.config.get("fail_after", float("inf")):
            raise RuntimeError("Inference failed")
        signal = np.random.default_rng(request.seed).normal(
            size=(len(request.signal.channels), request.signal.num_samples)
        )
        signal *= self.config.get("scale", 1)
        if self.config.get("bad_shape"):
            signal = signal[:, :-1]
        if self.config.get("bad_values"):
            signal[0, 0] = np.nan
        return GeneratedSignal(signal, request.signal, {"test_backend": True})

    def provenance(self):
        value = {"model": "test", "weights_sha256": "a" * 64, "device": "cpu"}
        if self.config.get("changing_provenance"):
            value["calls"] = len(self.generated)
        return value


@pytest.fixture
def setup(monkeypatch):
    instances = []

    def factory(config):
        instance = FakeGenerator(config)
        instances.append(instance)
        return instance

    monkeypatch.setattr(registry, "_REGISTERED", {})
    monkeypatch.setattr(registry, "entry_points", lambda **_: [])
    registry.register_generator("test.ppg@1", factory)
    plan = {
        "schema_version": "syntide.generation-plan@1",
        "analysis_id": "analysis",
        "source": {"source_sha256": "source", "role": "train"},
        "seed": 17,
        "targets": {"record_ids": ["r1"], "text_ids": ["q1"], "labels": ["requested-label"]},
        "requests": [
            {
                "method": "diffusion",
                "count": 5,
                "generator": "test.ppg@1",
                "modality": "ppg",
                "parameters": {"scale": 2},
            }
        ],
    }
    config = {
        "generator": "test.ppg@1",
        "signal": {
            "modality": "ppg",
            "channels": ["green", "infrared"],
            "sample_rate": 32,
            "num_samples": 64,
            "units": "normalized",
        },
        "conditions": [
            {"conditioning": {"rate": 60}, "source_record_ids": ["r1"], "source_text_ids": ["q1"]},
            {"conditioning": {"rate": 80}},
        ],
    }
    return plan, config, instances


def seal(plan):
    body = {key: value for key, value in plan.items() if key != "plan_id"}
    return {**body, "plan_id": content_id(body)}


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_full_run_freezes_exact_count_and_reopens_non_ecg_dataset(setup, tmp_path):
    plan, config, instances = setup
    out = tmp_path / "first"
    run = run_generation(seal(plan), config, out)
    assert run["count"] == 5
    assert run["status"] == "complete_pending_domain_and_label_review"
    assert run["unexecuted_request_indices"] == []
    assert run["config"]["scale"] == 2
    assert run["config_id"] == content_id(run["config"])
    assert len(instances[0].validated) == 2
    ledger = read_lines(out / "ledger.jsonl")
    assert [row["condition_index"] for row in ledger] == [0, 1, 0, 1, 0]
    assert [row["conditioning"]["rate"] for row in ledger] == [60, 80, 60, 80, 60]
    assert len({row["seed"] for row in ledger}) == 5
    assert all(row["conditioning_id"] == content_id(row["conditioning"]) for row in ledger)
    source = open_source({"dataset": "jsonl", "input": str(out / "catalog.jsonl")})
    recordings = source.load([row["record_id"] for row in source.rows])
    assert len(recordings) == 5
    assert all(row["signal"].shape == (2, 64) for row in recordings)
    assert all(row["signal"].dtype == np.float32 and row["labels"] == [] for row in recordings)
    assert all(
        row["texts"] == [] and row["metadata"]["qa_status"].startswith("pending") for row in recordings
    )
    assert recordings[0]["metadata"]["requested_labels"] == ["requested-label"]
    repeated = run_generation(seal(plan), config, tmp_path / "second")
    assert repeated == run
    for file in out.rglob("*"):
        if file.is_file():
            assert file.read_bytes() == (tmp_path / "second" / file.relative_to(out)).read_bytes()


def test_mixed_requests_require_explicit_diffusion_selection(setup, tmp_path):
    plan, config, _ = setup
    plan["requests"].insert(0, {"method": "augmentation", "count": 2})
    with pytest.raises(ValueError, match="explicit --request-index"):
        run_generation(seal(plan), config, tmp_path / "out")
    with pytest.raises(ValueError, match="method=diffusion"):
        run_generation(seal(plan), config, tmp_path / "out", request_index=0)
    result = run_generation(seal(plan), config, tmp_path / "out", request_index=1)
    assert result["count"] == 5 and result["unexecuted_request_indices"] == [0]


def test_multiple_diffusion_requests_are_not_silently_merged(setup, tmp_path):
    plan, config, _ = setup
    plan["requests"] *= 2
    with pytest.raises(ValueError, match="explicit --request-index"):
        run_generation(seal(plan), config, tmp_path / "out")
    for index in (-1, 2, True):
        with pytest.raises(ValueError, match="outside"):
            run_generation(seal(plan), config, tmp_path / "out", request_index=index)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c.update(generator="diffusets@1"), "generator conflicts"),
        (lambda c: c.update(scale=3), "scale conflicts"),
        (lambda c: c["signal"].update(modality="eeg"), "modality conflicts"),
        (lambda c: c["signal"].update(misspelled=True), "unknown"),
        (lambda c: c.update(misspelled=True), "Unknown backend fields"),
        (lambda c: c["conditions"][0].update(misspelled=True), "unknown fields"),
        (lambda c: c["conditions"][0].update(source_record_ids=["heldout"]), "subset"),
        (lambda c: c["conditions"][0].update(source_text_ids=["heldout"]), "subset"),
        (lambda c: c["conditions"][0].update(source_record_ids=["r1", "r1"]), "duplicate"),
        (lambda c: c.update(conditions=[]), "at least one"),
        (lambda c: c["conditions"][1].update(conditioning={"rate": -2}), "positive rate"),
    ],
)
def test_invalid_requests_fail_before_inference_or_output(setup, tmp_path, change, message):
    plan, config, instances = setup
    change(config)
    with pytest.raises(ValueError, match=message):
        run_generation(seal(plan), config, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert all(not instance.generated for instance in instances)
    assert not list(tmp_path.glob(".syntide-generation-*"))


def test_tampered_plan_and_existing_output_are_never_replaced(setup, tmp_path):
    plan, config, instances = setup
    frozen = seal(plan)
    frozen["seed"] += 1
    with pytest.raises(ValueError, match="checksum"):
        run_generation(frozen, config, tmp_path / "out")
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep").write_text("keep")
    with pytest.raises(FileExistsError, match="replace"):
        run_generation(seal(plan), config, out)
    assert (out / "keep").read_text() == "keep"
    assert not instances


@pytest.mark.parametrize(
    ("settings", "exception", "message"),
    [
        ({"fail_after": 1}, RuntimeError, "Inference failed"),
        ({"bad_shape": True}, ValueError, "shape"),
        ({"bad_values": True}, ValueError, "finite"),
        ({"changing_provenance": True}, ValueError, "provenance changed"),
    ],
)
def test_failed_generation_leaves_no_partial_bundle(setup, tmp_path, settings, exception, message):
    plan, config, _ = setup
    config.update(settings)
    with pytest.raises(exception, match=message):
        run_generation(seal(plan), config, tmp_path / "out")
    assert list(tmp_path.iterdir()) == []


def test_seed_stream_is_stable_when_count_changes(setup, tmp_path):
    plan, config, _ = setup
    run_generation(seal(plan), config, tmp_path / "long")
    plan["requests"][0]["count"] = 2
    run_generation(seal(plan), config, tmp_path / "short")
    first = read_lines(tmp_path / "long" / "ledger.jsonl")
    second = read_lines(tmp_path / "short" / "ledger.jsonl")
    assert [row["seed"] for row in first[:2]] == [row["seed"] for row in second]
    for index in range(2):
        name = f"signals/{index:08d}.npy"
        assert (tmp_path / "long" / name).read_bytes() == (tmp_path / "short" / name).read_bytes()


def test_signal_contract_rejects_ambiguous_or_invalid_shapes(setup):
    _, config, _ = setup
    for field, value in (
        ("modality", ""),
        ("channels", []),
        ("channels", ["I", "I"]),
        ("channels", "I"),
        ("sample_rate", 0),
        ("sample_rate", True),
        ("num_samples", 2.5),
        ("num_samples", True),
        ("units", ""),
    ):
        with pytest.raises(ValueError):
            SignalSpec.from_dict({**config["signal"], field: value})
    signal = SignalSpec.from_dict(config["signal"])
    assert signal.channels == ("green", "infrared")
    with pytest.raises(ValueError, match="seed"):
        GenerationRequest(2**32, signal, {})
    with pytest.raises(ValueError, match="finite"):
        GenerationRequest(0, signal, {"rate": float("nan")})


def test_registry_loads_only_selected_plugin_and_detects_collisions(monkeypatch):
    loaded = []
    entry = SimpleNamespace(name="plugin.ppg@2", load=lambda: loaded.append(True) or FakeGenerator)
    monkeypatch.setattr(registry, "entry_points", lambda **_: [entry])
    assert "plugin.ppg@2" in registry.available_generators()
    assert not loaded
    assert isinstance(registry.create_generator("plugin.ppg@2", {}), FakeGenerator)
    assert loaded == [True]
    monkeypatch.setattr(registry, "entry_points", lambda **_: [entry, entry])
    with pytest.raises(ValueError, match="Multiple"):
        registry.create_generator("plugin.ppg@2", {})
    for name in ("unversioned", "../generator.py", "broken@"):
        with pytest.raises(ValueError, match="versioned"):
            registry.create_generator(name, {})
    with pytest.raises(ValueError, match="already registered"):
        registry.register_generator("diffusets@1", FakeGenerator)


def test_cli_uses_frozen_files(setup, tmp_path, capsys):
    from syntide.generate.full.__main__ import main

    plan, config, _ = setup
    plan_path, config_path = tmp_path / "plan.json", tmp_path / "config.json"
    plan_path.write_text(json.dumps(seal(plan)))
    config_path.write_text(json.dumps(config))
    main(["--plan", str(plan_path), "--config", str(config_path), "--output", str(tmp_path / "out")])
    assert "Generated 5 signals" in capsys.readouterr().out
    assert (tmp_path / "out" / "run.json").is_file()


def test_cli_lists_generators_without_loading_models(setup, capsys):
    from syntide.generate.full.__main__ import main

    main(["--list"])
    assert capsys.readouterr().out.splitlines() == ["diffusets@1", "test.ppg@1"]
    assert not setup[2]
