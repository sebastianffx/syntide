"""Every shipped QA config parses into its schema, through the same code the CLI uses."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from syntide.eval.qa_holdout import QAEvalRules
from syntide.generate.qa.__main__ import _build_config
from syntide.generate.qa.build import COMPONENTS, SUITE_ROLES
from syntide.qa_trial import QAArmConfig, QAProtocolConfig

CONF = Path(__file__).parents[2] / "conf" / "qa"
BASELINE_MANIFEST = "6c46d78a53838895c076f66c5a2006753c50dcfbbfed081f19dc7dd4f42726c4"


@pytest.mark.parametrize(
    "path", sorted([*CONF.glob("corpus_*.yaml"), *CONF.glob("suite_*.yaml")]), ids=lambda p: p.stem
)
def test_corpus_and_suite_configs_parse_and_pin_the_baseline_split(path):
    source, build = _build_config(path)
    assert set(build.components) <= set(COMPONENTS)
    if source.get("kind") == "external":
        assert build.purpose == "evaluation" and build.suite_role == "external"
        return
    assert source["manifest_sha256"] == BASELINE_MANIFEST
    if path.stem.startswith("suite_"):
        assert (
            build.purpose == "evaluation"
            and build.suite_role in SUITE_ROLES
            and source["role"] in ("val", "test")
        )
    else:
        assert build.purpose == "training" and source["role"] == "pool"


@pytest.mark.parametrize("path", sorted(CONF.glob("protocol_*.yaml")), ids=lambda p: p.stem)
def test_protocols_parse_and_keep_the_baseline_budget(path):
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    raw["arms"] = {name: QAArmConfig(**arm) for name, arm in raw["arms"].items()}
    protocol = QAProtocolConfig(**raw)
    assert (protocol.batch_size, protocol.max_steps, protocol.val_every) == (16, 15531, 1000)
    assert protocol.decoding["prompt_version"] == "ecgqa-cot-single-joint-media-v1"
    assert len(protocol.seeds) >= 3
    for arm in protocol.arms.values():
        assert set(arm.corpora) <= set(protocol.corpora)


def test_eval_rules_parse_and_name_planned_arms():
    rules = QAEvalRules.from_yaml(CONF / "eval_ecgqa_cot_qa50.yaml")
    raw = OmegaConf.to_container(OmegaConf.load(CONF / "protocol_ecgqa_cot_qa50.yaml"), resolve=True)
    keys = {f"{arm}/{mode}" for arm in raw["arms"] for mode in raw["supervision"]}
    assert {*rules.primary, *(k for pair in rules.contrasts for k in pair)} <= keys
    assert rules.primary_suite in rules.suites and set(rules.controls) == {"context_only", "waveform_swap"}
