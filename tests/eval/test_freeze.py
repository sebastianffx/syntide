"""Evaluation rules must be frozen before test scoring begins."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import syntide.eval.__main__  # noqa: F401  (registers the evaluate schema)
from syntide.eval.analysis import AnalysisConfig, RulesError, check_rules
from syntide.eval.freeze import (
    CLAIM_FILE,
    RULES_FILE,
    FreezeError,
    freeze_rules,
    verify_frozen,
)

CONF = Path(__file__).resolve().parents[2] / "conf"


def _config(protocol: str, *overrides: str):
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        cfg = compose(
            "evaluate", overrides=[f"protocol={protocol}", *overrides]
        )
    return OmegaConf.to_object(cfg)


@pytest.mark.parametrize("protocol", ["ptbxl_artifact_50", "ptbxl_artifact_v2"])
def test_each_protocol_ships_rules_that_name_only_arms_it_runs(protocol):
    config = _config(
        protocol
    )  # protocol=<name> also selects conf/analysis/<name>.yaml
    check_rules(config.analysis, config.protocol)
    assert config.analysis.regime in config.protocol.regimes


@pytest.mark.parametrize(
    "override, message",
    [
        ("analysis.regime=scarcity25", "not a regime"),
        ("analysis.primary=[A3,A9]", "does not run arm A9"),
        (
            "analysis.diagnostics=[[A1@full100,A0]]",
            "regime full100 does not run arm A1",
        ),
        ("analysis.diagnostics=[[A1,A0,C1]]", "exactly two arms"),
    ],
)
def test_rules_naming_an_arm_a_regime_does_not_run_are_refused(
    override, message
):
    config = _config("ptbxl_artifact_50", override)
    with pytest.raises(RulesError, match=message):
        check_rules(config.analysis, config.protocol)


@pytest.fixture
def protocol_dir(tmp_path):
    (tmp_path / "plan.json").write_text(
        json.dumps({"protocol": "p", "trials": []})
    )
    return tmp_path


RULES = AnalysisConfig(regime="scarcity50")


def _claim(protocol_dir, claimed_at):
    (protocol_dir / CLAIM_FILE).write_text(
        json.dumps({"claimed_at": claimed_at, "trials": []})
    )


def test_rules_are_frozen_once_and_never_replaced(protocol_dir):
    path, written = freeze_rules(protocol_dir, RULES)
    assert written and path == protocol_dir / RULES_FILE
    assert json.loads(path.read_text())["rules_sha256"] == RULES.sha256
    assert freeze_rules(protocol_dir, RULES) == (
        path,
        False,
    )  # the same rules again: nothing written
    with pytest.raises(FreezeError, match="changed practical_target"):
        freeze_rules(protocol_dir, replace(RULES, practical_target=0.01))


def test_rules_cannot_be_frozen_after_the_test_fold_is_claimed(protocol_dir):
    _claim(protocol_dir, "2026-10-22T15:00:00+00:00")
    with pytest.raises(FreezeError, match="already claimed"):
        freeze_rules(protocol_dir, RULES)
    assert not (protocol_dir / RULES_FILE).exists()


def _set_frozen_at(protocol_dir, when):
    path = protocol_dir / RULES_FILE
    path.write_text(
        json.dumps({**json.loads(path.read_text()), "frozen_at": when})
    )


def test_report_requires_rules_frozen_before_test_claim(
    protocol_dir,
):
    with pytest.raises(FreezeError, match="run mode=freeze"):
        verify_frozen(protocol_dir, RULES, claimed=False)
    freeze_rules(protocol_dir, RULES)
    _set_frozen_at(protocol_dir, "2026-10-22T15:00:00+00:00")
    verify_frozen(
        protocol_dir, RULES, claimed=False
    )  # frozen, not claimed yet: holdout may claim now
    with pytest.raises(FreezeError, match="not scored yet"):
        verify_frozen(protocol_dir, RULES, claimed=True)

    _claim(protocol_dir, "2026-10-22T15:30:00+00:00")
    verify_frozen(protocol_dir, RULES, claimed=True)
    with pytest.raises(FreezeError, match="rules changed"):
        verify_frozen(protocol_dir, replace(RULES, n_boot=500), claimed=True)

    _set_frozen_at(
        protocol_dir, "2026-10-22T16:00:00+00:00"
    )  # frozen after the claim
    with pytest.raises(
        FreezeError, match=r"claimed .* before the rules were frozen"
    ):
        verify_frozen(protocol_dir, RULES, claimed=True)


def test_a_changed_plan_invalidates_the_frozen_rules(protocol_dir):
    freeze_rules(protocol_dir, RULES)
    (protocol_dir / "plan.json").write_text(
        json.dumps({"protocol": "p", "trials": [{"run_dir": "x"}]})
    )
    with pytest.raises(FreezeError, match=r"plan\.json changed"):
        verify_frozen(protocol_dir, RULES, claimed=False)


def test_a_rule_added_to_the_schema_later_keeps_earlier_freezes_valid(
    protocol_dir,
):
    freeze_rules(protocol_dir, RULES)
    path = protocol_dir / RULES_FILE
    frozen = json.loads(path.read_text())
    del frozen["rules"][
        "bootstrap_seed"
    ]  # as if frozen before the field existed (it has a default)
    path.write_text(json.dumps(frozen))
    verify_frozen(protocol_dir, RULES, claimed=False)
