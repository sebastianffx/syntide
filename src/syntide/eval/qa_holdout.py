"""Frozen QA evaluation on internal and external held-out suites: freeze, claim once, score, report.

1. **Freeze** (``eval/qa_rules.json``, create-only) after every planned QA trial finished: the rules
   (suites, contrasts, controls, bootstrap), their sha256, ``plan.json``'s sha256, the parser and
   selection metric, every suite's corpus id and ``corpus.json`` sha256 (dataset version, eligibility,
   label mapping and preprocessing are all inside the corpus identity), each run's best-checkpoint
   sha256, and each external suite's declared overlap with the generators' training cohorts.
2. **Verify**, on every route that reads held-out predictions or writes scores: all of the above are
   recomputed and compared. A changed rule, suite, mapping, checkpoint or plan is refused before any
   held-out data is loaded.
3. **Claim** (``qa_test_claim.json``, create-only): the frozen runs and suites are scored once; a
   different set is refused.
4. **Score** each run's best checkpoint with its trainer's predictor on every suite, plus two controls
   on tasks that need waveform evidence: ``context_only`` (all signals zeroed) and ``waveform_swap``
   (each task gets another task's waveforms, same family, different answer). Outputs are saved;
   ``scores`` files are create-only.
5. **Report** from the saved outputs only, re-parsed with the frozen parser: per run, per arm (mean and
   SD over seeds), per slice, the controls' drops, rationale contradictions, and the primary paired
   contrast with a patient-cluster bootstrap interval.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from syntide import __version__
from syntide.data.qa import QATask
from syntide.eval import qa_metrics
from syntide.explore.plans import content_id
from syntide.qa_trial import QAExample, QAPredictor, example, load_qa_spec
from syntide.trial import Checkpoint

log = logging.getLogger(__name__)
RULES_FILE = "eval/qa_rules.json"
CLAIM_FILE = "qa_test_claim.json"
CONTROLS = ("context_only", "waveform_swap")


class QAFreezeError(RuntimeError):
    """The QA evaluation inputs are not frozen, changed after freezing, or the claim does not match."""


@dataclass
class QAEvalRules:
    suites: dict[str, str] = field(default_factory=dict)  # name -> published evaluation corpus path
    primary_suite: str = ""
    primary: list[str] = field(default_factory=list)  # ["arm/supervision", "arm/supervision"]: a - b
    contrasts: list[list[str]] = field(default_factory=list)  # further reported contrasts
    controls: list[str] = field(default_factory=lambda: list(CONTROLS))
    external_overlap: dict[str, str] = field(
        default_factory=dict
    )  # suite -> declared generator-cohort overlap
    n_boot: int = 1000
    ci_level: float = 0.95
    bootstrap_seed: int = 0
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, path: str | Path) -> QAEvalRules:
        return OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.load(path)))

    @property
    def sha256(self) -> str:
        return content_id({k: v for k, v in asdict(self).items() if k != "device"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runs(protocol_dir: Path) -> list[dict[str, Any]]:
    plan = protocol_dir / "plan.json"
    if not plan.exists():
        raise QAFreezeError(f"{plan} does not exist; plan the QA protocol first")
    runs, unfinished = [], []
    for entry in json.loads(plan.read_text(encoding="utf-8"))["trials"]:
        run_dir = protocol_dir / entry["run_dir"]
        result_path = run_dir / "result.json"
        if not result_path.exists():
            unfinished.append(entry["run_dir"])
            continue
        spec = load_qa_spec(run_dir)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete" or not result["trial_id"] == spec.trial_id == entry["trial_id"]:
            raise QAFreezeError(f"{run_dir} does not hold the finished trial {entry['trial_id'][:12]}")
        best = result["provenance"]["checkpoints"]["best"]
        runs.append(
            {
                "run_dir": entry["run_dir"],
                "trial_id": spec.trial_id,
                "key": f"{spec.arm}/{spec.supervision}",
                "seed": spec.replicate_seed,
                "best_path": best["path"],
                "best_step": best["step"],
                "best_sha256": best["sha256"],
                "predictor": result["trainer"]["predictor"],
                "fingerprint": result["trainer"]["fingerprint"],
                "device": result["trainer"]["device"],
            }
        )
    if unfinished:
        raise QAFreezeError(f"{len(unfinished)} planned QA trials are not finished (e.g. {unfinished[:3]})")
    if len({r["fingerprint"] for r in runs}) > 1:
        raise QAFreezeError(
            "the QA runs come from several trainer fingerprints; compared arms must share one"
        )
    return runs


def _state(protocol_dir: Path, rules: QAEvalRules) -> dict[str, Any]:
    from syntide.generate.qa.corpus import verify_corpus_files

    suites = {}
    for name, path in sorted(rules.suites.items()):
        info = verify_corpus_files(path)
        role = info["identity"]["config"].get("suite_role")
        if info["identity"]["purpose"] != "evaluation" or role not in ("test", "external"):
            raise QAFreezeError(
                f"suite {name} is a {role or info['identity']['purpose']} corpus, not a test/external suite"
            )
        suites[name] = {
            "corpus_id": info["corpus_id"],
            "corpus_json_sha256": _sha256(Path(path) / "corpus.json"),
            "suite_role": role,
            "overlap": rules.external_overlap.get(name, "not_declared" if role == "external" else "n/a"),
        }
    if rules.primary_suite not in suites:
        raise QAFreezeError(
            f"primary_suite {rules.primary_suite!r} is not one of the suites {sorted(suites)}"
        )
    runs = _runs(protocol_dir)
    _check_suites_disjoint(protocol_dir, runs, rules)
    for run in runs:
        path = protocol_dir / run["run_dir"] / run["best_path"]
        if not path.is_file() or _sha256(path) != run["best_sha256"]:
            raise QAFreezeError(f"{path} is missing or changed after training")
    keys = {r["key"] for r in runs}
    for contrast in ([rules.primary] if rules.primary else []) + list(rules.contrasts):
        if len(contrast) != 2 or set(contrast) - keys:
            raise QAFreezeError(f"contrast {contrast} names arms that were not run ({sorted(keys)})")
    return {
        "rules": asdict(rules) | {"device": None},
        "rules_sha256": rules.sha256,
        "plan_sha256": _sha256(protocol_dir / "plan.json"),
        "parser": qa_metrics.PARSER,
        "selection_metric": qa_metrics.SELECTION_METRIC,
        "suites": suites,
        "runs": [{k: r[k] for k in ("run_dir", "trial_id", "best_sha256")} for r in runs],
    }


def _check_suites_disjoint(protocol_dir: Path, runs: Sequence[Mapping[str, Any]], rules: QAEvalRules) -> None:
    """No held-out suite shares a patient with any corpus a compared run trained on."""
    from syntide.data.leakage import LeakageError, check_disjoint_subjects
    from syntide.generate.qa.corpus import load_corpus

    paths = {e["path"] for run in runs for e in load_qa_spec(protocol_dir / run["run_dir"]).corpora}
    train = {
        s for p in sorted(paths) for ref in load_corpus(p).inputs.values() for s in ref.root_subject_uids
    }
    for name, path in sorted(rules.suites.items()):
        held = {s for ref in load_corpus(path).inputs.values() for s in ref.root_subject_uids}
        try:
            check_disjoint_subjects({"training corpora": train, f"suite {name}": held})
        except LeakageError as error:
            raise QAFreezeError(str(error)) from error


def freeze(protocol_dir: str | Path, rules: QAEvalRules) -> tuple[Path, bool]:
    protocol_dir = Path(protocol_dir)
    state = _state(protocol_dir, rules)
    path = protocol_dir / RULES_FILE
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if {k: frozen[k] for k in state} != state:
            raise QAFreezeError(
                f"{path} froze other QA evaluation inputs on {frozen['frozen_at']}; frozen stays frozen"
            )
        return path, False
    if (protocol_dir / CLAIM_FILE).exists():
        raise QAFreezeError(
            "the QA held-out suites are already claimed; rules frozen now are not pre-registered"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                **state,
                "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "syntide_version": __version__,
            },
            handle,
            indent=1,
        )
        handle.write("\n")
    return path, True


def verify(protocol_dir: str | Path, rules: QAEvalRules) -> dict[str, Any]:
    """Recompute every frozen input and compare. Called before any held-out read or score write."""
    protocol_dir = Path(protocol_dir)
    path = protocol_dir / RULES_FILE
    if not path.exists():
        raise QAFreezeError(f"{path} does not exist; freeze the QA evaluation before scoring held-out suites")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    state = _state(protocol_dir, rules)
    changed = sorted(k for k in state if frozen.get(k) != state[k])
    if changed:
        raise QAFreezeError(f"QA evaluation inputs changed since they were frozen: {', '.join(changed)}")
    claim = protocol_dir / CLAIM_FILE
    if claim.exists():
        claimed = json.loads(claim.read_text(encoding="utf-8"))
        if datetime.fromisoformat(claimed["claimed_at"]) < datetime.fromisoformat(frozen["frozen_at"]):
            raise QAFreezeError("the QA suites were claimed before the rules were frozen")
    return frozen


def claim(protocol_dir: Path, frozen: Mapping[str, Any]) -> Path:
    path = protocol_dir / CLAIM_FILE
    body = {"runs": frozen["runs"], "suites": frozen["suites"], "rules_sha256": frozen["rules_sha256"]}
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(
                {**body, "claimed_at": datetime.now(UTC).isoformat(timespec="seconds")}, handle, indent=1
            )
            handle.write("\n")
    except FileExistsError:
        claimed = json.loads(path.read_text(encoding="utf-8"))
        if {k: claimed[k] for k in body} != body:
            raise QAFreezeError(
                "the QA suites were claimed for another set of runs or suites; they are scored once"
            ) from None
    return path


def import_predictor(name: str) -> QAPredictor:
    module, _, function = name.partition(":")
    return getattr(importlib.import_module(module), function)


def control_examples(tasks: Sequence[QATask], corpus: Any, control: str, seed: int) -> list[QAExample]:
    """Student examples with the control applied. Only tasks with measured evidence are controlled."""
    examples = []
    by_family: dict[str, list[QATask]] = defaultdict(list)
    for task in tasks:
        by_family[task.family].append(task)
    rng = np.random.default_rng(seed)
    for task in tasks:
        signals = [corpus.signal(ref.waveform_id) for ref in task.inputs]
        if control == "context_only":
            signals = [np.zeros_like(s) for s in signals]
        elif control == "waveform_swap":
            others = [
                t
                for t in by_family[task.family]
                if t.answer_text != task.answer_text and len(t.inputs) == len(task.inputs)
            ]
            if others:
                donor = others[int(rng.integers(len(others)))]
                signals = [corpus.signal(ref.waveform_id) for ref in donor.inputs]
        else:
            raise ValueError(f"unknown control {control!r}")
        examples.append(example(task, None, target_mode=None, signals=signals))
    return examples


def needs_waveform(task: QATask, corpus: Any) -> bool:
    return any(corpus.evidence[e].kind == "measured" for e in task.required_evidence if e in corpus.evidence)


def score(
    protocol_dir: str | Path,
    rules: QAEvalRules,
    *,
    load_predictor: Callable[[str], QAPredictor] = import_predictor,
) -> list[Path]:
    """Freeze check, claim, then score every frozen run on every suite and control (once)."""
    from syntide.generate.qa.corpus import load_corpus

    protocol_dir = Path(protocol_dir)
    frozen = verify(protocol_dir, rules)
    claim(protocol_dir, frozen)
    suites = {name: load_corpus(path) for name, path in rules.suites.items()}
    written = []
    for run in _runs(protocol_dir):
        run_dir = protocol_dir / run["run_dir"]
        predict = load_predictor(run["predictor"])
        checkpoint = Checkpoint(run_dir / run["best_path"], run["best_step"])
        for name, corpus in suites.items():
            variants = {"": None, **{c: c for c in rules.controls if name == rules.primary_suite}}
            for suffix, control in variants.items():
                out = run_dir / f"qa_{name}{'__' + suffix if suffix else ''}.parquet"
                if out.exists():
                    continue
                tasks = (
                    corpus.tasks
                    if control is None
                    else [t for t in corpus.tasks if needs_waveform(t, corpus)]
                )
                if not tasks:
                    continue
                examples = (
                    [example(t, corpus.signal, target_mode=None) for t in tasks]
                    if control is None
                    else control_examples(tasks, corpus, control, rules.bootstrap_seed)
                )
                outputs = predict(checkpoint, examples, rules.device)
                rows = qa_metrics.score_outputs(tasks, outputs)
                table = pa.table({k: pa.array([row[k] for row in rows]) for k in rows[0]})
                partial = out.with_name(out.name + ".partial")
                pq.write_table(table, partial, compression="zstd")
                partial.replace(out)
                written.append(out)
    return written


def report(protocol_dir: str | Path, rules: QAEvalRules, *, split: str = "test") -> dict[str, Any]:
    """Results from saved outputs only, re-parsed with the frozen parser. ``split='val'`` reads selection."""
    from syntide.generate.qa.corpus import load_corpus

    protocol_dir = Path(protocol_dir)
    if split == "test":
        frozen = verify(protocol_dir, rules)
        if not (protocol_dir / CLAIM_FILE).exists():
            raise QAFreezeError("the QA suites are not scored yet; run the score step first")
        suites = {name: load_corpus(path) for name, path in rules.suites.items()}
    runs = _runs(protocol_dir) if split == "test" else _finished(protocol_dir)
    results: dict[str, Any] = {
        "protocol_dir": str(protocol_dir),
        "split": split,
        "parser": qa_metrics.PARSER,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "suites": {},
    }
    if split == "val":
        spec = load_qa_spec(protocol_dir / runs[0]["run_dir"])
        suites = {"selection": load_corpus(spec.selection["path"])}
    for name, corpus in suites.items():
        tasks = {t.task_id: t for t in corpus.tasks}
        per_run: dict[str, list[dict[str, Any]]] = {}
        suite: dict[str, Any] = {
            "corpus_id": corpus.corpus_id,
            "runs": {},
            "arms": {},
            "controls": {},
            "contrasts": [],
        }
        if split == "test":
            suite["overlap"] = frozen["suites"][name]["overlap"]
            suite["generator_cohorts"] = _training_cohorts(protocol_dir, runs)
        for run in runs:
            rows = _rows(protocol_dir / run["run_dir"], name, split, tasks)
            per_run[run["run_dir"]] = rows
            summary = qa_metrics.summarize(rows, tasks)
            summary["ci"] = qa_metrics.cluster_bootstrap(
                rows, n_boot=rules.n_boot, seed=rules.bootstrap_seed, level=rules.ci_level
            )
            summary["by_slice"] = _slices(rows)
            summary["rationale_contradiction_rate"] = float(
                np.mean(
                    [
                        bool(
                            qa_metrics.rationale_contradictions(
                                r["output"], tasks[r["task_id"]], corpus.evidence
                            )
                        )
                        for r in rows
                    ]
                )
            )
            suite["runs"][run["run_dir"]] = summary
            if split == "test" and name == rules.primary_suite:
                for control in rules.controls:
                    path = protocol_dir / run["run_dir"] / f"qa_{name}__{control}.parquet"
                    if path.exists():
                        ctl = _reparse(pq.read_table(path).to_pylist(), tasks)
                        same = [r for r in rows if r["task_id"] in {c["task_id"] for c in ctl}]
                        suite["controls"].setdefault(run["run_dir"], {})[control] = {
                            "n": len(ctl),
                            "accuracy_real": float(np.mean([r["correct"] for r in same])),
                            "accuracy_control": float(np.mean([r["correct"] for r in ctl])),
                        }
        by_key: dict[str, list[str]] = defaultdict(list)
        for run in runs:
            by_key[run["key"]].append(run["run_dir"])
        for key, run_dirs in sorted(by_key.items()):
            values = [suite["runs"][d]["accuracy"] for d in run_dirs]
            suite["arms"][key] = {
                "seeds": len(values),
                "accuracy_mean": float(np.mean(values)),
                "accuracy_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
        for a, b in ([rules.primary] if rules.primary else []) + list(rules.contrasts):
            if a in by_key and b in by_key:
                suite["contrasts"].append(
                    {
                        "a": a,
                        "b": b,
                        "primary": [a, b] == list(rules.primary),
                        **qa_metrics.paired_difference(
                            _mean_rows(by_key[a], per_run),
                            _mean_rows(by_key[b], per_run),
                            n_boot=rules.n_boot,
                            seed=rules.bootstrap_seed,
                            level=rules.ci_level,
                        ),
                    }
                )
        results["suites"][name] = suite
    out_dir = protocol_dir / "eval" / "qa" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=1, default=float) + "\n", encoding="utf-8"
    )
    (out_dir / "results.md").write_text(_markdown(results), encoding="utf-8")
    return results


def _finished(protocol_dir: Path) -> list[dict[str, Any]]:
    plan = json.loads((protocol_dir / "plan.json").read_text(encoding="utf-8"))
    runs = []
    for entry in plan["trials"]:
        if (protocol_dir / entry["run_dir"] / "result.json").exists():
            spec = load_qa_spec(protocol_dir / entry["run_dir"])
            runs.append(
                {
                    "run_dir": entry["run_dir"],
                    "key": f"{spec.arm}/{spec.supervision}",
                    "seed": spec.replicate_seed,
                }
            )
    return runs


def _rows(run_dir: Path, suite: str, split: str, tasks: Mapping[str, QATask]) -> list[dict[str, Any]]:
    if split == "val":
        table = [
            r
            for r in pq.read_table(run_dir / "predictions_val.parquet").to_pylist()
            if r["checkpoint"] == "best"
        ]
    else:
        table = pq.read_table(run_dir / f"qa_{suite}.parquet").to_pylist()
    rows = _reparse(table, tasks)
    if {r["task_id"] for r in rows} != set(tasks):
        raise QAFreezeError(f"{run_dir}: the saved outputs do not cover the {suite} suite")
    return rows


def _reparse(table: Sequence[Mapping[str, Any]], tasks: Mapping[str, QATask]) -> list[dict[str, Any]]:
    """Never trust a stored correctness flag: parse the raw outputs again."""
    chosen = [tasks[r["task_id"]] for r in table]
    return qa_metrics.score_outputs(chosen, [r["output"] for r in table])


def _slices(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for key in ("family", "composition", "answerability", "origin", "context_reveals_answer"):
        groups: dict[str, list] = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row["correct"])
        out[key] = {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in sorted(groups.items())}
    return out


def _mean_rows(run_dirs: Sequence[str], per_run: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Per task, the mean correctness over an arm's seeds (patients keep their cluster key)."""
    blocks = [{r["task_id"]: r for r in per_run[d]} for d in run_dirs]
    return [
        {
            "task_id": tid,
            "patients": row["patients"],
            "correct": float(np.mean([b[tid]["correct"] for b in blocks])),
        }
        for tid, row in blocks[0].items()
    ]


def _training_cohorts(protocol_dir: Path, runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Generator training cohorts behind any generated waveform the compared arms trained on."""
    from syntide.generate.qa.corpus import load_corpus

    seen, out = set(), []
    for run in runs:
        for entry in load_qa_spec(protocol_dir / run["run_dir"]).corpora:
            if entry["corpus_id"] in seen:
                continue
            seen.add(entry["corpus_id"])
            corpus = load_corpus(entry["path"])
            cohorts = {
                json.dumps(ref.lineage.get("generator_cohort"), sort_keys=True)
                for ref in corpus.inputs.values()
                if ref.origin == "generated"
            }
            for cohort in sorted(cohorts):
                out.append({"corpus_id": entry["corpus_id"], **json.loads(cohort)})
    return out


def _markdown(results: Mapping[str, Any]) -> str:
    lines = [
        f"# QA results ({results['split']})",
        "",
        f"Parser `{results['parser']}`. Every number is recomputed from saved outputs.",
        "",
    ]
    for name, suite in results["suites"].items():
        lines += [f"## {name} (`{suite['corpus_id']}`)", ""]
        if suite.get("overlap") not in (None, "n/a"):
            lines += [f"Generator-cohort overlap with this suite: **{suite['overlap']}**.", ""]
        lines += ["| Arm | Seeds | Accuracy (mean) | SD |", "|---|---:|---:|---:|"]
        for key, arm in suite["arms"].items():
            sd = "-" if arm["accuracy_sd"] is None else f"{arm['accuracy_sd']:.4f}"
            lines.append(f"| {key} | {arm['seeds']} | {arm['accuracy_mean']:.4f} | {sd} |")
        for contrast in suite["contrasts"]:
            tag = " (primary)" if contrast["primary"] else ""
            lines.append(
                f"\n{contrast['a']} - {contrast['b']}{tag}: {contrast['estimate']:+.4f} "
                f"[{contrast['low']:+.4f}, {contrast['high']:+.4f}], {contrast['clusters']} patient clusters"
            )
        for run_dir, controls in suite["controls"].items():
            for control, values in controls.items():
                lines.append(
                    f"\n{run_dir} {control}: {values['accuracy_real']:.4f} -> "
                    f"{values['accuracy_control']:.4f} "
                    f"on {values['n']} waveform-evidence tasks"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m syntide.eval.qa_holdout",
        description="Freeze, score once and report QA held-out suites.",
    )
    parser.add_argument("mode", choices=["freeze", "score", "report", "report-val"])
    parser.add_argument("protocol_dir")
    parser.add_argument(
        "--rules", required=True, help="QA evaluation rules YAML, e.g. conf/qa/eval_ecgqa_cot_qa50.yaml"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    rules = QAEvalRules.from_yaml(args.rules)
    try:
        if args.mode == "freeze":
            path, written = freeze(args.protocol_dir, rules)
            log.info("%s %s", "froze" if written else "already frozen:", path)
        elif args.mode == "score":
            for path in score(args.protocol_dir, rules):
                log.info("wrote %s", path)
        else:
            out = report(args.protocol_dir, rules, split="val" if args.mode == "report-val" else "test")
            log.info("wrote results for %s", ", ".join(out["suites"]))
    except QAFreezeError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
