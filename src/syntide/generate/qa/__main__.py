"""``python -m syntide.generate.qa``: build QA corpora and suites, measure coverage, allocate, plan trials.

    build        --config conf/qa/<corpus>.yaml --output DIR [--plan PLAN] [--parent DIR] [--max-records N]
    coverage     DIR
    gap          --suite DIR --run RUN_DIR --protected DIR [DIR ...] --output GAP.json
    allocate     --gap GAP.json --budget N --parent DIR --output PLAN.json [--seed S]
    plan-trials  --protocol conf/qa/protocol_<id>.yaml [--runs-dir runs] [--arms A ...] [--seeds 0 1]

A corpus config has a ``source`` block (registry, manifest, manifest_sha256, role, questions; or
``kind: external`` with catalog, dataset, label_mapping for an evaluation-only suite) and a ``build``
block (``syntide.generate.qa.build.BuildConfig``). ``build`` is restartable: rerun the same command after
an interruption and it resumes after the last committed unit. Nothing is published until every unit is
done, and a published corpus is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from omegaconf import OmegaConf

from syntide.generate.qa.build import BuildConfig


def _build_config(path: Path) -> tuple[dict, BuildConfig]:
    raw = OmegaConf.load(path)
    build = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(BuildConfig), raw.get("build", {})))
    return OmegaConf.to_container(raw.get("source", {}), resolve=True), build


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m syntide.generate.qa", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--config", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--plan", type=Path)
    b.add_argument("--parent", type=Path, help="published corpus the plan was made from")
    b.add_argument("--max-records", type=int)
    b.add_argument("--tracker", type=Path, help="conf/tracker/<backend>.yaml for build counters")
    c = commands.add_parser("coverage")
    c.add_argument("corpus", type=Path)
    g = commands.add_parser("gap")
    g.add_argument("--suite", type=Path, required=True, help="development suite (suite_role: development)")
    g.add_argument("--outputs", type=Path, required=True, help="parquet with task_id and output columns")
    g.add_argument(
        "--protected",
        type=Path,
        nargs="*",
        default=[],
        help="selection/test suites whose patients are off limits",
    )
    g.add_argument("--output", type=Path, required=True)
    a = commands.add_parser("allocate")
    a.add_argument("--gap", type=Path, required=True)
    a.add_argument("--budget", type=int, required=True)
    a.add_argument("--parent", type=Path, required=True)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("plan-trials")
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--arms", nargs="*")
    p.add_argument("--seeds", nargs="*", type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        _run(args)
    except (ValueError, FileExistsError, PermissionError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


def _run(args: argparse.Namespace) -> None:
    if args.command == "build":
        from syntide.generate.qa.build import build_corpus
        from syntide.generate.qa.sources import open_source

        source_cfg, cfg = _build_config(args.config)
        if args.max_records is not None:
            cfg.max_records = args.max_records
        if source_cfg.get("kind") == "external":
            from syntide.generate.qa.sources import open_external_source

            source = open_external_source(
                source_cfg["catalog"], source_cfg["dataset"], label_mapping=source_cfg.get("label_mapping")
            )
        else:
            source = open_source(
                source_cfg.get("registry", "datasets/timenet-registry"),
                source_cfg["manifest"],
                source_cfg["role"],
                purpose=cfg.purpose,
                questions=source_cfg.get("questions", True),
                manifest_sha256=source_cfg.get("manifest_sha256"),
            )
        plan = json.loads(args.plan.read_text()) if args.plan else None
        parent = None
        if args.parent:
            from syntide.generate.qa.corpus import verify_corpus_files

            parent = verify_corpus_files(args.parent)["corpus_id"]
        tracker = None
        if args.tracker:
            from syntide.eval.tracking import TrackerConfig, make_tracker

            tracker = make_tracker(
                TrackerConfig.from_yaml(args.tracker),
                args.output.with_name(args.output.name + ".dashboard"),
                cfg.name,
                "qa-corpus",
                {"name": cfg.name},
            )
        info = build_corpus(cfg, source, args.output, plan=plan, parent_corpus_id=parent, tracker=tracker)
        print(f"Published {info['corpus_id']} at {args.output.resolve()}")
        print(json.dumps(info["counts"], indent=1))
    elif args.command == "coverage":
        from syntide.generate.qa.allocation import coverage
        from syntide.generate.qa.corpus import load_corpus

        corpus = load_corpus(args.corpus)
        print(
            json.dumps(
                {"corpus_id": corpus.corpus_id, "coverage": coverage(corpus.tasks, corpus.evidence)}, indent=1
            )
        )
    elif args.command == "gap":
        import pyarrow.parquet as pq

        from syntide.eval import qa_metrics
        from syntide.generate.qa.allocation import gap_report
        from syntide.generate.qa.corpus import load_corpus

        suite = load_corpus(args.suite)
        table = pq.read_table(args.outputs).to_pylist()
        by_id = {t.task_id: t for t in suite.tasks}
        rows = qa_metrics.score_outputs([by_id[r["task_id"]] for r in table], [r["output"] for r in table])
        protected = {
            s
            for path in args.protected
            for ref in load_corpus(path).inputs.values()
            for s in ref.root_subject_uids
        }
        report = gap_report(
            suite.info,
            suite.tasks,
            suite.evidence,
            {r["task_id"]: r["correct"] for r in rows},
            protected_subjects=protected,
        )
        args.output.write_text(json.dumps(report, indent=1) + "\n")
        print(f"Gap report {report['gap_report_id']}: {args.output}")
    elif args.command == "allocate":
        from syntide.explore.plans import write_frozen
        from syntide.generate.qa.allocation import make_plan
        from syntide.generate.qa.corpus import verify_corpus_files

        plan = make_plan(
            json.loads(args.gap.read_text()),
            args.budget,
            parent_corpus_id=verify_corpus_files(args.parent)["corpus_id"],
            seed=args.seed,
        )
        write_frozen(args.output, plan)
        print(f"Plan {plan['plan_id']}: {args.output} (builds a new corpus; nothing is executed)")
    else:
        from syntide.qa_trial import QAArmConfig, QAProtocolConfig, plan_qa_trials

        raw = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
        raw["arms"] = {name: QAArmConfig(**arm) for name, arm in raw.get("arms", {}).items()}
        protocol = QAProtocolConfig(**raw)
        written = plan_qa_trials(protocol, args.runs_dir, arms=args.arms, seeds=args.seeds)
        print(f"{len(written)} new QA trials under {args.runs_dir / protocol.id}")


if __name__ == "__main__":
    main()
