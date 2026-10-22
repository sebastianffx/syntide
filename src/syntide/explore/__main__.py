"""``python -m syntide.explore``: discover coverage, inspect signals, export a plan."""

import argparse
import json
import logging
from pathlib import Path

from syntide.explore.plans import make_plan, write_frozen


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser(
        "analyze", help="Embed a reproducible training sample and write an explorer"
    )
    analyze.add_argument("--dataset", choices=["ptbxl", "ecgqa", "jsonl"], default="ptbxl")
    analyze.add_argument("--registry", default="datasets/timenet-registry")
    analyze.add_argument("--dataset-ref")
    analyze.add_argument("--input", type=Path, help="Generic JSONL catalog")
    analyze.add_argument("--manifest", type=Path)
    analyze.add_argument("--role", help="Defaults to depleted with a manifest, otherwise train")
    analyze.add_argument("--scp-statements", type=Path)
    sample = analyze.add_mutually_exclusive_group()
    sample.add_argument("--sample", type=int, help="Recording count (default 512)")
    sample.add_argument("--sample-fraction", type=float)
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--backend", choices=["chronos", "opentslm"], default="chronos")
    analyze.add_argument("--model", help="Backend model ID; defaults to Chronos T5 tiny or OpenTSLM TSQA SP")
    analyze.add_argument("--revision", help="Pin a Hugging Face model revision")
    analyze.add_argument("--checkpoint", type=Path, help="Local OpenTSLM encoder checkpoint")
    analyze.add_argument(
        "--checkpoint-provenance",
        type=Path,
        help="JSON declaring checkpoint training_data and contains_ecgqa=false",
    )
    analyze.add_argument("--target-sample-rate", type=float, help="Explicitly resample mixed-rate signals")
    analyze.add_argument(
        "--context-length", type=int, help="Chunk length (never silently truncate recordings)"
    )
    analyze.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    analyze.add_argument("--batch-size", type=int, default=16)
    analyze.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    analyze.add_argument("--text-revision")
    analyze.add_argument("--text-fields", nargs="+", default=["question", "answer", "text"])
    analyze.add_argument(
        "--no-text", action="store_true", help="Keep tuples visible but skip semantic novelty"
    )
    analyze.add_argument("--projection", choices=["umap", "pca"], default="umap")
    analyze.add_argument("--clusters", type=int, default=8)
    analyze.add_argument("--neighbors", type=int, default=15)
    analyze.add_argument(
        "--waveform-points", type=int, default=256, help="Preview points per channel, max 1000"
    )
    analyze.add_argument("--target-count", type=int, default=0)
    analyze.add_argument("--target-fraction", type=float, default=0)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument(
        "--reuse", action="store_true", help="Reuse this exact frozen analysis if already present"
    )
    analyze.add_argument("--serve", action="store_true", help="Open a local explorer after analysis")
    analyze.add_argument("--port", type=int, default=8765)
    serve = commands.add_parser("serve", help="Serve a saved analysis locally")
    serve.add_argument("directory", type=Path)
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-open", action="store_true")
    plan = commands.add_parser(
        "plan", help="Validate a saved selection and export an immutable generation plan"
    )
    plan.add_argument("directory", type=Path)
    plan.add_argument("--selection", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parser().parse_args(argv)
    try:
        if args.command == "analyze":
            from syntide.explore.pipeline import run_analysis

            config = {
                "source": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in {
                        "dataset": args.dataset,
                        "registry": args.registry,
                        "dataset_ref": args.dataset_ref,
                        "input": args.input,
                        "manifest": args.manifest,
                        "role": args.role,
                        "scp_statements": args.scp_statements,
                    }.items()
                    if v is not None
                },
                "sampling": {
                    "size": args.sample
                    if args.sample is not None
                    else (None if args.sample_fraction is not None else 512),
                    "fraction": args.sample_fraction,
                    "seed": args.seed,
                },
                "embedding": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in {
                        "backend": args.backend,
                        "model": args.model,
                        "revision": args.revision,
                        "checkpoint": args.checkpoint,
                        "device": args.device,
                        "batch_size": args.batch_size,
                        "seed": args.seed,
                        "target_sample_rate": args.target_sample_rate,
                        "context_length": args.context_length,
                        "checkpoint_provenance": json.loads(args.checkpoint_provenance.read_text())
                        if args.checkpoint_provenance
                        else None,
                    }.items()
                    if v is not None
                },
                "text": {
                    "model": args.text_model,
                    "revision": args.text_revision,
                    "device": args.device,
                    "batch_size": args.batch_size,
                    "fields": args.text_fields,
                    "enabled": not args.no_text,
                },
                "analysis": {
                    "projection": args.projection,
                    "n_clusters": args.clusters,
                    "n_neighbors": args.neighbors,
                    "seed": args.seed,
                    "target_count": args.target_count,
                    "target_fraction": args.target_fraction,
                    "waveform_points": args.waveform_points,
                },
            }
            run_analysis(config, args.output, reuse=args.reuse)
            print(f"Explorer: {(args.output / 'index.html').resolve()}")
            if args.serve:
                from syntide.explore.server import serve

                serve(args.output, args.port)
        elif args.command == "serve":
            from syntide.explore.server import serve

            serve(args.directory, args.port, open_browser=not args.no_open)
        else:
            analysis = json.loads((args.directory / "analysis.json").read_text())
            selection = json.loads(args.selection.read_text())
            plan = make_plan(analysis, selection)
            write_frozen(args.output, plan)
            print(f"Plan {plan['plan_id']}: {args.output.resolve()}")
    except (ValueError, FileNotFoundError, FileExistsError, ImportError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
