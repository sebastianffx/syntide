"""Generate complete signals from one explicit request in a frozen explorer plan."""

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list", action="store_true", help="List registered full-signal generator IDs")
    parser.add_argument(
        "--request-index", type=int, help="Zero-based request index; required for mixed plans"
    )
    args = parser.parse_args(argv)
    if args.list:
        if any(value is not None for value in (args.plan, args.config, args.output, args.request_index)):
            parser.error("--list cannot be combined with execution options")
        from syntide.generate.full.registry import available_generators

        print("\n".join(available_generators()))
        return
    if any(value is None for value in (args.plan, args.config, args.output)):
        parser.error("execution requires --plan, --config and --output")
    from syntide.generate.full.runner import run_generation

    try:
        result = run_generation(
            json.loads(args.plan.read_text(encoding="utf-8")),
            json.loads(args.config.read_text(encoding="utf-8")),
            args.output,
            request_index=args.request_index,
        )
    except (ValueError, TypeError, OSError, ImportError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(f"Generated {result['count']} signals: {args.output.resolve()}")
    print(f"Run {result['run_id']}; domain and label review remains pending")


if __name__ == "__main__":
    main()
