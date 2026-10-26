"""Verify the archived classification experiments and summarize saved scores.

This helper uses only the Python standard library. It never trains a model or
reads patient waveforms. The table summarizes saved metrics rather than scoring
predictions again.
"""

import argparse
import collections
import hashlib
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent


def read_json(path):
    """Read a UTF-8 file containing a JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root=ROOT):
    """Check every packaged file and each recorded historical source hash.

    Args:
        root: Directory containing the archive manifest.

    Returns:
        Counts of verified files, recorded source hashes, and training trials.

    Raises:
        ValueError: A file is missing, changed, unexpected, or inconsistent with
            a saved experiment plan or historical source hash.
    """
    root = pathlib.Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    entries = manifest["files"]
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    # Input manifests inside runtime/ are included; only this manifest omits
    # its own hash to avoid a circular definition.
    actual.update(
        str(path.relative_to(root))
        for path in (root / "runtime").rglob("manifest.json")
    )
    if actual != set(entries):
        raise ValueError(
            f"File inventory differs: missing={sorted(set(entries) - actual)}, "
            f"unexpected={sorted(actual - set(entries))}"
        )
    for relative, expected in entries.items():
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"File is outside the archive: {relative}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"File content differs: {relative}")

    sources = read_json(root / "sources.json")
    for relative, expected in sources["sources"].items():
        if any(
            entries[relative][field] != expected[field]
            for field in ("bytes", "sha256")
        ):
            raise ValueError(f"Source copy differs: {relative}")
    for pin in sources["historical_code_pins"]:
        if entries[pin["path"]]["sha256"] != pin["expected_sha256"]:
            raise ValueError(f"Historical source differs: {pin['path']}")

    trial_count = 0
    for plan_path in (root / "runtime/runs").glob("*/plan.json"):
        plan = read_json(plan_path)
        for entry in plan["trials"]:
            trial_dir = plan_path.parent / entry["run_dir"]
            trial = read_json(trial_dir / "trial.json")
            result = read_json(trial_dir / "result.json")
            if not (
                trial["trial_id"] == result["trial_id"] == entry["trial_id"]
                and result["status"] == "complete"
            ):
                raise ValueError(
                    f"Trial identity or status differs: {trial_dir}"
                )
            trial_count += 1
    return {
        "status": "passed",
        "files": len(entries),
        "historical_code_pins": len(sources["historical_code_pins"]),
        "completed_trials": trial_count,
        "scope": "Packaged files and saved metrics; external inputs excluded.",
    }


def table(root=ROOT):
    """Summarize the saved validation and test scores for each training group.

    Args:
        root: Directory containing the verified reproduction archive.

    Returns:
        A Markdown table of mean area under the receiver operating
        characteristic curve, averaged across diagnostic classes and then seeds.
        Missing test scores are shown as an em dash.
    """
    root = pathlib.Path(root)
    verify(root)
    groups = collections.defaultdict(list)
    for path in sorted((root / "runtime/runs").glob("*/*/*/seed*/result.json")):
        result = read_json(path)
        key = (result["protocol"], result["regime"], result["arm"])
        score_path = path.with_name("test_scores.json")
        score = read_json(score_path) if score_path.exists() else {}
        groups[key].append((result, score))
    lines = [
        "| Experiment | Training group | Arm key | Seeds | Validation | Test |",
        "|---|---|---|---:|---:|---:|",
    ]
    for (protocol, regime, arm), rows in sorted(groups.items()):
        validation = statistics.mean(row[0]["best_val_auroc"] for row in rows)
        scores = [row[1]["test/auroc_overall"] for row in rows if row[1]]
        test = (
            f"{statistics.mean(scores):.6f}"
            if len(scores) == len(rows)
            else "—"
        )
        lines.append(
            f"| {protocol} | {regime} | {arm} | {len(rows)} | "
            f"{validation:.6f} | {test} |"
        )
    return "\n".join(lines)


def main():
    """Run the requested offline integrity check or saved-score summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "table"))
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(), indent=2))
    else:
        print(table())


if __name__ == "__main__":
    main()
