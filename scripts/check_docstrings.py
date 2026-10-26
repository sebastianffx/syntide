"""Check maintained docstrings for unexplained experiment variant codes."""

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "trainers", "scripts", "tests")
EXPERIMENT_CODE = re.compile(
    r"\b(?:A[0-3]|C[12]|E1|G[12]|R[12]|S[1-4]|T[1-6]W?)\b"
)
DOCUMENTED_NODES = (
    ast.Module,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)


def main() -> int:
    """Report experiment codes in docstrings and return a failing exit status.

    Archived experiment files are excluded because their original text forms
    part of their recorded source hashes. Configuration keys and saved trial
    identifiers are also outside the scope of this documentation check.

    Returns:
        Zero when every maintained docstring passes, otherwise one.
    """
    files = [ROOT / "noxfile.py"]
    for directory in SOURCE_DIRS:
        files.extend(sorted((ROOT / directory).rglob("*.py")))
    failures = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, DOCUMENTED_NODES):
                continue
            docstring = ast.get_docstring(node) or ""
            codes = sorted(set(EXPERIMENT_CODE.findall(docstring)))
            if codes:
                line = node.body[0].lineno
                location = f"{path.relative_to(ROOT)}:{line}"
                failures.append(f"{location}: spell out {', '.join(codes)}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Experiment-code docstring check passed for {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
