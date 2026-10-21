#!/usr/bin/env python3
"""Compare the converged solutions of two CI run trees.

Every ``.info`` file found under the new tree is paired with the file of the
same relative path under the reference tree. The numerical parameters of each
object section (``ns``, ``bh``, ``binary``, ``ns1``, ...) are compared and the
relative differences are written as a markdown table, e.g. for
``include/FUKA_Solvers/<SOLVER>/tests/regression/RESULTS.md``.

usage: compare_ci_runs.py <new_tree> <reference_tree> [--tol 1e-6] [--out FILE]
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

# Sections that only hold controls, settings or provenance, not results.
SKIPPED_SECTIONS = {
    "fields",
    "stages",
    "sequence_controls",
    "sequence_settings",
    "metadata",
    "differential_rotation",
}

log = logging.getLogger("compare_ci_runs")


def parse_info(path: Path) -> dict[str, dict[str, str]]:
    """Return ``{section path: {key: value}}`` of a Boost ``.info`` file.

    Nested sections are joined with ``/``; only key/value leaves are kept.
    """
    sections: dict[str, dict[str, str]] = {}
    stack: list[str] = []
    pending: str | None = None
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if line == "{":
                if pending is None:
                    raise ValueError(f"{path}: '{{' without a section name")
                stack.append(pending)
                pending = None
                continue
            if line == "}":
                stack.pop()
                continue
            key, _, value = line.partition(" ")
            if not value:
                pending = key
                continue
            sections.setdefault("/".join(stack), {})[key] = value.strip()
    return sections


def as_float(value: str) -> float | None:
    """Return ``value`` as a float, or None for non-numerical entries."""
    try:
        return float(value)
    except ValueError:
        return None


def relative_difference(new: float, ref: float) -> float:
    """Relative difference, absolute when the reference vanishes."""
    scale = abs(ref)
    return abs(new - ref) / scale if scale > 0.0 else abs(new - ref)


@dataclass
class Difference:
    """One compared parameter."""

    section: str
    key: str
    new: float
    ref: float

    @property
    def rel(self) -> float:
        return relative_difference(self.new, self.ref)


def compare_files(new: Path, ref: Path) -> list[Difference]:
    """Compare the numerical result parameters of two ``.info`` files."""
    new_sections, ref_sections = parse_info(new), parse_info(ref)
    diffs: list[Difference] = []
    for section, entries in new_sections.items():
        if section.split("/")[0] in SKIPPED_SECTIONS:
            continue
        ref_entries = ref_sections.get(section, {})
        for key, value in entries.items():
            new_val, ref_val = as_float(value), as_float(ref_entries.get(key, ""))
            if new_val is None or ref_val is None:
                continue
            if math.isnan(new_val) and math.isnan(ref_val):
                continue
            diffs.append(Difference(section, key, new_val, ref_val))
    return diffs


def version_of(path: Path) -> str:
    """FUKA version recorded in the metadata of an ``.info`` file."""
    return parse_info(path).get("metadata", {}).get("fuka_version", "unknown")


def write_report(new_root: Path, ref_root: Path, tol: float) -> tuple[str, int]:
    """Return the markdown report and the number of files above ``tol``."""
    lines = [
        f"New tree: `{new_root}`  ",
        f"Reference tree: `{ref_root}`  ",
        f"Tolerance (relative): {tol:g}",
        "",
        "| solution | versions (new / ref) | max rel. diff | parameter | new | ref | status |",
        "|---|---|---|---|---|---|---|",
    ]
    n_fail = 0
    for new in sorted(new_root.rglob("*.info")):
        rel_path = new.relative_to(new_root)
        ref = ref_root / rel_path
        if not ref.exists():
            lines.append(f"| `{rel_path}` | | | | | | no reference |")
            continue
        diffs = compare_files(new, ref)
        versions = f"{version_of(new)} / {version_of(ref)}"
        if not diffs:
            lines.append(f"| `{rel_path}` | {versions} | | | | | nothing to compare |")
            continue
        worst = max(diffs, key=lambda d: d.rel)
        status = "ok" if worst.rel <= tol else "DIFF"
        n_fail += status != "ok"
        lines.append(
            f"| `{rel_path}` | {versions} | {worst.rel:.2e} | "
            f"{worst.section}/{worst.key} | {worst.new:.12g} | {worst.ref:.12g} | "
            f"{status} |"
        )
    return "\n".join(lines) + "\n", n_fail


def main() -> int:
    """Command-line entry point; returns the number of files above tolerance."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("new_tree", type=Path, help="tree with the new solutions")
    parser.add_argument("reference_tree", type=Path, help="tree with the reference solutions")
    parser.add_argument("--tol", type=float, default=1e-6, help="relative tolerance")
    parser.add_argument("--out", type=Path, help="write the markdown report to this file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for root in (args.new_tree, args.reference_tree):
        if not root.is_dir():
            parser.error(f"{root} is not a directory")

    report, n_fail = write_report(args.new_tree, args.reference_tree, args.tol)
    if args.out:
        args.out.write_text(report)
        log.info("report written to %s", args.out)
    else:
        sys.stdout.write(report)
    return n_fail


if __name__ == "__main__":
    sys.exit(min(main(), 255))
