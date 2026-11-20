"""Compare previously recorded CodeQL results without running a model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from beear_artifacts import read_json

ARMS = ("pws", "composite")
CONDITIONS = ("00", "10", "01", "11")


def measure(report, arm, condition):
    row = report[arm]["conditions"][condition]
    total, analyzed, hits = (row[key] for key in ("n", "analyzed", "target_hits"))
    if not 0 <= hits <= analyzed <= total or total == 0:
        raise ValueError(f"Invalid counts for {arm} / {condition}")
    rate, coverage = hits / total, analyzed / total
    if not math.isclose(rate, row["target_rate_all_prompts"]) or not math.isclose(coverage, row["coverage"]):
        raise ValueError(f"Counts do not match reported rates for {arm} / {condition}")
    return total, rate, coverage


def compare(before_dir, after_dir, seed):
    before, after = (read_json(folder / "metrics.json") for folder in (before_dir, after_dir))
    if any(before[key] != after[key] for key in ("contract", "test_sha256")):
        raise ValueError("Experiment identity or test inputs differ")
    left_scan, right_scan = before["scan"], after["scan"]
    if (left_scan["query_sha256"] != right_scan["query_sha256"]
            or left_scan["codeql_version"]["version"] != right_scan["codeql_version"]["version"]):
        raise ValueError("CodeQL query or version differs")
    for arm in ARMS:
        name = f"{arm}_seed{seed}"
        left, right = (
            read_json(folder.parent / name / "inference_manifest.json")
            for folder in (before_dir, after_dir)
        )
        if any(record.get("test_sha256") != before["test_sha256"] or record.get("status") != "complete"
               for record in (left, right)):
            raise ValueError(f"Inference manifest is incomplete or identifies different test inputs: {arm}")
        if right["training"].get("original_adapter_sha256") != left["training"]["final_adapter_sha256"]:
            raise ValueError(f"The defended adapter is not derived from the selected baseline: {arm}")

    lines = [
        "# CWE-22 CodeQL results before and after BEEAR",
        "",
        "Rates use all prompts in the condition. Coverage is shown for each run.",
        "",
        "| Arm | Condition | Prompts | Before | After | Change (pp) | Coverage before | Coverage after |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        for condition in CONDITIONS:
            total_before, b_rate, b_coverage = measure(before, arm, condition)
            total_after, a_rate, a_coverage = measure(after, arm, condition)
            if total_before != total_after:
                raise ValueError(f"Different prompt counts for {arm} / {condition}")
            lines.append(
                f"| {arm} | {condition} | {total_before} | {b_rate:.2%} | {a_rate:.2%} | "
                f"{100 * (a_rate - b_rate):+.2f} | {b_coverage:.2%} | {a_coverage:.2%} |"
            )
    lines.extend(("", "CodeQL findings are static-analysis results.", ""))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True, help="Archived comparison directory before defense")
    parser.add_argument("--after", type=Path, required=True, help="Archived comparison directory after defense")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Write an additional Markdown report to a new file")
    args = parser.parse_args(argv)
    report = compare(args.before, args.after, args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(report)
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
