#!/usr/bin/env python3
"""
Collect the `summary` metrics from every llm_benchmark.py JSON result in this
directory into a single CSV, one row per metric and one column per file.

Usage:
    python3 summarize_results.py [results_dir] [--output metrics.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METRICS = [
    "mean_ttft_seconds",
    "median_ttft_seconds",
    "mean_tokens_per_second",
    "median_tokens_per_second",
    "mean_total_seconds",
    "mean_avg_rss_mb",
    "max_peak_rss_mb"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the benchmark JSON files. Default: this directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Default: <results_dir>/metrics_summary.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir
    output_path = args.output or results_dir / "metrics_summary.csv"

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in: {results_dir}")
        return 1

    summaries: dict[str, dict[str, float]] = {}
    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping {path.name}: {exc}")
            continue

        summary = data.get("summary")
        if not isinstance(summary, dict):
            print(f"Skipping {path.name}: no 'summary' object found")
            continue

        summaries[path.name] = summary

    if not summaries:
        print("No valid benchmark result files found.")
        return 1

    filenames = list(summaries.keys())

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", *filenames])
        for metric in METRICS:
            row = [metric]
            for filename in filenames:
                value = summaries[filename].get(metric, "")
                row.append(value)
            writer.writerow(row)

    print(f"Wrote {len(METRICS)} metrics x {len(filenames)} files -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
