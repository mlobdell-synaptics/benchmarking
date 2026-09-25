#!/usr/bin/env python3
"""
Collect the per-question `results` (question_number, question, response) from every
llm_benchmark.py JSON result in this directory into a single CSV.

One row per (file, question); columns are: file, question_number, question, response.

Usage:
    python3 collect_responses.py [results_dir] [--output responses.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

COLUMNS = ["question_number", "question", "response"]


def collapse_ws(value: object) -> object:
    """Collapse any run of whitespace (incl. newlines) to a single space."""
    if isinstance(value, str):
        return " ".join(value.split())
    return value


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
        help="CSV output path. Default: <results_dir>/responses.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir
    output_path = args.output or results_dir / "responses.csv"

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in: {results_dir}")
        return 1

    rows: list[list[object]] = []
    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping {path.name}: {exc}")
            continue

        results = data.get("results")
        if not isinstance(results, list):
            print(f"Skipping {path.name}: no 'results' list found")
            continue

        for entry in results:
            if not isinstance(entry, dict):
                continue
            rows.append(
                [
                    path.name,
                    entry.get("question_number", ""),
                    collapse_ws(entry.get("question", "")),
                    collapse_ws(entry.get("response", "")),
                ]
            )

    if not rows:
        print("No valid benchmark result files found.")
        return 1

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", *COLUMNS])
        writer.writerows(rows)

    file_count = len({row[0] for row in rows})
    print(f"Wrote {len(rows)} question rows from {file_count} files -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
