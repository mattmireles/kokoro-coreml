#!/usr/bin/env python3
"""Summarize external bakeoff result JSON files as markdown tables."""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external_bakeoff.schema import RUNTIME_BUCKETS, load_json  # noqa: E402


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="*", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/external_bakeoff"))
    args = parser.parse_args()

    files = args.results or sorted(args.results_dir.glob("results_*.json"))
    rows = []
    for path in files:
        payload = load_json(path)
        for record in payload.get("records", []):
            median_s = _median([float(x) for x in record.get("warm_wall_times_s", [])])
            rows.append((record["machine_id"], record["impl"], record["input_key"], median_s))

    print("| Machine | Impl | " + " | ".join(RUNTIME_BUCKETS) + " |")
    print("| --- | --- | " + " | ".join("---:" for _ in RUNTIME_BUCKETS) + " |")
    grouped: dict[tuple[str, str], dict[str, float | None]] = {}
    for machine, impl, key, median_s in rows:
        grouped.setdefault((machine, impl), {})[key] = median_s
    for (machine, impl), values in sorted(grouped.items()):
        cells = []
        for key in RUNTIME_BUCKETS:
            value = values.get(key)
            cells.append("missing" if value is None else f"{value * 1000:.1f} ms")
        print(f"| {machine} | {impl} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
