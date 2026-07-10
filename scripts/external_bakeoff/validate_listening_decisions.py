#!/usr/bin/env python3
"""Validate filled human listening decisions for the external bakeoff."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DECISIONS = (
    Path("outputs")
    / "external_bakeoff"
    / "listening"
    / "external_bakeoff_listening_decisions.csv"
)
PASSING_DECISIONS = {"pass", "caveat"}
FAILING_DECISION = "fail"
VALID_DECISIONS = PASSING_DECISIONS | {FAILING_DECISION}
REQUIRED_COLUMNS = {
    "machine_id",
    "input_key",
    "impl",
    "status",
    "wav_path",
    "human_decision",
    "notes",
}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        return list(reader)


def _row_label(row: dict[str, str]) -> str:
    return "/".join(
        [
            row.get("machine_id", ""),
            row.get("input_key", ""),
            row.get("impl", ""),
        ]
    )


def validate_rows(
    rows: list[dict[str, str]],
    *,
    allow_failures: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    error_rows = [row for row in rows if row.get("status") != "ok"]
    decisions = Counter((row.get("human_decision") or "").strip().lower() for row in ok_rows)

    for row in ok_rows:
        label = _row_label(row)
        decision = (row.get("human_decision") or "").strip().lower()
        if not decision:
            errors.append(f"{label}: missing human_decision")
            continue
        if decision not in VALID_DECISIONS:
            errors.append(
                f"{label}: invalid human_decision {decision!r}; "
                f"expected one of {sorted(VALID_DECISIONS)}"
            )
            continue
        if decision == FAILING_DECISION and not allow_failures:
            errors.append(f"{label}: human_decision=fail")
        if decision == "caveat" and not (row.get("notes") or "").strip():
            errors.append(f"{label}: human_decision=caveat requires notes")

    summary = {
        "rows": len(rows),
        "ok_rows": len(ok_rows),
        "error_rows": len(error_rows),
        "decision_counts": dict(sorted(decisions.items())),
        "valid": not errors,
    }
    return summary, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Treat explicit human_decision=fail rows as documented failures instead of a validation error.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        rows = _load_rows(args.decisions)
        summary, errors = validate_rows(rows, allow_failures=args.allow_failures)
    except Exception as exc:
        if args.json:
            print(json.dumps({"valid": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({**summary, "errors": errors[:50]}, indent=2, sort_keys=True))
    else:
        print(
            "listening_decisions "
            f"rows={summary['rows']} ok={summary['ok_rows']} "
            f"error={summary['error_rows']} decisions={summary['decision_counts']}"
        )
        for error in errors[:50]:
            print(f"ERROR: {error}", file=sys.stderr)
        if len(errors) > 50:
            print(f"ERROR: ... {len(errors) - 50} more", file=sys.stderr)

    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
