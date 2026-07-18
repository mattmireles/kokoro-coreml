#!/usr/bin/env python3
"""Verify whether the external bakeoff plan can be marked complete."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external_bakeoff.schema import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    RUNTIME_BUCKETS,
    load_json,
    validate_result_payload,
)
from scripts.external_bakeoff.validate_listening_decisions import (  # noqa: E402
    DEFAULT_DECISIONS,
    validate_rows,
)


MAC_MACHINES = ("m2-studio", "m2-air", "irvine-m1")
PRIMARY_IMPLS = (
    "config-f-reference",
    "mlx-audio",
    "soniqo-speech-swift-kokoro",
)
BACKUP_IMPL = "laishere-kokoro-coreml"
IPHONE_MACHINE_ID = "iphone-12-pro"
IPHONE_IMPL = "soniqo-speech-swift-kokoro-ios"


def _result_files(results_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in results_dir.glob("results_*.json")
        if not path.name.endswith("_30s_pure.json")
    )


def _load_records(results_dir: Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[str]]:
    errors: list[str] = []
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in _result_files(results_dir):
        try:
            payload = load_json(path)
            validate_result_payload(payload)
        except Exception as exc:
            errors.append(f"{path}: invalid result payload: {exc}")
            continue
        for record in payload.get("records", []):
            key = (
                str(record.get("machine_id")),
                str(record.get("impl")),
                str(record.get("input_key")),
            )
            records[key] = record
    return records, errors


def _check_mac_primary(records: dict[tuple[str, str, str], dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for machine in MAC_MACHINES:
        for impl in PRIMARY_IMPLS:
            for bucket in RUNTIME_BUCKETS:
                record = records.get((machine, impl, bucket))
                if record is None:
                    errors.append(f"missing result: {machine}/{impl}/{bucket}")
                    continue
                status = record.get("status")
                if status != "ok":
                    if impl == "mlx-audio" and bucket == "3s":
                        continue
                    errors.append(f"{machine}/{impl}/{bucket}: status={status!r}")
                    continue
                warm = record.get("warm_wall_times_s") or []
                if len(warm) < 5:
                    errors.append(f"{machine}/{impl}/{bucket}: expected at least 5 warm calls")
                if record.get("cold_wall_time_s") is None:
                    errors.append(f"{machine}/{impl}/{bucket}: missing cold latency")
    return errors


def _check_backup(records: dict[tuple[str, str, str], dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for machine in MAC_MACHINES:
        for bucket in RUNTIME_BUCKETS:
            record = records.get((machine, BACKUP_IMPL, bucket))
            if record is None:
                errors.append(f"missing backup result: {machine}/{BACKUP_IMPL}/{bucket}")
                continue
            if record.get("status") != "ok":
                errors.append(f"{machine}/{BACKUP_IMPL}/{bucket}: status={record.get('status')!r}")
    return errors


def _check_iphone(records: dict[tuple[str, str, str], dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for bucket in RUNTIME_BUCKETS:
        record = records.get((IPHONE_MACHINE_ID, IPHONE_IMPL, bucket))
        if record is None:
            errors.append(f"missing signed iPhone result: {IPHONE_MACHINE_ID}/{IPHONE_IMPL}/{bucket}")
            continue
        if record.get("status") != "ok":
            errors.append(
                f"{IPHONE_MACHINE_ID}/{IPHONE_IMPL}/{bucket}: status={record.get('status')!r}"
            )
        if len(record.get("warm_wall_times_s") or []) < 5:
            errors.append(f"{IPHONE_MACHINE_ID}/{IPHONE_IMPL}/{bucket}: expected 5 warm calls")
    return errors


def _check_listening(decisions_path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        with decisions_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        return {"valid": False, "error": str(exc)}, [f"{decisions_path}: {exc}"]
    summary, errors = validate_rows(rows)
    return summary, errors


def _check_preflight(results_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    path = results_dir / "ios_runner_preflight_latest.json"
    if not path.exists():
        return None, ["missing iOS preflight evidence"]
    try:
        payload = load_json(path)
    except Exception as exc:
        return None, [f"{path}: {exc}"]
    if payload.get("ok"):
        return payload, []
    blockers = payload.get("blockers") or []
    return payload, [f"iOS preflight not ready: {', '.join(blockers) or 'unknown blocker'}"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    records, result_errors = _load_records(args.results_dir)
    listening_summary, listening_errors = _check_listening(args.decisions)
    preflight, preflight_errors = _check_preflight(args.results_dir)
    errors = (
        result_errors
        + _check_mac_primary(records)
        + _check_backup(records)
        + _check_iphone(records)
        + [f"listening: {error}" for error in listening_errors]
        + preflight_errors
    )
    summary = {
        "valid": not errors,
        "result_record_count": len(records),
        "mac_machines": list(MAC_MACHINES),
        "primary_impls": list(PRIMARY_IMPLS),
        "runtime_buckets": list(RUNTIME_BUCKETS),
        "listening": listening_summary,
        "ios_preflight_ok": bool(preflight and preflight.get("ok")),
        "errors": errors[:100],
        "error_count": len(errors),
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "external_bakeoff_completion "
            f"valid={summary['valid']} records={summary['result_record_count']} "
            f"errors={summary['error_count']}"
        )
        for error in errors[:100]:
            print(f"ERROR: {error}", file=sys.stderr)
        if len(errors) > 100:
            print(f"ERROR: ... {len(errors) - 100} more", file=sys.stderr)
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
