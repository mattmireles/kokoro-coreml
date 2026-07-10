#!/usr/bin/env python3
"""Launch, poll, and ingest the Config F physical-iPhone runner when available."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs" / "external_bakeoff"
DEFAULT_DEVICE_ID = "F383FC46-FD64-5346-AEC6-59E3E2F8C9CA"
DEFAULT_APP_BUNDLE_ID = "com.kokoro.externalbakeoff.ConfigFIOSRunnerManual"
DEFAULT_REMOTE_RESULT = "Documents/config_f_ios_result.json"
DEFAULT_PULLED_JSON = DEFAULT_RESULTS_DIR / "config_f_ios_result_latest.json"
DEFAULT_INGESTED_JSON = DEFAULT_RESULTS_DIR / "results_config_f_reference_ios_iphone-12-pro.json"
DEFAULT_STATUS_JSON = DEFAULT_RESULTS_DIR / "config_f_ios_run_latest.json"


def _now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _run(command: list[str], *, timeout_s: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
    )


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(part for part in (result.stdout, result.stderr) if part)


def _is_locked_launch_error(text: str) -> bool:
    lowered = text.lower()
    return (
        "locked" in lowered
        and "unable to launch" in lowered
        and ("requestdenied" in lowered or "request denied" in lowered)
    )


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _base_status(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "app_bundle_id": args.app_bundle_id,
        "checked_at_local": _now_local(),
        "device_id": args.device_id,
        "ingested_json": str(args.ingested_json),
        "launch_ok": False,
        "ok": False,
        "pulled_json": str(args.pulled_json),
        "remote_result": args.remote_result,
        "runner": "Config F manual direct-swiftc iOS runner",
    }


def _device_lock_state(device_id: str) -> dict[str, Any]:
    result = _run(
        ["xcrun", "devicectl", "device", "info", "lockState", "--device", device_id],
        timeout_s=30,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "output": _combined_output(result).strip(),
    }


def _launch_app(args: argparse.Namespace) -> dict[str, Any]:
    result = _run(
        [
            "xcrun",
            "devicectl",
            "device",
            "process",
            "launch",
            "--device",
            args.device_id,
            args.app_bundle_id,
        ],
        timeout_s=args.launch_timeout_s,
    )
    output = _combined_output(result).strip()
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "output": output,
        "locked": _is_locked_launch_error(output),
    }


def _copy_result(args: argparse.Namespace) -> subprocess.CompletedProcess[str]:
    args.pulled_json.parent.mkdir(parents=True, exist_ok=True)
    return _run(
        [
            "xcrun",
            "devicectl",
            "device",
            "copy",
            "from",
            "--device",
            args.device_id,
            "--domain-type",
            "appDataContainer",
            "--domain-identifier",
            args.app_bundle_id,
            "--source",
            args.remote_result,
            "--destination",
            str(args.pulled_json),
        ],
        timeout_s=args.copy_timeout_s,
    )


def _poll_result(args: argparse.Namespace) -> dict[str, Any]:
    deadline = time.monotonic() + args.poll_timeout_s
    attempts = 0
    last_output = ""
    while True:
        attempts += 1
        result = _copy_result(args)
        last_output = _combined_output(result).strip()
        if result.returncode == 0 and args.pulled_json.exists():
            return {
                "ok": True,
                "attempts": attempts,
                "output": last_output,
            }
        if time.monotonic() >= deadline:
            return {
                "ok": False,
                "attempts": attempts,
                "output": last_output,
                "timed_out": True,
            }
        time.sleep(args.poll_interval_s)


def _ingest(args: argparse.Namespace) -> dict[str, Any]:
    result = _run(
        [
            "python",
            "scripts/external_bakeoff/ingest_ios_runner_result.py",
            "--input",
            str(args.pulled_json),
            "--machine-id",
            "iphone-12-pro",
            "--device-model",
            "iPhone 12 Pro (iPhone13,3)",
            "--version",
            "Config F manual iOS runner; staged compute units; exact duration packages",
            "--output",
            str(args.ingested_json),
        ],
        timeout_s=60,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "output": _combined_output(result).strip(),
    }


def _refresh_frontier() -> dict[str, Any]:
    result = _run(
        [
            "python",
            "scripts/external_bakeoff/summarize_competitive_frontier.py",
            "--output",
            "outputs/external_bakeoff/competitive_frontier.md",
            "--json-output",
            "outputs/external_bakeoff/competitive_frontier.json",
        ],
        timeout_s=60,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "output": _combined_output(result).strip(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    status = _base_status(args)
    status["lock_state"] = _device_lock_state(args.device_id)
    if args.check_only:
        status["ok"] = bool(status["lock_state"]["ok"])
        status["check_only"] = True
        return status

    launch = _launch_app(args)
    status["launch"] = launch
    status["launch_ok"] = launch["ok"]
    if not launch["ok"]:
        status["launch_blocker"] = "device_locked" if launch["locked"] else "launch_failed"
        status["launch_error"] = launch["output"]
        return status

    poll = _poll_result(args)
    status["poll_result"] = poll
    if not poll["ok"]:
        status["launch_blocker"] = "result_not_available"
        return status

    ingest = _ingest(args)
    status["ingest"] = ingest
    if not ingest["ok"]:
        status["launch_blocker"] = "ingest_failed"
        return status

    frontier = _refresh_frontier()
    status["frontier"] = frontier
    status["ok"] = frontier["ok"]
    if not frontier["ok"]:
        status["launch_blocker"] = "frontier_refresh_failed"
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    parser.add_argument("--app-bundle-id", default=DEFAULT_APP_BUNDLE_ID)
    parser.add_argument("--remote-result", default=DEFAULT_REMOTE_RESULT)
    parser.add_argument("--pulled-json", type=Path, default=DEFAULT_PULLED_JSON)
    parser.add_argument("--ingested-json", type=Path, default=DEFAULT_INGESTED_JSON)
    parser.add_argument("--status-json", type=Path, default=DEFAULT_STATUS_JSON)
    parser.add_argument("--poll-timeout-s", type=float, default=7200.0)
    parser.add_argument("--poll-interval-s", type=float, default=30.0)
    parser.add_argument("--launch-timeout-s", type=int, default=60)
    parser.add_argument("--copy-timeout-s", type=int, default=60)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    status = run(args)
    _write_status(args.status_json, status)
    print(
        json.dumps(
            {
                "launch_blocker": status.get("launch_blocker"),
                "launch_ok": status.get("launch_ok"),
                "ok": status.get("ok"),
                "status_json": str(args.status_json),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if status.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
