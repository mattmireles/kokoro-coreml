#!/usr/bin/env python3
"""Run Stage 0 expert-block read measurements for the MoE prefetch plan."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.moe_prefetch.schema import load_json, machine_info, utc_now, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default="outputs/moe_prefetch/stage0/model_inventory.json")
    parser.add_argument("--thresholds", default="outputs/moe_prefetch/stage0/thresholds.json")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/moe_prefetch/stage0"))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--queue-depth", type=int, default=1)
    parser.add_argument("--block-count", type=int, default=4)
    parser.add_argument("--skip-fs-usage", action="store_true")
    parser.add_argument(
        "--fs-usage-sudo-mode",
        choices=("noninteractive", "interactive"),
        default="noninteractive",
        help=(
            "Use sudo -n by default so automation never hangs. Use "
            "'interactive' only from a human terminal that can answer the "
            "sudo password prompt for root-only fs_usage capture."
        ),
    )
    parser.add_argument("--fs-usage-timeout-s", type=int, default=20)
    parser.add_argument("--capture-powermetrics", action="store_true")
    parser.add_argument("--powermetrics-sample-rate-ms", type=int, default=500)
    return parser.parse_args()


def _compile_benchmark(output_dir: Path) -> Path:
    binary = output_dir / "bin" / "direct_read_bench"
    binary.parent.mkdir(parents=True, exist_ok=True)
    source = Path("scripts/moe_prefetch/direct_read_bench.c")
    subprocess.run(
        ["cc", "-O2", "-Wall", "-Wextra", "-o", str(binary), str(source), "-lpthread"],
        check=True,
    )
    return binary


def _ensure_random_file(path: Path, size: int) -> None:
    if path.exists() and path.stat().st_size == size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = 1024 * 1024
    remaining = size
    with path.open("wb") as f:
        while remaining > 0:
            chunk = os.urandom(min(chunk_size, remaining))
            f.write(chunk)
            remaining -= len(chunk)


def fs_usage_command(*, pid: int, timeout_s: int, sudo_mode: str) -> list[str]:
    """Build the root-only `fs_usage -f diskio` command for a benchmark PID.

    Called by `_run_cell` after launching `direct_read_bench`. The
    noninteractive default is deliberately safe for automation; interactive
    mode exists so a human terminal can provide the sudo password and produce
    the required Stage 0 proof without changing the measurement contract.
    """
    sudo_cmd = ["sudo"] if sudo_mode == "interactive" else ["sudo", "-n"]
    return sudo_cmd + [
        "fs_usage",
        "-w",
        "-f",
        "diskio",
        "-t",
        str(timeout_s),
        str(pid),
    ]


def _run_cell(
    *,
    binary: Path,
    data_file: Path,
    output_dir: Path,
    block_size: int,
    iterations: int,
    queue_depth: int,
    pattern: str,
    capture_fs_usage: bool,
    fs_usage_sudo_mode: str,
    fs_usage_timeout_s: int,
) -> dict[str, Any]:
    command = [
        str(binary),
        "--file",
        str(data_file),
        "--block-size",
        str(block_size),
        "--iterations",
        str(iterations),
        "--pattern",
        pattern,
        "--queue-depth",
        str(queue_depth),
        "--start-delay-ms",
        "750",
    ]
    bench = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    fs_usage_path = output_dir / f"fs_usage_{pattern}_qd{queue_depth}.txt"
    fs_usage_error = ""
    if capture_fs_usage:
        fs_cmd = fs_usage_command(
            pid=bench.pid,
            timeout_s=fs_usage_timeout_s,
            sudo_mode=fs_usage_sudo_mode,
        )
        try:
            with fs_usage_path.open("w") as f:
                subprocess.run(fs_cmd, stdout=f, stderr=subprocess.PIPE, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            fs_usage_error = exc.stderr.strip()
        except FileNotFoundError as exc:
            fs_usage_error = str(exc)
    stdout, stderr = bench.communicate()
    cell: dict[str, Any] = {
        "pattern": pattern,
        "command": command,
        "returncode": bench.returncode,
        "stderr": stderr,
        "fs_usage_path": str(fs_usage_path) if fs_usage_path.exists() else "",
        "fs_usage_error": fs_usage_error,
    }
    if stdout.strip():
        cell["measurement"] = json.loads(stdout)
    else:
        cell["measurement"] = {}
    return cell


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = load_json(Path(args.inventory))
    thresholds = load_json(Path(args.thresholds))
    expert_bytes = int(inventory["expert_bytes"])
    file_size = expert_bytes * args.block_count
    data_file = output_dir / "expert_blocks_random.bin"
    binary = _compile_benchmark(output_dir)
    _ensure_random_file(data_file, file_size)

    powermetrics_path = output_dir / "powermetrics_stage0.plist"
    powermetrics_error = ""
    powermetrics_process = None
    if args.capture_powermetrics:
        powermetrics_process = subprocess.Popen(
            [
                "sudo",
                "-n",
                "powermetrics",
                "--samplers",
                "disk,cpu_power,gpu_power,ane_power,thermal",
                "--sample-rate",
                str(args.powermetrics_sample_rate_ms),
                "--format",
                "plist",
                "--output-file",
                str(powermetrics_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(args.powermetrics_sample_rate_ms / 1000.0)

    cells = []
    try:
        for pattern in ("sequential", "random"):
            cells.append(
                _run_cell(
                    binary=binary,
                    data_file=data_file,
                    output_dir=output_dir,
                    block_size=expert_bytes,
                    iterations=args.iterations,
                    queue_depth=args.queue_depth,
                    pattern=pattern,
                    capture_fs_usage=not args.skip_fs_usage,
                    fs_usage_sudo_mode=args.fs_usage_sudo_mode,
                    fs_usage_timeout_s=args.fs_usage_timeout_s,
                )
            )
    finally:
        if powermetrics_process is not None:
            powermetrics_process.terminate()
            _stdout, stderr = powermetrics_process.communicate(timeout=5)
            powermetrics_error = stderr.strip()

    payload = {
        "created_at": utc_now(),
        "machine": machine_info(None),
        "inventory": inventory,
        "thresholds": thresholds,
        "config": {
            "iterations": args.iterations,
            "queue_depth": args.queue_depth,
            "fs_usage_sudo_mode": args.fs_usage_sudo_mode,
            "block_count": args.block_count,
            "data_file": str(data_file),
            "binary": str(binary),
            "powermetrics_path": str(powermetrics_path) if powermetrics_path.exists() else "",
            "powermetrics_error": powermetrics_error,
        },
        "cells": cells,
    }
    write_json(output_dir / "results.json", payload)
    print(f"wrote {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
