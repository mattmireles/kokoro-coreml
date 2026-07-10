#!/usr/bin/env python3
"""Run the HAR-post rewrite promotion benchmark only on quiet remote Macs.

This is the safe wrapper for lower-end Mac promotion runs. It first applies the
same quiet-host gate as ``check_remote_host_quiet.py``. For each quiet host it
runs Config F with ``--generator-models-dir`` pointing at the rewritten
HAR-post packages; noisy hosts get a durable skip record instead of polluted
timing.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from scripts.external_bakeoff import check_remote_host_quiet


DEFAULT_REPO_PATH = Path("/Users/mm/Documents/GitHub/kokoro-coreml")
DEFAULT_GENERATOR_MODELS_DIR = Path("outputs/export_rewrite_smoke")
DEFAULT_OUTPUT_DIR = Path("outputs/external_bakeoff")
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUT_DIR / "rewrite_promotion_when_quiet_latest.json"
DEFAULT_SUMMARY_MD = DEFAULT_OUTPUT_DIR / "rewrite_promotion_when_quiet_latest.md"


def _run_ssh(target: str, command: str, timeout_s: int) -> subprocess.CompletedProcess[str]:
    """Run a command on a remote host through non-interactive SSH."""

    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={min(timeout_s, 30)}",
            target,
            command,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s + 10,
    )


def _run_scp(
    target: str,
    remote_path: Path,
    local_path: Path,
    timeout_s: int,
    *,
    recursive: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Copy one remote evidence artifact back to the local checkout."""

    local_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={min(timeout_s, 30)}",
    ]
    if recursive:
        command.append("-r")
    command.extend([f"{target}:{remote_path}", str(local_path)])
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s + 10,
    )


def _shell_join(command: list[str]) -> str:
    """Return a shell-safe command line."""

    return " ".join(shlex.quote(part) for part in command)


def _result_relative_path(machine_id: str) -> Path:
    """Return the Config F rewrite promotion result path for one machine."""

    return DEFAULT_OUTPUT_DIR / f"results_config_f_reference_{machine_id}_rewrite_ups_as_conv.json"


def _spotcheck_relative_dir(machine_id: str) -> Path:
    """Return the Config F rewrite promotion spotcheck WAV directory."""

    return DEFAULT_OUTPUT_DIR / "spotcheck_wavs" / f"config_f_reference_{machine_id}_rewrite_ups_as_conv"


def _remote_benchmark_command(
    *,
    repo_path: Path,
    machine_id: str,
    generator_models_dir: Path,
    input_keys: list[str],
    iterations: int,
    preflight_runs: int,
    compute_units: str,
) -> str:
    """Build the remote Config F rewrite-promotion command."""

    output = _result_relative_path(machine_id)
    command = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/external_bakeoff/run_config_f_reference.py",
        "--machine-id",
        f"{machine_id}_rewrite_ups_as_conv",
        "--output",
        str(output),
        "--compute-units",
        compute_units,
        "--preflight-runs",
        str(preflight_runs),
        "--iterations",
        str(iterations),
        "--generator-models-dir",
        str(generator_models_dir),
    ]
    for key in input_keys:
        command.extend(["--input-key", key])
    return f"cd {shlex.quote(str(repo_path))} && {_shell_join(command)}"


def _quiet_payload_for_host(args: argparse.Namespace, host: str) -> dict[str, Any]:
    """Run the quiet gate for one host string."""

    return check_remote_host_quiet.build_payload(
        argparse.Namespace(
            host=[host],
            max_load_1=args.max_load_1,
            max_noisy_cpu_pct=args.max_noisy_cpu_pct,
            max_swap_used_mb=args.max_swap_used_mb,
            min_memory_free_pct=args.min_memory_free_pct,
            allow_battery=args.allow_battery,
            timeout=args.quiet_timeout,
        )
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run quiet-gated rewrite promotion attempts."""

    host_values = args.host or list(check_remote_host_quiet.DEFAULT_HOSTS)
    input_keys = args.input_key or ["3s", "7s", "10s", "15s", "30s"]
    rows: list[dict[str, Any]] = []
    for host_value in host_values:
        machine_id, target = check_remote_host_quiet._parse_host(host_value)
        quiet_payload = _quiet_payload_for_host(args, host_value)
        quiet_row = quiet_payload["rows"][0]
        row: dict[str, Any] = {
            "machine_id": machine_id,
            "target": target,
            "quiet": bool(quiet_row.get("quiet")),
            "quiet_payload": quiet_row,
        }
        if not quiet_row.get("quiet"):
            row.update(
                {
                    "status": "skipped_noisy_host",
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                }
            )
            rows.append(row)
            continue

        command = _remote_benchmark_command(
            repo_path=args.repo_path,
            machine_id=machine_id,
            generator_models_dir=args.generator_models_dir,
            input_keys=input_keys,
            iterations=args.iterations,
            preflight_runs=args.preflight_runs,
            compute_units=args.compute_units,
        )
        row["command"] = command
        if args.dry_run:
            row.update(
                {
                    "status": "dry_run",
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                }
            )
            rows.append(row)
            continue

        result = _run_ssh(target, command, args.run_timeout)
        row.update(
            {
                "status": "ok" if result.returncode == 0 else "remote_error",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        if result.returncode != 0:
            rows.append(row)
            continue

        result_relative_path = _result_relative_path(machine_id)
        remote_result_path = args.repo_path / result_relative_path
        local_result_path = Path(result_relative_path)
        fetch = _run_scp(target, remote_result_path, local_result_path, args.fetch_timeout)
        row.update(
            {
                "remote_result_path": str(remote_result_path),
                "local_result_path": str(local_result_path),
                "fetch_returncode": fetch.returncode,
                "fetch_stdout": fetch.stdout,
                "fetch_stderr": fetch.stderr,
            }
        )
        if fetch.returncode != 0:
            row["status"] = "fetch_error"
            rows.append(row)
            continue

        spotcheck_relative_dir = _spotcheck_relative_dir(machine_id)
        remote_spotcheck_dir = args.repo_path / spotcheck_relative_dir
        local_spotcheck_dir = Path(spotcheck_relative_dir)
        spotcheck_fetch = _run_scp(
            target,
            remote_spotcheck_dir,
            local_spotcheck_dir.parent,
            args.fetch_timeout,
            recursive=True,
        )
        row.update(
            {
                "remote_spotcheck_dir": str(remote_spotcheck_dir),
                "local_spotcheck_dir": str(local_spotcheck_dir),
                "spotcheck_fetch_returncode": spotcheck_fetch.returncode,
                "spotcheck_fetch_stdout": spotcheck_fetch.stdout,
                "spotcheck_fetch_stderr": spotcheck_fetch.stderr,
            }
        )
        if spotcheck_fetch.returncode != 0:
            row["spotcheck_fetch_status"] = "error"
        else:
            row["spotcheck_fetch_status"] = "ok"
        rows.append(row)

    payload = {
        "repo_path": str(args.repo_path),
        "generator_models_dir": str(args.generator_models_dir),
        "input_keys": input_keys,
        "iterations": args.iterations,
        "preflight_runs": args.preflight_runs,
        "compute_units": args.compute_units,
        "dry_run": bool(args.dry_run),
        "rows": rows,
        "summary": {
            "hosts": len(rows),
            "quiet_hosts": sum(1 for row in rows if row["quiet"]),
            "ran_hosts": sum(1 for row in rows if row["status"] == "ok"),
            "skipped_noisy_hosts": sum(1 for row in rows if row["status"] == "skipped_noisy_host"),
            "failed_hosts": sum(1 for row in rows if row["status"] == "remote_error"),
            "fetch_failed_hosts": sum(1 for row in rows if row["status"] == "fetch_error"),
            "spotcheck_fetch_failed_hosts": sum(
                1 for row in rows if row.get("spotcheck_fetch_status") == "error"
            ),
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output.write_text(render_markdown(payload))
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a human-readable promotion summary."""

    lines = [
        "# Rewrite Promotion When Quiet",
        "",
        "This report runs, or skips, the HAR-post upsample rewrite promotion command",
        "based on the quiet-host gate. Skipped noisy hosts are not timing evidence.",
        "",
        f"- Generator models dir: `{payload['generator_models_dir']}`.",
        f"- Input keys: `{', '.join(payload['input_keys'])}`.",
        f"- Iterations: `{payload['iterations']}`.",
        f"- Preflight runs: `{payload['preflight_runs']}`.",
        f"- Compute units: `{payload['compute_units']}`.",
        f"- Dry run: `{payload['dry_run']}`.",
        "",
        "| Machine | Quiet | Status | Blockers |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        blockers = "; ".join(row.get("quiet_payload", {}).get("blockers") or []) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['machine_id']}`",
                    "`yes`" if row["quiet"] else "`no`",
                    f"`{row['status']}`",
                    blockers,
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", action="append", default=None, help="MACHINE=ssh-target")
    parser.add_argument("--repo-path", type=Path, default=DEFAULT_REPO_PATH)
    parser.add_argument("--generator-models-dir", type=Path, default=DEFAULT_GENERATOR_MODELS_DIR)
    parser.add_argument("--input-key", action="append", default=None)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--preflight-runs", type=int, default=3)
    parser.add_argument("--compute-units", default="staged")
    parser.add_argument("--max-load-1", type=float, default=1.5)
    parser.add_argument("--max-noisy-cpu-pct", type=float, default=5.0)
    parser.add_argument("--max-swap-used-mb", type=float, default=0.0)
    parser.add_argument("--min-memory-free-pct", type=int, default=10)
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("--quiet-timeout", type=int, default=5)
    parser.add_argument("--run-timeout", type=int, default=7200)
    parser.add_argument("--fetch-timeout", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_SUMMARY_JSON)
    args = parser.parse_args()

    payload = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                **payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
