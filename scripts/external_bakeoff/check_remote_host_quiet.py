#!/usr/bin/env python3
"""Check whether remote Mac hosts are quiet enough for publishable timing."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/external_bakeoff")
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "remote_host_quiet_latest.md"
DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "remote_host_quiet_latest.json"
DEFAULT_HOSTS = (
    "irvine-m1=mattmireles@irvine-m1.local",
    "m2-air=mattmireles@m2-air.local",
)
NOISY_PROCESS_PATTERNS = (
    "mds",
    "mdworker",
    "mediaanalysisd",
    "mediaanalysisd-access",
    "photoanalysisd",
)


@dataclass(frozen=True)
class ProcessSample:
    """One process row from the remote host."""

    cpu_pct: float
    command: str
    noisy: bool


def _split_sections(output: str) -> dict[str, str]:
    """Split the remote probe output into named sections."""

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in output.splitlines():
        if line.startswith("__") and line.endswith("__"):
            current = line.strip("_").lower()
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return {key: "\n".join(lines).strip() for key, lines in sections.items()}


def _run_ssh(target: str, command: str, timeout_s: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={timeout_s}",
            target,
            command,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s + 5,
    )


def _parse_load(uptime: str) -> tuple[float | None, float | None, float | None]:
    match = re.search(r"load averages?:\s*([0-9.]+)[, ]+([0-9.]+)[, ]+([0-9.]+)", uptime)
    if not match:
        return (None, None, None)
    return tuple(float(match.group(index)) for index in (1, 2, 3))  # type: ignore[return-value]


def _parse_processes(ps_output: str) -> list[ProcessSample]:
    rows: list[ProcessSample] = []
    for line in ps_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            cpu_pct = float(parts[0])
        except ValueError:
            continue
        command = parts[1]
        command_lower = command.lower()
        noisy = any(pattern in command_lower for pattern in NOISY_PROCESS_PATTERNS)
        rows.append(ProcessSample(cpu_pct=cpu_pct, command=command, noisy=noisy))
    return rows


def _parse_swap_used_mb(swapusage: str) -> float | None:
    match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", swapusage)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "K":
        return value / 1024.0
    if unit == "G":
        return value * 1024.0
    return value


def _parse_memory_free_pct(memory_pressure: str) -> int | None:
    match = re.search(r"System-wide memory free percentage:\s*([0-9]+)%", memory_pressure)
    if not match:
        return None
    return int(match.group(1))


def _parse_power_ac(power: str) -> bool | None:
    if not power:
        return None
    lower = power.lower()
    if "ac power" in lower:
        return True
    if "battery power" in lower:
        return False
    return None


def _thermal_ok(thermal: str) -> bool | None:
    if not thermal:
        return None
    lower = thermal.lower()
    thermal_warning = "thermal warning" in lower and "no thermal warning" not in lower
    performance_warning = "performance warning" in lower and "no performance warning" not in lower
    return not (thermal_warning or performance_warning)


def _host_status(
    machine_id: str,
    target: str,
    *,
    max_load_1: float,
    max_noisy_cpu_pct: float,
    max_swap_used_mb: float | None,
    min_memory_free_pct: int | None,
    require_ac_power: bool,
    timeout_s: int,
) -> dict[str, Any]:
    command = "\n".join(
        [
            "printf '__UPTIME__\\n'",
            "uptime",
            "printf '__PS__\\n'",
            "ps -Ao pcpu,comm | sort -nr | head -12",
            "printf '__SWAP__\\n'",
            "sysctl vm.swapusage 2>/dev/null || true",
            "printf '__MEMORY_PRESSURE__\\n'",
            "memory_pressure 2>/dev/null || true",
            "printf '__POWER__\\n'",
            "pmset -g batt 2>/dev/null | head -n 1 || true",
            "printf '__THERMAL__\\n'",
            "pmset -g therm 2>/dev/null || true",
        ]
    )
    result = _run_ssh(target, command, timeout_s)
    if result.returncode != 0:
        return {
            "machine_id": machine_id,
            "target": target,
            "ok": False,
            "quiet": False,
            "error": result.stderr.strip() or result.stdout.strip(),
            "returncode": result.returncode,
        }
    sections = _split_sections(result.stdout)
    if sections:
        uptime = sections.get("uptime", "")
        ps_output = sections.get("ps", "")
    else:
        lines = result.stdout.splitlines()
        uptime = lines[0] if lines else ""
        ps_output = "\n".join(lines[1:])
    swapusage = sections.get("swap", "")
    memory_pressure = sections.get("memory_pressure", "")
    power = sections.get("power", "")
    thermal = sections.get("thermal", "")
    load_1, load_5, load_15 = _parse_load(uptime)
    processes = _parse_processes(ps_output)
    noisy_processes = [row for row in processes if row.noisy and row.cpu_pct >= max_noisy_cpu_pct]
    swap_used_mb = _parse_swap_used_mb(swapusage)
    memory_free_pct = _parse_memory_free_pct(memory_pressure)
    ac_power = _parse_power_ac(power)
    thermals_ok = _thermal_ok(thermal)
    load_ok = load_1 is not None and load_1 <= max_load_1
    noisy_ok = not noisy_processes
    swap_ok = max_swap_used_mb is None or (swap_used_mb is not None and swap_used_mb <= max_swap_used_mb)
    memory_ok = min_memory_free_pct is None or (
        memory_free_pct is not None and memory_free_pct >= min_memory_free_pct
    )
    power_ok = (not require_ac_power) or ac_power is True
    thermal_gate_ok = thermals_ok is not False
    quiet = load_ok and noisy_ok and swap_ok and memory_ok and power_ok and thermal_gate_ok
    blockers: list[str] = []
    if not load_ok:
        blockers.append(f"load1 {load_1!r} exceeds {max_load_1:.2f}")
    for row in noisy_processes:
        blockers.append(f"{row.command} at {row.cpu_pct:.1f}% CPU")
    if not swap_ok:
        blockers.append(f"swap used {swap_used_mb!r} MB exceeds {max_swap_used_mb:.1f} MB")
    if not memory_ok:
        blockers.append(
            f"memory free {memory_free_pct!r}% below {min_memory_free_pct}% threshold"
        )
    if not power_ok:
        blockers.append("not drawing from AC power")
    if not thermal_gate_ok:
        blockers.append("thermal or performance warning recorded")
    return {
        "machine_id": machine_id,
        "target": target,
        "ok": True,
        "quiet": quiet,
        "blockers": blockers,
        "thresholds": {
            "max_load_1": max_load_1,
            "max_noisy_cpu_pct": max_noisy_cpu_pct,
            "max_swap_used_mb": max_swap_used_mb,
            "min_memory_free_pct": min_memory_free_pct,
            "require_ac_power": require_ac_power,
            "noisy_process_patterns": list(NOISY_PROCESS_PATTERNS),
        },
        "uptime": uptime,
        "load": {
            "one_minute": load_1,
            "five_minutes": load_5,
            "fifteen_minutes": load_15,
        },
        "swap": {
            "raw": swapusage,
            "used_mb": swap_used_mb,
        },
        "memory_pressure": {
            "free_pct": memory_free_pct,
            "raw": memory_pressure,
        },
        "power": {
            "ac_power": ac_power,
            "raw": power,
        },
        "thermal": {
            "ok": thermals_ok,
            "raw": thermal,
        },
        "processes": [asdict(row) for row in processes],
    }


def _parse_host(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("host must be MACHINE=ssh-target")
    machine_id, target = value.split("=", 1)
    if not machine_id or not target:
        raise argparse.ArgumentTypeError("host must be MACHINE=ssh-target")
    return machine_id, target


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Build quiet-host status for all requested hosts."""

    host_values = args.host if args.host else list(DEFAULT_HOSTS)
    hosts = [_parse_host(value) for value in host_values]
    max_swap_used_mb = getattr(args, "max_swap_used_mb", None)
    min_memory_free_pct = getattr(args, "min_memory_free_pct", None)
    allow_battery = getattr(args, "allow_battery", True)
    rows = [
        _host_status(
            machine_id,
            target,
            max_load_1=args.max_load_1,
            max_noisy_cpu_pct=args.max_noisy_cpu_pct,
            max_swap_used_mb=max_swap_used_mb,
            min_memory_free_pct=min_memory_free_pct,
            require_ac_power=not allow_battery,
            timeout_s=args.timeout,
        )
        for machine_id, target in hosts
    ]
    return {
        "checked_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "publishable_timing_allowed": all(row.get("quiet") for row in rows),
        "rows": rows,
    }


def _fmt_load(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _fmt_optional(value: object, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value}{suffix}"


def render_markdown(payload: dict[str, Any]) -> str:
    """Render quiet-host status as Markdown."""

    lines = [
        "# Remote Host Quiet Status",
        "",
        f"Checked at local time: `{payload['checked_at_local']}`.",
        "",
        "| Machine | Quiet | Load 1/5/15 | Swap | Mem free | Power | Thermal | Blockers |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        if not row.get("ok"):
            blockers = row.get("error") or "ssh failed"
            load = "n/a"
            swap = "n/a"
            memory_free = "n/a"
            power = "n/a"
            thermal = "n/a"
        else:
            load_payload = row["load"]
            load = "/".join(
                [
                    _fmt_load(load_payload["one_minute"]),
                    _fmt_load(load_payload["five_minutes"]),
                    _fmt_load(load_payload["fifteen_minutes"]),
                ]
            )
            swap = _fmt_optional(row["swap"]["used_mb"], " MB")
            memory_free = _fmt_optional(row["memory_pressure"]["free_pct"], "%")
            power = "AC" if row["power"]["ac_power"] else "battery/unknown"
            thermal = "ok" if row["thermal"]["ok"] is not False else "warning"
            blockers = "; ".join(row.get("blockers") or []) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['machine_id']}`",
                    "`yes`" if row.get("quiet") else "`no`",
                    load,
                    swap,
                    memory_free,
                    power,
                    thermal,
                    blockers,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Publishable lower-end Mac timing is allowed only when every target row is",
            "`quiet=yes`. If a host is noisy, skip warmed frontier promotion and record",
            "the blocker instead.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", action="append", default=None, help="MACHINE=ssh-target")
    parser.add_argument("--max-load-1", type=float, default=1.5)
    parser.add_argument("--max-noisy-cpu-pct", type=float, default=5.0)
    parser.add_argument("--max-swap-used-mb", type=float, default=0.0)
    parser.add_argument("--min-memory-free-pct", type=int, default=10)
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    payload = build_payload(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(payload))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "publishable_timing_allowed": payload["publishable_timing_allowed"],
                "quiet_hosts": sum(1 for row in payload["rows"] if row.get("quiet")),
                "total_hosts": len(payload["rows"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
