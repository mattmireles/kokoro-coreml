#!/usr/bin/env python3
"""Shared schema helpers for the MoE SSD/DRAM prefetch experiment.

This module is used by `model_inventory.py` in Phase 0 and by later stage
summaries. Keep it dependency-free: Stage 0 must be runnable before any heavy
MoE model packages are installed.
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_SPEED_WIN_PERCENT = 25.0
DEFAULT_TRIVIAL_MARGIN_PERCENT = 10.0
DEFAULT_ENERGY_REGRESSION_ALLOWED = False


def repo_root() -> Path:
    """Return the repository root for scripts under `scripts/moe_prefetch/`."""
    return Path(__file__).resolve().parents[2]


def utc_now() -> str:
    """Return a UTC timestamp suitable for JSON provenance records."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from `path`."""
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable, newline-terminated JSON object to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _command_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True).strip()
    except Exception:
        return "unknown"


def machine_info(machine_id: str | None = None) -> dict[str, Any]:
    """Capture lightweight machine provenance for benchmark artifacts."""
    info: dict[str, Any] = {
        "machine_id": machine_id or platform.node() or "unknown",
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }
    for key, command in {
        "sw_vers": ["sw_vers"],
        "hw_model": ["sysctl", "-n", "hw.model"],
        "memory_bytes": ["sysctl", "-n", "hw.memsize"],
        "cpu_brand": ["sysctl", "-n", "machdep.cpu.brand_string"],
    }.items():
        value = _command_text(command)
        if key == "memory_bytes" and value.isdigit():
            info[key] = int(value)
        else:
            info[key] = value
    return info


def estimate_expert_bytes(expert_parameters: int, quantization_bits: float) -> int:
    """Estimate bytes for one expert block from parameter count and bit width."""
    if expert_parameters <= 0:
        raise ValueError("expert_parameters must be positive")
    if quantization_bits <= 0:
        raise ValueError("quantization_bits must be positive")
    return int(math.ceil(expert_parameters * quantization_bits / 8.0))


def thresholds_payload(
    *,
    speed_win_percent: float = DEFAULT_SPEED_WIN_PERCENT,
    trivial_margin_percent: float = DEFAULT_TRIVIAL_MARGIN_PERCENT,
    energy_regression_allowed: bool = DEFAULT_ENERGY_REGRESSION_ALLOWED,
) -> dict[str, Any]:
    """Build the pre-registered thresholds record for all later gates."""
    if speed_win_percent < 0:
        raise ValueError("speed_win_percent must be non-negative")
    if trivial_margin_percent < 0:
        raise ValueError("trivial_margin_percent must be non-negative")
    return {
        "created_at": utc_now(),
        "speed_win_percent": float(speed_win_percent),
        "trivial_margin_percent": float(trivial_margin_percent),
        "energy_regression_allowed": bool(energy_regression_allowed),
        "energy_target": "demand_paging_baseline",
    }


def model_inventory_payload(
    *,
    model_id: str,
    quantization_bits: float,
    active_experts_per_token: int,
    target_tokens_per_second: float,
    expert_bytes: int,
    expert_parameters: int | None,
    target_device: str,
    estimate_source: str,
    notes: str,
    machine: dict[str, Any],
) -> dict[str, Any]:
    """Build the model inventory record required before Stage 0 measurements."""
    if not model_id:
        raise ValueError("model_id is required")
    if quantization_bits <= 0:
        raise ValueError("quantization_bits must be positive")
    if active_experts_per_token <= 0:
        raise ValueError("active_experts_per_token must be positive")
    if target_tokens_per_second <= 0:
        raise ValueError("target_tokens_per_second must be positive")
    if expert_bytes <= 0:
        raise ValueError("expert_bytes must be positive")
    active_expert_bytes_per_token = active_experts_per_token * expert_bytes
    return {
        "created_at": utc_now(),
        "model_id": model_id,
        "quantization_bits": float(quantization_bits),
        "active_experts_per_token": int(active_experts_per_token),
        "target_tokens_per_second": float(target_tokens_per_second),
        "expert_bytes": int(expert_bytes),
        "expert_parameters": expert_parameters,
        "active_expert_bytes_per_token": int(active_expert_bytes_per_token),
        "target_device": target_device,
        "estimate_source": estimate_source,
        "notes": notes,
        "machine": machine,
    }


def validate_phase0_ready(
    inventory: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[str]:
    """Return missing Phase 0 readiness fields; empty means Stage 0 may run."""
    missing: list[str] = []
    for key in (
        "model_id",
        "quantization_bits",
        "active_experts_per_token",
        "target_tokens_per_second",
        "expert_bytes",
        "active_expert_bytes_per_token",
        "target_device",
    ):
        if inventory.get(key) in (None, "", 0):
            missing.append(f"model_inventory.{key}")
    for key in (
        "speed_win_percent",
        "trivial_margin_percent",
        "energy_regression_allowed",
        "energy_target",
    ):
        if key not in thresholds:
            missing.append(f"thresholds.{key}")
    return missing

