#!/usr/bin/env python3
"""Measure a conservative one-layer compute budget for Stage 0 hideability.

This script benchmarks the active Mixtral-style MoE feed-forward math for one
token and the configured number of active experts per layer. It intentionally
does not download a Hugging Face checkpoint or trace a router. Stage 0 only
needs a resident-compute lead-time budget to compare against expert-block SSD
fetch latency; Stage 1 owns real router traces and model-specific prediction.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.moe_prefetch.schema import load_json, machine_info, utc_now, write_json

MIXTRAL_HIDDEN_SIZE = 4096
MIXTRAL_INTERMEDIATE_SIZE = 14336
MIXTRAL_ACTIVE_EXPERTS_PER_LAYER = 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default="outputs/moe_prefetch/stage0/model_inventory.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/moe_prefetch/stage0/compute.json"))
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=MIXTRAL_HIDDEN_SIZE)
    parser.add_argument("--intermediate-size", type=int, default=MIXTRAL_INTERMEDIATE_SIZE)
    parser.add_argument(
        "--active-experts-per-layer",
        type=int,
        default=MIXTRAL_ACTIVE_EXPERTS_PER_LAYER,
    )
    return parser.parse_args()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _torch_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but torch.backends.mps.is_available() is false")
    return requested


def _synchronize(device: str) -> None:
    if device == "mps":
        import torch

        torch.mps.synchronize()


def _active_moe_ffn(x: Any, w1: Any, w2: Any, w3: Any) -> Any:
    import torch

    out = torch.zeros_like(x)
    for expert_index in range(w1.shape[0]):
        gate = torch.nn.functional.silu(x @ w1[expert_index])
        up = x @ w3[expert_index]
        out = out + ((gate * up) @ w2[expert_index])
    return out / float(w1.shape[0])


def main() -> int:
    args = _parse_args()
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.hidden_size <= 0 or args.intermediate_size <= 0:
        raise SystemExit("--hidden-size and --intermediate-size must be positive")
    if args.active_experts_per_layer <= 0:
        raise SystemExit("--active-experts-per-layer must be positive")

    import torch

    inventory = load_json(Path(args.inventory))
    device = _torch_device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    torch.manual_seed(0)

    shape_in = (args.active_experts_per_layer, args.hidden_size, args.intermediate_size)
    shape_out = (args.active_experts_per_layer, args.intermediate_size, args.hidden_size)
    x = torch.randn((args.hidden_size,), device=device, dtype=dtype)
    w1 = torch.randn(shape_in, device=device, dtype=dtype) * 0.01
    w3 = torch.randn(shape_in, device=device, dtype=dtype) * 0.01
    w2 = torch.randn(shape_out, device=device, dtype=dtype) * 0.01
    _synchronize(device)

    with torch.no_grad():
        for _ in range(args.warmup):
            y = _active_moe_ffn(x, w1, w2, w3)
        _synchronize(device)

        latencies_ns: list[int] = []
        for _ in range(args.iterations):
            start_ns = time.perf_counter_ns()
            y = _active_moe_ffn(x, w1, w2, w3)
            _synchronize(device)
            latencies_ns.append(time.perf_counter_ns() - start_ns)
        checksum = float(y.float().sum().item())

    latencies_ms = [value / 1_000_000.0 for value in latencies_ns]
    payload: dict[str, Any] = {
        "created_at": utc_now(),
        "machine": machine_info(None),
        "inventory": inventory,
        "benchmark": "synthetic_active_moe_ffn",
        "model_shape": {
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "active_experts_per_layer": args.active_experts_per_layer,
        },
        "config": {
            "device": device,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "torch_version": torch.__version__,
        },
        "latencies_ns": latencies_ns,
        "latency_p50_ms": statistics.median(latencies_ms),
        "latency_p95_ms": _percentile(latencies_ms, 0.95),
        "checksum": checksum,
        "notes": (
            "Conservative Stage 0 lead-time proxy: one token through the active "
            "Mixtral-style MoE FFN math for resident weights. This is not a "
            "router trace or full-model decode benchmark."
        ),
    }
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(f"one_layer_compute_p50_ms={payload['latency_p50_ms']:.6f}")
    print(f"one_layer_compute_p95_ms={payload['latency_p95_ms']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
