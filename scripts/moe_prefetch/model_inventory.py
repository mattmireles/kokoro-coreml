#!/usr/bin/env python3
"""Write Phase 0 model inventory and pre-registered threshold artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.moe_prefetch.schema import (
    DEFAULT_ENERGY_REGRESSION_ALLOWED,
    DEFAULT_SPEED_WIN_PERCENT,
    DEFAULT_TRIVIAL_MARGIN_PERCENT,
    estimate_expert_bytes,
    machine_info,
    model_inventory_payload,
    thresholds_payload,
    validate_phase0_ready,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze Phase 0 assumptions for the MoE SSD/DRAM prefetch "
            "experiment. Provide either --expert-bytes or --expert-parameters; "
            "the tool refuses to guess expert size."
        )
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--quantization-bits", type=float, required=True)
    parser.add_argument("--active-experts-per-token", type=int, required=True)
    parser.add_argument("--target-tokens-per-second", type=float, required=True)
    parser.add_argument("--target-device", default=None)
    parser.add_argument("--machine-id", default=None)
    parser.add_argument("--expert-bytes", type=int, default=None)
    parser.add_argument("--expert-parameters", type=int, default=None)
    parser.add_argument("--estimate-source", default="manual")
    parser.add_argument("--notes", default="")
    parser.add_argument("--speed-win-percent", type=float, default=DEFAULT_SPEED_WIN_PERCENT)
    parser.add_argument(
        "--trivial-margin-percent",
        type=float,
        default=DEFAULT_TRIVIAL_MARGIN_PERCENT,
    )
    parser.add_argument(
        "--energy-regression-allowed",
        action="store_true",
        default=DEFAULT_ENERGY_REGRESSION_ALLOWED,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for model_inventory.json",
    )
    parser.add_argument(
        "--thresholds-output",
        type=Path,
        default=None,
        help="Path for thresholds.json; defaults beside --output.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.expert_bytes is None and args.expert_parameters is None:
        raise SystemExit("provide --expert-bytes or --expert-parameters")
    if args.expert_bytes is not None and args.expert_parameters is not None:
        raise SystemExit("provide only one of --expert-bytes or --expert-parameters")

    expert_bytes = (
        args.expert_bytes
        if args.expert_bytes is not None
        else estimate_expert_bytes(args.expert_parameters, args.quantization_bits)
    )
    machine = machine_info(args.machine_id)
    target_device = args.target_device or machine.get("machine_id", "unknown")
    inventory = model_inventory_payload(
        model_id=args.model_id,
        quantization_bits=args.quantization_bits,
        active_experts_per_token=args.active_experts_per_token,
        target_tokens_per_second=args.target_tokens_per_second,
        expert_bytes=expert_bytes,
        expert_parameters=args.expert_parameters,
        target_device=str(target_device),
        estimate_source=args.estimate_source,
        notes=args.notes,
        machine=machine,
    )
    thresholds = thresholds_payload(
        speed_win_percent=args.speed_win_percent,
        trivial_margin_percent=args.trivial_margin_percent,
        energy_regression_allowed=args.energy_regression_allowed,
    )
    missing = validate_phase0_ready(inventory, thresholds)
    if missing:
        raise SystemExit("phase0 readiness failed: " + ", ".join(missing))

    thresholds_output = args.thresholds_output or args.output.parent / "thresholds.json"
    write_json(args.output, inventory)
    write_json(thresholds_output, thresholds)
    print(f"wrote {args.output}")
    print(f"wrote {thresholds_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
