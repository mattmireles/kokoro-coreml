#!/usr/bin/env python3
"""Histogram MIL operation types for an MLProgram .mlpackage (ANE optimization Phase 0).

Walks ``spec.mlProgram`` function blocks for reproducible Phase 3 comparisons.

Example::

    uv run python scripts/count_mil_ops.py coreml/kokoro_decoder_har_post_3s.mlpackage
    uv run python scripts/count_mil_ops.py --probe-conv-lowering
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn


def _iter_operation_types(spec) -> list[str]:
    mp = spec.mlProgram
    types: list[str] = []
    for fn in mp.functions.values():
        for block in fn.block_specializations.values():
            for op in block.operations:
                types.append(op.type)
    return types


def histogram_for_mlpackage(path: Path) -> tuple[Counter, int]:
    model = ct.models.MLModel(str(path), skip_model_load=True)
    spec = model.get_spec()
    types = _iter_operation_types(spec)
    return Counter(types), len(types)


def _export_minimal_and_top_ops(module: nn.Module, x: torch.Tensor) -> list[str]:
    m = module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(m, (x,), strict=False)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=x.shape, dtype=np.float32)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
    )
    spec = ml.get_spec()
    types = _iter_operation_types(spec)
    return sorted(set(types))


class _TinyConv1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(4, 8, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(x)


class _TinyConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(4, 8, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(x)


def probe_conv_lowering() -> dict:
    x1 = torch.zeros(1, 4, 16, dtype=torch.float32)
    x2 = torch.zeros(1, 4, 1, 16, dtype=torch.float32)
    t1 = _export_minimal_and_top_ops(_TinyConv1d(), x1)
    t2 = _export_minimal_and_top_ops(_TinyConv2d(), x2)
    return {
        "conv1d_unique_op_types": t1,
        "conv2d_unique_op_types": t2,
        "both_have_conv": "conv" in t1 and "conv" in t2,
        "type_sets_equal": sorted(t1) == sorted(t2),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mlpackage", nargs="?", type=Path, help="Path to .mlpackage directory")
    p.add_argument("--json", action="store_true", help="Histogram as JSON")
    p.add_argument(
        "--probe-conv-lowering",
        action="store_true",
        help="Minimal Conv1d vs Conv2d MIL op comparison (Phase 0 gate)",
    )
    args = p.parse_args(argv)

    print(f"coremltools {ct.__version__}", file=sys.stderr)

    if args.probe_conv_lowering:
        out = probe_conv_lowering()
        print(json.dumps(out, indent=2))
        if not out["both_have_conv"]:
            return 2
        return 0

    if args.mlpackage is None:
        p.error("mlpackage path required unless --probe-conv-lowering")

    path = args.mlpackage
    if not path.is_dir():
        print(f"Not a directory: {path}", file=sys.stderr)
        return 1

    counts, total = histogram_for_mlpackage(path)
    if args.json:
        print(json.dumps({"total_ops": total, "counts": dict(sorted(counts.items()))}, indent=2))
    else:
        print(f"total_ops\t{total}")
        for op, n in counts.most_common():
            print(f"{op}\t{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
