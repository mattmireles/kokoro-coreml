#!/usr/bin/env python3
"""Compare Core ML MLProgram graph surfaces.

This is a structural companion to latency probes. It loads `.mlpackage` specs,
counts MIL operation types, records I/O shapes, package size, and highlights
ops that commonly affect Apple Silicon placement (`instance_norm`, `tile`,
`conv_transpose`, trigonometric ops, palettized weight decompression, etc.).
It does not run inference or prove compute-unit residency.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

FOCUS_OPS = (
    "conv",
    "conv_transpose",
    "linear",
    "matmul",
    "instance_norm",
    "reduce_mean",
    "sqrt",
    "real_div",
    "tile",
    "reshape",
    "split",
    "concat",
    "sin",
    "cos",
    "pow",
    "exp",
    "cast",
    "constexpr_lut_to_dense",
)

DTYPES = {
    65552: "FLOAT16",
    65568: "FLOAT32",
    131104: "INT32",
}


def _package_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _shape(feature: Any) -> dict[str, Any]:
    array = feature.type.multiArrayType
    ranges = []
    for item in array.shapeRange.sizeRanges:
        ranges.append({"lower": int(item.lowerBound), "upper": int(item.upperBound)})
    return {
        "name": feature.name,
        "dtype": DTYPES.get(int(array.dataType), int(array.dataType)),
        "shape": [int(v) for v in array.shape],
        "shape_ranges": ranges,
    }


def _summarize(label: str, path: Path) -> dict[str, Any]:
    from coremltools.models.utils import load_spec

    spec = load_spec(str(path))
    if spec.WhichOneof("Type") != "mlProgram":
        raise ValueError(f"{path} is {spec.WhichOneof('Type')}, not mlProgram")

    block_summaries = []
    total_counts: Counter[str] = Counter()
    for function_name, function in spec.mlProgram.functions.items():
        for block_key, block in function.block_specializations.items():
            counts = Counter(op.type for op in block.operations)
            total_counts.update(counts)
            block_summaries.append(
                {
                    "function": function_name,
                    "block": block_key,
                    "op_count": int(sum(counts.values())),
                    "unique_op_count": int(len(counts)),
                    "op_counts": dict(sorted(counts.items())),
                    "focus_op_counts": {
                        op: int(counts[op])
                        for op in FOCUS_OPS
                        if counts[op]
                    },
                    "top_ops": counts.most_common(20),
                }
            )

    return {
        "label": label,
        "path": str(path),
        "specification_version": int(spec.specificationVersion),
        "package_size_bytes": int(_package_size_bytes(path)),
        "inputs": [_shape(item) for item in spec.description.input],
        "outputs": [_shape(item) for item in spec.description.output],
        "op_count": int(sum(total_counts.values())),
        "unique_op_count": int(len(total_counts)),
        "focus_op_counts": {
            op: int(total_counts[op])
            for op in FOCUS_OPS
            if total_counts[op]
        },
        "top_ops": total_counts.most_common(20),
        "blocks": block_summaries,
    }


def _parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, raw_path = value.split("=", 1)
    return label, Path(raw_path)


def _print_table(summaries: list[dict[str, Any]]) -> None:
    focus = ("op_count", "conv", "conv_transpose", "instance_norm", "reduce_mean", "tile", "sin", "cos", "constexpr_lut_to_dense")
    print("| Model | Spec | Size MB | Ops | Conv | ConvT | InstNorm | ReduceMean | Tile | Sin | Cos | LUT |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in summaries:
        counts = item["focus_op_counts"]
        print(
            f"| {item['label']} | {item['specification_version']} | "
            f"{item['package_size_bytes'] / 1_000_000.0:.1f} | "
            f"{item['op_count']} | "
            f"{counts.get('conv', 0)} | {counts.get('conv_transpose', 0)} | "
            f"{counts.get('instance_norm', 0)} | {counts.get('reduce_mean', 0)} | "
            f"{counts.get('tile', 0)} | {counts.get('sin', 0)} | "
            f"{counts.get('cos', 0)} | {counts.get('constexpr_lut_to_dense', 0)} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model path or label=path. Repeat for each package.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summaries = [_summarize(label, path) for label, path in map(_parse_model_arg, args.model)]
    payload = {"models": summaries}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output}")
    _print_table(summaries)


if __name__ == "__main__":
    main()
