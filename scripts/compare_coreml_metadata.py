#!/usr/bin/env python3
"""Compare normalized ``xcrun coremlcompiler metadata`` surfaces."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


FOCUS_OPS = (
    "conv",
    "convTranspose",
    "linear",
    "instanceNorm",
    "reduceMean",
    "tile",
    "sin",
    "cos",
    "constexprLutToDense",
)


def _metadata(path: Path) -> dict[str, Any]:
    """Return the first metadata object from ``coremlcompiler metadata``."""

    output = subprocess.check_output(
        ["xcrun", "coremlcompiler", "metadata", str(path)],
        text=True,
    )
    payload = json.loads(output)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"metadata output for {path} was not a non-empty list")
    first = payload[0]
    if not isinstance(first, dict):
        raise ValueError(f"metadata output for {path} did not contain an object")
    return first


def _suffix_op_name(name: str) -> str:
    """Normalize ``Ios17.conv`` and ``Tile`` style op keys."""

    suffix = name.split(".")[-1]
    return suffix[:1].lower() + suffix[1:]


def _normalized_histogram(histogram: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, value in histogram.items():
        normalized = _suffix_op_name(str(key))
        counts[normalized] = counts.get(normalized, 0) + int(value)
    return dict(sorted(counts.items()))


def _schema_rows(schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in schema:
        rows.append(
            {
                "name": item.get("name"),
                "dataType": item.get("dataType"),
                "shape": item.get("shape"),
                "hasShapeFlexibility": item.get("hasShapeFlexibility"),
                "shapeFlexibility": item.get("shapeFlexibility"),
                "shapeRange": item.get("shapeRange"),
            }
        )
    return rows


def summarize_metadata(label: str, path: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized metadata summary for one Core ML package."""

    raw = metadata if metadata is not None else _metadata(path)
    histogram = _normalized_histogram(raw.get("mlProgramOperationTypeHistogram") or {})
    return {
        "label": label,
        "path": str(path),
        "generatedClassName": raw.get("generatedClassName"),
        "specificationVersion": raw.get("specificationVersion"),
        "storagePrecision": raw.get("storagePrecision"),
        "computePrecision": raw.get("computePrecision"),
        "availability": raw.get("availability") or {},
        "userDefinedMetadata": raw.get("userDefinedMetadata") or {},
        "inputSchema": _schema_rows(raw.get("inputSchema") or []),
        "outputSchema": _schema_rows(raw.get("outputSchema") or []),
        "opHistogram": histogram,
        "focusOps": {
            op: histogram.get(op, 0)
            for op in FOCUS_OPS
            if histogram.get(op, 0)
        },
    }


def parse_model_arg(value: str) -> tuple[str, Path]:
    """Parse ``label=path`` or use the package stem as label."""

    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, raw_path = value.split("=", 1)
    return label, Path(raw_path)


def render_markdown(models: list[dict[str, Any]]) -> str:
    """Render a compact metadata comparison table."""

    lines = [
        "# Core ML Metadata Comparison",
        "",
        "| Model | Spec | Storage | Inputs | Flexible inputs | Outputs | Focus ops |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for model in models:
        inputs = ", ".join(
            f"{item['name']}:{item['dataType']} {item['shape']}"
            for item in model["inputSchema"]
        )
        flexible = ", ".join(
            f"{item['name']}={item['shapeFlexibility'] or item['shapeRange']}"
            for item in model["inputSchema"]
            if str(item.get("hasShapeFlexibility")) == "1"
        )
        outputs = ", ".join(
            f"{item['name']}:{item['dataType']} {item['shape']}"
            for item in model["outputSchema"]
        )
        focus = ", ".join(f"{key}={value}" for key, value in model["focusOps"].items())
        lines.append(
            "| "
            + " | ".join(
                [
                    model["label"],
                    str(model["specificationVersion"]),
                    str(model["storagePrecision"]),
                    inputs,
                    flexible or "none",
                    outputs,
                    focus or "none",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="Model path or label=path.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    models = [
        summarize_metadata(label, path)
        for label, path in map(parse_model_arg, args.model)
    ]
    payload = {"models": models}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown = render_markdown(models)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown)
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
