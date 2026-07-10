#!/usr/bin/env python3
"""Summarize the exact HAR -> pre-noise folding surface.

This is a planning/reporting tool, not a model exporter. It quantifies whether
the current generator can avoid materializing HAR frames when folding STFT/HAR
directly into the first ``noise_convs`` outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

from probe_generator_exact_geometry import _load_kmodel  # noqa: E402


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _tensor_shape(manifest: dict[str, Any], name: str) -> list[int]:
    for row in manifest.get("tensors") or []:
        if row.get("name") == name:
            return [int(v) for v in row["shape"]]
    raise ValueError(f"manifest missing tensor shape for {name}")


def _conv_output_length(input_len: int, *, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return ((input_len + 2 * padding - dilation * (kernel - 1) - 1) // stride) + 1


def _touched_frames(input_len: int, *, output_len: int, kernel: int, stride: int, padding: int, dilation: int) -> set[int]:
    frames: set[int] = set()
    for output_index in range(output_len):
        base = output_index * stride - padding
        for kernel_index in range(kernel):
            frame = base + kernel_index * dilation
            if 0 <= frame < input_len:
                frames.add(frame)
    return frames


def _conv_geometry(conv: Any) -> dict[str, Any]:
    weight = conv.weight.detach().cpu().numpy().astype(np.float64)
    channel_l1 = np.sum(np.abs(weight), axis=(0, 2))
    return {
        "weight_shape": [int(v) for v in weight.shape],
        "out_channels": int(weight.shape[0]),
        "in_channels": int(weight.shape[1]),
        "kernel": int(conv.kernel_size[0]),
        "stride": int(conv.stride[0]),
        "padding": int(conv.padding[0]),
        "dilation": int(conv.dilation[0]),
        "nonzero_input_channels_l1_gt_1e_9": int(np.sum(channel_l1 > 1e-9)),
        "input_channel_l1": [float(v) for v in channel_l1],
    }


def build_summary(dump_dirs: list[Path]) -> dict[str, Any]:
    """Build the folding-surface summary for all provided dump directories."""

    gen = _load_kmodel().decoder.generator.eval()
    convs = [_conv_geometry(conv) for conv in gen.noise_convs]
    rows = []
    for dump_dir in dump_dirs:
        manifest = _load_manifest(dump_dir / "tensor_manifest.json")
        metadata = manifest.get("metadata") or {}
        har_shape = _tensor_shape(manifest, "har_padded")
        har_channels = int(har_shape[1])
        har_time = int(har_shape[2])
        conv_rows = []
        union_frames: set[int] = set()
        total_pre_noise_values = 0
        for index, conv in enumerate(convs):
            output_len = _conv_output_length(
                har_time,
                kernel=conv["kernel"],
                stride=conv["stride"],
                padding=conv["padding"],
                dilation=conv["dilation"],
            )
            touched = _touched_frames(
                har_time,
                output_len=output_len,
                kernel=conv["kernel"],
                stride=conv["stride"],
                padding=conv["padding"],
                dilation=conv["dilation"],
            )
            union_frames.update(touched)
            pre_noise_values = int(conv["out_channels"] * output_len)
            total_pre_noise_values += pre_noise_values
            conv_rows.append(
                {
                    "index": index,
                    "output_length": output_len,
                    "touched_har_frames": len(touched),
                    "touched_har_frame_pct": len(touched) / har_time * 100.0,
                    "pre_noise_values": pre_noise_values,
                    "pre_noise_bytes_fp16": pre_noise_values * 2,
                    "pre_noise_bytes_fp32": pre_noise_values * 4,
                    "geometry": conv,
                }
            )
        full_har_values = int(har_channels * har_time)
        rows.append(
            {
                "dump_dir": str(dump_dir),
                "input_key": metadata.get("input_key") or dump_dir.name,
                "bucket_seconds": metadata.get("bucket_seconds"),
                "har_shape": har_shape,
                "full_har_values": full_har_values,
                "full_har_bytes_fp16": full_har_values * 2,
                "full_har_bytes_fp32": full_har_values * 4,
                "union_touched_har_frames": len(union_frames),
                "union_touched_har_frame_pct": len(union_frames) / har_time * 100.0,
                "total_pre_noise_values": total_pre_noise_values,
                "total_pre_noise_bytes_fp16": total_pre_noise_values * 2,
                "total_pre_noise_bytes_fp32": total_pre_noise_values * 4,
                "pre_noise_to_har_value_ratio": total_pre_noise_values / full_har_values,
                "conv_rows": conv_rows,
            }
        )
    return {
        "stft": {
            "filter_length": int(gen.stft.filter_length),
            "hop_length": int(gen.stft.hop_length),
            "win_length": int(gen.stft.win_length),
            "har_channels": 22,
        },
        "noise_convs": convs,
        "rows": rows,
        "decision": _decision(rows),
    }


def _decision(rows: list[dict[str, Any]]) -> str:
    if all(float(row["union_touched_har_frame_pct"]) >= 99.9 for row in rows):
        return "fold_for_memory_locality_not_frame_skipping"
    return "fold_may_skip_har_frames"


def _fmt_bytes(value: int) -> str:
    mib = value / (1024.0 * 1024.0)
    return f"{mib:.2f} MiB"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a markdown report."""

    lines = [
        "# Pre-Noise Folding Surface",
        "",
        "This report quantifies the exact HAR frames consumed by the generator",
        "`noise_convs` before any AdaIN residual work.",
        "",
        f"- Decision: `{summary['decision']}`.",
        (
            "- STFT/HAR geometry: "
            f"filter `{summary['stft']['filter_length']}`, "
            f"hop `{summary['stft']['hop_length']}`, "
            f"channels `{summary['stft']['har_channels']}`."
        ),
        "",
        "| Bucket | HAR shape | HAR fp16 | Touched HAR frames | Pre-noise fp16 | Pre-noise/HAR values |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['input_key']}`",
                    "`" + "x".join(str(v) for v in row["har_shape"]) + "`",
                    _fmt_bytes(int(row["full_har_bytes_fp16"])),
                    f"{row['union_touched_har_frames']} ({row['union_touched_har_frame_pct']:.2f}%)",
                    _fmt_bytes(int(row["total_pre_noise_bytes_fp16"])),
                    f"{row['pre_noise_to_har_value_ratio']:.2f}x",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Conv Geometry",
            "",
            "| Conv | Weight | Stride | Padding | Output frames touched | Input channels with nonzero L1 |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    first_row = summary["rows"][0]
    for conv_row in first_row["conv_rows"]:
        geom = conv_row["geometry"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(conv_row["index"]),
                    "`" + "x".join(str(v) for v in geom["weight_shape"]) + "`",
                    str(geom["stride"]),
                    str(geom["padding"]),
                    f"{conv_row['touched_har_frames']} ({conv_row['touched_har_frame_pct']:.2f}%)",
                    str(geom["nonzero_input_channels_l1_gt_1e_9"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Because `noise_conv_1` is a 1x1 stride-1 convolution over every HAR",
            "frame, folding cannot skip most HAR frames. The useful implementation",
            "target is memory locality and boundary removal: compute the exact",
            "STFT/HAR receptive-field dot products directly into pre-noise outputs",
            "without materializing a standalone HAR tensor or adding another Core",
            "ML prediction boundary.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-dir",
        action="append",
        type=Path,
        dest="dump_dirs",
        help="Tensor dump directory. Defaults to all canonical bucket dumps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/external_bakeoff/pre_noise_folding_surface.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("outputs/external_bakeoff/pre_noise_folding_surface.json"),
    )
    args = parser.parse_args()
    dump_dirs = args.dump_dirs or [
        Path("outputs/generator_isolation/dumps/3s"),
        Path("outputs/generator_isolation/dumps/7s"),
        Path("outputs/generator_isolation/dumps/10s"),
        Path("outputs/generator_isolation/dumps/15s"),
        Path("outputs/generator_isolation/dumps/30s"),
    ]
    summary = build_summary(dump_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "decision": summary["decision"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
