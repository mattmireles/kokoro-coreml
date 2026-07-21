#!/usr/bin/env python3
"""Trace and convert the real LFM2.5 Stage 0 blocks to MLProgram packages."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from coremltools.converters.mil.mil.scope import ScopeSource

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lfm2_surgical.blocks import (
    CoreMLAttentionBlock,
    CoreMLConvBlock,
    causal_mask,
    rope_tables,
)
from scripts.lfm2_surgical.checkpoint import (
    checkpoint_identity,
    load_block_tensors,
    load_config,
)
from scripts.lfm2_surgical.constants import (
    CAUSAL_MASK_NEGATIVE,
    CONV_LAYER_INDEX,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_STAGE0_DIR,
    GQA_LAYER_INDEX,
    PREFILL_BUCKETS,
    validate_frozen_protocol,
)

TRACE_BUCKET = PREFILL_BUCKETS[0]
GQA_FP32_MODULE_NAMES = frozenset(
    {"operator_norm", "attention_positioning", "ffn_norm"}
)


def parse_args() -> argparse.Namespace:
    """Parse export scope and checkpoint/output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_STAGE0_DIR))
    parser.add_argument(
        "--block", choices=("conv", "gqa", "both"), default="both"
    )
    parser.add_argument(
        "--all-buckets",
        action="store_true",
        help="affirm the frozen enumerated-shape bucket set",
    )
    parser.add_argument(
        "--fixed-bucket",
        type=int,
        choices=PREFILL_BUCKETS,
        default=None,
        help="export a fixed-shape diagnostic twin for per-bucket compute plans",
    )
    parser.add_argument(
        "--conv-rms",
        choices=("layernorm", "explicit"),
        default="explicit",
        help="select the canonical explicit RMSNorm or the drift diagnostic",
    )
    return parser.parse_args()


def enumerated_shapes(shape_builder) -> ct.EnumeratedShapes:
    """Build a Core ML enumerated-shape object over the frozen buckets."""

    return ct.EnumeratedShapes(shapes=[shape_builder(bucket) for bucket in PREFILL_BUCKETS])


def gqa_mixed_precision_selector(operation) -> bool:
    """Keep only named RMSNorm/RoPE module scopes fp32."""

    module_scopes = operation.scopes.get(
        ScopeSource.TORCHSCRIPT_MODULE_NAME, tuple()
    )
    is_approved_fp32_island = any(
        component in GQA_FP32_MODULE_NAMES
        for scope in module_scopes
        for component in str(scope).split(".")
    )
    return not is_approved_fp32_island


def package_size_bytes(package: Path) -> int:
    """Return the total byte size of a generated model package."""

    return sum(path.stat().st_size for path in package.rglob("*") if path.is_file())


def replace_package(model: ct.models.MLModel, package: Path) -> None:
    """Save ``model`` to the exact generated-artifact path, replacing only it."""

    if package.exists():
        shutil.rmtree(package)
    package.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(package))


def enumerated_counts(model: ct.models.MLModel) -> dict[str, int]:
    """Return the number of enumerated variants declared for each input."""

    counts: dict[str, int] = {}
    for feature in model.get_spec().description.input:
        counts[feature.name] = len(feature.type.multiArrayType.enumeratedShapes.shapes)
    return counts


def convert_conv(
    checkpoint_dir: Path,
    out_dir: Path,
    fixed_bucket: int | None = None,
    conv_rms: str = "explicit",
) -> dict:
    """Convert layer 0 as the canonical enumerated model or a diagnostic twin."""

    config = load_config(checkpoint_dir)
    tensors = load_block_tensors(checkpoint_dir, CONV_LAYER_INDEX, "conv")
    module = CoreMLConvBlock(
        config, tensors, explicit_rms=conv_rms == "explicit"
    ).eval()
    # Trace once at the smallest frozen shape; conversion owns specialization.
    trace_bucket = TRACE_BUCKET
    example_hidden = torch.zeros(
        1, config.hidden_size, 1, trace_bucket, dtype=torch.float16
    )
    with torch.inference_mode():
        traced = torch.jit.trace(module, (example_hidden,), strict=True)
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="hidden_states",
                shape=(1, config.hidden_size, 1, fixed_bucket)
                if fixed_bucket
                else enumerated_shapes(
                    lambda bucket: (1, config.hidden_size, 1, bucket)
                ),
                dtype=np.float16,
            )
        ],
        outputs=[
            ct.TensorType(name="hidden_out", dtype=np.float16),
            ct.TensorType(name="conv_state_out", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )
    suffix = f"_{fixed_bucket}" if fixed_bucket else ""
    if conv_rms == "layernorm":
        suffix += "_layernorm_rms"
    package = out_dir / f"conv_block{suffix}.mlpackage"
    replace_package(model, package)
    counts = enumerated_counts(model)
    if fixed_bucket is None and counts != {"hidden_states": len(PREFILL_BUCKETS)}:
        raise RuntimeError(f"conv export lost the frozen enumerated shapes: {counts}")
    return {
        "layer_index": CONV_LAYER_INDEX,
        "package": str(package),
        "bytes": package_size_bytes(package),
        "enumerated_input_counts": counts,
        "fixed_bucket": fixed_bucket,
        "rms_implementation": conv_rms,
    }


def convert_gqa(
    checkpoint_dir: Path, out_dir: Path, fixed_bucket: int | None = None
) -> dict:
    """Convert layer 2 as the canonical enumerated model or a diagnostic twin."""

    config = load_config(checkpoint_dir)
    tensors = load_block_tensors(checkpoint_dir, GQA_LAYER_INDEX, "full_attention")
    module = CoreMLAttentionBlock(config, tensors).eval()
    # Trace once at the smallest frozen shape; conversion owns specialization.
    trace_bucket = TRACE_BUCKET
    example_hidden = torch.zeros(
        1, config.hidden_size, 1, trace_bucket, dtype=torch.float16
    )
    example_cosine, example_sine = rope_tables(
        trace_bucket, config.head_dim, config.rope_theta, torch.float32
    )
    example_mask = causal_mask(
        trace_bucket, CAUSAL_MASK_NEGATIVE, torch.float16
    )
    with torch.inference_mode():
        traced = torch.jit.trace(
            module,
            (example_hidden, example_cosine, example_sine, example_mask),
            strict=True,
        )
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="hidden_states",
                shape=(1, config.hidden_size, 1, fixed_bucket)
                if fixed_bucket
                else enumerated_shapes(
                    lambda bucket: (1, config.hidden_size, 1, bucket)
                ),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="cosine",
                shape=(1, fixed_bucket, config.head_dim)
                if fixed_bucket
                else enumerated_shapes(lambda bucket: (1, bucket, config.head_dim)),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="sine",
                shape=(1, fixed_bucket, config.head_dim)
                if fixed_bucket
                else enumerated_shapes(lambda bucket: (1, bucket, config.head_dim)),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=(1, 1, fixed_bucket, fixed_bucket)
                if fixed_bucket
                else enumerated_shapes(lambda bucket: (1, 1, bucket, bucket)),
                dtype=np.float16,
            ),
        ],
        outputs=[
            ct.TensorType(name="hidden_out", dtype=np.float16),
            ct.TensorType(name="key_out", dtype=np.float16),
            ct.TensorType(name="value_out", dtype=np.float16),
        ],
        compute_precision=ct.transform.FP16ComputePrecision(
            op_selector=gqa_mixed_precision_selector
        ),
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )
    suffix = f"_{fixed_bucket}" if fixed_bucket else ""
    package = out_dir / f"gqa_block{suffix}.mlpackage"
    replace_package(model, package)
    counts = enumerated_counts(model)
    expected_counts = {
        "hidden_states": len(PREFILL_BUCKETS),
        "cosine": len(PREFILL_BUCKETS),
        "sine": len(PREFILL_BUCKETS),
        "attention_mask": len(PREFILL_BUCKETS),
    }
    if fixed_bucket is None and counts != expected_counts:
        raise RuntimeError(f"GQA export lost the frozen enumerated shapes: {counts}")
    return {
        "layer_index": GQA_LAYER_INDEX,
        "package": str(package),
        "bytes": package_size_bytes(package),
        "enumerated_input_counts": counts,
        "fixed_bucket": fixed_bucket,
        "mixed_precision_fp32_modules": sorted(GQA_FP32_MODULE_NAMES),
    }


def main() -> None:
    """Export the selected blocks and write the Stage 0 export manifest."""

    args = parse_args()
    validate_frozen_protocol()
    if not args.all_buckets:
        raise SystemExit("pass --all-buckets to affirm the frozen protocol")
    manifest: dict[str, object] = {
        "buckets": list(PREFILL_BUCKETS),
        "checkpoint": checkpoint_identity(args.checkpoint),
        "coremltools": ct.__version__,
        "minimum_deployment_target": "iOS18",
        "blocks": {},
    }
    if args.block in ("conv", "both"):
        manifest["blocks"]["conv"] = convert_conv(
            args.checkpoint, args.out_dir, args.fixed_bucket, args.conv_rms
        )
    if args.block in ("gqa", "both"):
        manifest["blocks"]["gqa"] = convert_gqa(
            args.checkpoint, args.out_dir, args.fixed_bucket
        )
    manifest_name = (
        f"export_fixed_{args.fixed_bucket}.json"
        if args.fixed_bucket
        else "export_manifest.json"
    )
    if args.conv_rms == "layernorm":
        manifest_name = manifest_name.removesuffix(".json") + "_layernorm_rms.json"
    manifest_path = args.out_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
