#!/usr/bin/env python3
"""Compare fp16 Core ML blocks with independent fp32 PyTorch references."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lfm2_surgical.blocks import (
    ReferenceAttentionBlock,
    ReferenceConvBlock,
    causal_mask,
    rope_tables,
)
from scripts.lfm2_surgical.checkpoint import (
    checkpoint_identity,
    load_block_tensors,
    load_config,
    load_embedding,
)
from scripts.lfm2_surgical.constants import (
    CAUSAL_MASK_NEGATIVE,
    CONV_LAYER_INDEX,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_STAGE0_DIR,
    GQA_INPUT_LAYER_INDEX,
    GQA_LAYER_INDEX,
    NUMERICS_MAX_ABS_GATE,
    NUMERICS_PROMPT_COUNT,
    PREFILL_BUCKETS,
    REAL_PROMPTS,
    validate_frozen_protocol,
)


def parse_args() -> argparse.Namespace:
    """Parse real-checkpoint, package, prompt-count, and output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--stage0-dir", type=Path, default=Path(DEFAULT_STAGE0_DIR))
    parser.add_argument("--prompts", type=int, default=NUMERICS_PROMPT_COUNT)
    return parser.parse_args()


def load_tokenizer(checkpoint_dir: Path) -> PreTrainedTokenizerFast:
    """Load the checkpoint's tokenizer JSON while ignoring v5-only metadata."""

    return PreTrainedTokenizerFast(
        tokenizer_file=str(checkpoint_dir / "tokenizer.json"),
        bos_token="<|startoftext|>",
        eos_token="<|im_end|>",
        pad_token="<|pad|>",
    )


def fixed_bucket_ids(
    tokenizer: PreTrainedTokenizerFast, prompt: str, bucket: int
) -> torch.Tensor:
    """Tokenize a real prompt and repeat it deterministically to fill ``bucket``."""

    token_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    if not token_ids:
        raise ValueError(f"prompt produced no token IDs: {prompt!r}")
    repetitions = (bucket + len(token_ids) - 1) // len(token_ids)
    return torch.tensor((token_ids * repetitions)[:bucket], dtype=torch.long)


def tensor_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Return max-absolute error and cosine similarity for one output tensor."""

    reference_flat = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate_flat = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if reference_flat.shape != candidate_flat.shape:
        raise ValueError(
            f"output shape mismatch: {reference_flat.shape} != {candidate_flat.shape}"
        )
    max_abs = float(np.max(np.abs(reference_flat - candidate_flat)))
    denominator = float(
        np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat)
    )
    cosine = float(np.dot(reference_flat, candidate_flat) / denominator)
    return {"max_abs": max_abs, "cosine": cosine}


def merge_metric(aggregate: dict[str, float], current: dict[str, float]) -> None:
    """Update an aggregate with worst-case max-abs and cosine values."""

    aggregate["max_abs"] = max(aggregate["max_abs"], current["max_abs"])
    aggregate["cosine"] = min(aggregate["cosine"], current["cosine"])


def empty_metric() -> dict[str, float]:
    """Return the identity values for worst-case metric aggregation."""

    return {"max_abs": 0.0, "cosine": 1.0}


def package_tree_sha256(package: Path) -> str:
    """Hash package-relative paths and bytes to bind parity to exact graphs."""

    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in package.rglob("*") if candidate.is_file()):
        digest.update(str(path.relative_to(package)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as package_file:
            for chunk in iter(lambda: package_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def run_numerics(
    checkpoint_dir: Path, stage0_dir: Path, prompt_count: int
) -> dict:
    """Run every frozen bucket on real prompts through both block formulations."""

    if prompt_count < 1 or prompt_count > NUMERICS_PROMPT_COUNT:
        raise ValueError(
            f"--prompts must be between 1 and {NUMERICS_PROMPT_COUNT}"
        )
    config = load_config(checkpoint_dir)
    tokenizer = load_tokenizer(checkpoint_dir)
    embedding = load_embedding(checkpoint_dir).float()
    conv_reference = ReferenceConvBlock(
        config, load_block_tensors(checkpoint_dir, CONV_LAYER_INDEX, "conv")
    ).eval()
    gqa_input_reference = ReferenceConvBlock(
        config, load_block_tensors(checkpoint_dir, GQA_INPUT_LAYER_INDEX, "conv")
    ).eval()
    gqa_reference = ReferenceAttentionBlock(
        config,
        load_block_tensors(checkpoint_dir, GQA_LAYER_INDEX, "full_attention"),
    ).eval()
    # These packages have fp16 external I/O. On the experiment's macOS 26.5.2
    # host, Core ML traps (SIGTRAP) when the fp16 graph is compiled CPU-only.
    # CPU_AND_GPU is the narrowest stable runtime configuration for isolating
    # fp16 conversion drift without also making ANE scheduling part of G0c.
    validation_compute_units = ct.ComputeUnit.CPU_AND_GPU
    conv_package = stage0_dir / "conv_block.mlpackage"
    gqa_package = stage0_dir / "gqa_block.mlpackage"
    conv_model = ct.models.MLModel(
        str(conv_package),
        compute_units=validation_compute_units,
    )
    gqa_model = ct.models.MLModel(
        str(gqa_package),
        compute_units=validation_compute_units,
    )

    buckets: dict[str, dict] = {}
    for bucket in PREFILL_BUCKETS:
        cosine, sine = rope_tables(
            bucket, config.head_dim, config.rope_theta, torch.float32
        )
        mask = causal_mask(bucket, CAUSAL_MASK_NEGATIVE, torch.float32)
        aggregate = {
            "conv_hidden": empty_metric(),
            "conv_state": empty_metric(),
            "gqa_hidden": empty_metric(),
            "gqa_key": empty_metric(),
            "gqa_value": empty_metric(),
        }
        for prompt in REAL_PROMPTS[:prompt_count]:
            token_ids = fixed_bucket_ids(tokenizer, prompt, bucket)
            hidden = embedding[token_ids].unsqueeze(0)
            with torch.inference_mode():
                conv_hidden_ref, conv_state_ref = conv_reference(hidden)
                gqa_input, _ = gqa_input_reference(conv_hidden_ref)
                gqa_hidden_ref, gqa_key_ref, gqa_value_ref = gqa_reference(
                    gqa_input, cosine, sine, mask
                )
            hidden_ane = hidden.to(torch.float16).permute(0, 2, 1).unsqueeze(2)
            gqa_input_ane = (
                gqa_input.to(torch.float16).permute(0, 2, 1).unsqueeze(2)
            )
            conv_output = conv_model.predict(
                {"hidden_states": hidden_ane.numpy()},
            )
            gqa_output = gqa_model.predict(
                {
                    "hidden_states": gqa_input_ane.numpy(),
                    "cosine": cosine.numpy(),
                    "sine": sine.numpy(),
                    "attention_mask": mask.to(torch.float16).numpy(),
                }
            )
            merge_metric(
                aggregate["conv_hidden"],
                tensor_metrics(
                    conv_hidden_ref.numpy(),
                    np.asarray(conv_output["hidden_out"]).squeeze(2).transpose(0, 2, 1),
                ),
            )
            merge_metric(
                aggregate["conv_state"],
                tensor_metrics(conv_state_ref.numpy(), conv_output["conv_state_out"]),
            )
            merge_metric(
                aggregate["gqa_hidden"],
                tensor_metrics(
                    gqa_hidden_ref.numpy(),
                    np.asarray(gqa_output["hidden_out"]).squeeze(2).transpose(0, 2, 1),
                ),
            )
            merge_metric(
                aggregate["gqa_key"],
                tensor_metrics(gqa_key_ref.numpy(), gqa_output["key_out"]),
            )
            merge_metric(
                aggregate["gqa_value"],
                tensor_metrics(gqa_value_ref.numpy(), gqa_output["value_out"]),
            )
        aggregate["prompt_count"] = prompt_count
        buckets[str(bucket)] = aggregate
        print(f"bucket {bucket}: {json.dumps(aggregate, sort_keys=True)}")

    # G0c applies to every public block output, including state handoff tensors.
    gate_outputs = [
        "conv_hidden",
        "conv_state",
        "gqa_hidden",
        "gqa_key",
        "gqa_value",
    ]
    failures = [
        {
            "bucket": int(bucket),
            "output": output,
            "max_abs": metrics[output]["max_abs"],
        }
        for bucket, metrics in buckets.items()
        for output in gate_outputs
        if metrics[output]["max_abs"] > NUMERICS_MAX_ABS_GATE
    ]
    return {
        "activation_sources": {
            "conv": "trained token embeddings at layer 0 input",
            "gqa": "fp32 reference outputs after layers 0 and 1",
        },
        "buckets": buckets,
        "checkpoint": checkpoint_identity(checkpoint_dir),
        "compute_units": "CPU_AND_GPU",
        "fp16_max_abs_gate": NUMERICS_MAX_ABS_GATE,
        "gate_outputs": gate_outputs,
        "packages": {
            "conv": {
                "path": str(conv_package),
                "sha256": package_tree_sha256(conv_package),
            },
            "gqa": {
                "path": str(gqa_package),
                "sha256": package_tree_sha256(gqa_package),
            },
        },
        "failures": failures,
        "passed": not failures,
        "prompt_count": prompt_count,
    }


def main() -> None:
    """Run the full numerical gate and save ``numerics.json``."""

    args = parse_args()
    validate_frozen_protocol()
    report = run_numerics(args.checkpoint, args.stage0_dir, args.prompts)
    output_path = args.stage0_dir / "numerics.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output_path}")
    if not report["passed"]:
        raise SystemExit("G0c failed; see numerics.json")


if __name__ == "__main__":
    main()
