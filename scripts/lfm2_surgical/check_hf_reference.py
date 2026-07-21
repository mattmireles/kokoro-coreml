#!/usr/bin/env python3
"""Validate the independent fp32 block oracle against Transformers 5.5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from packaging.version import Version
from transformers import AutoTokenizer, Lfm2Model, __version__ as transformers_version

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
    REAL_PROMPTS,
    validate_frozen_protocol,
)

REFERENCE_SEQUENCE_LENGTH = 128
REFERENCE_MAX_ABS_GATE = 1.0e-5


def parse_args() -> argparse.Namespace:
    """Parse the frozen checkpoint and output-manifest paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(DEFAULT_STAGE0_DIR) / "hf_reference.json",
    )
    return parser.parse_args()


def repeated_prompt_ids(checkpoint_dir: Path, sequence_length: int) -> torch.Tensor:
    """Tokenize one real prompt and repeat/truncate it to the frozen probe length."""

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    token_ids = tokenizer(REAL_PROMPTS[0], add_special_tokens=True).input_ids
    repeated = (token_ids * ((sequence_length + len(token_ids) - 1) // len(token_ids)))[
        :sequence_length
    ]
    return torch.tensor(repeated, dtype=torch.long).unsqueeze(0)


def maximum_absolute_error(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the maximum absolute difference between equal-shaped tensors."""

    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    return float(torch.max(torch.abs(left.float() - right.float())).item())


def run_reference_check(checkpoint_dir: Path) -> dict:
    """Compare the independent first-three-layer oracle with official HF LFM2."""

    if Version(transformers_version) < Version("5.5.0"):
        raise RuntimeError(
            "official LFM2 validation requires Transformers >= 5.5.0; "
            f"found {transformers_version}"
        )
    config = load_config(checkpoint_dir)
    model = Lfm2Model.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    input_ids = repeated_prompt_ids(checkpoint_dir, REFERENCE_SEQUENCE_LENGTH)
    hidden = load_embedding(checkpoint_dir).float()[input_ids]
    position_ids = torch.arange(REFERENCE_SEQUENCE_LENGTH).unsqueeze(0)
    cosine, sine = rope_tables(
        REFERENCE_SEQUENCE_LENGTH,
        config.head_dim,
        config.rope_theta,
        torch.float32,
    )
    mask = causal_mask(
        REFERENCE_SEQUENCE_LENGTH, CAUSAL_MASK_NEGATIVE, torch.float32
    )
    layer_zero = ReferenceConvBlock(
        config, load_block_tensors(checkpoint_dir, CONV_LAYER_INDEX, "conv")
    ).eval()
    layer_one = ReferenceConvBlock(
        config,
        load_block_tensors(checkpoint_dir, GQA_INPUT_LAYER_INDEX, "conv"),
    ).eval()
    layer_two = ReferenceAttentionBlock(
        config,
        load_block_tensors(checkpoint_dir, GQA_LAYER_INDEX, "full_attention"),
    ).eval()

    with torch.inference_mode():
        official = model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        ).hidden_states
        reference_zero, _ = layer_zero(hidden)
        reference_one, _ = layer_one(reference_zero)
        reference_two, _, _ = layer_two(
            reference_one, cosine, sine, mask
        )

    errors = {
        "embedding": maximum_absolute_error(official[0], hidden),
        "layer_0_conv": maximum_absolute_error(official[1], reference_zero),
        "layer_1_conv": maximum_absolute_error(official[2], reference_one),
        "layer_2_gqa": maximum_absolute_error(official[3], reference_two),
    }
    failures = {
        name: error for name, error in errors.items() if error > REFERENCE_MAX_ABS_GATE
    }
    return {
        "checkpoint": checkpoint_identity(checkpoint_dir),
        "transformers": transformers_version,
        "sequence_length": REFERENCE_SEQUENCE_LENGTH,
        "max_abs_gate": REFERENCE_MAX_ABS_GATE,
        "max_abs": errors,
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    """Run the official-reference check and write its durable manifest."""

    args = parse_args()
    validate_frozen_protocol()
    report = run_reference_check(args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("official Hugging Face reference parity failed")


if __name__ == "__main__":
    main()
