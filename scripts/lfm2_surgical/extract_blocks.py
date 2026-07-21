#!/usr/bin/env python3
"""Load the real LFM2.5 checkpoint and instantiate the Stage 0 blocks.

This command is intentionally a cheap pre-export gate. It validates checkpoint
identity, layer classes, weight names, parameter counts, and a small real-token
PyTorch parity probe before ``export_blocks.py`` pays conversion cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lfm2_surgical.blocks import (
    CoreMLAttentionBlock,
    CoreMLConvBlock,
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

PROBE_SEQUENCE_LENGTH = 128


def parse_args() -> argparse.Namespace:
    """Parse checkpoint and manifest paths for the extraction gate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument(
        "--out", type=Path, default=Path(DEFAULT_STAGE0_DIR) / "extraction_manifest.json"
    )
    return parser.parse_args()


def load_tokenizer(checkpoint_dir: Path) -> PreTrainedTokenizerFast:
    """Load the real tokenizer JSON without depending on Transformers 5 metadata."""

    return PreTrainedTokenizerFast(
        tokenizer_file=str(checkpoint_dir / "tokenizer.json"),
        bos_token="<|startoftext|>",
        eos_token="<|im_end|>",
        pad_token="<|pad|>",
    )


def real_prompt_hidden_states(
    checkpoint_dir: Path, sequence_length: int
) -> torch.Tensor:
    """Turn a real natural-language prompt into trained embedding activations."""

    tokenizer = load_tokenizer(checkpoint_dir)
    token_ids = tokenizer(REAL_PROMPTS[0], add_special_tokens=True).input_ids
    if not token_ids:
        raise ValueError("real prompt tokenization produced no token IDs")
    repeated = (token_ids * ((sequence_length + len(token_ids) - 1) // len(token_ids)))[
        :sequence_length
    ]
    embedding = load_embedding(checkpoint_dir).float()
    return embedding[torch.tensor(repeated, dtype=torch.long)].unsqueeze(0)


def maximum_absolute_error(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the largest absolute difference between identically shaped tensors."""

    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    return float(torch.max(torch.abs(left.float() - right.float())).item())


def parameter_count(module: torch.nn.Module) -> int:
    """Count all learned parameters in ``module``."""

    return sum(parameter.numel() for parameter in module.parameters())


def run_probe(checkpoint_dir: Path) -> dict:
    """Instantiate both formulations and prove their small-prompt PyTorch parity."""

    config = load_config(checkpoint_dir)
    conv_tensors = load_block_tensors(checkpoint_dir, CONV_LAYER_INDEX, "conv")
    gqa_tensors = load_block_tensors(
        checkpoint_dir, GQA_LAYER_INDEX, "full_attention"
    )
    reference_conv = ReferenceConvBlock(config, conv_tensors).eval()
    coreml_conv = CoreMLConvBlock(config, conv_tensors).eval()
    gqa_input_reference = ReferenceConvBlock(
        config, load_block_tensors(checkpoint_dir, GQA_INPUT_LAYER_INDEX, "conv")
    ).eval()
    reference_gqa = ReferenceAttentionBlock(config, gqa_tensors).eval()
    coreml_gqa = CoreMLAttentionBlock(config, gqa_tensors).eval()

    hidden = real_prompt_hidden_states(checkpoint_dir, PROBE_SEQUENCE_LENGTH)
    cosine, sine = rope_tables(
        PROBE_SEQUENCE_LENGTH,
        config.head_dim,
        config.rope_theta,
        torch.float32,
    )
    mask = causal_mask(PROBE_SEQUENCE_LENGTH, CAUSAL_MASK_NEGATIVE, torch.float32)
    hidden_ane = hidden.to(torch.float16).permute(0, 2, 1).unsqueeze(2)

    with torch.inference_mode():
        conv_reference_output, conv_reference_state = reference_conv(hidden)
        conv_candidate_output, conv_candidate_state = coreml_conv(hidden_ane)
        gqa_input, _ = gqa_input_reference(conv_reference_output)
        gqa_input_ane = gqa_input.to(torch.float16).permute(0, 2, 1).unsqueeze(2)
        gqa_reference_output, gqa_reference_key, gqa_reference_value = reference_gqa(
            gqa_input, cosine, sine, mask
        )
        gqa_candidate_output, gqa_candidate_key, gqa_candidate_value = coreml_gqa(
            gqa_input_ane,
            cosine,
            sine,
            mask.to(torch.float16),
        )

    return {
        "checkpoint": checkpoint_identity(checkpoint_dir),
        "checkpoint_layer_types": list(config.layer_types),
        "conv": {
            "layer_index": CONV_LAYER_INDEX,
            "reference_parameters": parameter_count(reference_conv),
            "candidate_parameters": parameter_count(coreml_conv),
            "hidden_max_abs": maximum_absolute_error(
                conv_reference_output,
                conv_candidate_output.squeeze(2).permute(0, 2, 1),
            ),
            "state_max_abs": maximum_absolute_error(
                conv_reference_state, conv_candidate_state
            ),
        },
        "gqa": {
            "layer_index": GQA_LAYER_INDEX,
            "reference_parameters": parameter_count(reference_gqa),
            "candidate_parameters": parameter_count(coreml_gqa),
            "hidden_max_abs": maximum_absolute_error(
                gqa_reference_output,
                gqa_candidate_output.squeeze(2).permute(0, 2, 1),
            ),
            "key_max_abs": maximum_absolute_error(
                gqa_reference_key, gqa_candidate_key
            ),
            "value_max_abs": maximum_absolute_error(
                gqa_reference_value, gqa_candidate_value
            ),
        },
    }


def main() -> None:
    """Run the real-checkpoint extraction probe and write its JSON manifest."""

    args = parse_args()
    validate_frozen_protocol()
    manifest = run_probe(args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
