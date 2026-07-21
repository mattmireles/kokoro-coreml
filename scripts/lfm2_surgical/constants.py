"""Frozen scientific constants for plan 010.

This module is the executable copy of the pre-registered variables in
``README/Plans/010-lfm2-surgical-prefill-plan.md``. Changing a value here is a
protocol change and requires a plan/spec version bump.
"""

from __future__ import annotations

CHECKPOINT_REPO = "LiquidAI/LFM2.5-350M"
CHECKPOINT_REVISION = "b9d6e4e2d75f440b12a2b4d731c808004ecbbd89"
CHECKPOINT_MODEL_SHA256 = "1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e"
CHECKPOINT_CONFIG_SHA256 = "720b43d6ddc2ed25be23eed355aefcf342434a176dedad23dbe0a5e3ac24bbb8"
CHECKPOINT_TOKENIZER_SHA256 = "df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd"
PREFILL_BUCKETS = (128, 256, 512, 1024, 2048)
CONV_LAYER_INDEX = 0
GQA_INPUT_LAYER_INDEX = 1
GQA_LAYER_INDEX = 2
EXPECTED_LAYER_TYPES = (
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
)
NUMERICS_PROMPT_COUNT = 32
NUMERICS_MAX_ABS_GATE = 1.0e-2
CAUSAL_MASK_NEGATIVE = -65504.0
DEFAULT_CHECKPOINT_DIR = "outputs/lfm2_surgical/hf_model"
DEFAULT_STAGE0_DIR = "outputs/lfm2_surgical/stage0"

REAL_PROMPTS = (
    "Explain why fixed tensor shapes can help a compiler target an accelerator.",
    "Write a concise comparison of convolution and attention for sequence models.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Summarize the tradeoff between latency, throughput, and energy on a phone.",
    "What evidence would falsify a claim that an operation ran on the neural engine?",
    "Describe grouped-query attention to an engineer who already understands transformers.",
    "Give three failure modes of numerical validation in mixed-precision inference.",
    "A spacecraft receives a weak signal from a distant probe and must decode it reliably.",
    "Translate this engineering principle into plain language: measure before optimizing.",
    "Why can a faster subgraph make an end-to-end pipeline slower after decomposition?",
    "A red bicycle leaned against the old brick wall throughout the afternoon rain.",
    "List the assumptions behind comparing energy measurements on two hardware targets.",
    "How should cold compilation time be separated from warmed inference latency?",
    "Write a short dialogue between a compiler engineer and a model researcher.",
    "What is the purpose of an inversion control in a heterogeneous-compute experiment?",
    "Describe a robust test for checking whether two model outputs preserve ranking.",
    "The library opened at dawn, and the first visitor asked for a book about astronomy.",
    "Explain why a growing key-value cache changes the economics of autoregressive decode.",
    "Give a practical reason to prefer a small number of static buckets over a range shape.",
    "What should a negative experimental result contain so another team can reproduce it?",
    "A chef adjusted the recipe one ingredient at a time to identify the source of bitterness.",
    "Compare a permission to use hardware with proof that the hardware actually executed work.",
    "Explain the difference between conversion success and runtime placement success.",
    "Write two sentences about causal depthwise convolution in a language model block.",
    "Why is a real trained checkpoint necessary for a numerical and performance experiment?",
    "The weather station recorded temperature, pressure, wind speed, and battery voltage.",
    "Describe how residual connections affect the interpretation of a block-output error.",
    "What does it mean for an experiment to preregister its success and kill criteria?",
    "A small robot crossed the warehouse while avoiding people, pallets, and closed doors.",
    "Explain why tensor layout changes can cost more than the arithmetic they accelerate.",
    "Give a debugging ladder for an unsupported operation during PyTorch to Core ML conversion.",
    "State the simplest next experiment when a hybrid conv-attention model has unknown placement.",
)


def validate_frozen_protocol() -> None:
    """Raise when the executable constants drift from the pre-registered plan."""

    if len(REAL_PROMPTS) != NUMERICS_PROMPT_COUNT:
        raise ValueError(
            f"expected {NUMERICS_PROMPT_COUNT} real prompts, found {len(REAL_PROMPTS)}"
        )
    if EXPECTED_LAYER_TYPES[CONV_LAYER_INDEX] != "conv":
        raise ValueError("the frozen conv layer index no longer selects a conv block")
    if EXPECTED_LAYER_TYPES[GQA_LAYER_INDEX] != "full_attention":
        raise ValueError("the frozen GQA layer index no longer selects an attention block")
