"""Real-checkpoint loading and validation for LFM2.5 surgical blocks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open

from .constants import (
    CHECKPOINT_CONFIG_SHA256,
    CHECKPOINT_MODEL_SHA256,
    CHECKPOINT_REPO,
    CHECKPOINT_REVISION,
    CHECKPOINT_TOKENIZER_SHA256,
    EXPECTED_LAYER_TYPES,
)


@dataclass(frozen=True)
class Lfm2Config:
    """The checkpoint fields that define the isolated block graphs."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    norm_eps: float
    conv_l_cache: int
    conv_bias: bool
    rope_theta: float
    vocab_size: int
    layer_types: tuple[str, ...]


@lru_cache(maxsize=4)
def checkpoint_identity(checkpoint_dir: Path) -> dict[str, str]:
    """Verify and return the frozen revision plus graph-defining file digests."""

    resolved_directory = checkpoint_dir.resolve()
    expected_digests = {
        "model.safetensors": CHECKPOINT_MODEL_SHA256,
        "config.json": CHECKPOINT_CONFIG_SHA256,
        "tokenizer.json": CHECKPOINT_TOKENIZER_SHA256,
    }
    observed_digests: dict[str, str] = {}
    for filename, expected_digest in expected_digests.items():
        artifact_path = resolved_directory / filename
        metadata_path = (
            resolved_directory / f".cache/huggingface/download/{filename}.metadata"
        )
        if not artifact_path.is_file():
            raise FileNotFoundError(f"frozen checkpoint artifact missing: {artifact_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Hugging Face revision metadata missing: {metadata_path}"
            )
        metadata_lines = metadata_path.read_text(encoding="utf-8").splitlines()
        revision = metadata_lines[0] if metadata_lines else ""
        if revision != CHECKPOINT_REVISION:
            raise ValueError(
                f"{filename} revision changed; expected {CHECKPOINT_REVISION}, "
                f"found {revision or '<empty>'}"
            )
        with artifact_path.open("rb") as artifact_file:
            digest = hashlib.file_digest(artifact_file, "sha256").hexdigest()
        if digest != expected_digest:
            raise ValueError(
                f"{filename} digest changed; expected {expected_digest}, found {digest}"
            )
        observed_digests[filename] = digest
    return {
        "repository": CHECKPOINT_REPO,
        "revision": CHECKPOINT_REVISION,
        "model_sha256": observed_digests["model.safetensors"],
        "config_sha256": observed_digests["config.json"],
        "tokenizer_sha256": observed_digests["tokenizer.json"],
    }


def effective_intermediate_size(config: dict) -> int:
    """Return the trained SwiGLU width after Liquid's automatic adjustment."""

    intermediate_size = int(config.get("intermediate_size") or config["block_ff_dim"])
    if not config.get("block_auto_adjust_ff_dim", True):
        return intermediate_size
    intermediate_size = int(2 * intermediate_size / 3)
    intermediate_size = int(
        float(config.get("block_ffn_dim_multiplier", 1.0)) * intermediate_size
    )
    multiple = int(config.get("block_multiple_of", 256))
    return multiple * ((intermediate_size + multiple - 1) // multiple)


def load_config(checkpoint_dir: Path) -> Lfm2Config:
    """Load and validate the frozen 350M configuration from ``checkpoint_dir``."""

    checkpoint_identity(checkpoint_dir)
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"real checkpoint config missing: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    layer_types = tuple(raw["layer_types"])
    if layer_types != EXPECTED_LAYER_TYPES:
        raise ValueError(
            "checkpoint layer_types changed; refusing to move the preregistered target: "
            f"{layer_types}"
        )
    head_dim = int(
        raw.get("head_dim")
        or int(raw["hidden_size"]) // int(raw["num_attention_heads"])
    )
    rope_parameters = raw.get("rope_parameters") or {}
    return Lfm2Config(
        hidden_size=int(raw["hidden_size"]),
        intermediate_size=effective_intermediate_size(raw),
        num_attention_heads=int(raw["num_attention_heads"]),
        num_key_value_heads=int(raw["num_key_value_heads"]),
        head_dim=head_dim,
        norm_eps=float(raw.get("norm_eps", 1.0e-5)),
        conv_l_cache=int(raw["conv_L_cache"]),
        conv_bias=bool(raw.get("conv_bias", False)),
        rope_theta=float(rope_parameters.get("rope_theta", 1_000_000.0)),
        vocab_size=int(raw["vocab_size"]),
        layer_types=layer_types,
    )


def load_tensors(checkpoint_dir: Path, names: set[str]) -> dict[str, torch.Tensor]:
    """Read only ``names`` from the real safetensors checkpoint into CPU memory."""

    checkpoint_identity(checkpoint_dir)
    weights_path = checkpoint_dir / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"real checkpoint weights missing: {weights_path}")
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        missing = names - available
        if missing:
            raise KeyError(f"checkpoint is missing tensors: {sorted(missing)}")
        for name in sorted(names):
            tensors[name] = handle.get_tensor(name)
    return tensors


def block_tensor_names(layer_index: int, layer_type: str) -> set[str]:
    """Return the exact checkpoint tensor names required by one decoder block."""

    prefix = f"model.layers.{layer_index}"
    names = {
        f"{prefix}.operator_norm.weight",
        f"{prefix}.ffn_norm.weight",
        f"{prefix}.feed_forward.w1.weight",
        f"{prefix}.feed_forward.w2.weight",
        f"{prefix}.feed_forward.w3.weight",
    }
    if layer_type == "conv":
        names.update(
            {
                f"{prefix}.conv.in_proj.weight",
                f"{prefix}.conv.conv.weight",
                f"{prefix}.conv.out_proj.weight",
            }
        )
    elif layer_type == "full_attention":
        names.update(
            {
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.out_proj.weight",
                f"{prefix}.self_attn.q_layernorm.weight",
                f"{prefix}.self_attn.k_layernorm.weight",
            }
        )
    else:
        raise ValueError(f"unsupported layer type: {layer_type}")
    return names


def load_block_tensors(
    checkpoint_dir: Path, layer_index: int, layer_type: str
) -> dict[str, torch.Tensor]:
    """Load one real decoder block and strip its common checkpoint prefix."""

    prefix = f"model.layers.{layer_index}."
    loaded = load_tensors(checkpoint_dir, block_tensor_names(layer_index, layer_type))
    return {name.removeprefix(prefix): tensor for name, tensor in loaded.items()}


def load_embedding(checkpoint_dir: Path) -> torch.Tensor:
    """Load the trained token embedding used to turn real prompts into block inputs."""

    name = "model.embed_tokens.weight"
    return load_tensors(checkpoint_dir, {name})[name]
