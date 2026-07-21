"""Independent fp32 references and ANE-layout fp16 LFM2.5 blocks.

The reference modules preserve Hugging Face's linear-layer formulation. The
Core ML candidates use 1x1/depthwise Conv2d while keeping sequence on the last
axis. Comparing both in PyTorch before conversion distinguishes architecture
errors from converter/runtime drift.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .checkpoint import Lfm2Config


def _copy_parameter(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Copy ``source`` into ``destination`` using the destination dtype."""

    with torch.no_grad():
        destination.copy_(source.to(dtype=destination.dtype))


def _copy_linear_weight(destination: nn.Linear, source: torch.Tensor) -> None:
    """Copy a checkpoint linear matrix into an ``nn.Linear`` module."""

    _copy_parameter(destination.weight, source)


def _copy_conv2d_weight(destination: nn.Conv2d, source: torch.Tensor) -> None:
    """Copy a checkpoint linear or convolution tensor into ``nn.Conv2d``."""

    if source.ndim == 2:
        source = source[:, :, None, None]
    elif source.ndim == 3:
        source = source[:, :, None, :]
    _copy_parameter(destination.weight, source)


def rms_norm_reference(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Apply the checkpoint's RMSNorm equation in fp32 reference layout."""

    variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden_states.float() * torch.rsqrt(variance + epsilon)
    return normalized * weight.float()


def rms_norm_ane_layout(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Apply RMSNorm via LayerNorm while preserving ``[B,C,1,S]`` externally."""

    sequence_last = hidden_states.permute(0, 2, 3, 1)
    doubled = torch.cat((sequence_last, -sequence_last), dim=-1)
    normalized = functional.layer_norm(
        doubled,
        normalized_shape=(2 * hidden_states.shape[1],),
        weight=None,
        bias=None,
        eps=epsilon,
    )
    first_half, _ = torch.chunk(normalized, 2, dim=-1)
    weighted = first_half * weight.reshape(1, 1, 1, -1)
    return weighted.permute(0, 3, 1, 2)


def rms_norm_ane_explicit(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Apply canonical RMSNorm with fp32 accumulation and restored I/O dtype."""

    working = hidden_states.float()
    variance = working.pow(2).mean(dim=1, keepdim=True)
    normalized = working * torch.rsqrt(variance + epsilon)
    weighted = normalized * weight.float().reshape(1, -1, 1, 1)
    return weighted.to(hidden_states.dtype)


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the attention head dimension for RoPE."""

    midpoint = hidden_states.shape[-1] // 2
    first = hidden_states[..., :midpoint]
    second = hidden_states[..., midpoint:]
    return torch.cat((-second, first), dim=-1)


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply precomputed rotary embeddings to query and key tensors."""

    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return (
        query * cosine + rotate_half(query) * sine,
        key * cosine + rotate_half(key) * sine,
    )


def rope_tables(
    sequence_length: int,
    head_dim: int,
    theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic RoPE cosine/sine inputs for one prefill bucket."""

    positions = torch.arange(sequence_length, dtype=torch.float32)
    frequencies = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(positions, frequencies)
    doubled = torch.cat((angles, angles), dim=-1).unsqueeze(0)
    return doubled.cos().to(dtype), doubled.sin().to(dtype)


def causal_mask(
    sequence_length: int, negative_value: float, dtype: torch.dtype
) -> torch.Tensor:
    """Create an additive causal mask shaped ``[1,1,S,S]``."""

    mask = torch.full(
        (sequence_length, sequence_length), negative_value, dtype=dtype
    )
    mask = torch.triu(mask, diagonal=1)
    return mask.reshape(1, 1, sequence_length, sequence_length)


class FP32ANELayoutRMSNorm(nn.Module):
    """Named fp32 RMSNorm island with an explicit fp16 boundary."""

    def __init__(self, weight: torch.Tensor, epsilon: float) -> None:
        """Freeze one channel-axis RMSNorm weight and its checkpoint epsilon."""

        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("weight", weight.float())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize ``[B,C,1,S]`` in fp32 and return an fp16 activation."""

        working = hidden_states.float()
        variance = working.pow(2).mean(dim=1, keepdim=True)
        normalized = working * torch.rsqrt(variance + self.epsilon)
        weighted = normalized * self.weight.reshape(1, -1, 1, 1)
        return weighted.to(torch.float16)


class FP32NormalizedRoPE(nn.Module):
    """Named Q/K RMSNorm-plus-RoPE island with explicit fp16 outputs."""

    def __init__(
        self,
        query_weight: torch.Tensor,
        key_weight: torch.Tensor,
        epsilon: float,
    ) -> None:
        """Freeze the Q/K normalization weights and checkpoint epsilon."""

        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("query_weight", query_weight.float())
        self.register_buffer("key_weight", key_weight.float())

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize Q/K, apply RoPE in fp32, then return fp16 tensors."""

        query_working = query.float()
        query_variance = query_working.pow(2).mean(dim=-1, keepdim=True)
        query_normalized = query_working * torch.rsqrt(
            query_variance + self.epsilon
        )
        query_normalized = query_normalized * self.query_weight.reshape(
            1, 1, 1, -1
        )
        key_working = key.float()
        key_variance = key_working.pow(2).mean(dim=-1, keepdim=True)
        key_normalized = key_working * torch.rsqrt(key_variance + self.epsilon)
        key_normalized = key_normalized * self.key_weight.reshape(1, 1, 1, -1)
        rotated_query, rotated_key = apply_rope(
            query_normalized, key_normalized, cosine.float(), sine.float()
        )
        return rotated_query.to(torch.float16), rotated_key.to(torch.float16)


class ReferenceMLP(nn.Module):
    """Liquid's fp32 SwiGLU feed-forward sublayer."""

    def __init__(self, config: Lfm2Config, tensors: dict[str, torch.Tensor]) -> None:
        """Initialize the MLP from real checkpoint tensors."""

        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        _copy_linear_weight(self.w1, tensors["feed_forward.w1.weight"])
        _copy_linear_weight(self.w3, tensors["feed_forward.w3.weight"])
        _copy_linear_weight(self.w2, tensors["feed_forward.w2.weight"])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the fp32 SwiGLU equation."""

        return self.w2(functional.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class ANELayoutMLP(nn.Module):
    """The same SwiGLU sublayer expressed as 1x1 Conv2d operations."""

    def __init__(self, config: Lfm2Config, tensors: dict[str, torch.Tensor]) -> None:
        """Initialize fp16 Conv2d projections from real checkpoint matrices."""

        super().__init__()
        self.w1 = nn.Conv2d(
            config.hidden_size, config.intermediate_size, 1, bias=False, dtype=torch.float16
        )
        self.w3 = nn.Conv2d(
            config.hidden_size, config.intermediate_size, 1, bias=False, dtype=torch.float16
        )
        self.w2 = nn.Conv2d(
            config.intermediate_size, config.hidden_size, 1, bias=False, dtype=torch.float16
        )
        _copy_conv2d_weight(self.w1, tensors["feed_forward.w1.weight"])
        _copy_conv2d_weight(self.w3, tensors["feed_forward.w3.weight"])
        _copy_conv2d_weight(self.w2, tensors["feed_forward.w2.weight"])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the fp16 1x1-Conv2d SwiGLU equation."""

        return self.w2(functional.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class ReferenceConvBlock(nn.Module):
    """Independent fp32 reference for one full LFM2.5 short-conv block."""

    def __init__(self, config: Lfm2Config, tensors: dict[str, torch.Tensor]) -> None:
        """Initialize the reference block from real checkpoint tensors."""

        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.in_proj = nn.Linear(hidden, 3 * hidden, bias=config.conv_bias)
        self.out_proj = nn.Linear(hidden, hidden, bias=config.conv_bias)
        self.mlp = ReferenceMLP(config, tensors)
        self.register_buffer("operator_norm_weight", tensors["operator_norm.weight"].float())
        self.register_buffer("ffn_norm_weight", tensors["ffn_norm.weight"].float())
        self.register_buffer("conv_weight", tensors["conv.conv.weight"].float())
        _copy_linear_weight(self.in_proj, tensors["conv.in_proj.weight"])
        _copy_linear_weight(self.out_proj, tensors["conv.out_proj.weight"])

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run full-sequence causal conv and return hidden output plus decode state."""

        residual = hidden_states.float()
        normalized = rms_norm_reference(
            residual, self.operator_norm_weight, self.config.norm_eps
        )
        projected = self.in_proj(normalized).transpose(1, 2)
        gate_b, gate_c, values = torch.chunk(projected, 3, dim=1)
        gated_values = gate_b * values
        padded = functional.pad(gated_values, (self.config.conv_l_cache - 1, 0))
        convolved = functional.conv1d(
            padded,
            self.conv_weight,
            bias=None,
            groups=self.config.hidden_size,
        )
        operator_output = self.out_proj((gate_c * convolved).transpose(1, 2))
        hidden_states = residual + operator_output
        ffn_input = rms_norm_reference(
            hidden_states, self.ffn_norm_weight, self.config.norm_eps
        )
        hidden_states = hidden_states + self.mlp(ffn_input)
        state_width = self.config.conv_l_cache - 1
        return hidden_states, gated_values[..., -state_width:]


class CoreMLConvBlock(nn.Module):
    """Fp16 ANE-layout candidate for one full LFM2.5 short-conv block."""

    def __init__(
        self,
        config: Lfm2Config,
        tensors: dict[str, torch.Tensor],
        explicit_rms: bool = True,
    ) -> None:
        """Initialize the Core ML candidate from real checkpoint tensors."""

        super().__init__()
        self.config = config
        self.explicit_rms = explicit_rms
        hidden = config.hidden_size
        self.in_proj = nn.Conv2d(
            hidden, 3 * hidden, 1, bias=config.conv_bias, dtype=torch.float16
        )
        self.depthwise = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(1, config.conv_l_cache),
            groups=hidden,
            bias=config.conv_bias,
            dtype=torch.float16,
        )
        self.out_proj = nn.Conv2d(
            hidden, hidden, 1, bias=config.conv_bias, dtype=torch.float16
        )
        self.mlp = ANELayoutMLP(config, tensors)
        self.register_buffer(
            "operator_norm_weight", tensors["operator_norm.weight"].to(torch.float16)
        )
        self.register_buffer(
            "ffn_norm_weight", tensors["ffn_norm.weight"].to(torch.float16)
        )
        _copy_conv2d_weight(self.in_proj, tensors["conv.in_proj.weight"])
        _copy_conv2d_weight(self.depthwise, tensors["conv.conv.weight"])
        _copy_conv2d_weight(self.out_proj, tensors["conv.out_proj.weight"])

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the fp16 block on ``[B,C,1,S]`` and emit the final live state."""

        residual = hidden_states
        rms_norm = (
            rms_norm_ane_explicit if self.explicit_rms else rms_norm_ane_layout
        )
        normalized = rms_norm(
            residual, self.operator_norm_weight, self.config.norm_eps
        )
        gate_b, gate_c, values = torch.chunk(self.in_proj(normalized), 3, dim=1)
        gated_values = gate_b * values
        padded = functional.pad(
            gated_values, (self.config.conv_l_cache - 1, 0, 0, 0)
        )
        convolved = self.depthwise(padded)
        hidden_states = residual + self.out_proj(gate_c * convolved)
        ffn_input = rms_norm(
            hidden_states, self.ffn_norm_weight, self.config.norm_eps
        )
        hidden_states = hidden_states + self.mlp(ffn_input)
        state_width = self.config.conv_l_cache - 1
        conv_state = gated_values[:, :, 0, -state_width:]
        return hidden_states, conv_state


class ReferenceAttentionBlock(nn.Module):
    """Independent fp32 reference for one full LFM2.5 GQA block."""

    def __init__(self, config: Lfm2Config, tensors: dict[str, torch.Tensor]) -> None:
        """Initialize the GQA reference from real checkpoint tensors."""

        super().__init__()
        self.config = config
        hidden = config.hidden_size
        query_width = config.num_attention_heads * config.head_dim
        key_value_width = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(hidden, query_width, bias=False)
        self.k_proj = nn.Linear(hidden, key_value_width, bias=False)
        self.v_proj = nn.Linear(hidden, key_value_width, bias=False)
        self.out_proj = nn.Linear(query_width, hidden, bias=False)
        self.mlp = ReferenceMLP(config, tensors)
        self.register_buffer("operator_norm_weight", tensors["operator_norm.weight"].float())
        self.register_buffer("ffn_norm_weight", tensors["ffn_norm.weight"].float())
        self.register_buffer(
            "query_norm_weight", tensors["self_attn.q_layernorm.weight"].float()
        )
        self.register_buffer(
            "key_norm_weight", tensors["self_attn.k_layernorm.weight"].float()
        )
        _copy_linear_weight(self.q_proj, tensors["self_attn.q_proj.weight"])
        _copy_linear_weight(self.k_proj, tensors["self_attn.k_proj.weight"])
        _copy_linear_weight(self.v_proj, tensors["self_attn.v_proj.weight"])
        _copy_linear_weight(self.out_proj, tensors["self_attn.out_proj.weight"])

    def forward(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run full-sequence GQA and emit the prefill K/V cache seed."""

        config = self.config
        residual = hidden_states.float()
        normalized = rms_norm_reference(
            residual, self.operator_norm_weight, config.norm_eps
        )
        batch_size, sequence_length, _ = normalized.shape
        query = self.q_proj(normalized).reshape(
            batch_size, sequence_length, config.num_attention_heads, config.head_dim
        ).transpose(1, 2)
        key = self.k_proj(normalized).reshape(
            batch_size, sequence_length, config.num_key_value_heads, config.head_dim
        ).transpose(1, 2)
        value = self.v_proj(normalized).reshape(
            batch_size, sequence_length, config.num_key_value_heads, config.head_dim
        ).transpose(1, 2)
        query = rms_norm_reference(query, self.query_norm_weight, config.norm_eps)
        key = rms_norm_reference(key, self.key_norm_weight, config.norm_eps)
        query, key = apply_rope(query, key, cosine.float(), sine.float())
        repeat_count = config.num_attention_heads // config.num_key_value_heads
        expanded_key = torch.repeat_interleave(key, repeat_count, dim=1)
        expanded_value = torch.repeat_interleave(value, repeat_count, dim=1)
        scores = torch.matmul(query, expanded_key.transpose(-1, -2)) / math.sqrt(
            config.head_dim
        )
        probabilities = torch.softmax(scores + attention_mask.float(), dim=-1)
        attended = torch.matmul(probabilities, expanded_value)
        attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        hidden_states = residual + self.out_proj(attended)
        ffn_input = rms_norm_reference(
            hidden_states, self.ffn_norm_weight, config.norm_eps
        )
        return hidden_states + self.mlp(ffn_input), key, value


class CoreMLAttentionBlock(nn.Module):
    """Fp16 ANE-layout candidate for one full LFM2.5 GQA prefill block."""

    def __init__(self, config: Lfm2Config, tensors: dict[str, torch.Tensor]) -> None:
        """Initialize fp16 Conv2d projections from real checkpoint matrices."""

        super().__init__()
        self.config = config
        hidden = config.hidden_size
        query_width = config.num_attention_heads * config.head_dim
        key_value_width = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Conv2d(hidden, query_width, 1, bias=False, dtype=torch.float16)
        self.k_proj = nn.Conv2d(hidden, key_value_width, 1, bias=False, dtype=torch.float16)
        self.v_proj = nn.Conv2d(hidden, key_value_width, 1, bias=False, dtype=torch.float16)
        self.out_proj = nn.Conv2d(query_width, hidden, 1, bias=False, dtype=torch.float16)
        self.mlp = ANELayoutMLP(config, tensors)
        self.operator_norm = FP32ANELayoutRMSNorm(
            tensors["operator_norm.weight"], config.norm_eps
        )
        self.attention_positioning = FP32NormalizedRoPE(
            tensors["self_attn.q_layernorm.weight"],
            tensors["self_attn.k_layernorm.weight"],
            config.norm_eps,
        )
        self.ffn_norm = FP32ANELayoutRMSNorm(
            tensors["ffn_norm.weight"], config.norm_eps
        )
        self.register_buffer(
            "attention_scale",
            torch.tensor(1.0 / math.sqrt(config.head_dim), dtype=torch.float16),
        )
        _copy_conv2d_weight(self.q_proj, tensors["self_attn.q_proj.weight"])
        _copy_conv2d_weight(self.k_proj, tensors["self_attn.k_proj.weight"])
        _copy_conv2d_weight(self.v_proj, tensors["self_attn.v_proj.weight"])
        _copy_conv2d_weight(self.out_proj, tensors["self_attn.out_proj.weight"])

    def forward(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run fp16 GQA on ``[B,C,1,S]`` and emit K/V cache seeds."""

        config = self.config
        residual = hidden_states
        normalized = self.operator_norm(residual)
        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[-1]
        query = self.q_proj(normalized).reshape(
            batch_size, config.num_attention_heads, config.head_dim, sequence_length
        ).permute(0, 1, 3, 2)
        key = self.k_proj(normalized).reshape(
            batch_size, config.num_key_value_heads, config.head_dim, sequence_length
        ).permute(0, 1, 3, 2)
        value = self.v_proj(normalized).reshape(
            batch_size, config.num_key_value_heads, config.head_dim, sequence_length
        ).permute(0, 1, 3, 2)
        # These named submodules are the only mixed-precision islands. Their
        # explicit fp16 boundaries prevent promotion from leaking into scores,
        # cache outputs, residual arithmetic, or the MLP.
        query, key = self.attention_positioning(query, key, cosine, sine)
        repeat_count = config.num_attention_heads // config.num_key_value_heads
        expanded_key = torch.repeat_interleave(key, repeat_count, dim=1)
        expanded_value = torch.repeat_interleave(value, repeat_count, dim=1)
        scores = torch.matmul(query, expanded_key.transpose(-1, -2)) * self.attention_scale
        probabilities = torch.softmax(scores + attention_mask, dim=-1)
        attended = torch.matmul(probabilities, expanded_value)
        attended = attended.permute(0, 1, 3, 2).reshape(
            batch_size,
            config.num_attention_heads * config.head_dim,
            1,
            sequence_length,
        )
        hidden_states = residual + self.out_proj(attended)
        ffn_input = self.ffn_norm(hidden_states)
        return hidden_states + self.mlp(ffn_input), key, value
