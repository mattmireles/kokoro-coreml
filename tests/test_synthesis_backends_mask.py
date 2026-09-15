"""Regression tests for mask geometry in the Python Decoder-HAR bridge."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from kokoro.synthesis_backends import build_decoder_har_post_inputs_np


class _MaskCapture(nn.Module):
    def __init__(self, upsample: bool = False):
        super().__init__()
        self.upsample_type = "2x" if upsample else "none"
        self.seen: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, x, _style, m=None, m_up=None):
        assert m is not None and m_up is not None
        self.seen = (m.detach().clone(), m_up.detach().clone())
        return x.repeat_interleave(2, dim=2) if self.upsample_type != "none" else x


class _F0Upsample(nn.Module):
    scale_factor = 24_000

    def forward(self, x):
        return x


class _Source(nn.Module):
    def forward(self, f0):
        return torch.zeros_like(f0), None, None


class _STFT:
    @staticmethod
    def transform(source):
        value = source.unsqueeze(1)
        return value, value


def test_decoder_har_builder_masks_decoder_and_doubles_generator_validity():
    encode = _MaskCapture()
    decode = _MaskCapture(upsample=True)
    conv = nn.Conv1d(1, 1, kernel_size=1, bias=False)
    generator = SimpleNamespace(f0_upsamp=_F0Upsample(), m_source=_Source(), stft=_STFT())
    decoder = SimpleNamespace(
        generator=generator,
        F0_conv=conv,
        N_conv=conv,
        encode=encode,
        asr_res=nn.Identity(),
        decode=[decode],
    )
    inputs = {
        "asr": np.zeros((1, 512, 2), dtype=np.float32),
        "f0_curve": np.zeros((1, 2), dtype=np.float32),
        "n": np.zeros((1, 2), dtype=np.float32),
        "ref_s": np.zeros((1, 256), dtype=np.float32),
    }

    _, _, _, _, frame_count, generator_mask = build_decoder_har_post_inputs_np(
        decoder, inputs, sec=3, asr_len=6, har_t=3, warn_geometry=False
    )

    assert frame_count == 3
    assert encode.seen is not None and decode.seen is not None
    assert encode.seen[0].flatten().tolist() == [1, 1, 0]
    assert decode.seen[1].flatten().tolist() == [1, 1, 1, 1, 0, 0]
    assert generator_mask.flatten().tolist() == [1, 1, 1, 1, 0, 0]


def test_all_decoder_har_builder_calls_unpack_the_mask():
    """Changing the shared builder arity must not strand benchmark scripts."""
    root = Path(__file__).resolve().parents[1]
    assignments: list[tuple[Path, int]] = []
    for path in [root / "kokoro" / "synthesis_backends.py", *(root / "scripts").glob("*.py")]:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            function = node.value.func
            if isinstance(function, ast.Name) and function.id == "build_decoder_har_post_inputs_np":
                target = node.targets[0]
                assert isinstance(target, ast.Tuple), f"{path}:{node.lineno} must unpack the builder result"
                assignments.append((path, len(target.elts)))

    assert assignments, "no build_decoder_har_post_inputs_np callers found"
    assert all(count == 6 for _, count in assignments), assignments
