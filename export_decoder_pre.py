#!/usr/bin/env python3
"""Export the decoder pre-processing stack to CoreML (.mlpackage).

The decoder pre-processing runs F0_conv + N_conv + encode + decode blocks
to produce x_pre from (asr, F0, N, ref_s). This is the compute-heavy part
of ``build_decoder_har_post_inputs_np()`` that currently runs on PyTorch CPU.

Architecture:
    - F0_conv: Conv1d(1, 1, k=3, s=2, p=1) -> downsamples F0 by ~2x
    - N_conv:  Conv1d(1, 1, k=3, s=2, p=1) -> downsamples N by ~2x
    - cat([asr, F0, N], dim=1) -> (1, 514, T)
    - encode: AdainResBlk1d -> initial encoding with style conditioning
    - asr_res: Conv1d -> residual projection
    - decode: list of AdainResBlk1d blocks (some with upsample)
    - Returns: x_pre (1, C, T')

Usage::

    uv run python export_decoder_pre.py                    # export for 3s + 10s
    uv run python export_decoder_pre.py --buckets 3 10     # specific buckets
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent


def _load_kmodel():
    """Load KModel."""
    from kokoro._export_utils import load_kokoro_for_export
    _, _, kokoro_model = load_kokoro_for_export(suffix="_decoder_pre")
    KModel = kokoro_model.KModel
    cfg = _ROOT / "checkpoints/config.json"
    ckpt = _ROOT / "checkpoints/kokoro-v1_0.pth"
    try:
        if cfg.is_file() and ckpt.is_file():
            return KModel(config=str(cfg), model=str(ckpt), disable_complex=True)
        elif cfg.is_file():
            return KModel(config=str(cfg), disable_complex=True)
    except OSError:
        pass
    return KModel(disable_complex=True)


def _remove_training_ops(model: nn.Module) -> None:
    """Recursively replace Dropout with Identity."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Dropout):
            parts = name.rsplit(".", 1)
            parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
            setattr(parent, parts[-1], nn.Identity())
        elif isinstance(module, (nn.BatchNorm1d, nn.LSTM)):
            module.training = False


class DecoderPreWrapper(nn.Module):
    """Wraps the decoder pre-processing for CoreML export.

    Runs F0_conv + N_conv + encode + decode blocks to produce x_pre.

    ``mask`` is an optional (B, 1, frame_count) validity mask threaded through
    every ``AdainResBlk1d`` call; ``decode[-1]`` upsamples 2x, so its ``norm2``
    sees the 2x-repeated mask. On the exported package ``mask`` carries an
    all-ones ``default_value``, which keeps it optional for existing consumers.
    All-ones means full fill: a caller that pads must pass the real mask.
    """

    def __init__(self, decoder):
        super().__init__()
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv
        self.encode_block = decoder.encode
        self.asr_res = decoder.asr_res
        self.decode = decoder.decode

    def forward(
        self,
        asr: torch.FloatTensor,   # (1, 512, frame_count)
        f0: torch.FloatTensor,    # (1, 1, full_f0_len)
        n_input: torch.FloatTensor,     # (1, 1, full_f0_len)
        ref_s: torch.FloatTensor, # (1, 256)
        mask: torch.Tensor | None = None,  # (1, 1, frame_count) float, optional
    ) -> torch.Tensor:
        s = ref_s[:, :128]  # baseline embedding

        F0 = self.F0_conv(f0)
        N = self.N_conv(n_input)

        x = torch.cat([asr, F0, N], dim=1)
        x = self.encode_block(x, s, m=mask, m_up=mask)
        asr_res = self.asr_res(asr)

        mask_up = mask.repeat_interleave(2, dim=2) if mask is not None else None

        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], dim=1)
            if block.upsample_type != "none":
                x = block(x, s, m=mask, m_up=mask_up)
                res = False
            else:
                x = block(x, s, m=mask, m_up=mask)

        return x


def export_decoder_pre(bucket_sec: int, output_dir: Path | None = None) -> Path | None:
    """Export DecoderPre for a specific bucket."""
    import coremltools as ct
    from kokoro.conv_length import conv1d_output_length_from_module

    if output_dir is None:
        output_dir = _ROOT / "coreml"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading KModel...")
    kmodel = _load_kmodel()
    dec = kmodel.decoder
    gen = dec.generator

    # Compute bucket geometry
    f0_samples_per_step = int(round(float(gen.f0_upsamp.scale_factor)))
    bucket_samples = bucket_sec * 24000
    full_f0_len = int(round(bucket_samples / float(f0_samples_per_step)))
    frame_count = conv1d_output_length_from_module(full_f0_len, dec.F0_conv)
    print(f"Bucket {bucket_sec}s: full_f0_len={full_f0_len}, frame_count={frame_count}")

    wrapper = DecoderPreWrapper(dec)
    wrapper.train(False)
    _remove_training_ops(wrapper)
    for m in wrapper.modules():
        m.train(False)

    # Dummy inputs; the mask traces as all-ones (full fill).
    asr_dummy = torch.randn(1, 512, frame_count, dtype=torch.float32)
    f0_dummy = torch.randn(1, 1, full_f0_len, dtype=torch.float32)
    n_dummy = torch.randn(1, 1, full_f0_len, dtype=torch.float32)
    ref_s_dummy = torch.randn(1, 256, dtype=torch.float32)
    mask_dummy = torch.ones(1, 1, frame_count, dtype=torch.float32)

    # Test forward
    with torch.no_grad():
        x_pre_test = wrapper(asr_dummy, f0_dummy, n_dummy, ref_s_dummy, mask_dummy)
        print(f"Forward pass OK: x_pre {x_pre_test.shape}")

    # Trace
    print("Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (asr_dummy, f0_dummy, n_dummy, ref_s_dummy, mask_dummy), strict=False)

    # Convert
    print("Converting to CoreML...")
    try:
        ml = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="asr", shape=(1, 512, frame_count), dtype=np.float32),
                ct.TensorType(name="f0", shape=(1, 1, full_f0_len), dtype=np.float32),
                ct.TensorType(name="n_input", shape=(1, 1, full_f0_len), dtype=np.float32),
                ct.TensorType(name="ref_s", shape=(1, 256), dtype=np.float32),
                # The all-ones default keeps `mask` optional for existing consumers.
                # It means full fill: a caller that pads and omits the mask gets
                # the padding contamination back, silently.
                ct.TensorType(name="mask", shape=(1, 1, frame_count), dtype=np.float32,
                              default_value=np.ones((1, 1, frame_count), dtype=np.float32)),
            ],
            outputs=[
                ct.TensorType(name="x_pre"),
            ],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS12,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL,
        )
    except Exception as e:
        print(f"\nCoreML conversion FAILED: {e}")
        print("This is the known AdaIN export risk.")
        return None

    out_path = output_dir / f"kokoro_decoder_pre_{bucket_sec}s.mlpackage"
    ml.save(str(out_path))
    print(f"Saved: {out_path}")

    # Numeric validation
    print("\nNumeric validation...")
    validate_decoder_pre(traced, ml, frame_count, full_f0_len)

    return out_path


def validate_decoder_pre(traced, ml_model, frame_count: int, full_f0_len: int, n_tests: int = 3) -> None:
    """Compare PyTorch traced vs CoreML predict."""
    torch.manual_seed(42)
    correlations = []

    for i in range(n_tests):
        asr = torch.randn(1, 512, frame_count, dtype=torch.float32)
        f0 = torch.randn(1, 1, full_f0_len, dtype=torch.float32)
        n_in = torch.randn(1, 1, full_f0_len, dtype=torch.float32)
        ref_s = torch.randn(1, 256, dtype=torch.float32)
        # Validated at full fill (mask=ones), where masking must be a no-op.
        mask = torch.ones(1, 1, frame_count, dtype=torch.float32)

        with torch.no_grad():
            pt_out = traced(asr, f0, n_in, ref_s, mask)
        pt_np = pt_out.numpy().flatten()

        coreml_out = ml_model.predict({
            "asr": asr.numpy(),
            "f0": f0.numpy(),
            "n_input": n_in.numpy(),
            "ref_s": ref_s.numpy(),
            "mask": mask.numpy(),
        })
        cm_np = np.asarray(coreml_out["x_pre"]).flatten()

        corr = float(np.corrcoef(pt_np, cm_np)[0, 1])
        correlations.append(corr)
        print(f"  Test {i}: corr={corr:.6f}")

    mean_corr = np.mean(correlations)
    print(f"\n  Mean correlation: {mean_corr:.6f}")
    if mean_corr < 0.99:
        print(f"  WARNING: correlation {mean_corr:.4f} < 0.99")
    else:
        print(f"  PASS: correlation > 0.99")


def main():
    parser = argparse.ArgumentParser(description="Export DecoderPre to CoreML")
    parser.add_argument("--buckets", type=int, nargs="+", default=[3, 10])
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None

    for sec in args.buckets:
        print(f"\n{'='*60}")
        print(f"Exporting DecoderPre for {sec}s bucket")
        print(f"{'='*60}")
        result = export_decoder_pre(sec, output_dir=out_dir)
        if result is None:
            print(f"\nDecoderPre {sec}s export failed. Keeping PyTorch bridge.")


if __name__ == "__main__":
    main()
