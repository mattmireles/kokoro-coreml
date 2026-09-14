#!/usr/bin/env python3
"""Export the F0Ntrain prosody prediction sub-model to CoreML (.mlpackage).

F0Ntrain is the only neural-network call between the Duration model output and
the decoder pre-processing input.  It predicts pitch (F0) and noise (N) curves
from aligned duration features and a style embedding.

Architecture:
    - Shared bidirectional LSTM (input: d_hid + style_dim = 640, output: d_hid = 512)
    - Two parallel branches (F0 and N), each with:
      - 3 x AdainResBlk1d (style-conditioned, one with 2x upsample)
      - 1 x Conv1d(d_hid//2, 1, 1) projection
    - Output shape per branch: (1, 2 * T_frames) due to the 2x upsample

Inputs:
    en:  (1, 640, T)  float32  -- aligned duration features from d @ alignment
    s:   (1, 128)     float32  -- style embedding (ref_s[:, 128:])

Outputs:
    F0_pred:  (1, 2*T)  float32  -- pitch contour
    N_pred:   (1, 2*T)  float32  -- noise contour

Usage::

    uv run python export_f0ntrain.py                    # export for 3s + 10s buckets
    uv run python export_f0ntrain.py --t-frames 400     # custom fixed T dimension
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from export_synth.wrappers import MaskedBidirectionalLSTM

_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_kmodel():
    """Load KModel with the same logic as export_duration.py."""
    from kokoro._export_utils import load_kokoro_for_export

    _, _, kokoro_model = load_kokoro_for_export(suffix="_f0ntrain")
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
    """Recursively replace Dropout with Identity (same pattern as export_duration.py)."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Dropout):
            parts = name.rsplit(".", 1)
            parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
            child_name = parts[-1]
            setattr(parent, child_name, nn.Identity())
        elif isinstance(module, (nn.BatchNorm1d, nn.LSTM)):
            module.eval()


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class F0NtrainWrapper(nn.Module):
    """``predictor.F0Ntrain`` with explicit tensor I/O and bucket masking.

    The shared BiLSTM runs as ``MaskedBidirectionalLSTM``, so its backward
    direction starts at the last valid frame instead of walking the padding,
    and each ``AdainResBlk1d`` gets the mask at its own resolution (each branch
    is [no-upsample, 2x-upsample, no-upsample]). ``mask=None`` means full fill.
    """

    def __init__(self, predictor):
        super().__init__()
        self.shared = MaskedBidirectionalLSTM(predictor.shared)
        self.F0 = predictor.F0
        self.N = predictor.N
        self.F0_proj = predictor.F0_proj
        self.N_proj = predictor.N_proj

    def forward(self, en: torch.FloatTensor, s: torch.FloatTensor, mask: torch.Tensor | None = None):
        """en: (1, 640, T); s: (1, 128); mask: (1, 1, T) float, 1.0 on valid frames.

        Returns (F0_pred, N_pred), each (1, 2*T).
        """
        if mask is None:
            mask = en.new_ones(en.shape[0], 1, en.shape[-1])
        x = self.shared(en.transpose(-1, -2), mask.squeeze(1))
        mask_up = mask.repeat_interleave(2, dim=2)
        per_block = ((mask, mask), (mask, mask_up), (mask_up, mask_up))
        F0 = x.transpose(-1, -2)
        for block, (m_in, m_out) in zip(self.F0, per_block):
            F0 = block(F0, s, m_in, m_out)
        F0 = self.F0_proj(F0)
        N = x.transpose(-1, -2)
        for block, (m_in, m_out) in zip(self.N, per_block):
            N = block(N, s, m_in, m_out)
        N = self.N_proj(N)
        return F0.squeeze(1), N.squeeze(1)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_f0ntrain(t_frames: int = 120, output_dir: Path | None = None) -> Path:
    """Export F0Ntrain wrapper to CoreML.

    Args:
        t_frames: Fixed time dimension for aligned features input.
        output_dir: Where to save the .mlpackage (default: coreml/).

    Returns:
        Path to the saved .mlpackage.
    """
    import coremltools as ct

    if output_dir is None:
        output_dir = _ROOT / "coreml"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading KModel...")
    kmodel = _load_kmodel()

    wrapper = F0NtrainWrapper(kmodel.predictor)
    wrapper.eval()
    _remove_training_ops(wrapper)
    for m in wrapper.modules():
        m.eval()

    # Dummy inputs; the mask traces as all-ones (full fill).
    en_dummy = torch.randn(1, 640, t_frames, dtype=torch.float32)
    s_dummy = torch.randn(1, 128, dtype=torch.float32)
    mask_dummy = torch.ones(1, 1, t_frames, dtype=torch.float32)

    # Test forward pass
    with torch.no_grad():
        F0_test, N_test = wrapper(en_dummy, s_dummy, mask_dummy)
        print(f"Forward pass OK: F0 {F0_test.shape}, N {N_test.shape}")
        assert F0_test.shape == (1, 2 * t_frames), f"Expected F0 (1, {2*t_frames}), got {F0_test.shape}"
        assert N_test.shape == (1, 2 * t_frames), f"Expected N (1, {2*t_frames}), got {N_test.shape}"

    # Trace
    print(f"Tracing with T={t_frames}...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (en_dummy, s_dummy, mask_dummy), strict=False)

    # Convert to CoreML
    print("Converting to CoreML...")
    ml = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="en", shape=(1, 640, t_frames), dtype=np.float32),
            ct.TensorType(name="s", shape=(1, 128), dtype=np.float32),
            # The all-ones default keeps `mask` optional for existing consumers.
            # It means full fill: a caller that pads and omits the mask gets
            # the padding contamination back, silently.
            ct.TensorType(name="mask", shape=(1, 1, t_frames), dtype=np.float32,
                          default_value=np.ones((1, 1, t_frames), dtype=np.float32)),
        ],
        outputs=[
            ct.TensorType(name="F0_pred"),
            ct.TensorType(name="N_pred"),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS12,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )

    out_path = output_dir / f"kokoro_f0ntrain_t{t_frames}.mlpackage"
    ml.save(str(out_path))
    print(f"Saved: {out_path}")

    # Numeric validation
    print("\nNumeric validation...")
    validate_f0ntrain(traced, ml, t_frames)

    return out_path


def validate_f0ntrain(traced, ml_model, t_frames: int, n_tests: int = 5) -> None:
    """Compare PyTorch traced vs CoreML predict on random inputs."""
    torch.manual_seed(42)
    correlations_f0 = []
    correlations_n = []

    for i in range(n_tests):
        en = torch.randn(1, 640, t_frames, dtype=torch.float32)
        s = torch.randn(1, 128, dtype=torch.float32)
        # Validated at full fill (mask=ones), where masking must be a no-op.
        mask = torch.ones(1, 1, t_frames, dtype=torch.float32)

        with torch.no_grad():
            pt_f0, pt_n = traced(en, s, mask)
        pt_f0_np = pt_f0.numpy().flatten()
        pt_n_np = pt_n.numpy().flatten()

        coreml_out = ml_model.predict({"en": en.numpy(), "s": s.numpy(), "mask": mask.numpy()})
        cm_f0 = np.asarray(coreml_out["F0_pred"]).flatten()
        cm_n = np.asarray(coreml_out["N_pred"]).flatten()

        corr_f0 = float(np.corrcoef(pt_f0_np, cm_f0)[0, 1])
        corr_n = float(np.corrcoef(pt_n_np, cm_n)[0, 1])
        correlations_f0.append(corr_f0)
        correlations_n.append(corr_n)
        print(f"  Test {i}: F0 corr={corr_f0:.6f}, N corr={corr_n:.6f}")

    mean_f0 = np.mean(correlations_f0)
    mean_n = np.mean(correlations_n)
    print(f"\n  Mean correlation: F0={mean_f0:.6f}, N={mean_n:.6f}")

    if mean_f0 < 0.99:
        print(f"  WARNING: F0 correlation {mean_f0:.4f} < 0.99 threshold")
    if mean_n < 0.99:
        print(f"  WARNING: N correlation {mean_n:.4f} < 0.99 threshold")
    if mean_f0 >= 0.99 and mean_n >= 0.99:
        print("  PASS: Both F0 and N exceed 0.99 correlation")


# ---------------------------------------------------------------------------
# Duration model benchmark
# ---------------------------------------------------------------------------


def bench_duration_coreml(n_warmup: int = 5, n_iter: int = 20) -> None:
    """Micro-benchmark the existing Duration CoreML model predict time.

    Loads kokoro_duration.mlpackage, warms it, and times predict calls.
    This fills in the UNKNOWN latency in the Swift prefix rewrite plan.
    """
    import coremltools as ct

    pkg_path = _ROOT / "coreml" / "kokoro_duration.mlpackage"
    if not pkg_path.exists():
        print(f"Duration model not found at {pkg_path}. Run: python scripts/download_models.py --coreml")
        return

    print(f"\nLoading Duration model: {pkg_path}")
    model = ct.models.MLModel(str(pkg_path), compute_units=ct.ComputeUnit.ALL)

    # Fixed 128-token inputs (matching the exported model)
    input_ids = np.zeros((1, 128), dtype=np.int32)
    input_ids[0, :10] = np.array([1, 5, 23, 42, 67, 12, 3, 55, 8, 2], dtype=np.int32)
    attention_mask = np.zeros((1, 128), dtype=np.int32)
    attention_mask[0, :10] = 1
    ref_s = np.random.randn(1, 256).astype(np.float32)
    speed = np.array([1.0], dtype=np.float32)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ref_s": ref_s,
        "speed": speed,
    }

    # Warmup
    print(f"Warming up ({n_warmup} calls)...")
    for _ in range(n_warmup):
        model.predict(inputs)

    # Timed runs
    print(f"Timing {n_iter} predict calls...")
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        model.predict(inputs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times.sort()
    median = times[len(times) // 2]
    p5 = times[0]
    p95 = times[-1]
    mean = sum(times) / len(times)

    print(f"\nDuration CoreML predict latency ({n_iter} calls):")
    print(f"  Median: {median:.2f} ms")
    print(f"  Mean:   {mean:.2f} ms")
    print(f"  Min:    {p5:.2f} ms")
    print(f"  Max:    {p95:.2f} ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Export F0Ntrain to CoreML")
    parser.add_argument(
        "--t-frames", type=int, nargs="+", default=[120, 400],
        help="Fixed T dimension(s) for the aligned features input. "
             "120 covers the 3s bucket, 400 covers the 10s bucket.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: coreml/)",
    )
    parser.add_argument(
        "--bench-duration", action="store_true",
        help="Also benchmark the Duration CoreML model predict latency",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None

    for t in args.t_frames:
        print(f"\n{'='*60}")
        print(f"Exporting F0Ntrain with T={t}")
        print(f"{'='*60}")
        export_f0ntrain(t_frames=t, output_dir=out_dir)

    if args.bench_duration:
        bench_duration_coreml()


if __name__ == "__main__":
    main()
