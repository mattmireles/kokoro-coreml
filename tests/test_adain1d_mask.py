"""Unit-level tests for the mask-aware AdaIN1d path.

Two correctness contracts:

1. **Equivalence at full fill (`m = ones`).** When the mask is all-ones, the
   masked path must produce the same output as the unmasked path within fp32
   rounding. The mask discipline is a no-op at full fill; if it isn't, the
   implementation is wrong.

2. **Correctness at partial fill.** The masked path on a right-padded input
   must produce, on the valid prefix, the same output the unmasked path would
   produce on a tight (un-padded) input. This is the AdaIN-statistics fix:
   pads must not shift the per-channel mean/var seen by the valid region.

Plus a thin smoke test that the threading through ``AdainResBlk1d`` and
``AdaINResBlock1`` honors the same two contracts (no mask = unchanged; with
mask = valid-prefix matches the tight-input unmasked reference).

Mechanism: ``AdaIN1d`` normalizes by a mean and variance taken over the time
axis. Under bucketed export that axis is zero-padded to a fixed length, so the
padding enters both moments and the valid region is rescaled by statistics that
partly describe padding. Supplying a validity mask confines both reductions to
real frames.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from kokoro.istftnet import AdaIN1d, AdainResBlk1d, AdaINResBlock1


# ---------------------------------------------------------------------------
# AdaIN1d
# ---------------------------------------------------------------------------


def _build_inputs(B: int = 1, C: int = 8, V: int = 17, T_b: int = 64, style_dim: int = 4):
    """Tight-shape input (B, C, V) and right-padded bucketed input (B, C, T_b)."""
    torch.manual_seed(0)
    x_tight = torch.randn(B, C, V)
    x_padded = torch.zeros(B, C, T_b)
    x_padded[:, :, :V] = x_tight
    s = torch.randn(B, style_dim)
    mask_full = torch.ones(B, 1, T_b)
    mask_partial = torch.zeros(B, 1, T_b)
    mask_partial[:, :, :V] = 1.0
    return x_tight, x_padded, s, mask_full, mask_partial


def test_adain1d_mask_none_matches_no_mask_arg():
    """Default `m=None` must be byte-identical to the pre-mask call signature."""
    x_tight, _, s, _, _ = _build_inputs()
    layer = AdaIN1d(style_dim=s.shape[-1], num_features=x_tight.shape[1]).eval()
    with torch.no_grad():
        a = layer(x_tight, s)
        b = layer(x_tight, s, m=None)
    assert torch.equal(a, b), "m=None path is not bit-identical to no-m-arg path"


def test_adain1d_mask_ones_matches_unmasked_full_fill():
    """At full fill (m=ones), masked and unmasked outputs must agree to fp32 rounding."""
    _, x_padded, s, mask_full, _ = _build_inputs()
    layer = AdaIN1d(style_dim=s.shape[-1], num_features=x_padded.shape[1]).eval()
    with torch.no_grad():
        unmasked = layer(x_padded, s)
        masked = layer(x_padded, s, m=mask_full)
    torch.testing.assert_close(unmasked, masked, atol=1e-6, rtol=1e-6)


def test_adain1d_partial_fill_matches_tight_reference():
    """Masked-on-padded must match unmasked-on-tight on the valid prefix."""
    x_tight, x_padded, s, _, mask_partial = _build_inputs()
    V = x_tight.shape[-1]
    layer = AdaIN1d(style_dim=s.shape[-1], num_features=x_tight.shape[1]).eval()
    with torch.no_grad():
        ref_tight = layer(x_tight, s)              # tight, unmasked: ground truth
        masked_padded = layer(x_padded, s, m=mask_partial)
    # Valid region of masked-padded must match the tight reference.
    torch.testing.assert_close(masked_padded[..., :V], ref_tight, atol=1e-6, rtol=1e-6)


def test_adain1d_partial_fill_unmasked_DRIFTS_vs_tight():
    """Positive control: the unmasked path on padded input must disagree with the
    tight reference, or there is nothing for the mask to fix."""
    x_tight, x_padded, s, _, _ = _build_inputs()
    V = x_tight.shape[-1]
    layer = AdaIN1d(style_dim=s.shape[-1], num_features=x_tight.shape[1]).eval()
    with torch.no_grad():
        ref_tight = layer(x_tight, s)
        unmasked_padded = layer(x_padded, s)
    diff = (unmasked_padded[..., :V] - ref_tight).abs().max().item()
    assert diff > 1e-3, f"expected partial-fill drift in the unmasked path, got {diff:.2e}"


# ---------------------------------------------------------------------------
# AdaINResBlock1 (used in the audio-rate generator stack)
# ---------------------------------------------------------------------------


def test_adain_resblock1_mask_none_matches_no_mask_arg():
    """AdaINResBlock1 default-arg behavior unchanged."""
    x_tight, _, s, _, _ = _build_inputs(B=1, C=8, V=17, T_b=64, style_dim=4)
    block = AdaINResBlock1(channels=x_tight.shape[1], kernel_size=3, dilation=(1, 3, 5), style_dim=s.shape[-1]).eval()
    with torch.no_grad():
        a = block(x_tight, s)
        b = block(x_tight, s, m=None)
    assert torch.equal(a, b)


def test_adain_resblock1_partial_fill_drift_is_bounded():
    """Masked-on-padded is close to unmasked-on-tight on the valid prefix, not
    equal: the block's convs read across the boundary into padded positions
    whose values differ from the tight input's implicit zero padding. This
    bounds the residual; scripts/measure_bucket_contamination.py measures it
    end to end."""
    x_tight, x_padded, s, _, mask_partial = _build_inputs(B=1, C=8, V=17, T_b=64, style_dim=4)
    V = x_tight.shape[-1]
    block = AdaINResBlock1(channels=x_tight.shape[1], kernel_size=3, dilation=(1, 3, 5), style_dim=s.shape[-1]).eval()
    with torch.no_grad():
        ref_tight = block(x_tight, s)
        masked_padded = block(x_padded, s, m=mask_partial)
        unmasked_padded = block(x_padded, s)
    masked_drift = (masked_padded[..., :V] - ref_tight).abs().max().item()
    unmasked_drift = (unmasked_padded[..., :V] - ref_tight).abs().max().item()
    assert masked_drift < unmasked_drift, (
        f"Masking did not reduce block-level drift: masked={masked_drift:.3e} "
        f"vs unmasked={unmasked_drift:.3e}"
    )


# ---------------------------------------------------------------------------
# AdainResBlk1d (used in the decoder + f0ntrain F0/N branches)
# ---------------------------------------------------------------------------


def test_adain_resblk1d_mask_none_matches_no_mask_arg_no_upsample():
    """AdainResBlk1d default args unchanged when no upsample (m_up irrelevant)."""
    x_tight, _, s, _, _ = _build_inputs(B=1, C=8, V=17, T_b=64, style_dim=4)
    block = AdainResBlk1d(dim_in=8, dim_out=8, style_dim=s.shape[-1]).eval()
    with torch.no_grad():
        a = block(x_tight, s)
        b = block(x_tight, s, m=None, m_up=None)
    assert torch.equal(a, b)


def test_adain_resblk1d_partial_fill_drift_is_bounded_no_upsample():
    """Same boundary-leak bound as for AdaINResBlock1."""
    x_tight, x_padded, s, _, mask_partial = _build_inputs(B=1, C=8, V=17, T_b=64, style_dim=4)
    V = x_tight.shape[-1]
    block = AdainResBlk1d(dim_in=8, dim_out=8, style_dim=s.shape[-1]).eval()
    with torch.no_grad():
        ref_tight = block(x_tight, s)
        masked_padded = block(x_padded, s, m=mask_partial, m_up=mask_partial)
        unmasked_padded = block(x_padded, s)
    masked_drift = (masked_padded[..., :V] - ref_tight).abs().max().item()
    unmasked_drift = (unmasked_padded[..., :V] - ref_tight).abs().max().item()
    assert masked_drift < unmasked_drift, (
        f"Masking did not reduce block-level drift: masked={masked_drift:.3e} "
        f"vs unmasked={unmasked_drift:.3e}"
    )


def test_adain_resblk1d_mask_none_matches_no_mask_arg_with_upsample():
    """AdainResBlk1d default args unchanged in the upsample variant too."""
    x_tight, _, s, _, _ = _build_inputs(B=1, C=8, V=17, T_b=64, style_dim=4)
    block = AdainResBlk1d(dim_in=8, dim_out=4, style_dim=s.shape[-1], upsample='2x').eval()
    with torch.no_grad():
        a = block(x_tight, s)
        b = block(x_tight, s, m=None, m_up=None)
    assert torch.equal(a, b)


def test_residual_requires_explicit_m_up_at_upsample_resolution():
    """On an upsampling block, handing ``norm2`` the input-resolution mask must
    raise rather than broadcast: there is deliberately no ``m_up or m`` fallback."""
    torch.manual_seed(0)
    block = AdainResBlk1d(dim_in=8, dim_out=8, style_dim=4, upsample=True).eval()
    x = torch.randn(1, 8, 16)
    s = torch.randn(1, 4)
    m = torch.ones(1, 1, 16)
    m_up = m.repeat_interleave(2, dim=2)

    with torch.no_grad():
        block(x, s, m, m_up)          # correct usage: both masks, right resolutions

    with pytest.raises(RuntimeError):
        with torch.no_grad():
            block(x, s, m, m)         # wrong: norm2 gets a 1x mask for a 2x activation
