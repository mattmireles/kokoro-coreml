"""Probe: the Kokoro generator folded into rows so it fits the Neural Engine.

Why: the ANE rejects any tensor axis over 16,384, and the generator's time
axis reaches 144,001 at 30 s. Every past ANE attempt hit that limit and fell
back to the CPU (see README/Notes/coreml-compute-unit-ablation.md).

How: the x_pre time axis is cut into H rows of ``core`` frames, each widened
by ``halo`` frames of real neighbour data on both sides, giving tensors of
shape (1, C, H, W). Every Conv1d becomes a (1, k) Conv2d over W, so rows never
mix and each row's halo absorbs the zero-padding damage at its edges. The
generator's receptive field is about 8 x_pre frames, so a halo of 10 keeps
every core frame exact. AdaIN needs per-channel statistics over the whole
utterance; because all rows are in one predict call, a stats mask (core and
valid frames only) gives the exact global statistics. A separate zero mask
reproduces the zero-padding of the unfolded run outside the valid audio.

The model stops at the conv_post logits (22 channels at 4800 Hz); the
exp/sin/iSTFT tail (about 1 ms) runs on the host.

Usage:
    uv run --no-sync python scripts/probe_generator_ane_fold.py \
        --weights k/kokoro-v1_0.pth --config k/config.json --voice k/voices/af_heart.pt \
        --input-json outputs/swift_bench_inputs/30s.json --core 240 --halo 10 --out DIR

Writes DIR/gen_fold.mlpackage, DIR/in/<input>.bin (float32) and DIR/ref.npz
(reference logits + audio) and prints PyTorch-folded vs unfolded parity.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from kokoro.model import KModel  # noqa: E402

STAGE1 = 10  # ups[0] rate: x_pre frame -> 10 samples
STAGE2 = 60  # ups[0] * ups[1]
EXTRA2 = 5   # stage-2 width is 60*W0 + 5: +1 reflect sample, +4 zeros so W % 5 == 0


# --------------------------------------------------------------------------
# Folded (4-D) building blocks
# --------------------------------------------------------------------------

def conv2d_from(c: nn.Conv1d) -> nn.Conv2d:
    out = nn.Conv2d(c.in_channels, c.out_channels, (1, c.kernel_size[0]), stride=(1, c.stride[0]),
                    padding=(0, c.padding[0]), dilation=(1, c.dilation[0]), bias=c.bias is not None)
    out.weight.data = c.weight.detach().unsqueeze(2).clone()
    if c.bias is not None:
        out.bias.data = c.bias.detach().clone()
    return out


def convT2d_from(c: nn.ConvTranspose1d) -> nn.ConvTranspose2d:
    out = nn.ConvTranspose2d(c.in_channels, c.out_channels, (1, c.kernel_size[0]), stride=(1, c.stride[0]),
                             padding=(0, c.padding[0]), bias=c.bias is not None)
    out.weight.data = c.weight.detach().unsqueeze(2).clone()
    if c.bias is not None:
        out.bias.data = c.bias.detach().clone()
    return out


# ANE-safe op rewrites, all exact up to float rounding. Tensors are
# (rows, C, 1, W): rows ride the batch axis so a reshape of W into (W/d, d)
# needs no transpose (W/d stays under the 16,384 axis cap). Measured on the
# A17 Pro ANE at stage-2 size: dilation 5 by height reshape 14.7 ms vs 76 ms by
# space-to-batch transposes; native dilation 3 13 ms.
#
# Snake x + sin^2(a x)/a is folded: the AdaIN before it emits z = a*y (or
# u = a*y/pi for the polynomial form) through its per-channel affine, and the
# conv after it carries the 1/a (or pi/a) in its weights. What remains on the
# full tensor is z + sin(z)^2: sin, mul, add.
SIN_MODES = ("native", "poly", "torch")  # torch: unfolded reference math
# sin^2(pi f) for |f| <= 0.5 as f^2 * P(f^2), P from a least-squares fit.
# Degree trades elementwise ANE passes (2 per degree) for accuracy.
SIN2_COEFS = {
    3: (9.81377168, -30.95882638, 30.94114192),                              # err 2.0e-3
    4: (9.86724579, -32.3704018, 41.42608277, -23.29753983),                 # err 6.2e-5
    5: (9.86954128, -32.46588447, 42.65359441, -29.47994363, 10.67763048),  # err 1.3e-6
}
SIN2_COEF = SIN2_COEFS[3]


def snake_folded(z, mode: str):
    """Snake on pre-scaled input. native: z = a*y, returns z + sin(z)^2.
    poly: z = a*y/pi, returns z + sin^2(pi z)/pi (period 1 in z)."""
    if mode == "native":
        sz = torch.sin(z)
        return z + sz * sz
    f = z - torch.round(z)
    f2 = f * f
    p = SIN2_COEF[-1] / math.pi
    for c in reversed(SIN2_COEF[:-1]):
        p = p * f2 + c / math.pi
    return z + p * f2


def snake_scale(a, mode: str):
    """Per-channel factor the AdaIN folds into its output for ``mode``."""
    return a / math.pi if mode == "poly" else a


def scaled_conv(c: nn.Conv1d, in_scale) -> nn.Module:
    """``c`` applied to (in_scale * x): weights divided per input channel."""
    w = c.weight.detach() / in_scale.reshape(1, -1, 1)
    c2 = nn.Conv1d(c.in_channels, c.out_channels, c.kernel_size[0], padding=c.padding[0],
                   dilation=c.dilation[0], bias=c.bias is not None)
    c2.weight.data = w
    if c.bias is not None:
        c2.bias.data = c.bias.detach().clone()
    return ane_conv(c2)


class DilatedConv(nn.Module):
    """Dilation-d conv as an undilated (k, 1) conv on W reshaped to (W/d, d).
    W must be a multiple of d. Rows are in the batch axis, so no transpose."""

    def __init__(self, c: nn.Conv1d) -> None:
        super().__init__()
        self.d = c.dilation[0]
        k = c.kernel_size[0]
        self.conv = nn.Conv2d(c.in_channels, c.out_channels, (k, 1), padding=((k - 1) // 2, 0), bias=c.bias is not None)
        self.conv.weight.data = c.weight.detach().unsqueeze(3).clone()
        if c.bias is not None:
            self.conv.bias.data = c.bias.detach().clone()

    def forward(self, x):
        n, ch, _, w = x.shape
        y = self.conv(x.reshape(n, ch, w // self.d, self.d))
        return y.reshape(n, y.shape[1], 1, w)


class StridedConv(nn.Module):
    """Stride-s conv as space-to-depth then a stride-1 conv (kernel ceil(k/s))."""

    def __init__(self, c: nn.Conv1d) -> None:
        super().__init__()
        s, k, p = c.stride[0], c.kernel_size[0], c.padding[0]
        self.s, self.p = s, p
        km = -(-k // s)
        w = torch.zeros(c.out_channels, c.in_channels, s, km)
        for m in range(km):
            for e in range(s):
                if m * s + e < k:
                    w[:, :, e, m] = c.weight.detach()[:, :, m * s + e]
        self.conv = nn.Conv2d(c.in_channels * s, c.out_channels, (1, km), bias=True)
        self.conv.weight.data = w.reshape(c.out_channels, c.in_channels * s, 1, km)
        self.conv.bias.data = c.bias.detach().clone()

    def forward(self, x, out_w: int):
        n, ch, _, w = x.shape
        s = self.s
        total = -(-(w + self.p) // s) * s + s  # room for the last tap
        x = torch.cat([torch.zeros(n, ch, 1, self.p), x, torch.zeros(n, ch, 1, total - w - self.p)], dim=3)
        y = x.reshape(n, ch, total // s, s).permute(0, 1, 3, 2).reshape(n, ch * s, 1, total // s)
        return self.conv(y)[..., :out_w]


class PolyphaseConvT(nn.Module):
    """ConvTranspose1d(stride s) as one conv to Cout*s channels plus a width shuffle."""

    def __init__(self, c: nn.ConvTranspose1d) -> None:
        super().__init__()
        s, k, p = c.stride[0], c.kernel_size[0], c.padding[0]
        wt = c.weight.detach()  # (Cin, Cout, K)
        cin, cout = wt.shape[0], wt.shape[1]
        # y[q*s + r] = sum_i x[i] W[:, :, q*s + r + p - i*s]; tap offset o = i - q.
        taps = {}
        for r in range(s):
            for kk in range(k):
                num = r + p - kk
                if num % s == 0:
                    taps.setdefault((r, num // s), kk)
        offs = [o for (_, o) in taps]
        self.omin, self.omax = min(offs), max(offs)
        kw = self.omax - self.omin + 1
        w = torch.zeros(cout, s, cin, kw)
        for (r, o), kk in taps.items():
            w[:, r, :, o - self.omin] = wt[:, :, kk].t()
        self.conv = nn.Conv2d(cin, cout * s, (1, kw), bias=True)
        self.conv.weight.data = w.reshape(cout * s, cin, 1, kw)
        self.conv.bias.data = c.bias.detach().repeat_interleave(s)
        self.s, self.cout = s, cout

    def forward(self, x):
        n, ch, _, q = x.shape
        x = torch.cat([torch.zeros(n, ch, 1, -self.omin), x, torch.zeros(n, ch, 1, self.omax)], dim=3)
        y = self.conv(x)  # (n, cout*s, 1, q)
        return y.reshape(n, self.cout, self.s, q).permute(0, 1, 3, 2).reshape(n, self.cout, 1, q * self.s)


def dense_from_dilated(c: nn.Conv1d) -> nn.Conv1d:
    """A small dilated kernel as an equivalent dense one (zeros between taps)."""
    k, d = c.kernel_size[0], c.dilation[0]
    kd = (k - 1) * d + 1
    out = nn.Conv1d(c.in_channels, c.out_channels, kd, padding=(kd - 1) // 2, bias=c.bias is not None)
    w = torch.zeros(c.out_channels, c.in_channels, kd)
    w[:, :, ::d] = c.weight.detach()
    out.weight.data = w
    if c.bias is not None:
        out.bias.data = c.bias.detach().clone()
    return out


def ane_conv(c: nn.Conv1d) -> nn.Module:
    k, d = c.kernel_size[0], c.dilation[0]
    # k=3 at dilation 3 or 5 as a dense width-7/11 kernel: the M1 ANE ran the
    # native k=3 d=3 conv at 516 ms (off the ANE) vs ~15 ms dense.
    if d > 1 and (k - 1) * d + 1 <= 11:
        return conv2d_from(dense_from_dilated(c))
    if d not in (1, 2, 3, 4, 8):
        return DilatedConv(c)
    return conv2d_from(c)


def masked_mean(v, sm, fill):
    """Mean over the stats mask: width, then rows (two short reductions)."""
    return (v * sm).mean(dim=3, keepdim=True).mean(dim=0, keepdim=True) / fill


class AdaIN2d(nn.Module):
    """AdaIN over (rows, C, 1, W) with statistics from the stats mask only.
    ``out_scale`` (1, C, 1, 1) multiplies the output (the folded Snake scale)."""

    def __init__(self, src, out_scale=None) -> None:
        super().__init__()
        self.fc = src.fc
        self.eps = src.eps
        self.c = src.num_features
        self.out_scale = torch.ones(1, self.c, 1, 1) if out_scale is None else out_scale

    def forward(self, x, s, sm, zm, fill):
        # Two-pass centring. Some channels have |mean| ~3 but std ~0.08, and
        # the ANE's fp16 mean is off by ~1e-3 relative: that shifts the whole
        # normalised channel by ~0.04 (AdaIN-only SNR 40 dB vs 60 on the GPU).
        # Re-centring on the residual's mean cancels on small numbers instead.
        # Do not pre-scale c before squaring: the noise branch has std ~0.1 and
        # (c / 16)^2 falls into fp16 subnormals, which the ANE flushes to zero.
        c = x - masked_mean(x, sm, fill)
        # The residual mean m2 and the second moment come from the same masked
        # tensor d, and m2 is folded into the per-channel shift: two fewer
        # full-tensor passes than re-centring c. var = E[d^2] - m2^2 does not
        # cancel badly because m2 is tiny after the first centring.
        d = c * sm
        m2 = d.mean(dim=3, keepdim=True).mean(dim=0, keepdim=True) / fill
        var = (d * d).mean(dim=3, keepdim=True).mean(dim=0, keepdim=True) / fill - m2 * m2
        h = self.fc(s).view(1, 2 * self.c, 1, 1)
        gamma, beta = h[:, : self.c], h[:, self.c:]
        scale = (1.0 + gamma) * self.out_scale * torch.rsqrt(var + self.eps)  # (1, C, 1, 1)
        shift = beta * self.out_scale - m2 * scale
        if getattr(AdaIN2d, "skip_zm", False):
            return c * scale + shift
        return (c * scale + shift) * zm


class ResBlock2d(nn.Module):
    def __init__(self, src, mode: str = "torch") -> None:
        super().__init__()
        self.mode = mode
        a1 = [a.detach().unsqueeze(-1) for a in src.alpha1]  # (1, C, 1, 1)
        a2 = [a.detach().unsqueeze(-1) for a in src.alpha2]
        if mode == "torch":
            self.c1 = nn.ModuleList(conv2d_from(c) for c in src.convs1)
            self.c2 = nn.ModuleList(conv2d_from(c) for c in src.convs2)
            self.n1 = nn.ModuleList(AdaIN2d(n) for n in src.adain1)
            self.n2 = nn.ModuleList(AdaIN2d(n) for n in src.adain2)
            self.a1, self.a2 = a1, a2
        else:
            k1 = [snake_scale(a, mode) for a in a1]
            k2 = [snake_scale(a, mode) for a in a2]
            self.c1 = nn.ModuleList(scaled_conv(c, k.flatten()) for c, k in zip(src.convs1, k1))
            self.c2 = nn.ModuleList(scaled_conv(c, k.flatten()) for c, k in zip(src.convs2, k2))
            self.n1 = nn.ModuleList(AdaIN2d(n, k) for n, k in zip(src.adain1, k1))
            self.n2 = nn.ModuleList(AdaIN2d(n, k) for n, k in zip(src.adain2, k2))

    def act(self, x, a):
        if self.mode == "torch":
            return x + (1 / a) * torch.sin(a * x) ** 2
        return snake_folded(x, self.mode)

    def forward(self, x, s, sm, zm, fill):
        for i, (c1, c2, n1, n2) in enumerate(zip(self.c1, self.c2, self.n1, self.n2)):
            xt = self.act(n1(x, s, sm, zm, fill), self.a1[i] if self.mode == "torch" else None)
            xt = c1(xt)
            xt = self.act(n2(xt, s, sm, zm, fill), self.a2[i] if self.mode == "torch" else None)
            xt = c2(xt)
            x = xt + x
        return x


class FoldedGenerator(nn.Module):
    """Generator body on folded rows (rows in the batch axis). Inputs (float):
    x_pre (R,512,1,W0), har (R,22,1,60*W0+5), ref_s (1,256),
    sm1/zm1 (R,1,1,10*W0), sm2/zm2 (R,1,1,60*W0+5). Output logits (R,22,1,60*W0+5).
    ``mode``: torch (reference math), native (ANE with sin), poly (ANE without sin)."""

    def __init__(self, gen, mode: str = "torch", taps: bool = False) -> None:
        super().__init__()
        ane = mode != "torch"
        self.taps = taps
        self.noise_convs = nn.ModuleList(
            StridedConv(c) if ane and c.stride[0] > 1 else conv2d_from(c) for c in gen.noise_convs)
        self.noise_res = nn.ModuleList(ResBlock2d(r, mode) for r in gen.noise_res)
        self.ups = nn.ModuleList(PolyphaseConvT(u) if ane else convT2d_from(u) for u in gen.ups)
        self.resblocks = nn.ModuleList(ResBlock2d(r, mode) for r in gen.resblocks)
        self.conv_post = conv2d_from(gen.conv_post)
        self.num_kernels = gen.num_kernels

    def forward(self, x_pre, har, ref_s, sm1, zm1, sm2, zm2):
        s = ref_s[:, :128]
        masks = ((sm1, zm1, sm1.mean(dim=3, keepdim=True).mean(dim=0, keepdim=True)),
                 (sm2, zm2, sm2.mean(dim=3, keepdim=True).mean(dim=0, keepdim=True)))
        x = x_pre
        tapped = []
        for i in range(2):
            sm, zm, fill = masks[i]
            x = F.leaky_relu(x, 0.1)
            nc = self.noise_convs[i]
            xs_ = nc(har, sm.shape[3]) if isinstance(nc, StridedConv) else nc(har)
            if not getattr(self, "skip_noise_res", False):
                xs_ = self.noise_res[i](xs_, s, sm, zm, fill)
            x = self.ups[i](x)
            if i == 1:
                # Reflect pad (1, 0), then zero-extend to the stage-2 width
                # (a multiple of 5 for the dilation-5 reshape).
                n, ch, _, w = x.shape
                x = torch.cat([x[..., 1:2], x, torch.zeros(n, ch, 1, sm.shape[3] - w - 1)], dim=3)
            # Zero outside the valid audio so the next layer sees the unfolded run's zero padding.
            x = (x + xs_) * zm
            acc = None
            for j in range(self.num_kernels):
                y = self.resblocks[i * self.num_kernels + j](x, s, sm, zm, fill)
                acc = y if acc is None else acc + y
            x = acc * (zm / self.num_kernels)
            tapped.append(x)
        x = F.leaky_relu(x)
        if self.taps:
            return self.conv_post(x), tapped[0], tapped[1]
        return self.conv_post(x)


# --------------------------------------------------------------------------
# Unfolded reference (same math as export_synth.wrappers.GeneratorFromHar)
# --------------------------------------------------------------------------

def reference_logits(gen, x, s, har):
    for i in range(gen.num_upsamples):
        x = F.leaky_relu(x, negative_slope=0.1)
        xs_ = gen.noise_res[i](gen.noise_convs[i](har), s)
        x = gen.ups[i](x)
        if i == gen.num_upsamples - 1:
            x = gen.reflection_pad(x)
        x = x + xs_[:, :, : x.shape[2]]
        acc = None
        for j in range(gen.num_kernels):
            y = gen.resblocks[i * gen.num_kernels + j](x, s)
            acc = y if acc is None else acc + y
        x = acc / gen.num_kernels
    return gen.conv_post(F.leaky_relu(x))


def tail(gen, logits):
    n = gen.post_n_fft // 2 + 1
    return gen.stft.inverse(torch.exp(logits[:, :n]), torch.sin(logits[:, n:])).flatten()


# --------------------------------------------------------------------------
# Folding helpers (host side; the Swift runtime will mirror these)
# --------------------------------------------------------------------------

def fold(arr: np.ndarray, rate: int, extra: int, rows: int, core: int, halo: int) -> np.ndarray:
    """arr (C, T) at ``rate`` samples per x_pre frame -> (C, rows, W)."""
    c, t = arr.shape
    w = (core + 2 * halo) * rate + extra
    out = np.zeros((c, rows, w), dtype=np.float32)
    for r in range(rows):
        g0 = (r * core - halo) * rate
        lo, hi = max(g0, 0), min(g0 + w, t)
        if hi > lo:
            out[:, r, lo - g0: hi - g0] = arr[:, lo:hi]
    return out


def masks(rate: int, extra: int, rows: int, core: int, halo: int, valid: int) -> tuple[np.ndarray, np.ndarray]:
    """Zero mask: global index in [0, valid). Stats mask: zero mask and owned by
    this row (the same owner rule ``unfold`` uses), so each index counts once."""
    w = (core + 2 * halo) * rate + extra
    j = np.arange(w)
    zm = np.zeros((rows, w), np.float32)
    sm = np.zeros((rows, w), np.float32)
    for r in range(rows):
        g = (r * core - halo) * rate + j
        z = (g >= 0) & (g < valid)
        owner = np.minimum(np.maximum(g, 0) // (core * rate), rows - 1)
        zm[r] = z
        sm[r] = z & (owner == r)
    return sm, zm


def unfold(arr: np.ndarray, rate: int, extra: int, core: int, halo: int, total: int) -> np.ndarray:
    """(C, rows, W) -> (C, total) taking each global index from its core row."""
    c, rows, w = arr.shape
    g = np.arange(total)
    r = np.minimum(g // (core * rate), rows - 1)
    j = g - (r * core - halo) * rate
    return arr[:, r, j]


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--core", type=int, default=240)
    ap.add_argument("--halo", type=int, default=10)
    ap.add_argument("--bucket-frames", type=int, default=0, help="padded x_pre frames (0 = natural)")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--no-convert", action="store_true")
    ap.add_argument("--mode", default="native", choices=SIN_MODES,
                    help="native: ANE graph with sin (A15+/A17); poly: ANE graph without sin (M1/M2/A14); torch: reference")
    ap.add_argument("--taps", action="store_true", help="also output the stage-1 and stage-2 activations")
    ap.add_argument("--poly-degree", type=int, default=3, choices=sorted(SIN2_COEFS))
    ap.add_argument("--ablate", default="", choices=["", "nosnake", "noadain", "convonly", "nonoise", "nozm"],
                    help="timing-only ablation (breaks the math): drop Snake and/or AdaIN")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    cfg = json.load(open(a.config))
    km = KModel(repo_id="hexgrad/Kokoro-82M", config=cfg, model=a.weights, disable_complex=True).eval()
    gen = km.decoder.generator
    inp = json.load(open(a.input_json))
    ids = torch.tensor([inp["input_ids"][: inp["num_tokens"]]])
    ref_s = torch.tensor(inp["ref_s"], dtype=torch.float32)[None]

    # Capture the generator's real inputs (x_pre, style, f0) from a full run.
    cap = {}
    hook = gen.register_forward_pre_hook(lambda m, args: cap.update(x=args[0], s=args[1], f0=args[2]))
    with torch.no_grad():
        km.forward_with_tokens(ids, ref_s)
        hook.remove()
        f0 = gen.f0_upsamp(cap["f0"][:, None]).transpose(1, 2)
        har_source, _, _ = gen.m_source(f0)
        spec, phase = gen.stft.transform(har_source.transpose(1, 2).squeeze(1))
        har = torch.cat([spec, phase], dim=1)
        x, s = cap["x"], cap["s"]
        ref = reference_logits(gen, x, s, har)
        ref_audio = tail(gen, ref)
    t_valid = x.shape[2]
    t0 = max(a.bucket_frames, t_valid)
    rows = math.ceil(t0 / a.core)
    print(f"x_pre frames {t_valid} (bucket {t0}), rows {rows}, core {a.core}, halo {a.halo}, "
          f"W2 {(a.core + 2 * a.halo) * STAGE2 + EXTRA2}")

    def rows_first(arr):  # (C, rows, W) -> (rows, C, 1, W)
        return np.ascontiguousarray(arr.transpose(1, 0, 2)[:, :, None, :])

    feeds = {
        "x_pre": rows_first(fold(x[0].numpy(), 1, 0, rows, a.core, a.halo)),
        "har": rows_first(fold(har[0].numpy(), STAGE2, EXTRA2, rows, a.core, a.halo)),
        "ref_s": ref_s.numpy(),
    }
    sm1, zm1 = masks(STAGE1, 0, rows, a.core, a.halo, t_valid * STAGE1)
    sm2, zm2 = masks(STAGE2, EXTRA2, rows, a.core, a.halo, t_valid * STAGE2 + 1)
    feeds.update(sm1=sm1[:, None, None], zm1=zm1[:, None, None], sm2=sm2[:, None, None], zm2=zm2[:, None, None])

    global SIN2_COEF, snake_folded
    SIN2_COEF = SIN2_COEFS[a.poly_degree]
    if a.ablate in ("nosnake", "convonly"):
        snake_folded = lambda z, mode: z  # noqa: E731
    if a.ablate in ("noadain", "convonly"):
        AdaIN2d.forward = lambda self, x, s, sm, zm, fill: x  # noqa: E731
    folded = FoldedGenerator(gen, mode=a.mode, taps=a.taps).eval()
    folded.skip_noise_res = a.ablate == "nonoise"
    AdaIN2d.skip_zm = a.ablate == "nozm"
    with torch.no_grad():
        tf = {k: torch.from_numpy(v) for k, v in feeds.items()}
        res = folded(**tf)
        out = (res[0] if a.taps else res)[:, :, 0, :].numpy().transpose(1, 0, 2)
        if a.taps:
            np.savez(a.out / "taps.npz", t1=res[1].numpy(), t2=res[2].numpy())
    got = unfold(out, STAGE2, EXTRA2, a.core, a.halo, ref.shape[2])
    want = ref[0].numpy()
    err = np.abs(got - want)
    print(f"torch folded vs unfolded logits: max abs {err.max():.3e}, worst frame {err.max(0).argmax()}")
    with torch.no_grad():
        audio = tail(gen, torch.from_numpy(got)[None]).numpy()
    ra = ref_audio.numpy()
    snr = 10 * np.log10((ra ** 2).sum() / ((audio - ra) ** 2).sum())
    print(f"torch folded audio SNR {snr:.1f} dB, corr {np.corrcoef(audio, ra)[0, 1]:.6f}")

    (a.out / "in").mkdir(exist_ok=True)
    for k, v in feeds.items():
        v.astype(np.float32).tofile(a.out / "in" / f"{k}.bin")
    np.savez(a.out / "ref.npz", logits=want, audio=ra, core=a.core, halo=a.halo, rows=rows)
    if a.no_convert:
        return

    import coremltools as ct
    traced = torch.jit.trace(folded, tuple(tf[k] for k in ["x_pre", "har", "ref_s", "sm1", "zm1", "sm2", "zm2"]))
    mlm = ct.convert(
        traced,
        inputs=[ct.TensorType(name=k, shape=feeds[k].shape) for k in ["x_pre", "har", "ref_s", "sm1", "zm1", "sm2", "zm2"]],
        outputs=[ct.TensorType(name=n) for n in (["logits", "t1", "t2"] if a.taps else ["logits"])],
        compute_precision=ct.precision.FLOAT16 if a.precision == "fp16" else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,  # the host may lack the target's ANE ops (sin); test on device
    )
    # FoldGeometry(model:) in swift/Sources/KokoroPipeline/FoldedGenerator.swift reads this.
    mlm.user_defined_metadata["kokoro.fold.halo"] = str(a.halo)
    mlm.save(str(a.out / "gen_fold.mlpackage"))
    print("saved", a.out / "gen_fold.mlpackage")


if __name__ == "__main__":
    main()
