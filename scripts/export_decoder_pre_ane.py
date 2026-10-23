"""Export DecoderPre in the ANE layout, same runtime contract as export_decoder_pre.py.

Why: the shipping DecoderPre (~75 GFLOP at 30 s) is a plain Conv1d / (B, C, T)
trace and runs 34-53 ms on CPU+ANE at 30 s, far from the ~15 ms its FLOPs need
on the ANE. Same recipe as scripts/export_f0ntrain_split.py (README/Notes/
ane-folded-generator.md, "F0Ntrain split, ANE layout").

DecoderPre is everything in kokoro Decoder before the iSTFTNet generator:
  F0_conv, N_conv  Conv1d(1, 1, k=3, s=2, p=1) on f0 / n_input (1, 1, 2T) -> (1, 1, T)
  encode           AdainResBlk1d(514 -> 1024) on cat(asr, F0, N)
  asr_res          Conv1d(512 -> 64, k=1)
  decode[0..2]     AdainResBlk1d(1090 -> 1024) on cat(x, asr_res, F0, N)
  decode[3]        AdainResBlk1d(1090 -> 512, upsample x2) on the same cat -> x_pre (1, 512, 2T)

Layout: (1, C, 1, T) tensors, Conv1d(k) -> Conv2d((1, k)) (the stride-2 F0/N
convs keep their stride), style fc as a 1x1 conv on s (1, 128, 1, 1), masked
two-pass AdaIN, and the depthwise ConvTranspose upsample as two phases
interleaved by nearest upsampling and 0/1 phase masks. Masking matches the
shipping export (AdaIN statistics over valid frames, padded frames zeroed after
each AdaIN and after the upsample).

Contract (unchanged): asr (1, 512, T), f0 (1, 1, 2T), n_input (1, 1, 2T),
ref_s (1, 256), mask (1, 1, T), all fp32 -> x_pre (1, 512, 2T) fp32.

Usage:
    uv run --no-sync python scripts/export_decoder_pre_ane.py --weights k/kokoro-v1_0.pth \
        --config k/config.json --out DIR [--buckets 3,7,10,15,30] [--check] [--no-export]
    uv run --no-sync python scripts/export_decoder_pre_ane.py --out DIR --coreml-check NAME [--shipping]
        # numerics only, one package per process (coremltools + torch GIL crash)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BUCKET_T = {3: 120, 7: 280, 10: 400, 15: 600, 30: 1200}  # bucket seconds -> frame_count T
BENCH = {"3s": 3, "7s": 7, "10s": 10, "15s": 15, "30s": 30}  # bench input -> bucket seconds


def build(weights: str, config: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    sys.path.insert(0, str(REPO))
    from export_decoder_pre import DecoderPreWrapper
    from kokoro.model import KModel

    def conv2d(c1: nn.Module) -> nn.Conv2d:
        """Conv1d (weight_norm resolved) -> Conv2d with a (1, k) kernel, same stride."""
        w = c1.weight.detach()
        conv = nn.Conv2d(w.shape[1], w.shape[0], (1, w.shape[2]), stride=(1, c1.stride[0]),
                         padding=(0, c1.padding[0]), bias=c1.bias is not None)
        conv.weight.data = w[:, :, None, :].clone()
        if c1.bias is not None:
            conv.bias.data = c1.bias.detach().clone()
        return conv

    def up2(t):
        return F.interpolate(t, scale_factor=(1.0, 2.0), mode="nearest")

    class AdaIN(nn.Module):
        """Masked AdaIN1d on (1, C, 1, T): two-pass centring, then the style affine."""

        def __init__(self, norm):
            super().__init__()
            self.fc = nn.Conv2d(norm.fc.in_features, norm.fc.out_features, 1)
            self.fc.weight.data = norm.fc.weight.detach()[:, :, None, None].clone()
            self.fc.bias.data = norm.fc.bias.detach().clone()
            self.c, self.eps = norm.num_features, norm.eps

        def forward(self, x, s4, m, inv_fill):
            c = (x - (x * m).mean(dim=3, keepdim=True) * inv_fill) * m
            c = (c - c.mean(dim=3, keepdim=True) * inv_fill) * m
            var = (c * c).mean(dim=3, keepdim=True) * inv_fill
            gamma, beta = self.fc(s4).split(self.c, dim=1)
            return ((1.0 + gamma) * (c * torch.rsqrt(var + self.eps)) + beta) * m

    class Block(nn.Module):
        """AdainResBlk1d (masked path) in (1, C, 1, T) layout."""

        def __init__(self, b):
            super().__init__()
            self.up = b.upsample_type != "none"
            self.norm1, self.norm2 = AdaIN(b.norm1), AdaIN(b.norm2)
            self.conv1, self.conv2 = conv2d(b.conv1), conv2d(b.conv2)
            self.sc = conv2d(b.conv1x1) if b.learned_sc else None
            if self.up:
                # ConvTranspose1d(k=3, s=2, p=1, op=1, depthwise):
                #   y[2i] = w1 x[i] + b,  y[2i+1] = w2 x[i] + w0 x[i+1] + b
                w = b.pool.weight.detach()[:, 0, :]  # (C, 3)
                self.register_buffer("w0", w[:, 0].view(1, -1, 1, 1).clone())
                self.register_buffer("w1", w[:, 1].view(1, -1, 1, 1).clone())
                self.register_buffer("w2", w[:, 2].view(1, -1, 1, 1).clone())
                self.register_buffer("pb", b.pool.bias.detach().view(1, -1, 1, 1).clone())

        def pool(self, x, even, odd):
            nxt = F.pad(x[..., 1:], (0, 1))
            return up2(x * self.w1) * even + up2(x * self.w2 + nxt * self.w0) * odd + self.pb

        def forward(self, x, s4, m, inv, mo, even, odd):
            h = F.leaky_relu(self.norm1(x, s4, m, inv), 0.2)
            if self.up:
                h = self.pool(h, even, odd) * mo
            h = self.conv1(h)
            h = self.conv2(F.leaky_relu(self.norm2(h, s4, mo, inv), 0.2))
            sc = up2(x) if self.up else x
            if self.sc is not None:
                sc = self.sc(sc)
            return (h + sc) * (2.0 ** -0.5)

    class DecoderPreANE(nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.f0_conv, self.n_conv = conv2d(dec.F0_conv), conv2d(dec.N_conv)
            self.encode = Block(dec.encode)
            self.asr_res = conv2d(dec.asr_res[0])
            self.decode = nn.ModuleList(Block(b) for b in dec.decode)

        def forward(self, asr, f0, n_input, ref_s, mask):
            t = asr.shape[-1]
            m = mask.unsqueeze(2)  # (1, 1, 1, T)
            mu = up2(m)
            # 1 / fill as a (1, 1, 1, 1) factor (same at T and 2T); the clamp only
            # guards an all-padding mask.
            inv = 1.0 / m.mean(dim=3, keepdim=True).clamp(min=1e-4)
            phase = torch.arange(2 * t, dtype=asr.dtype).remainder(2).view(1, 1, 1, 2 * t)
            even, odd = 1.0 - phase, phase
            s4 = ref_s[:, :128].reshape(1, 128, 1, 1)
            a4 = asr.unsqueeze(2)
            f0c = self.f0_conv(f0.unsqueeze(2))
            nc = self.n_conv(n_input.unsqueeze(2))
            x = self.encode(torch.cat([a4, f0c, nc], dim=1), s4, m, inv, m, even, odd)
            res = self.asr_res(a4)
            for b in self.decode:  # every decode block gets the residual cat
                x = b(torch.cat([x, res, f0c, nc], dim=1), s4, m, inv, mu if b.up else m, even, odd)
            return x.squeeze(2)

    km = KModel(repo_id="hexgrad/Kokoro-82M", config=json.load(open(config)), model=weights,
                disable_complex=True).eval()
    ship, ane = DecoderPreWrapper(km.decoder).eval(), DecoderPreANE(km.decoder).eval()
    for mod in (ship, ane):
        for q in mod.parameters():
            q.requires_grad_(False)
    return km, ship, ane


def bench_case(km, name: str):
    """Unpadded asr (1, 512, frames), F0/N (1, 2*frames), ref_s, as KModel.forward_with_tokens."""
    import torch

    inp = json.load(open(REPO / "outputs/swift_bench_inputs" / f"{name}.json"))
    ids = torch.tensor(inp["input_ids"][: inp["num_tokens"]])[None]
    ref_s = torch.tensor(inp["ref_s"])[None]
    with torch.no_grad():
        n = ids.shape[1]
        lengths = torch.tensor([n])
        text_mask = torch.zeros(1, n, dtype=torch.bool)
        d_en = km.bert_encoder(km.bert(ids, attention_mask=torch.ones_like(ids))).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = km.predictor.text_encoder(d_en, s, lengths, text_mask)
        x, _ = km.predictor.lstm(d)
        dur = torch.round(torch.sigmoid(km.predictor.duration_proj(x)).sum(-1)).clamp(min=1).long().squeeze()
        idx = torch.repeat_interleave(torch.arange(n), dur)
        aln = torch.zeros(n, idx.shape[0])
        aln[idx, torch.arange(idx.shape[0])] = 1
        en = d.transpose(-1, -2) @ aln[None]
        f0, nn_ = km.predictor.F0Ntrain(en, s)
        asr = km.text_encoder(ids, lengths, text_mask) @ aln[None]
    return asr, f0, nn_, ref_s


def padded(asr, f0, n, t: int):
    import torch

    frames = asr.shape[-1]
    assert frames <= t, (frames, t)
    a = torch.zeros(1, 512, t)
    a[..., :frames] = asr
    fp, np_ = torch.zeros(1, 1, 2 * t), torch.zeros(1, 1, 2 * t)
    fp[..., : 2 * frames] = f0
    np_[..., : 2 * frames] = n
    mask = torch.zeros(1, 1, t)
    mask[..., :frames] = 1
    return a, fp, np_, mask


def check(km, ship, ane, out: Path) -> None:
    import torch

    for name, sec in BENCH.items():
        t = BUCKET_T[sec]
        asr, f0, n, ref_s = bench_case(km, name)
        frames = asr.shape[-1]
        a, fp, np_, mask = padded(asr, f0, n, t)
        with torch.no_grad():
            ref = ship(a, fp, np_, ref_s, mask)
            got = ane(a, fp, np_, ref_s, mask)
        v = 2 * frames
        d = (ref - got)[..., :v].abs()
        print(f"{name} frames={frames} T={t}: x_pre max {d.max().item():.2e} mean {d.mean().item():.2e} "
              f"(ref max |x| {ref[..., :v].abs().max().item():.1f}); padded-tail max "
              f"{(ref - got)[..., v:].abs().max().item() if v < 2 * t else 0:.2e}", flush=True)
        np.savez(out / f"ref_{name}.npz", asr=a.numpy(), f0=fp.numpy(), n_input=np_.numpy(),
                 ref_s=ref_s.numpy(), mask=mask.numpy(), x_pre=ref.numpy(), frames=frames, sec=sec)


def export(ane, sec: int, out: Path) -> None:
    import coremltools as ct
    import torch

    t = BUCKET_T[sec]
    ex = (torch.randn(1, 512, t), torch.randn(1, 1, 2 * t), torch.randn(1, 1, 2 * t),
          torch.randn(1, 256), torch.ones(1, 1, t))
    traced = torch.jit.trace(ane, ex, check_trace=False)
    ml = ct.convert(traced, inputs=[ct.TensorType(name="asr", shape=(1, 512, t), dtype=np.float32),
                                    ct.TensorType(name="f0", shape=(1, 1, 2 * t), dtype=np.float32),
                                    ct.TensorType(name="n_input", shape=(1, 1, 2 * t), dtype=np.float32),
                                    ct.TensorType(name="ref_s", shape=(1, 256), dtype=np.float32),
                                    ct.TensorType(name="mask", shape=(1, 1, t), dtype=np.float32)],
                    outputs=[ct.TensorType(name="x_pre", dtype=np.float32)],
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS16,
                    compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
    ml.save(str(out / f"kokoro_decoder_pre_ane_{sec}s.mlpackage"))
    print("saved", sec, flush=True)


def coreml_check(out: Path, name: str, shipping: bool) -> None:
    """Core ML numerics vs the PyTorch fp32 reference (no timing). Torch-free process."""
    import coremltools as ct

    r = np.load(out / f"ref_{name}.npz")
    sec, v = int(r["sec"]), 2 * int(r["frames"])
    path = (REPO / f"coreml/kokoro_decoder_pre_{sec}s.mlpackage" if shipping
            else out / f"kokoro_decoder_pre_ane_{sec}s.mlpackage")
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    o = model.predict({k: r[k] for k in ("asr", "f0", "n_input", "ref_s", "mask")})["x_pre"]
    d = np.abs(o[..., :v] - r["x_pre"][..., :v])
    print(f"{name} {'shipping' if shipping else 'ane'} dtype={o.dtype} shape={o.shape}: x_pre max {d.max():.3f} "
          f"p99 {np.percentile(d, 99):.4f} mean {d.mean():.5f} (ref max |x| {np.abs(r['x_pre'][..., :v]).max():.1f})",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--config")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--buckets", default="3,7,10,15,30")
    ap.add_argument("--check", action="store_true", help="PyTorch parity on the bench inputs; writes ref_*.npz")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--coreml-check", metavar="NAME", help="Core ML numerics vs ref_NAME.npz, then exit")
    ap.add_argument("--shipping", action="store_true", help="with --coreml-check: the shipping package")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.coreml_check:
        coreml_check(a.out, a.coreml_check, a.shipping)
        return
    km, ship, ane = build(a.weights, a.config)
    if a.check:
        check(km, ship, ane, a.out)
    if not a.no_export:
        for sec in [int(v) for v in a.buckets.split(",")]:
            export(ane, sec, a.out)


if __name__ == "__main__":
    main()
