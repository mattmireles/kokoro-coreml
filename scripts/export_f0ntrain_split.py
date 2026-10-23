"""Export F0Ntrain split in two: the shared BiLSTM on the CPU, the blocks on the ANE.

Why: F0Ntrain runs CPU-only today (M2 Air t1200: 32 ms CPU, 40 ms with the ANE
allowed, 65 ms GPU). Most of its ~21 GFLOP at t1200 is the AdainResBlk1d convs,
which the ANE should do fast; the sequential LSTM cannot run there and drags
the whole graph back to the CPU. Same recipe as scripts/export_duration_ane.py.

  front (CPU):  en (1, 640, T), mask (1, 1, T) -> x (1, 512, T)
                MaskedBidirectionalLSTM, exactly as export_f0ntrain.F0NtrainWrapper.
  back  (ANE):  x (1, 512, T), s (1, 128), mask (1, 1, T)
                -> F0_pred (1, 2T), N_pred (1, 2T)

The back re-expresses the F0 and N branches in the ANE layout: (1, C, 1, T)
tensors, Conv1d(k) -> Conv2d((1, k)), the depthwise transposed-conv upsample as
two phases interleaved by nearest upsampling and 0/1 phase masks, and AdaIN
with masked two-pass centring (README/Notes/ane-folded-generator.md, "fp16
precision of AdaIN on the ANE"). Masking matches the shipping export: AdaIN
statistics over valid frames only, padded frames zeroed after each AdaIN and
after the upsample.

Usage:
    uv run --no-sync python scripts/export_f0ntrain_split.py --weights k/kokoro-v1_0.pth \
        --config k/config.json --out DIR [--frames 120,280,400,600,1200] [--check] [--no-export]
        [--front-fp32]
    uv run --no-sync python scripts/export_f0ntrain_split.py --coreml-check DIR   # numerics only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BENCH = {"3s": 120, "10s": 400, "30s": 1200}  # bench input -> bucket T


def build(weights: str, config: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    sys.path.insert(0, str(REPO))
    from coreml_export_lstm import MaskedBidirectionalLSTM
    from export_f0ntrain import F0NtrainWrapper
    from kokoro.model import KModel

    def conv2d(c1: nn.Module) -> nn.Conv2d:
        """Conv1d (weight_norm resolved) -> Conv2d with a (1, k) kernel."""
        w = c1.weight.detach()
        conv = nn.Conv2d(w.shape[1], w.shape[0], (1, w.shape[2]), padding=(0, c1.padding[0]),
                         bias=c1.bias is not None)
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

        def forward(self, x, s4, m, inv, mu, inv_u, even, odd):
            h = F.leaky_relu(self.norm1(x, s4, m, inv), 0.2)
            if self.up:
                h = self.pool(h, even, odd) * mu
            h = self.conv1(h)
            h = self.conv2(F.leaky_relu(self.norm2(h, s4, mu, inv_u), 0.2))
            sc = up2(x) if self.up else x
            if self.sc is not None:
                sc = self.sc(sc)
            return (h + sc) * (2.0 ** -0.5)

    class Front(nn.Module):
        def __init__(self, p):
            super().__init__()
            self.shared = MaskedBidirectionalLSTM(p.shared)

        def forward(self, en, mask):
            return self.shared(en.transpose(-1, -2), mask.squeeze(1)).transpose(-1, -2)

    class Back(nn.Module):
        def __init__(self, p):
            super().__init__()
            self.f0 = nn.ModuleList(Block(b) for b in p.F0)
            self.n = nn.ModuleList(Block(b) for b in p.N)
            self.f0_proj, self.n_proj = conv2d(p.F0_proj), conv2d(p.N_proj)

        def forward(self, x, s, mask):
            t = x.shape[-1]
            m = mask.unsqueeze(2)  # (1, 1, 1, T)
            mu = up2(m)
            # 1 / fill as a (1, 1, 1, 1) factor; the clamp only guards an all-padding mask.
            inv = 1.0 / m.mean(dim=3, keepdim=True).clamp(min=1e-4)
            phase = torch.arange(2 * t, dtype=x.dtype).remainder(2).view(1, 1, 1, 2 * t)
            even, odd = 1.0 - phase, phase
            s4 = s.view(1, -1, 1, 1)
            x4 = x.unsqueeze(2)
            outs = []
            for blocks, proj in ((self.f0, self.f0_proj), (self.n, self.n_proj)):
                h = x4
                for i, b in enumerate(blocks):
                    mi, ii = (m, inv) if i < 2 else (mu, inv)
                    mo = m if i == 0 else mu
                    h = b(h, s4, mi, ii, mo, inv, even, odd)
                outs.append(proj(h).view(1, 2 * t))
            return outs[0], outs[1]

    km = KModel(repo_id="hexgrad/Kokoro-82M", config=json.load(open(config)), model=weights,
                disable_complex=True).eval()
    p = km.predictor
    whole, front, back = F0NtrainWrapper(p).eval(), Front(p).eval(), Back(p).eval()
    for mod in (whole, front, back):
        for q in mod.parameters():
            q.requires_grad_(False)
    return km, whole, front, back


def bench_case(km, name: str):
    """en (1, 640, frames) and s (1, 128) as the runtime feeds F0Ntrain, unpadded."""
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
    return en, s


def padded(en, t: int):
    import torch

    frames = en.shape[-1]
    assert frames <= t, (frames, t)
    e = torch.zeros(1, en.shape[1], t)
    e[..., :frames] = en
    mask = torch.zeros(1, 1, t)
    mask[..., :frames] = 1
    return e, mask


def check(km, whole, front, back, out: Path) -> None:
    import torch

    for name, t in BENCH.items():
        en, s = bench_case(km, name)
        frames = en.shape[-1]
        e, mask = padded(en, t)
        with torch.no_grad():
            ref = whole(e, s, mask)
            x = front(e, mask)
            got = back(x, s, mask)
            native = km.predictor.F0Ntrain(en, s)
        v = 2 * frames
        msg = [f"{name} frames={frames} T={t}:"]
        for label, a, b in (("F0", ref[0], got[0]), ("N", ref[1], got[1])):
            d = (a - b)[..., :v].abs()
            msg.append(f"{label} max {d.max().item():.2e} (ref max |{label}| {a[..., :v].abs().max().item():.1f})")
        msg.append(f"| whole-vs-native F0 {(ref[0][..., :v] - native[0]).abs().max().item():.2e}")
        print(" ".join(msg), flush=True)
        np.savez(out / f"ref_{name}.npz", en=e.numpy(), s=s.numpy(), mask=mask.numpy(), x=x.numpy(),
                 F0=ref[0].numpy(), N=ref[1].numpy(), frames=frames, t=t)


def export(front, back, t: int, out: Path, front_fp32: bool) -> None:
    import coremltools as ct
    import torch

    en, mask, s = torch.randn(1, 640, t), torch.ones(1, 1, t), torch.randn(1, 128)
    with torch.no_grad():
        x = front(en, mask)
    tf = torch.jit.trace(front, (en, mask), check_trace=False)
    mf = ct.convert(tf, inputs=[ct.TensorType(name="en", shape=(1, 640, t)),
                                ct.TensorType(name="mask", shape=(1, 1, t))],
                    outputs=[ct.TensorType(name="x")],
                    compute_precision=ct.precision.FLOAT32 if front_fp32 else ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.iOS18,
                    compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
    suffix = "_fp32" if front_fp32 else ""
    mf.save(str(out / f"kokoro_f0ntrain_front_t{t}{suffix}.mlpackage"))
    tb = torch.jit.trace(back, (x, s, mask), check_trace=False)
    mb = ct.convert(tb, inputs=[ct.TensorType(name="x", shape=(1, 512, t)),
                                ct.TensorType(name="s", shape=(1, 128)),
                                ct.TensorType(name="mask", shape=(1, 1, t))],
                    outputs=[ct.TensorType(name="F0_pred"), ct.TensorType(name="N_pred")],
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS18,
                    compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
    mb.save(str(out / f"kokoro_f0ntrain_ane_back_t{t}.mlpackage"))
    print("saved", t, flush=True)


def coreml_check(out: Path) -> None:
    """Core ML numerics vs the PyTorch fp32 reference (no timing). Torch-free process."""
    import coremltools as ct

    cu = ct.ComputeUnit
    for name, t in BENCH.items():
        r = np.load(out / f"ref_{name}.npz")
        v = 2 * int(r["frames"])
        whole = ct.models.MLModel(str(REPO / f"coreml/kokoro_f0ntrain_t{t}.mlpackage"), compute_units=cu.CPU_ONLY)
        front = ct.models.MLModel(str(out / f"kokoro_f0ntrain_front_t{t}.mlpackage"), compute_units=cu.CPU_ONLY)
        back = ct.models.MLModel(str(out / f"kokoro_f0ntrain_ane_back_t{t}.mlpackage"), compute_units=cu.CPU_AND_NE)
        w = whole.predict({"en": r["en"], "s": r["s"], "mask": r["mask"]})
        x = front.predict({"en": r["en"], "mask": r["mask"]})["x"]
        sp = back.predict({"x": x, "s": r["s"], "mask": r["mask"]})
        sx = back.predict({"x": r["x"], "s": r["s"], "mask": r["mask"]})  # back alone on exact x
        line = [f"{name} T={t} x(front) max {np.abs(x - r['x'])[..., : v // 2].max():.2e} |"]
        for key, ref in (("F0", r["F0"]), ("N", r["N"])):
            k = f"{key}_pred"
            for label, o in (("whole", w), ("split", sp), ("back", sx)):
                d = np.abs(o[k][..., :v] - ref[..., :v])
                line.append(f"{key} {label} max {d.max():.3f} p99 {np.percentile(d, 99):.3f} mean {d.mean():.4f};")
        print(" ".join(line), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--config")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frames", default="120,280,400,600,1200")
    ap.add_argument("--check", action="store_true", help="PyTorch parity on 3s/10s/30s; writes ref_*.npz")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--front-fp32", action="store_true", help="fp32 front (default fp16)")
    ap.add_argument("--coreml-check", action="store_true", help="Core ML numerics vs ref_*.npz, then exit")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.coreml_check:
        coreml_check(a.out)
        return
    km, whole, front, back = build(a.weights, a.config)
    if a.check:
        check(km, whole, front, back, a.out)
    if not a.no_export:
        for t in [int(v) for v in a.frames.split(",")]:
            export(front, back, t, a.out, a.front_fp32)


if __name__ == "__main__":
    main()
