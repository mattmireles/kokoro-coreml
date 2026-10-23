"""Export the hybrid generator: Neural Engine convolutions + host-side AdaIN/Snake.

Why: on the ANE an elementwise pass costs about as much as a whole k=11 conv,
and the generator has ~10 of them per conv (AdaIN statistics, Snake). The GPU
does those passes at memory bandwidth with native `sin` and fp32 statistics.
Measured on an M1 per stage-2 unit: ANE conv 11 ms + Metal AdaIN+Snake 5 ms,
vs ~45 ms with everything on the ANE (README/Notes/ane-folded-generator.md).

So the generator is cut at every AdaIN. The ANE runs only convolutions (one
multifunction package, one function per conv layer); the host (Metal, in
swift/Sources/KokoroPipeline/HybridGenerator.swift) runs AdaIN + Snake and
the branch joins. Tensors stay row-folded, (rows, C, 1, W), as in
probe_generator_ane_fold.py, so every axis fits the ANE's 16,384 cap.

Snake is folded as in the probe: the host emits z = a * AdaIN(x) and computes
z + sin(z)^2; the following conv carries 1/a in its weights.

Outputs in --out:
  hybrid_generator_r{R}.mlpackage   multifunction, fixed shapes for R rows; functions in plan.json
  adain.bin                    float32 AdaIN tables (fc weight, fc bias, alpha), see plan.json
  plan.json                    geometry, function list, AdaIN table offsets
and, with --input-json, in/*.bin + ref.npz for parity checks (``--check``
runs the hybrid step sequence in PyTorch fp32 against the unfolded generator).

Usage:
    uv run --no-sync python scripts/export_hybrid_generator.py --weights k/kokoro-v1_0.pth \
        --config k/config.json --input-json outputs/swift_bench_inputs/30s.json --check --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import probe_generator_ane_fold as P  # noqa: E402
from kokoro.model import KModel  # noqa: E402

CORE = 224
HALO = 8
W0 = CORE + 2 * HALO            # 240 x_pre frames per row
W1 = W0 * 10                    # 2400
EXTRA2 = 480                    # +1 reflect sample, rounded so W2 % 480 == 0
W2 = W0 * 60 + EXTRA2           # 14880 (< 16,384)
# Stage 2 runs on a crop of the host's W2-wide rows: CROP2 columns off each side.
# The stage-0 halo (8 frames) is sized for the whole network's receptive field
# (~480 stage-2 samples); after stage 1 consumed its share, stage 2 needs only
# ~60-120 valid halo samples, so the outer 240 are dead weight. W2S (13,920)
# stays a multiple of 480.
CROP2 = 240
W2S = W2 - 2 * CROP2            # 13920: stage-2 tensors
# W1, W2 and W2S are multiples of 480 = 32 * 15: every dilation-3 / dilation-5 phase row
# (W/d samples) is a multiple of 32 fp16 values, i.e. 64-byte IOSurface rows.
ROWS = (2, 3, 4, 6, 10, 11)     # rows at core 224: 3/7/10/15 s buckets, 10 for 22-28 s, 11 for 30 s


# --------------------------------------------------------------------------
# ANE functions (pure convolutions)
# --------------------------------------------------------------------------

class Leaky(nn.Module):
    def __init__(self, slope, inner):
        super().__init__(); self.slope = slope; self.inner = inner
    def forward(self, x):
        return self.inner(F.leaky_relu(x, self.slope))


class PolyRaw(nn.Module):
    """leaky -> polyphase upsample conv, WITHOUT the width shuffle: output
    (rows, Cout*s, 1, Q), channel cout*s + r holding output sample q*s + r.
    The host interleaves it (Metal `unshuffle`): the in-graph transpose cost
    the ANE 10-27 ms per upsampler on an M1."""
    def __init__(self, ups):
        super().__init__(); self.p = P.PolyphaseConvT(ups)
    def forward(self, x):
        x = F.leaky_relu(x, 0.1)
        n, ch, _, q = x.shape
        x = torch.cat([torch.zeros(n, ch, 1, -self.p.omin), x, torch.zeros(n, ch, 1, self.p.omax)], dim=3)
        return self.p.conv(x)


class NoiseConv0(nn.Module):
    """Stride-6 noise conv on host-built space-to-depth input (rows, 22*6, 1, Q6):
    the in-graph space-to-depth ran at 0.09 TFLOPS on an M1 ANE (37 ms)."""
    def __init__(self, c):
        super().__init__(); self.c = P.StridedConv(c)
    def forward(self, har6):
        return self.c.conv(har6)[..., :W1]


S2D = 6
HAR_PAD = 3                                   # noise_convs[0] padding
Q6 = (-(-(W2 + HAR_PAD) // S2D) + 1 + 31) // 32 * 32  # space-to-depth columns: spare k=2 tap, 64-byte IOSurface rows


def space_to_depth(har):
    """(rows, 22, 1, W2) -> (rows, 22*6, 1, Q6): [n, c*6+e, q] = pad(har)[n, c, q*6+e]."""
    n, c, _, w = har.shape
    total = Q6 * S2D
    padded = torch.cat([torch.zeros(n, c, 1, HAR_PAD), har, torch.zeros(n, c, 1, total - w - HAR_PAD)], dim=3)
    return padded.reshape(n, c, Q6, S2D).permute(0, 1, 3, 2).reshape(n, c * S2D, 1, Q6)


def unshuffle(poly, s, reflect: bool, width: int):
    """(rows, C*s, 1, Q) polyphase output -> (rows, C, 1, width) interleaved; with
    `reflect`, prepend the reflect-pad sample and zero-extend (ups1)."""
    n, cs, _, q = poly.shape
    y = poly.reshape(n, cs // s, s, q).permute(0, 1, 3, 2).reshape(n, cs // s, 1, q * s)
    if reflect:
        y = torch.cat([y[..., 1:2], y], dim=3)
    return torch.cat([y, torch.zeros(n, y.shape[1], 1, width - y.shape[3])], dim=3) if y.shape[3] < width else y


class PhaseConv(nn.Module):
    """Dilation-d conv on a phase-major row: the host (Metal) stores each row's
    samples as d phases of W/d (sample q*d + p at p*W/d + q), so a contiguous
    reshape to (rows, C, d, W/d) turns the dilated conv into a plain (1, k)
    conv over d wide rows. On an M1 ANE the native dilation-3 conv ran at
    ~2.2 TFLOPS and the earlier (W/d, d) reshape at ~2.5; plain wide-row convs
    run at 6-8."""
    def __init__(self, c: nn.Conv1d, in_scale):
        super().__init__()
        self.d = c.dilation[0]
        k = c.kernel_size[0]
        self.conv = nn.Conv2d(c.in_channels, c.out_channels, (1, k), padding=(0, (k - 1) // 2), bias=c.bias is not None)
        self.conv.weight.data = (c.weight.detach() / in_scale.reshape(1, -1, 1)).unsqueeze(2).clone()
        if c.bias is not None:
            self.conv.bias.data = c.bias.detach().clone()

    def forward(self, x):
        n, ch, _, w = x.shape
        y = self.conv(x.reshape(n, ch, self.d, w // self.d))
        return y.reshape(n, y.shape[1], 1, w)


def to_phase(x, d):
    """Natural (n, c, 1, w) -> phase-major rows (same shape)."""
    if d == 1:
        return x
    n, c, _, w = x.shape
    return x.reshape(n, c, w // d, d).permute(0, 1, 3, 2).reshape(n, c, 1, w)


def from_phase(x, d):
    if d == 1:
        return x
    n, c, _, w = x.shape
    return x.reshape(n, c, d, w // d).permute(0, 1, 3, 2).reshape(n, c, 1, w)


class ConvRes(nn.Module):
    """conv(t) + residual: the second conv of each resblock pair. Moving the add
    to Metal was measured: no faster on M1, A14 or A17 (README/Notes)."""
    def __init__(self, conv):
        super().__init__(); self.conv = conv
    def forward(self, t, r):
        return self.conv(t) + r


PHASE = {}  # function name -> dilation d whose input/output rows are phase-major
# --resid-gpu: export each pair's second conv without its residual add; the host
# adds it on the GPU. On an A14 the residual input cost the ANE +3.7 ms (k3) and
# +4.4 ms (k11) per stage-2 call after a GPU pass, and the ANE is the bottleneck.
RESID_GPU = False


def build(gen):
    """Return (functions: name -> (module, input specs), adain list in host order)."""
    fns = {}
    adain = []

    def resblock(prefix, rb, c):
        for k in range(3):
            a1 = rb.alpha1[k].detach().flatten()
            a2 = rb.alpha2[k].detach().flatten()
            adain.append((f"{prefix}_n1_{k}", rb.adain1[k], a1))
            conv = rb.convs1[k]
            ks, d = conv.kernel_size[0], conv.dilation[0]
            if d > 1 and (ks - 1) * d + 1 > 11:      # small kernels go dense (P.ane_conv); phase-major k3 measured no faster
                fns[f"{prefix}_c1_{k}"] = (PhaseConv(conv, a1), [("t", c)])
                PHASE[f"{prefix}_c1_{k}"] = d
            else:
                fns[f"{prefix}_c1_{k}"] = (P.scaled_conv(conv, a1), [("t", c)])
            adain.append((f"{prefix}_n2_{k}", rb.adain2[k], a2))
            if RESID_GPU:   # the host adds the residual on the GPU (x += y after the call)
                fns[f"{prefix}_c2_{k}"] = (P.scaled_conv(rb.convs2[k], a2), [("t", c)])
            else:
                fns[f"{prefix}_c2_{k}"] = (ConvRes(P.scaled_conv(rb.convs2[k], a2)), [("t", c), ("r", c)])

    # Stage 1 (256 channels at W1), stage 2 (128 channels at W2).
    fns["nc0"] = (NoiseConv0(gen.noise_convs[0]), [("har6", "har6")])
    resblock("nr0", gen.noise_res[0], "s1")
    fns["ups0"] = (PolyRaw(gen.ups[0]), [("x", "x_pre")])
    for j in range(3):
        resblock(f"rb{j}", gen.resblocks[j], "s1")
    fns["nc1"] = (P.conv2d_from(gen.noise_convs[1]), [("har", "har")])
    resblock("nr1", gen.noise_res[1], "s2")
    fns["ups1"] = (PolyRaw(gen.ups[1]), [("x", "s1")])
    for j in range(3, 6):
        resblock(f"rb{j}", gen.resblocks[j], "s2")
    fns["post"] = (Leaky(0.01, P.conv2d_from(gen.conv_post)), [("x", "s2")])
    return fns, adain


SHAPES = {"x_pre": lambda r: (r, 512, 1, W0), "har": lambda r: (r, 22, 1, W2S),
          "har6": lambda r: (r, 22 * S2D, 1, Q6),
          "s1": lambda r: (r, 256, 1, W1), "s2": lambda r: (r, 128, 1, W2S)}


def crop2(t):
    """Stage-2 window of a W2-wide tensor (see CROP2)."""
    return t[..., CROP2:CROP2 + W2S]


# --------------------------------------------------------------------------
# Host steps, as the Swift runtime runs them (fp32 here)
# --------------------------------------------------------------------------

def adasnake(x, s, sm, zm, adain_mod, a):
    """Exact masked AdaIN, scaled by a, then z + sin(z)^2."""
    fill = sm.sum()
    mean = (x * sm).sum(dim=(0, 3), keepdim=True) / fill
    var = (((x - mean) ** 2) * sm).sum(dim=(0, 3), keepdim=True) / fill
    h = adain_mod.fc(s).view(1, -1, 1, 1)
    c = h.shape[1] // 2
    gamma, beta = h[:, :c], h[:, c:]
    z = (((x - mean) * torch.rsqrt(var + adain_mod.eps)) * (1 + gamma) + beta) * a.view(1, -1, 1, 1) * zm
    return z + torch.sin(z) ** 2


def run_hybrid(fns, adain, feeds, s):
    """Mirror of HybridGenerator.swift's step order."""
    ada = iter(adain)
    sm1, zm1, sm2, zm2 = feeds["sm1"], feeds["zm1"], feeds["sm2"], feeds["zm2"]

    def resblock(prefix, x, sm, zm):
        for k in range(3):
            d = PHASE.get(f"{prefix}_c1_{k}", 1)
            _, m1, a1 = next(ada)
            t = fns[f"{prefix}_c1_{k}"][0](to_phase(adasnake(x, s, sm, zm, m1, a1), d))
            _, m2, a2 = next(ada)
            u = adasnake(from_phase(t, d), s, sm, zm, m2, a2)
            x = fns[f"{prefix}_c2_{k}"][0](u) + x if RESID_GPU else fns[f"{prefix}_c2_{k}"][0](u, x)
        return x

    xs = resblock("nr0", fns["nc0"][0](space_to_depth(feeds["har"])), sm1, zm1)
    x = (unshuffle(fns["ups0"][0](feeds["x_pre"]), 10, False, W1) + xs) * zm1
    x = sum(resblock(f"rb{j}", x, sm1, zm1) for j in range(3)) * (zm1 / 3)
    sm2, zm2 = crop2(sm2), crop2(zm2)
    xs = resblock("nr1", fns["nc1"][0](crop2(feeds["har"])), sm2, zm2)
    x = (crop2(unshuffle(fns["ups1"][0](x), 6, True, W2)) + xs) * zm2
    x = sum(resblock(f"rb{j}", x, sm2, zm2) for j in range(3, 6)) * (zm2 / 3)
    return F.pad(fns["post"][0](x), (CROP2, CROP2))   # back to the host's W2 layout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-json")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-convert", action="store_true")
    ap.add_argument("--rows", default=",".join(str(r) for r in ROWS),
                    help="row counts to export, one fixed-shape package each (the ANE drops enumerated shapes)")
    ap.add_argument("--resid-gpu", action="store_true", help="second convs without the residual add (host adds it)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    global RESID_GPU
    RESID_GPU = a.resid_gpu
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    km = KModel(repo_id="hexgrad/Kokoro-82M", config=json.load(open(a.config)), model=a.weights,
                disable_complex=True).eval()
    gen = km.decoder.generator
    fns, adain = build(gen)
    for m, _ in fns.values():
        m.eval()

    # AdaIN tables, in host order: fc weight (2C x 128), fc bias (2C), alpha (C).
    table, entries, off = [], [], 0
    for name, mod, alpha in adain:
        w = mod.fc.weight.detach().float().numpy().ravel()
        b = mod.fc.bias.detach().float().numpy().ravel()
        al = alpha.float().numpy().ravel()
        entries.append({"name": name, "channels": int(al.size), "eps": mod.eps,
                        "weight": off, "bias": off + w.size, "alpha": off + w.size + b.size})
        table += [w, b, al]
        off += w.size + b.size + al.size
    np.concatenate(table).astype(np.float32).tofile(a.out / "adain.bin")
    # FLOPs per function at one row (scale by rows), for per-function efficiency tables.
    from torch.utils.flop_counter import FlopCounterMode
    flops = {}
    for name, (mod, ins) in fns.items():
        fc = FlopCounterMode(display=False)
        with fc, torch.no_grad():
            mod(*[torch.zeros(*SHAPES[kind](1)) for _, kind in ins])
        flops[name] = fc.get_total_flops()
    plan = {"core": CORE, "halo": HALO, "w0": W0, "w1": W1, "w2": W2, "extra2": EXTRA2,
            "rows": list(ROWS), "functions": list(fns), "adain": entries, "flops_per_row": flops,
            "q6": Q6, "har_pad": HAR_PAD, "phase": PHASE, "crop2": CROP2, "resid_gpu": RESID_GPU}
    (a.out / "plan.json").write_text(json.dumps(plan, indent=1))

    if a.input_json and a.check:
        check(a, km, gen, fns, adain)
    if a.no_convert:
        return
    for r in [int(v) for v in a.rows.split(",")]:
        export(a, fns, r)


def check(a, km, gen, fns, adain):
    inp = json.load(open(a.input_json))
    ids = torch.tensor([inp["input_ids"][: inp["num_tokens"]]])
    ref_s = torch.tensor(inp["ref_s"], dtype=torch.float32)[None]
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
        ref = P.reference_logits(gen, x, s, har)
        ref_audio = P.tail(gen, ref).numpy()
    t_valid = x.shape[2]
    rows = min(r for r in ROWS if r * CORE >= t_valid)  # pad up to a packaged row count, as Swift does

    def rows_first(arr):
        return np.ascontiguousarray(arr.transpose(1, 0, 2)[:, :, None, :])
    feeds = {"x_pre": rows_first(P.fold(x[0].numpy(), 1, 0, rows, CORE, HALO)),
             "har": rows_first(P.fold(har[0].numpy(), 60, EXTRA2, rows, CORE, HALO))}
    sm1, zm1 = P.masks(10, 0, rows, CORE, HALO, t_valid * 10)
    sm2, zm2 = P.masks(60, EXTRA2, rows, CORE, HALO, t_valid * 60 + 1)
    feeds.update(sm1=sm1[:, None, None], zm1=zm1[:, None, None], sm2=sm2[:, None, None], zm2=zm2[:, None, None])
    tf = {k: torch.from_numpy(v) for k, v in feeds.items()}
    with torch.no_grad():
        out = run_hybrid(fns, adain, tf, s)[:, :, 0, :].numpy().transpose(1, 0, 2)
        got = P.unfold(out, 60, EXTRA2, CORE, HALO, ref.shape[2])
        audio = P.tail(gen, torch.from_numpy(got)[None]).numpy()
    snr = 10 * np.log10((ref_audio ** 2).sum() / ((audio - ref_audio) ** 2).sum())
    print(f"hybrid fp32 vs unfolded: audio SNR {snr:.1f} dB, rows {rows}")
    (a.out / "in").mkdir(exist_ok=True)
    for k, v in feeds.items():
        v.astype(np.float32).tofile(a.out / "in" / f"{k}.bin")
    s.numpy().astype(np.float32).tofile(a.out / "in" / "style.bin")
    np.savez(a.out / "ref.npz", logits=ref[0].numpy(), audio=ref_audio, core=CORE, halo=HALO, rows=rows)


def export(a, fns, rows: int):
    """One multifunction package with fixed shapes for ``rows`` rows. Enumerated
    shapes were tried first: on an M1 they ran ~5x slower (ups0 522 ms), i.e. off
    the ANE, so every row count gets its own fixed-shape package."""
    import coremltools as ct
    import tempfile
    desc = None
    tmp = Path(tempfile.mkdtemp())
    for name, (mod, ins) in fns.items():
        example = tuple(torch.rand(*SHAPES[kind](rows)) for _, kind in ins)
        traced = torch.jit.trace(mod, example, check_trace=False)
        inputs = [ct.TensorType(name=n, dtype=np.float16, shape=SHAPES[kind](rows)) for n, kind in ins]
        m = ct.convert(traced, inputs=inputs, outputs=[ct.TensorType(name="y", dtype=np.float16)],
                       compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS18,
                       compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
        path = tmp / f"{name}.mlpackage"
        m.save(str(path))
        if desc is None:
            desc = ct.utils.MultiFunctionDescriptor()
        desc.add_function(str(path), src_function_name="main", target_function_name=name)
        print("converted", name, flush=True)
    desc.default_function_name = "post"
    path = a.out / f"hybrid_generator_r{rows}.mlpackage"
    ct.utils.save_multifunction(desc, str(path))
    print("saved", path)


if __name__ == "__main__":
    main()
