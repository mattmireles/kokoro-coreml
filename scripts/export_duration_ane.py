"""Export the duration front (PL-BERT + text-encoder CNN) in the ANE transformer layout.

Why: the naive split front (scripts/export_duration_split.py) ran BERT on the
ANE at 60 ms (t64) and 496 ms (t512), slower than the whole duration model on
the CPU. BERT is ~63 of the ~76 GFLOP at t512, which the ANE should do in
~15 ms. The exported graph was ANE-hostile, not the math.

This file re-expresses the same weights in Apple's ml-ane-transformers layout:
  - every tensor is (B, C, 1, S), channels first, never transposed again;
  - nn.Linear -> 1x1 Conv2d, Conv1d(k) -> Conv2d((1, k));
  - LayerNorm reduces over dim 1 (channels);
  - attention is split per head; scores are (B, S_k, 1, S_q) so softmax runs
    over dim 1, and the padding mask is an additive bias on the key axis.

The PL-BERT config (Kokoro-82M config.json "plbert"): ALBERT, embedding 128,
hidden 768, 12 heads of 64, intermediate 2048, 12 layers sharing one layer's
weights, gelu_new, LayerNorm eps 1e-12.

  front (ANE): input_ids, attention_mask -> d_en (1, 512, T), t_cnn (1, 512, T)
  back  (CPU): unchanged Back from export_duration_split.py (the LSTMs), in fp16:
               ~1.7x faster than fp32 on the M2 Air CPU, and pred_dur matches the
               shipping fp16 whole model (both are off by one frame on 1 of 476
               tokens of the 30 s bench input vs PyTorch fp32).

Usage:
    uv run --no-sync python scripts/export_duration_ane.py --weights k/kokoro-v1_0.pth \
        --config k/config.json --tokens 64,128,256,512 --out DIR [--check] [--no-export]
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
sys.path.insert(0, str(REPO / "scripts"))

from coreml_export_duration import DurationModel  # noqa: E402
from export_duration_split import Back, Front  # noqa: E402
from kokoro.model import KModel  # noqa: E402

BENCH = ["3s", "10s", "30s"]
MASK_BIAS = -1e4  # fp16-safe; exp(-1e4) is 0, same as the reference's finfo.min


def conv1x1(linear: nn.Linear) -> nn.Conv2d:
    conv = nn.Conv2d(linear.in_features, linear.out_features, 1)
    conv.weight.data = linear.weight.data[:, :, None, None].clone()
    conv.bias.data = linear.bias.data.clone()
    return conv


class ChannelNorm(nn.Module):
    """LayerNorm over dim 1 of a (B, C, 1, S) tensor."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, eps: float):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone().view(1, -1, 1, 1))
        self.bias = nn.Parameter(bias.detach().clone().view(1, -1, 1, 1))
        self.eps = eps

    def forward(self, x):
        x = x - x.mean(dim=1, keepdim=True)
        var = (x * x).mean(dim=1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight + self.bias


def norm_from(ln: nn.LayerNorm) -> ChannelNorm:
    return ChannelNorm(ln.weight, ln.bias, ln.eps)


class AneAlbertLayer(nn.Module):
    """One AlbertLayer (attention + FFN, post-norm) in (B, C, 1, S) layout."""

    def __init__(self, layer, heads: int):
        super().__init__()
        att = layer.attention
        self.heads = heads
        self.q, self.k, self.v = conv1x1(att.query), conv1x1(att.key), conv1x1(att.value)
        self.o = conv1x1(att.dense)
        self.norm1 = norm_from(att.LayerNorm)
        self.ffn, self.ffn_out = conv1x1(layer.ffn), conv1x1(layer.ffn_output)
        self.norm2 = norm_from(layer.full_layer_layer_norm)

    def forward(self, x, key_bias):
        d = x.shape[1] // self.heads
        q = self.q(x) * (1.0 / math.sqrt(d))
        qs, ks, vs = q.split(d, 1), self.k(x).split(d, 1), self.v(x).split(d, 1)
        ctx = []
        for qh, kh, vh in zip(qs, ks, vs):
            w = torch.einsum("bchq,bkhc->bkhq", qh, kh.transpose(1, 3)) + key_bias
            w = w.softmax(dim=1)
            ctx.append(torch.einsum("bkhq,bchk->bchq", w, vh))
        h = self.norm1(x + self.o(torch.cat(ctx, dim=1)))
        f = self.ffn_out(F.gelu(self.ffn(h), approximate="tanh"))
        return self.norm2(h + f)


class AneFront(nn.Module):
    """Same outputs as export_duration_split.Front, ANE layout throughout."""

    def __init__(self, dm: DurationModel):
        super().__init__()
        km = dm.kmodel
        cfg = km.bert.config
        assert cfg.num_hidden_groups == 1 and cfg.inner_group_num == 1, "expects one shared layer"
        assert cfg.hidden_act == "gelu_new" and cfg.position_embedding_type == "absolute"
        emb = km.bert.embeddings
        self.word = emb.word_embeddings
        self.pos = nn.Parameter(emb.position_embeddings.weight.detach().clone())
        self.tok0 = nn.Parameter(emb.token_type_embeddings.weight[0].detach().clone())
        self.emb_norm = norm_from(emb.LayerNorm)
        enc = km.bert.encoder
        self.map_in = conv1x1(enc.embedding_hidden_mapping_in)
        self.layer = AneAlbertLayer(enc.albert_layer_groups[0].albert_layers[0], cfg.num_attention_heads)
        self.n_layers = cfg.num_hidden_layers
        self.proj = conv1x1(km.bert_encoder)
        te = km.text_encoder
        self.t_emb = te.embedding
        self.cnn = nn.ModuleList()
        for block in te.cnn:
            c1, ln = block[0], block[1]
            w = c1.weight.detach()  # weight_norm parametrization resolves on access
            conv = nn.Conv2d(w.shape[1], w.shape[0], (1, w.shape[2]), padding=(0, c1.padding[0]))
            conv.weight.data = w[:, :, None, :].clone()
            conv.bias.data = c1.bias.detach().clone()
            self.cnn.append(nn.ModuleList([conv, ChannelNorm(ln.gamma, ln.beta, ln.eps)]))

    def forward(self, input_ids, attention_mask):
        t = input_ids.shape[1]
        m = attention_mask.to(torch.float32)
        key_bias = ((1.0 - m) * MASK_BIAS).view(1, t, 1, 1)
        keep = m.view(1, 1, 1, t)
        e = self.word(input_ids) + self.pos[:t] + self.tok0  # (1, T, 128)
        x = self.emb_norm(e.transpose(1, 2).unsqueeze(2))
        x = self.map_in(x)
        for _ in range(self.n_layers):
            x = self.layer(x, key_bias)
        d_en = self.proj(x).squeeze(2)
        y = self.t_emb(input_ids).transpose(1, 2).unsqueeze(2) * keep
        for conv, norm in self.cnn:
            y = F.leaky_relu(norm(conv(y)), 0.2) * keep
        return d_en, y.squeeze(2)


def bench_inputs(name: str):
    inp = json.load(open(REPO / "outputs/swift_bench_inputs" / f"{name}.json"))
    n = inp["num_tokens"]
    t = 1 << (n - 1).bit_length()
    ids = torch.zeros(1, t, dtype=torch.int32)
    ids[0, :n] = torch.tensor(inp["input_ids"][:n])
    mask = torch.zeros(1, t, dtype=torch.int32)
    mask[0, :n] = 1
    return n, ids, mask, torch.tensor(inp["ref_s"])[None]


def check(dm, front, ane, back) -> None:
    for name in BENCH:
        n, ids, mask, ref_s = bench_inputs(name)
        speed = torch.ones(1)
        with torch.no_grad():
            want = dm(ids, ref_s, speed, mask)
            f_ref, f_ane = front(ids, mask), ane(ids, mask)
            got = back(*f_ane, ref_s, speed, mask)
        diffs = [(a - b)[..., :n].abs().max().item() for a, b in zip(f_ref, f_ane)]
        mism = (want[0] != got[0])[..., :n].sum().item()
        print(f"{name} n={n} t={ids.shape[1]}: d_en {diffs[0]:.2e}  t_cnn {diffs[1]:.2e}  "
              f"pred_dur mismatches {mism}/{n}  t_en {(want[2] - got[2]).abs().max().item():.2e}", flush=True)


def export(ane, back, t: int, out: Path) -> None:
    import coremltools as ct

    ids = torch.ones(1, t, dtype=torch.int32)
    mask = torch.ones(1, t, dtype=torch.int32)
    with torch.no_grad():
        d_en, t_cnn = ane(ids, mask)
    tf = torch.jit.trace(ane, (ids, mask), check_trace=False)
    mf = ct.convert(tf, inputs=[ct.TensorType(name="input_ids", shape=(1, t), dtype=np.int32),
                                ct.TensorType(name="attention_mask", shape=(1, t), dtype=np.int32)],
                    outputs=[ct.TensorType(name="d_en"), ct.TensorType(name="t_cnn")],
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS18,
                    compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
    mf.save(str(out / f"kokoro_duration_ane_front_t{t}.mlpackage"))
    ex = (d_en, t_cnn, torch.rand(1, 256), torch.ones(1), mask)
    tb = torch.jit.trace(back, ex, check_trace=False)
    mb = ct.convert(tb, inputs=[ct.TensorType(name="d_en", shape=tuple(d_en.shape)),
                                ct.TensorType(name="t_cnn", shape=tuple(t_cnn.shape)),
                                ct.TensorType(name="ref_s", shape=(1, 256)),
                                ct.TensorType(name="speed", shape=(1,)),
                                ct.TensorType(name="attention_mask", shape=(1, t), dtype=np.int32)],
                    outputs=[ct.TensorType(name=n) for n in ["pred_dur", "d", "t_en", "s", "ref_s_out"]],
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS18,
                    compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
    mb.save(str(out / f"kokoro_duration_back_t{t}.mlpackage"))
    print("saved", t, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokens", default="64,128,256,512")
    ap.add_argument("--check", action="store_true", help="PyTorch parity on 3s/10s/30s bench inputs")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    km = KModel(repo_id="hexgrad/Kokoro-82M", config=json.load(open(a.config)), model=a.weights,
                disable_complex=True).eval()
    dm = DurationModel(km).eval()
    for p in dm.parameters():
        p.requires_grad_(False)
    front, back, ane = Front(dm).eval(), Back(dm).eval(), AneFront(dm).eval()
    for p in ane.parameters():
        p.requires_grad_(False)
    if a.check:
        check(dm, front, ane, back)
    if not a.no_export:
        for t in [int(v) for v in a.tokens.split(",")]:
            export(ane, back, t, a.out)


if __name__ == "__main__":
    main()
