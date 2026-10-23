"""Export the duration model split in two: a Neural Engine front and a CPU back.

Why: on the CPU the padded duration graph is dominated by BERT, not the LSTMs
(M1, CPU only: 59 of 86 ms at 512 tokens, 9.7 of 14.3 ms at 64 tokens with the
LSTMs removed). BERT runs well on the ANE, the LSTMs do not run there at all,
and iOS has failed ANECCompile on the whole duration graph. So:

  front (ANE):  input_ids, attention_mask -> d_en (1, 512, T), t_cnn (1, 512, T)
                BERT + bert_encoder, and the text encoder's embedding + CNN.
  back  (CPU):  d_en, t_cnn, ref_s, speed, attention_mask
                -> pred_dur, d, t_en, s, ref_s_out (the unsplit model's outputs)

Same math as coreml_export_duration.DurationModel, cut where the LSTMs start.

Usage:
    uv run --no-sync python scripts/export_duration_split.py --weights k/kokoro-v1_0.pth \
        --config k/config.json --tokens 64,512 --out DIR [--check outputs/swift_bench_inputs/30s.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from coreml_export_duration import DurationModel  # noqa: E402
from kokoro.model import KModel  # noqa: E402


class Front(nn.Module):
    def __init__(self, dm: DurationModel):
        super().__init__()
        self.km = dm.kmodel

    def forward(self, input_ids, attention_mask):
        km = self.km
        bert = km.bert(input_ids, attention_mask=attention_mask, token_type_ids=torch.zeros_like(input_ids))
        d_en = km.bert_encoder(bert).transpose(-1, -2)
        te = km.text_encoder
        pad = (attention_mask == 0).unsqueeze(1)
        x = te.embedding(input_ids).transpose(1, 2).masked_fill(pad, 0.0)
        for conv in te.cnn:
            x = conv(x).masked_fill(pad, 0.0)
        return d_en, x


class Back(nn.Module):
    def __init__(self, dm: DurationModel):
        super().__init__()
        self.dm = dm

    def forward(self, d_en, t_cnn, ref_s, speed, attention_mask):
        km = self.dm.kmodel
        input_lengths = attention_mask.sum(dim=-1).to(torch.long)
        text_mask = attention_mask == 0
        style = ref_s[:, 128:]
        d = km.predictor.text_encoder(d_en, style, input_lengths, text_mask)
        h = self.dm.duration_lstm(d, attention_mask)
        duration = km.predictor.duration_proj(h)
        pred_dur = torch.round(torch.sigmoid(duration).sum(axis=-1) / speed).clamp(min=1).long()
        te = km.text_encoder
        valid = (~text_mask).to(dtype=torch.long)
        t_en = te.lstm(t_cnn.transpose(1, 2), valid).transpose(-1, -2)
        t_en = t_en.masked_fill(text_mask.unsqueeze(1), 0.0)
        return pred_dur, d, t_en, style, ref_s + torch.zeros_like(ref_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokens", default="64,512")
    ap.add_argument("--check", help="bench input JSON: compare split vs unsplit outputs in PyTorch")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import coremltools as ct

    km = KModel(repo_id="hexgrad/Kokoro-82M", config=json.load(open(a.config)), model=a.weights,
                disable_complex=True).eval()
    dm = DurationModel(km).eval()
    for p in dm.parameters():
        p.requires_grad_(False)
    front, back = Front(dm).eval(), Back(dm).eval()

    if a.check:
        inp = json.load(open(a.check))
        n = inp["num_tokens"]
        t = 1 << (n - 1).bit_length()
        ids = torch.zeros(1, t, dtype=torch.int32); ids[0, :n] = torch.tensor(inp["input_ids"][:n])
        mask = torch.zeros(1, t, dtype=torch.int32); mask[0, :n] = 1
        ref_s = torch.tensor(inp["ref_s"])[None]
        speed = torch.ones(1)
        with torch.no_grad():
            want = dm(ids, ref_s, speed, mask)
            got = back(*front(ids, mask), ref_s, speed, mask)
        for name, w, g in zip(["pred_dur", "d", "t_en"], want, got):
            print(f"check {name}: max abs diff {(w.float() - g.float()).abs().max().item():.2e}")

    for t in [int(v) for v in a.tokens.split(",")]:
        ids = torch.ones(1, t, dtype=torch.int32)
        mask = torch.ones(1, t, dtype=torch.int32)
        with torch.no_grad():
            d_en, t_cnn = front(ids, mask)
        tf = torch.jit.trace(front, (ids, mask), check_trace=False)
        mf = ct.convert(tf, inputs=[ct.TensorType(name="input_ids", shape=(1, t), dtype=np.int32),
                                    ct.TensorType(name="attention_mask", shape=(1, t), dtype=np.int32)],
                        outputs=[ct.TensorType(name="d_en"), ct.TensorType(name="t_cnn")],
                        compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS18,
                        compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
        mf.save(str(a.out / f"kokoro_duration_front_t{t}.mlpackage"))
        ex = (d_en, t_cnn, torch.rand(1, 256), torch.ones(1), mask)
        tb = torch.jit.trace(back, ex, check_trace=False)
        mb = ct.convert(tb, inputs=[ct.TensorType(name="d_en", shape=tuple(d_en.shape)),
                                    ct.TensorType(name="t_cnn", shape=tuple(t_cnn.shape)),
                                    ct.TensorType(name="ref_s", shape=(1, 256)),
                                    ct.TensorType(name="speed", shape=(1,)),
                                    ct.TensorType(name="attention_mask", shape=(1, t), dtype=np.int32)],
                        outputs=[ct.TensorType(name=n) for n in ["pred_dur", "d", "t_en", "s", "ref_s_out"]],
                        compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.iOS18,
                        compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
        mb.save(str(a.out / f"kokoro_duration_back_t{t}.mlpackage"))
        print("saved", t, flush=True)


if __name__ == "__main__":
    main()
