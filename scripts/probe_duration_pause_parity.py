#!/usr/bin/env python3
"""Probe per-token duration parity for a frozen bench input.

Motivated by README/Notes/cs1-audio-quality-evaluation-2026-07-14.md: the 15s
bucket renders phrase-boundary pauses 40-110 ms longer than the PyTorch
reference. This probe isolates WHERE the deviation enters by comparing
per-token pred_dur across arms that differ by exactly one factor:

  A. PyTorch KModel eager, exact length, float32            (reference)
  B. PyTorch DurationModel wrapper, padded, float32         (wrapper/mask semantics)
  C. Core ML padded_tN, FP16, CPU_ONLY                      (B + FP16 quantization)
  D. Core ML padded_tN, FP16, ALL                           (C + accelerator placement)
  E. Core ML exact_tN, FP16, CPU_ONLY (if package exists)   (A + FP16, native LSTM)

Frame rate is 40 fps (25 ms per frame), so a +2 frame deviation on a pause
token is +50 ms of silence.

Usage:
  .venv/bin/python scripts/probe_duration_pause_parity.py --key 15s
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

FRAME_MS = 25.0  # 24 kHz audio, 600-sample hop -> 40 fps duration frames

def load_input(key: str) -> dict:
    return json.loads((ROOT / "outputs" / "swift_bench_inputs" / f"{key}.json").read_text())


def load_vocab() -> dict[int, str]:
    vocab = json.loads((ROOT / "_kokoro_vocab.json").read_text())["vocab"]
    # vocab maps phoneme -> id
    return {v: k for k, v in vocab.items()}


def pytorch_eager_pred_dur(kmodel, input_ids: list[int], ref_s: np.ndarray, speed: float) -> np.ndarray:
    ids = torch.LongTensor([input_ids])
    ref = torch.from_numpy(ref_s)
    with torch.no_grad():
        input_lengths = torch.full((1,), ids.shape[-1], dtype=torch.long)
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(1, -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
        bert_dur = kmodel.bert(ids, attention_mask=(~text_mask).int())
        d_en = kmodel.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref[:, 128:]
        d = kmodel.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = kmodel.predictor.lstm(d)
        duration = kmodel.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        return pred_dur.numpy(), duration.squeeze().numpy()


def coreml_pred_dur(pkg: Path, input_ids: list[int], ref_s: np.ndarray, speed: float,
                    pad_to: int | None, compute_units) -> np.ndarray:
    import coremltools as ct
    model = ct.models.MLModel(str(pkg), compute_units=compute_units)
    n = len(input_ids)
    if pad_to is None:
        ids = np.array([input_ids], dtype=np.int32)
        feed = {"input_ids": ids, "ref_s": ref_s, "speed": np.array([speed], dtype=np.float32)}
    else:
        ids = np.zeros((1, pad_to), dtype=np.int32)
        ids[0, :n] = input_ids
        mask = np.zeros((1, pad_to), dtype=np.int32)
        mask[0, :n] = 1
        feed = {"input_ids": ids, "ref_s": ref_s, "speed": np.array([speed], dtype=np.float32),
                "attention_mask": mask}
    out = model.predict(feed)
    pred = np.asarray(out["pred_dur"]).reshape(-1)[:n]
    return pred.astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="15s")
    ap.add_argument("--skip-wrapper", action="store_true",
                    help="skip arm B (PyTorch padded wrapper), which is slow to build")
    args = ap.parse_args()

    import coremltools as ct

    data = load_input(args.key)
    # input_ids in the frozen JSON are already right-padded; num_tokens is the
    # valid count. The PyTorch reference arm runs the exact-length sequence,
    # matching kmodel(phonemes, ...) in scripts/bakeoff_harness._run_pytorch.
    n = int(data["num_tokens"])
    input_ids = data["input_ids"][:n]
    assert all(v == 0 for v in data["input_ids"][n:]), "padding must be zeros"
    rev_vocab = load_vocab()
    speed = float(data.get("speed", 1.0))

    # The frozen bench input embeds the exact ref_s used by both arms.
    voice = data["voice"]
    ref_s = np.asarray(data["ref_s"], dtype=np.float32).reshape(1, 256)

    # Same weights source as scripts/bakeoff_harness.PyTorchContext: HF
    # hexgrad/Kokoro-82M (the checkpoints/ symlinks are dead on this host).
    from kokoro.model import KModel
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()

    print(f"key={args.key} tokens={n} voice={voice} speed={speed}")

    arms: dict[str, np.ndarray] = {}
    ref_dur, raw = pytorch_eager_pred_dur(kmodel, input_ids, ref_s, speed)
    arms["A_pytorch_eager"] = ref_dur

    if not args.skip_wrapper:
        import export_duration as ed
        pad_to = next(s for s in [32, 64, 128, 256, 512] if s >= n)
        wrapper = ed._eval_export_model(ed.DurationModel(
            KModel(config=str(cfg), model=str(ckpt), disable_complex=True)))
        ids = torch.zeros((1, pad_to), dtype=torch.long)
        ids[0, :n] = torch.LongTensor(input_ids)
        mask = torch.zeros((1, pad_to), dtype=torch.long)
        mask[0, :n] = 1
        with torch.no_grad():
            out = wrapper(ids, torch.from_numpy(ref_s), torch.tensor([speed]), mask)
        arms["B_wrapper_padded_fp32"] = out[0].squeeze().numpy()[:n]

    pad_to = next(s for s in [32, 64, 128, 256, 512] if s >= n)
    padded_pkg = ROOT / "coreml" / f"kokoro_duration_t{pad_to}.mlpackage"
    arms[f"C_coreml_padded_t{pad_to}_cpu"] = coreml_pred_dur(
        padded_pkg, input_ids, ref_s, speed, pad_to, ct.ComputeUnit.CPU_ONLY)
    arms[f"D_coreml_padded_t{pad_to}_all"] = coreml_pred_dur(
        padded_pkg, input_ids, ref_s, speed, pad_to, ct.ComputeUnit.ALL)

    exact_pkg = ROOT / "coreml" / f"kokoro_duration_exact_t{n}.mlpackage"
    if exact_pkg.exists():
        arms[f"E_coreml_exact_t{n}_cpu"] = coreml_pred_dur(
            exact_pkg, input_ids, ref_s, speed, None, ct.ComputeUnit.CPU_ONLY)

    ref = arms["A_pytorch_eager"]
    print(f"\ntotal frames: " + ", ".join(f"{k}={v.sum()}" for k, v in arms.items()))

    for name, arm in arms.items():
        if name == "A_pytorch_eager":
            continue
        diff = arm.astype(np.int64) - ref.astype(np.int64)
        nz = np.nonzero(diff)[0]
        print(f"\n=== {name}: {len(nz)} tokens differ, sum|diff|={np.abs(diff).sum()} frames "
              f"({np.abs(diff).sum() * FRAME_MS:.0f} ms redistributed)")
        for i in nz:
            ph = rev_vocab.get(input_ids[i], f"<{input_ids[i]}>")
            print(f"  tok[{i:3d}] {ph!r:8s} ref={ref[i]:3d} arm={arm[i]:3d} diff={diff[i]:+d} "
                  f"({diff[i] * FRAME_MS:+.0f} ms) raw_sigmoid_sum={raw[i]:.3f}")


if __name__ == "__main__":
    main()
