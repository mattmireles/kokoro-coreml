#!/usr/bin/env python3
"""Plan 014 quality gate: Swift pipeline audio vs the PyTorch fp32 generator.

usage: quality_014.py DUMP_DIR [DUMP_DIR ...] [--json OUT]

For each `kokoro-bench --dump-tensors` directory: run the fp32 PyTorch
generator (GeneratorFromHar, unmasked) on the dump's own x_pre, ref_s and har
cropped to the valid frames (2 * natural_frames x_pre frames, 60 * that + 1
har frames) -- exactly what the folded and hybrid generators compute -- and
score the dump's waveform against it. Gate: SNR >= 40 dB on every clip.
This checks the generator (the part plan 014's arms change); the front stages
are checked by predicted frames (natural_frames) matching across arms.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from audio_parity_tensor_io import load_tensor_dump  # noqa: E402

GATE_DB = 40.0


def generator():
    from export_synth.wrappers import GeneratorFromHar
    from kokoro import KModel
    return GeneratorFromHar(KModel(disable_complex=True).eval().decoder.generator).eval()


def score(gen, d):
    meta, t = load_tensor_dump(d)
    md = meta.get("metadata", {})
    v = 2 * int(md["natural_frames"])
    hv = v * 60 + 1
    x = torch.from_numpy(t["x_pre"].astype(np.float32)[..., :v])
    har = torch.from_numpy(t["har"].astype(np.float32)[..., :hv])
    s = torch.from_numpy(t["ref_s"].astype(np.float32))
    with torch.no_grad():
        ref = gen(x_pre=x, ref_s=s, har=har).numpy().reshape(-1).astype(np.float64)
    got = t["waveform"].astype(np.float64).reshape(-1)
    n = min(ref.size, got.size)
    ref, got = ref[:n], got[:n]
    snr = 10 * np.log10((ref ** 2).sum() / max(((got - ref) ** 2).sum(), 1e-30))
    return {"dump": str(d), "input_key": md.get("input_key"), "natural_frames": md.get("natural_frames"),
            "samples": int(n), "snr_db": round(float(snr), 2), "corr": round(float(np.corrcoef(ref, got)[0, 1]), 6),
            "pass": bool(snr >= GATE_DB)}


def main():
    args = sys.argv[1:]
    out = None
    if "--json" in args:
        i = args.index("--json"); out = args[i + 1]; del args[i:i + 2]
    gen = generator()
    rows = [score(gen, Path(a)) for a in args]
    for r in rows:
        print(f"{r['input_key']:>4} frames {r['natural_frames']} SNR {r['snr_db']:.2f} dB corr {r['corr']} {'PASS' if r['pass'] else 'FAIL'}  {r['dump']}")
    if rows:
        worst = min(rows, key=lambda r: r["snr_db"])
        print(f"worst clip: {worst['input_key']} {worst['snr_db']} dB ({worst['dump']})")
    if out:
        Path(out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
