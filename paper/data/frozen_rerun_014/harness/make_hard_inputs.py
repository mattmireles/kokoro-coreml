#!/usr/bin/env python3
"""Plan 014 hard inputs for the quality gate, in the swift_bench_inputs format.

usage: uv run --no-sync python outputs/frozen_rerun_014/make_hard_inputs.py OUT_DIR
Same tokenization as scripts/prepare_swift_bench_inputs.py (first pipeline
chunk, BOS/EOS, padded to the enumerated sizes); canonical duration from the
PyTorch duration predictor (frames / 40).
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

HARD = {
    "hard_numbers": "On 3/14/2026 at 9:45 a.m., invoice #10,482 for $1,299.95 was paid, and 42% of 7,500 units shipped by Q3.",
    "hard_punct": "Wait... what?! No; that's (truly) impossible: \"yes,\" she said, 'absolutely!' Then, silence?",
    "hard_short": "Hi.",
}
ENUM_SIZES = [32, 64, 128, 256, 320, 384, 512]


def main():
    from kokoro import KModel, KPipeline
    from kokoro.pipeline import voice_embedding_for_phoneme_string
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    pipe = KPipeline(lang_code="a")
    km = KModel().eval()
    ref = json.loads((ROOT / "outputs/swift_bench_inputs/3s.json").read_text())
    voice_pack = pipe.load_voice("af_heart")
    for key, text in HARD.items():
        phonemes = next(ps for _, ps, _ in pipe(text, "af_heart", 1.0))
        ref_s = voice_embedding_for_phoneme_string(voice_pack, phonemes)
        ids = [0] + [i for i in (km.vocab.get(p) for p in phonemes) if i is not None] + [0]
        enum_t = next(s for s in ENUM_SIZES if s >= len(ids))
        with torch.no_grad():
            o = km.forward_with_tokens(torch.LongTensor([ids]), ref_s.reshape(1, -1) if ref_s.dim() == 1 else ref_s, 1.0)
        frames = int(o[1].sum())
        entry = {"key": key, "text": text, "voice": "af_heart", "speed": 1.0, "phonemes": phonemes,
                 "input_ids": ids + [0] * (enum_t - len(ids)), "attention_mask": [1] * len(ids) + [0] * (enum_t - len(ids)),
                 "ref_s": ref_s.cpu().numpy().flatten().tolist(), "canonical_duration_s": frames / 40.0,
                 "num_tokens": len(ids), "hnsf_weights_sha256": ref["hnsf_weights_sha256"]}
        (out / f"{key}.json").write_text(json.dumps(entry))
        print(key, len(ids), "tokens", frames / 40.0, "s", phonemes)


if __name__ == "__main__":
    main()
