#!/usr/bin/env python3
"""Prepare pre-tokenized inputs for the Swift benchmark.

Runs the Python tokenizer on the 4 bakeoff inputs and saves the results
as JSON files that the Swift benchmark can load directly.

Usage::

    uv run python scripts/prepare_swift_bench_inputs.py

Output: outputs/swift_bench_inputs/{key}.json
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BAKEOFF_INPUTS = {
    "3s": "The quick brown fox jumps over the dog.",
    "7s": (
        "The morning sun cast long shadows across the garden as birds began "
        "their chorus in the ancient oak tree."
    ),
    "15s": (
        "The ancient lighthouse stood alone on the rocky cliff, its beam sweeping "
        "across dark waters with the patience of centuries. Ships had come and gone, "
        "storms had battered its walls, yet still it turned, guiding sailors home."
    ),
    "30s": (
        "When the last train departed that evening, the platform fell silent. "
        "The old stationmaster locked the ticket office with trembling hands, "
        "running his fingers along the worn counter where countless journeys "
        "had begun. Outside, autumn wind scattered golden leaves across the "
        "empty tracks. He had spent forty years here, watching the world rush "
        "past in a blur of faces and farewells. The station would stand a while "
        "longer, its clock still ticking, its roof sheltering the pigeons."
    ),
}
# Duration model enumerated sizes — pad tokens to nearest.
ENUM_SIZES = [32, 64, 128, 256, 320, 384, 512]
VOICE = "af_heart"
SPEED = 1.0


def _hnsf_weights_sha256(linear_weights: list[float], linear_bias: float) -> str:
    payload = json.dumps(
        {"linear_weights": linear_weights, "linear_bias": linear_bias},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main():
    from kokoro import KModel, KPipeline
    from kokoro.pipeline import voice_embedding_for_phoneme_string

    output_dir = _ROOT / "outputs" / "swift_bench_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = KPipeline(lang_code="a")
    kmodel = KModel()

    # Save hn-nsf linear weights
    gen = kmodel.decoder.generator
    linear_w = gen.m_source.l_linear.weight.detach().numpy().flatten().tolist()
    linear_b = float(gen.m_source.l_linear.bias.detach().numpy().flatten()[0])
    hnsf_hash = _hnsf_weights_sha256(linear_w, linear_b)

    hnsf_config = {
        "linear_weights": linear_w,
        "linear_bias": linear_b,
        "weights_sha256": hnsf_hash,
    }
    with open(output_dir / "hnsf_weights.json", "w") as f:
        json.dump(hnsf_config, f)
    print(f"Saved hn-nsf weights: {len(linear_w)} weights, bias={linear_b:.6f}")

    runtime_manifest = _ROOT / "outputs" / "external_bakeoff" / "runtime_input_manifest.json"
    if runtime_manifest.exists():
        manifest = json.loads(runtime_manifest.read_text())
        bakeoff_inputs = {
            key: value["text"]
            for key, value in manifest.get("inputs", {}).items()
        }
    else:
        bakeoff_inputs = DEFAULT_BAKEOFF_INPUTS

    for key, text in bakeoff_inputs.items():
        print(f"\n--- {key}: {text!r:.60} ---")

        # Tokenize
        voice_pack = pipeline.load_voice(VOICE)
        phonemes = None
        for _, ps, _ in pipeline(text, VOICE, SPEED):
            phonemes = ps
            break

        if not phonemes:
            print(f"  FAILED to tokenize {key}")
            continue

        # Get voice embedding
        ref_s = voice_embedding_for_phoneme_string(voice_pack, phonemes)

        # Build input IDs
        input_ids = list(filter(lambda i: i is not None,
                               map(lambda p: kmodel.vocab.get(p), phonemes)))
        input_ids = [0] + input_ids + [0]  # BOS/EOS

        # Pad to nearest enumerated size
        enum_T = next((s for s in ENUM_SIZES if s >= len(input_ids)), ENUM_SIZES[-1])
        padded_ids = input_ids[:enum_T] + [0] * max(0, enum_T - len(input_ids))
        attention_mask = [1] * min(len(input_ids), enum_T) + [0] * max(0, enum_T - len(input_ids))

        ref_s_list = ref_s.cpu().numpy().flatten().tolist()

        # Compute canonical duration via extract_vocoder_inputs (ground truth).
        # Try the bakeoff input manifest first to avoid loading all CoreML models.
        canonical_dur = None
        manifest_path = _ROOT / "outputs" / "bakeoff" / "input_manifest.json"
        if manifest_path.exists():
            import json as _json
            manifest = _json.loads(manifest_path.read_text())
            # The manifest uses different input keys; search by text match.
            for _ik, _iv in manifest.get("inputs", {}).items():
                if _iv.get("text") == text:
                    canonical_dur = _iv["canonical_duration_s"]
                    break
        if canonical_dur is None:
            # Fallback: use HybridTTSPipeline. Skip if SKIP_VOCODER_INPUTS is set
            # (avoids loading all CoreML models on memory-constrained machines).
            import os as _os
            if _os.environ.get("SKIP_VOCODER_INPUTS") != "1":
                from kokoro.coreml_pipeline import HybridTTSPipeline
                pipe = HybridTTSPipeline()
                vi = pipe.extract_vocoder_inputs(text, VOICE, SPEED)
                if vi is not None:
                    T_f0 = int(vi["f0_curve"].shape[-1])
                    canonical_dur = T_f0 / 80.0

        entry = {
            "key": key,
            "text": text,
            "voice": VOICE,
            "speed": SPEED,
            "phonemes": phonemes,
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            "ref_s": ref_s_list,
            "canonical_duration_s": canonical_dur,
            "num_tokens": len(input_ids),
            "hnsf_weights_sha256": hnsf_hash,
        }

        with open(output_dir / f"{key}.json", "w") as f:
            json.dump(entry, f)
        print(f"  Tokens: {len(input_ids)}, phonemes: {phonemes}")

    print(f"\nAll inputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
