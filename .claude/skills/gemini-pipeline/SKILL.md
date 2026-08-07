---
name: gemini-pipeline
description: >-
  Operate Kokoro's direct Gemini audio-judge workflow and diagnose Gemini API
  transport, response parsing, rubric, and artifact-boundary failures. Use for
  scripts/gemini_audio_judge_direct.py, audio quality judging, judge fixtures,
  or Gemini API behavior; do not use for Vertex endpoint routing, Cloudflare,
  or generic model-export work.
---

# Gemini Audio Judge Pipeline

1. Read [the pipeline anchors](references/index.md).
2. Bind the request to the audio file, transcript/reference text, model name,
   prompt/rubric revision, and output artifact before invoking Gemini.
3. Run the checked-in direct judge and its focused tests. Treat provider
   failures, safety blocks, malformed responses, and missing text as explicit
   failures—not silent passes.
4. Keep the Gemini judgment distinct from deterministic waveform, numerical
   parity, and device-performance gates.
5. Preserve request metadata and redacted diagnostics; never print API keys or
   raw private audio/transcripts.

