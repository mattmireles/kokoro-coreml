# External Bakeoff Adapters

Disposable adapters for `README/Plans/kokoro-external-bakeoff-v1.md`.

## Runtime Inputs

```bash
uv run --with-requirements requirements-bakeoff.txt --no-sync \
  python scripts/bakeoff_harness.py prepare-inputs
uv run --with-requirements requirements-bakeoff.txt --no-sync \
  python scripts/external_bakeoff/prepare_runtime_inputs.py
```

The runtime manifest must contain `3s`, `7s`, `10s`, `15s`, and `30s`, and each
input must route to its named bucket.

## MLX

Use a disposable venv and a pinned clone:

```bash
uv venv /tmp/kokoro-external-bakeoff/mlx-venv
uv pip install --python /tmp/kokoro-external-bakeoff/mlx-venv/bin/python \
  -r scripts/external_bakeoff/requirements_mlx_audio.txt
/tmp/kokoro-external-bakeoff/mlx-venv/bin/python \
  scripts/external_bakeoff/run_mlx_audio.py --machine-id m2-studio \
  --spotcheck-dir outputs/external_bakeoff/spotcheck_wavs/mlx_audio_m2-studio
```

## Soniqo Speech Swift

Clone and pin `soniqo/speech-swift` outside the repo, then run:

```bash
python scripts/external_bakeoff/run_speech_swift_kokoro.py \
  --machine-id m2-studio \
  --speech-swift /tmp/kokoro-external-bakeoff/speech-swift \
  --spotcheck-dir outputs/external_bakeoff/spotcheck_wavs/soniqo_speech_swift_kokoro_m2-studio
```

For the physical iPhone runner, run the signing/device preflight before any
build attempt:

```bash
SPEECH_SWIFT_PATH=/tmp/kokoro-external-bakeoff/speech-swift \
  python scripts/external_bakeoff/preflight_ios_runner.py \
  --generate-project \
  --output outputs/external_bakeoff/ios_runner_preflight_latest.json
```

After the signed app runs, paste its copied JSON into
`outputs/external_bakeoff/ios_runner_payload_latest.json` and ingest it:

```bash
python scripts/external_bakeoff/ingest_ios_runner_result.py \
  --input outputs/external_bakeoff/ios_runner_payload_latest.json \
  --machine-id iphone-12-pro
```

## Laishere Core ML Backup

Clone and convert `laishere/kokoro-coreml` outside the repo. The converter can
take a long time while Core ML compiles large models; do not interrupt it just
because output is quiet.

```bash
cd /tmp/kokoro-external-bakeoff/laishere-kokoro-coreml
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python convert.py --max-frames 2000
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python \
  /path/to/kokoro-coreml/scripts/external_bakeoff/run_laishere_kokoro_coreml.py \
  --machine-id m2-studio \
  --laishere-repo /tmp/kokoro-external-bakeoff/laishere-kokoro-coreml \
  --manifest /path/to/kokoro-coreml/outputs/external_bakeoff/runtime_input_manifest.json \
  --spotcheck-dir /path/to/kokoro-coreml/outputs/external_bakeoff/spotcheck_wavs/laishere_kokoro_coreml_m2-studio
```

The adapter times the seven-stage Core ML chain only; G2P and feed preparation
are outside the timed calls, matching laishere's public benchmark.

## Config F

Config F uses the existing Swift benchmark binary and prepared Swift inputs:

```bash
cd swift && swift build -c release --product kokoro-bench
cd ..
uv run --with-requirements requirements-bakeoff.txt --no-sync \
  python scripts/prepare_swift_bench_inputs.py
python scripts/external_bakeoff/run_config_f_reference.py --machine-id m2-studio
```

By default this runs the fastest production-shaped Config F policy:
`--compute-units staged` and exact duration model discovery. `staged` loads
duration, F0Ntrain, and generator with CPU+GPU and loads decoder-pre with
CPU+ANE. Use `--compute-units all --use-padded-duration-models` only when
reproducing the original slower bakeoff cells.

The adapter also defaults to `--preflight-runs 3`, so each bucket discards
three calls before recording the cold marker and five warm inference calls. Keep
that default for paper comparisons; warmed inference is the apples-to-apples
boundary.

For the five runtime fixtures, the fast path expects exact duration packages for
the observed token counts. In particular, the `10s` fixture needs
`coreml/kokoro_duration_exact_t156.mlpackage`; without it, the benchmark falls
back to `padded_t256` and spends hundreds of extra milliseconds in Duration on
M2 Studio-class hardware.

By default, each adapter writes the last warm output for each successful input
to `outputs/external_bakeoff/spotcheck_wavs/<impl>_<machine_id>/<bucket>.wav`.
Pass `--spotcheck-dir` to use an explicit collection directory.

## Summarize

```bash
python scripts/external_bakeoff/summarize_external.py
```

## Completion Gate

Before marking `README/Plans/kokoro-external-bakeoff-v1.md` complete, run:

```bash
python scripts/external_bakeoff/verify_external_bakeoff_completion.py
```

This checks the Mac result matrix, the documented MLX 3s error exception, the
laishere backup records, signed iPhone result ingestion, the iOS preflight
status, and filled human listening decisions.

## Human Listening Review

After result JSONs, spot-check WAVs, and `scripts/audio_quality_probe.py`
reports exist, generate the TTS-only listening review:

```bash
python scripts/external_bakeoff/create_listening_review.py
```

This writes:

- `outputs/external_bakeoff/listening/external_bakeoff_listening_review.md`
- `outputs/external_bakeoff/listening/external_bakeoff_listening_review.html`
- `outputs/external_bakeoff/listening/external_bakeoff_listening_decisions.csv`

The review uses only collected Kokoro TTS WAVs and waveform-quality reports. It
does not use Whisper, ASR, VAD, or Soniqo's echo-demo dependency graph.
Regeneration preserves existing `human_decision` and `notes` values for matching
rows. Use `--reset-decisions` only when intentionally discarding prior human
listening work.

Fill `human_decision` for every successful audio row in the CSV with one of:

- `pass`: comparable to the same-machine Config F reference.
- `caveat`: usable only with the caveat documented in `notes`.
- `fail`: not quality parity.

Then validate the filled decision sheet:

```bash
python scripts/external_bakeoff/validate_listening_decisions.py
```

The validator fails while any successful audio row is blank, has an invalid
decision, has `human_decision=caveat` without notes, or has
`human_decision=fail`. Use `--allow-failures` only when intentionally publishing
documented non-parity rows instead of claiming quality parity.
