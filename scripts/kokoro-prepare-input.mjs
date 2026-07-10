#!/usr/bin/env node

import { existsSync } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createInterface } from 'node:readline/promises';

const EnumSizes = [32, 64, 128, 256, 320, 384, 512];
// packages/contracts MAX_TTS_CHUNK_TOKENS
const MaxCallerChunkTokens = 450;
const VoiceEmbeddingDim = 256;
const RepoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DefaultRuntimeRoot = path.join(RepoRoot, 'packages/kokoro-coreml-runtime');

function parseArgs(argv) {
  const args = new Map();
  const flags = new Set();
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    if (!item.startsWith('--')) {
      continue;
    }
    const key = item.slice(2);
    if (key === 'serve') {
      flags.add(key);
      continue;
    }
    args.set(key, argv[index + 1]);
    index += 1;
  }
  return { args, flags };
}

function runtimeRoot() {
  const explicit = process.env.KOKORO_COREML_ROOT?.trim();
  if (explicit) {
    return explicit;
  }
  return DefaultRuntimeRoot;
}

async function loadPhonemizer(root) {
  const candidates = [
    path.join(root, 'kokoro.js/src/phonemize.js'),
    path.join(DefaultRuntimeRoot, 'kokoro.js/src/phonemize.js')
  ];
  for (const candidate of candidates) {
    if (!existsSync(candidate)) {
      continue;
    }
    const module = await import(pathToFileURL(candidate).href);
    if (typeof module.phonemize === 'function') {
      return module.phonemize;
    }
  }
  throw new Error(`Kokoro phonemizer not found under ${root}`);
}

async function loadVocab(root) {
  const candidates = [
    path.join(root, 'outputs/hnsf_validation/config.json'),
    path.join(root, 'checkpoints/config.json'),
    path.join(DefaultRuntimeRoot, 'outputs/hnsf_validation/config.json')
  ];
  for (const candidate of candidates) {
    if (!existsSync(candidate)) {
      continue;
    }
    const parsed = JSON.parse(await readFile(candidate, 'utf8'));
    if (parsed.vocab && typeof parsed.vocab === 'object') {
      return parsed.vocab;
    }
  }
  throw new Error(`Kokoro vocab config not found under ${root}`);
}

async function loadVoiceRows(root, voice) {
  const candidates = [
    path.join(root, 'kokoro.js/voices', `${voice}.bin`),
    path.join(DefaultRuntimeRoot, 'kokoro.js/voices', `${voice}.bin`)
  ];
  for (const candidate of candidates) {
    if (!existsSync(candidate)) {
      continue;
    }
    const bytes = await readFile(candidate);
    if (bytes.byteLength % (VoiceEmbeddingDim * 4) !== 0) {
      throw new Error(`Voice embedding file has invalid byte length: ${candidate}`);
    }
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }
  throw new Error(`Kokoro voice embedding not found for ${voice}`);
}

function voiceEmbeddingForPhonemes(voiceRows, phonemes) {
  const rowCount = voiceRows.length / VoiceEmbeddingDim;
  const rowIndex = Math.max(0, Math.min(rowCount - 1, phonemes.length - 1));
  const start = rowIndex * VoiceEmbeddingDim;
  return Array.from(voiceRows.slice(start, start + VoiceEmbeddingDim));
}

function languageForVoice(voice) {
  return voice.startsWith('b') ? 'b' : 'a';
}

async function prepareEntries({ texts, key, voice, speed, vocab, voiceRows, phonemize }) {
  const entries = [];
  for (const [index, text] of texts.entries()) {
    const phonemes = await phonemize(text, languageForVoice(voice));
    if (!phonemes) {
      throw new Error('Kokoro phonemizer returned no phonemes');
    }
    const ids = [];
    for (const phoneme of phonemes) {
      const token = vocab[phoneme];
      if (Number.isInteger(token)) {
        ids.push(token);
      }
    }
    const inputIds = [0, ...ids, 0];
    if (inputIds.length > MaxCallerChunkTokens) {
      throw new Error(
        `Kokoro chunk exceeds MAX_TTS_CHUNK_TOKENS (${MaxCallerChunkTokens}): ${inputIds.length} tokens`
      );
    }
    const enumLength = EnumSizes.find((size) => size >= inputIds.length) ?? EnumSizes.at(-1);
    const padded = [...inputIds, ...Array(Math.max(0, enumLength - inputIds.length)).fill(0)];
    const attentionMask = [
      ...Array(inputIds.length).fill(1),
      ...Array(Math.max(0, enumLength - inputIds.length)).fill(0)
    ];
    const refS = voiceEmbeddingForPhonemes(voiceRows, phonemes);
    // Emit only what Swift `PreparedKokoroInput` decodes. Phonemes,
    // num_tokens, and the sha256s used to ship as debug metadata; nothing
    // in the worker or runtime read them, so they were dead payload (and
    // phonemes are an unnecessary phonetic mirror of `text`).
    entries.push({
      key: texts.length === 1 ? key : `${key}-${String(index).padStart(3, '0')}`,
      text,
      voice,
      speed,
      input_ids: padded,
      attention_mask: attentionMask,
      ref_s: refS,
      canonical_duration_s: null
    });
  }
  return entries;
}

async function runServe() {
  const root = runtimeRoot();
  const phonemize = await loadPhonemizer(root);
  const vocab = await loadVocab(root);
  const voices = new Map();
  const lineReader = createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of lineReader) {
    if (!line.trim()) {
      continue;
    }
    let requestId = null;
    try {
      const request = JSON.parse(line);
      requestId = String(request.id);
      const texts = request.texts;
      const voice = String(request.voice);
      const speed = Number(request.speed);
      if (!Array.isArray(texts) || !texts.every((text) => typeof text === 'string')) {
        throw new Error('texts must be a string array');
      }
      if (!voices.has(voice)) {
        voices.set(voice, await loadVoiceRows(root, voice));
      }
      const inputs = await prepareEntries({
        texts,
        key: 'service',
        voice,
        speed,
        vocab,
        voiceRows: voices.get(voice),
        phonemize
      });
      process.stdout.write(`${JSON.stringify({ id: requestId, inputs })}\n`);
    } catch (error) {
      process.stdout.write(`${JSON.stringify({ id: requestId, error: error instanceof Error ? error.message : String(error) })}\n`);
    }
  }
}

async function runOnce(args) {
  const root = runtimeRoot();
  const textFile = args.get('text-file');
  const textListFile = args.get('text-list-file');
  const output = args.get('output');
  const key = args.get('key');
  const voice = args.get('voice');
  const speed = Number(args.get('speed'));
  if ((!textFile && !textListFile) || (textFile && textListFile) || !output || !key || !voice || !Number.isFinite(speed)) {
    throw new Error('provide exactly one input file plus --output, --key, --voice, and --speed');
  }
  const texts = textFile
    ? [await readFile(textFile, 'utf8')]
    : JSON.parse(await readFile(textListFile, 'utf8'));
  const phonemize = await loadPhonemizer(root);
  const vocab = await loadVocab(root);
  const voiceRows = await loadVoiceRows(root, voice);
  const entries = await prepareEntries({ texts, key, voice, speed, vocab, voiceRows, phonemize });
  await writeFile(output, `${JSON.stringify(textFile ? entries[0] : entries, null, 2)}\n`, 'utf8');
}

const { args, flags } = parseArgs(process.argv.slice(2));
try {
  if (flags.has('serve')) {
    await runServe();
  } else {
    await runOnce(args);
  }
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
}
