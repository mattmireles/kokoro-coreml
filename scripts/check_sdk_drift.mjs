#!/usr/bin/env node

import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const files = {
  pipeline: 'swift/Sources/KokoroPipeline/KokoroPipeline.swift',
  package: 'swift-tts/Package.swift',
  rootPackage: 'Package.swift',
  consumerFixturePackage: 'examples/KokoroConsumerFixture/Package.swift',
  voice: 'swift-tts/Sources/KokoroTTS/KokoroVoiceID.swift',
  textChunker: 'swift-tts/Sources/KokoroTTS/TextChunker.swift',
  manifestDecoder: 'swift-tts/Sources/KokoroTTS/KokoroRuntimeManifest.swift',
  manifestSchema: 'schemas/KokoroRuntimeManifest.schema.json',
  downloadedStore: 'swift-tts/Sources/KokoroTTS/KokoroDownloadedModelStore.swift',
  tts: 'swift-tts/Sources/KokoroTTS/KokoroTTS.swift',
  errors: 'swift-tts/Sources/KokoroTTS/KokoroErrors.swift',
  buildBundle: 'scripts/build_sdk_bundle.mjs',
  sdkProfiles: 'scripts/sdk_profiles.json',
  jsPrep: 'scripts/kokoro-prepare-input.mjs',
  pyPrep: 'scripts/kokoro-prepare-input.py',
};
// Gitignored local export of the hybrid generator; absent in a fresh clone.
const hybridPlanPath = path.join(repoRoot, 'coreml-hybrid/plan.json');

function fail(message) {
  throw new Error(message);
}

function parseIntConstant(source, name) {
  const match = source.match(new RegExp(`static let ${name}(?:: Int)? = ([0-9_]+)`));
  if (!match) {
    fail(`missing Swift Int constant ${name}`);
  }
  return Number.parseInt(match[1].replaceAll('_', ''), 10);
}

function parseSwiftIntArray(source, name) {
  const match = source.match(new RegExp(`static let ${name}: \\[Int\\] = \\[([^\\]]+)\\]`));
  if (!match) {
    fail(`missing Swift Int array ${name}`);
  }
  return match[1].split(',').map((item) => Number.parseInt(item.trim(), 10));
}

function parsePythonIntArray(source, name) {
  const match = source.match(new RegExp(`${name} = \\[([^\\]]+)\\]`));
  if (!match) {
    fail(`missing Python Int array ${name}`);
  }
  return match[1].split(',').map((item) => Number.parseInt(item.trim(), 10));
}

function parseJsConstInt(source, name) {
  const match = source.match(new RegExp(`const ${name} = ([0-9_]+);`));
  if (!match) {
    fail(`missing JS Int constant ${name}`);
  }
  return Number.parseInt(match[1].replaceAll('_', ''), 10);
}

function parseJsConstArray(source, name) {
  const match = source.match(new RegExp(`const ${name} = \\[([^\\]]+)\\];`));
  if (!match) {
    fail(`missing JS array constant ${name}`);
  }
  return match[1].split(',').map((item) => Number.parseInt(item.trim(), 10));
}

function parsePythonIntConstant(source, name) {
  const match = source.match(new RegExp(`${name} = ([0-9_]+)`));
  if (!match) {
    fail(`missing Python Int constant ${name}`);
  }
  return Number.parseInt(match[1].replaceAll('_', ''), 10);
}

function requireIncludes(source, file, snippet) {
  if (!source.includes(snippet)) {
    fail(`${file} is missing required SDK contract snippet: ${snippet}`);
  }
}

function parseSwiftIntMap(source, name) {
  const match = source.match(new RegExp(`static let ${name}: \\[Int: Int\\] = \\[([^\\]]+)\\]`));
  if (!match) {
    fail(`missing Swift [Int: Int] ${name}`);
  }
  return Object.fromEntries([...match[1].matchAll(/(\d+)\s*:\s*(\d+)/g)].map((m) => [m[1], Number(m[2])]));
}

/** Returns the `kind: "value"` pin after the MisakiSwift package URL. */
function misakiPin(source, file) {
  const match = source.match(/url: "https:\/\/github\.com\/mattmireles\/MisakiSwift",\s*(\w+): "([^"]+)"/);
  if (!match) {
    fail(`${file} is missing the MisakiSwift package pin`);
  }
  return `${match[1]}: ${match[2]}`;
}

function manifestSchemaRequiredKeys(schemaSource) {
  const schema = JSON.parse(schemaSource);
  return schema.required || [];
}

function decoderCodingKeys(decoderSource) {
  const match = decoderSource.match(/private enum CodingKeys: String, CodingKey \{([\s\S]*?)\n    \}/);
  if (!match) {
    fail('missing KokoroRuntimeManifest CodingKeys');
  }
  return [...match[1].matchAll(/case\s+\w+(?:\s*=\s*"([^"]+)")?/g)].map((item) => {
    if (item[1]) {
      return item[1];
    }
    return item[0].replace(/^case\s+/, '').trim();
  });
}

function sameArray(a, b) {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function assertArrayEqual(label, a, b) {
  if (!sameArray(a, b)) {
    fail(`${label} drift: ${JSON.stringify(a)} != ${JSON.stringify(b)}`);
  }
}

const sources = Object.fromEntries(
  await Promise.all(Object.entries(files).map(async ([key, rel]) => [
    key,
    await readFile(path.join(repoRoot, rel), 'utf8'),
  ]))
);

const contract = {
  iOS: '18.0',
  macOS: '15.0',
  voiceEmbeddingDim: parseIntConstant(sources.pipeline, 'voiceEmbeddingDim'),
  maxCallerChunkTokens: parseIntConstant(sources.pipeline, 'maxCallerChunkTokens'),
  durationTokenSizes: parseSwiftIntArray(sources.pipeline, 'durationTokenSizes'),
  sdkDurationTokenSizes: [parseIntConstant(sources.tts, 'runtimeDurationTokenLength')],
  fullBuckets: parseSwiftIntArray(sources.pipeline, 'defaultBuckets'),
  starterBuckets: [15],
  starterVoice: 'af_heart',
  tFramesForBucket: parseSwiftIntMap(sources.pipeline, 'tFramesForBucket'),
};

const sdkProfiles = JSON.parse(sources.sdkProfiles);
assertArrayEqual('sdk_profiles starter buckets', sdkProfiles.profiles.starter.buckets, contract.starterBuckets);
assertArrayEqual('sdk_profiles full buckets', sdkProfiles.profiles.full.buckets, contract.fullBuckets);
assertArrayEqual('sdk_profiles duration sizes', sdkProfiles.duration_token_sizes, contract.sdkDurationTokenSizes);
assertArrayEqual('JS prep duration sizes', parseJsConstArray(sources.jsPrep, 'EnumSizes'), contract.durationTokenSizes);
assertArrayEqual('Python prep duration sizes', parsePythonIntArray(sources.pyPrep, 'ENUM_SIZES'), contract.durationTokenSizes);

if (parseJsConstInt(sources.jsPrep, 'MaxCallerChunkTokens') !== contract.maxCallerChunkTokens) {
  fail('JS prep max caller chunk token cap drifted');
}
if (parsePythonIntConstant(sources.pyPrep, 'MAX_CALLER_CHUNK_TOKENS') !== contract.maxCallerChunkTokens) {
  fail('Python prep max caller chunk token cap drifted');
}
if (parseJsConstInt(sources.jsPrep, 'VoiceEmbeddingDim') !== contract.voiceEmbeddingDim) {
  fail('JS prep voice embedding dimension drifted');
}

requireIncludes(sources.package, files.package, `.macOS("${contract.macOS}")`);
requireIncludes(sources.package, files.package, `.iOS("${contract.iOS}")`);
requireIncludes(sources.consumerFixturePackage, files.consumerFixturePackage, '.product(name: "KokoroTTS", package: "swift-tts")');
requireIncludes(sources.voice, files.voice, `KokoroVoiceID("${contract.starterVoice}")`);
requireIncludes(sources.voice, files.voice, 'public static let starterVoices: [KokoroVoiceID] = [.afHeart]');
requireIncludes(sources.buildBundle, files.buildBundle, `minimum_platforms: { iOS: '${contract.iOS}', macOS: '${contract.macOS}' }`);
requireIncludes(sources.buildBundle, files.buildBundle, 'refusing to build an SDK bundle from a dirty working tree');

for (const key of manifestSchemaRequiredKeys(sources.manifestSchema)) {
  requireIncludes(sources.buildBundle, files.buildBundle, key);
  if (!decoderCodingKeys(sources.manifestDecoder).includes(key)) {
    fail(`${files.manifestDecoder} is missing CodingKey for manifest field: ${key}`);
  }
}

requireIncludes(sources.downloadedStore, files.downloadedStore, 'expectedManifestSHA256');
requireIncludes(sources.downloadedStore, files.downloadedStore, 'allowInsecureLocalDevelopment');
requireIncludes(sources.errors, files.errors, 'insecureManifestURL');
requireIncludes(sources.errors, files.errors, 'downloadTooLarge');

// custom-profile geometry must match what the Swift pipeline loads.
const custom = sdkProfiles.custom;
for (const [bucket, tFrames] of Object.entries(custom.t_frames)) {
  if (contract.tFramesForBucket[bucket] !== tFrames) {
    fail(`sdk_profiles custom.t_frames[${bucket}]=${tFrames} != Swift tFramesForBucket[${bucket}]=${contract.tFramesForBucket[bucket]}`);
  }
}
// Hybrid row count per bucket: ceil(x_pre frames / core), x_pre = 2 * tFrames.
// core is plan.json's `core`; 224 when the gitignored export is absent (see
// HybridGenerator's FoldGeometry in swift/Sources/KokoroPipeline/HybridGenerator.swift).
const hybridCore = existsSync(hybridPlanPath) ? JSON.parse(await readFile(hybridPlanPath, 'utf8')).core : 224;
for (const [bucket, rows] of Object.entries(custom.hybrid_rows)) {
  const expected = Math.ceil((2 * custom.t_frames[bucket]) / hybridCore);
  if (rows !== expected) {
    fail(`sdk_profiles custom.hybrid_rows[${bucket}]=${rows} != ceil(2*${custom.t_frames[bucket]}/${hybridCore})=${expected}`);
  }
}

// The root Package.swift mirrors swift-tts/Package.swift for SwiftPM consumers.
if (misakiPin(sources.rootPackage, files.rootPackage) !== misakiPin(sources.package, files.package)) {
  fail(`${files.rootPackage} MisakiSwift pin != ${files.package} pin`);
}

console.log('SDK drift check passed');
