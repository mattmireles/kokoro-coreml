#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const files = {
  pipeline: 'swift/Sources/KokoroPipeline/KokoroPipeline.swift',
  package: 'swift-tts/Package.swift',
  consumerFixturePackage: 'examples/KokoroConsumerFixture/Package.swift',
  voice: 'swift-tts/Sources/KokoroTTS/KokoroVoiceID.swift',
  textChunker: 'swift-tts/Sources/KokoroTTS/TextChunker.swift',
  synthesisOptions: 'swift-tts/Sources/KokoroTTS/KokoroSynthesisOptions.swift',
  manifestDecoder: 'swift-tts/Sources/KokoroTTS/KokoroRuntimeManifest.swift',
  manifestSchema: 'schemas/KokoroRuntimeManifest.schema.json',
  buildBundle: 'scripts/build_sdk_bundle.mjs',
  downloadModels: 'scripts/download_models.py',
  jsPrep: 'scripts/kokoro-prepare-input.mjs',
  pyPrep: 'scripts/kokoro-prepare-input.py',
  sdkReadme: 'README/SDK.md',
  modelCard: 'README/hf-model-card.md',
};

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

function parseDoubleConstant(source, name) {
  const match = source.match(new RegExp(`(?:public )?static let ${name} = ([0-9.]+)`));
  if (!match) {
    fail(`missing Swift Double constant ${name}`);
  }
  return Number.parseFloat(match[1]);
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

function parseJsProfile(source, profile, field) {
  const profileMatch = source.match(new RegExp(`${profile}: \\{([\\s\\S]*?)\\n  \\}`));
  if (!profileMatch) {
    fail(`missing JS bundle profile ${profile}`);
  }
  const fieldMatch = profileMatch[1].match(new RegExp(`${field}: \\[([^\\]]+)\\]`));
  if (!fieldMatch) {
    fail(`missing ${field} for JS bundle profile ${profile}`);
  }
  return fieldMatch[1]
    .split(',')
    .map((item) => item.trim().replaceAll("'", '').replaceAll('"', ''))
    .filter(Boolean)
    .map((item) => (/^[0-9]+$/.test(item) ? Number.parseInt(item, 10) : item));
}

function requireIncludes(source, file, snippet) {
  if (!source.includes(snippet)) {
    fail(`${file} is missing required SDK contract snippet: ${snippet}`);
  }
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
  sampleRate: parseIntConstant(sources.pipeline, 'sampleRate'),
  voiceEmbeddingDim: parseIntConstant(sources.pipeline, 'voiceEmbeddingDim'),
  maxCallerChunkTokens: parseIntConstant(sources.pipeline, 'maxCallerChunkTokens'),
  durationTokenSizes: parseSwiftIntArray(sources.pipeline, 'durationTokenSizes'),
  fullBuckets: parseSwiftIntArray(sources.pipeline, 'defaultBuckets'),
  starterBuckets: [15],
  starterVoice: 'af_heart',
  defaultChunkSeconds: parseDoubleConstant(sources.synthesisOptions, 'defaultMaxChunkSeconds'),
};

assertArrayEqual(
  'build_sdk_bundle starter buckets',
  parseJsProfile(sources.buildBundle, 'starter', 'buckets'),
  contract.starterBuckets
);
assertArrayEqual(
  'build_sdk_bundle full buckets',
  parseJsProfile(sources.buildBundle, 'full', 'buckets'),
  contract.fullBuckets
);
assertArrayEqual(
  'download_models starter buckets',
  parsePythonIntArray(sources.downloadModels, 'STARTER_BUCKET_SECONDS'),
  contract.starterBuckets
);
assertArrayEqual(
  'download_models duration sizes',
  parsePythonIntArray(sources.downloadModels, 'SDK_DURATION_TOKEN_SIZES'),
  contract.durationTokenSizes
);
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

for (const key of manifestSchemaRequiredKeys(sources.manifestSchema)) {
  requireIncludes(sources.buildBundle, files.buildBundle, key);
  if (!decoderCodingKeys(sources.manifestDecoder).includes(key)) {
    fail(`${files.manifestDecoder} is missing CodingKey for manifest field: ${key}`);
  }
}

const docs = [files.sdkReadme, files.modelCard];
for (const file of docs) {
  const source = sources[Object.entries(files).find(([, rel]) => rel === file)[0]];
  requireIncludes(source, file, 'KokoroTTS');
  requireIncludes(source, file, `iOS ${contract.iOS}+`);
  requireIncludes(source, file, `macOS ${contract.macOS}+`);
  requireIncludes(source, file, String(contract.sampleRate));
  requireIncludes(source, file, contract.starterVoice);
  requireIncludes(source, file, contract.durationTokenSizes.join(','));
  requireIncludes(source, file, contract.fullBuckets.join(','));
  requireIncludes(source, file, String(contract.maxCallerChunkTokens));
  requireIncludes(source, file, String(contract.voiceEmbeddingDim));
}

requireIncludes(sources.sdkReadme, files.sdkReadme, '.product(name: "KokoroTTS", package: "swift-tts")');
requireIncludes(sources.modelCard, files.modelCard, 'matching Git release commit');

console.log('SDK drift check passed');
