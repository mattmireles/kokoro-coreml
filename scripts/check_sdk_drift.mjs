#!/usr/bin/env node

import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const files = {
  pipeline: 'swift/Sources/KokoroPipeline/KokoroPipeline.swift',
  rootPackage: 'Package.swift',
  package: 'swift-tts/Package.swift',
  consumerFixturePackage: 'examples/KokoroConsumerFixture/Package.swift',
  sdkProfiles: 'sdk_profiles.json',
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

async function compileDocumentedSDKConsumer(manifestSHA256) {
  const tempRoot = await mkdtemp(path.join(tmpdir(), 'kokoro-sdk-docs-compile-'));
  try {
    await mkdir(path.join(tempRoot, 'Sources/DocsCompile'), { recursive: true });
    await writeFile(path.join(tempRoot, 'Package.swift'), `// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "KokoroDocsCompile",
    platforms: [.macOS("15.0"), .iOS("18.0")],
    dependencies: [
        .package(name: "kokoro-coreml", path: ${JSON.stringify(repoRoot)})
    ],
    targets: [
        .executableTarget(
            name: "DocsCompile",
            dependencies: [
                .product(name: "KokoroTTS", package: "kokoro-coreml")
            ]
        )
    ]
)
`);
    await writeFile(path.join(tempRoot, 'Sources/DocsCompile/main.swift'), `import Foundation
import KokoroTTS

@main
struct DocsCompile {
    static func main() async throws {
        let bundleURL = URL(fileURLWithPath: "/tmp/KokoroRuntime", isDirectory: true)
        let cacheURL = URL(fileURLWithPath: "/tmp/KokoroTTSCache", isDirectory: true)
        let bundled = KokoroResourceProvider.directory(bundleURL, compiledModelsDirectory: cacheURL)
        let tts = try await KokoroTTS.load(resources: bundled)
        try await tts.prewarm(text: "Hello world.", voice: .afHeart)
        let audio = try await tts.synthesize("Hello world.", voice: .afHeart)
        _ = try audio.makePCMBuffer()
        _ = try await KokoroDownloadedModelStore(
            manifestURL: URL(string: "https://huggingface.co/mattmireles/kokoro-coreml/resolve/main/HostedManifest.json")!,
            expectedManifestSHA256: "${manifestSHA256}",
            cacheDirectory: cacheURL
        ).hydrate()
    }
}
`);
    const result = spawnSync('swift', ['build', '--package-path', tempRoot], {
      encoding: 'utf8',
      maxBuffer: 20 * 1024 * 1024,
    });
    if (result.status !== 0) {
      fail(`documented SDK consumer snippet failed to compile:\n${result.stdout}\n${result.stderr}`);
    }
  } finally {
    await rm(tempRoot, { recursive: true, force: true });
  }
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
const profileConfig = JSON.parse(sources.sdkProfiles);
assertArrayEqual('sdk_profiles duration sizes', profileConfig.duration_token_sizes, contract.durationTokenSizes);
assertArrayEqual('sdk_profiles starter buckets', profileConfig.profiles.starter.buckets, contract.starterBuckets);
assertArrayEqual('sdk_profiles full buckets', profileConfig.profiles.full.buckets, contract.fullBuckets);
assertArrayEqual('sdk_profiles starter voices', profileConfig.profiles.starter.voices, [contract.starterVoice]);

requireIncludes(sources.buildBundle, files.buildBundle, 'sdk_profiles.json');
requireIncludes(sources.buildBundle, files.buildBundle, 'sdkProfiles.profiles.starter.buckets');
requireIncludes(sources.buildBundle, files.buildBundle, 'sdkProfiles.profiles.full.buckets');
requireIncludes(sources.downloadModels, files.downloadModels, 'sdk_profiles.json');
requireIncludes(sources.downloadModels, files.downloadModels, 'SDK_PROFILE_CONFIG');
requireIncludes(sources.downloadModels, files.downloadModels, 'FULL_BUCKET_SECONDS');
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
requireIncludes(sources.rootPackage, files.rootPackage, 'name: "kokoro-coreml"');
requireIncludes(sources.rootPackage, files.rootPackage, '.library(name: "KokoroTTS", targets: ["KokoroTTS"])');
requireIncludes(sources.consumerFixturePackage, files.consumerFixturePackage, '.product(name: "KokoroTTS", package: "kokoro-coreml")');
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

requireIncludes(sources.sdkReadme, files.sdkReadme, '.product(name: "KokoroTTS", package: "kokoro-coreml")');
requireIncludes(sources.modelCard, files.modelCard, 'matching Git release commit');

const manifestHashMatch = sources.sdkReadme.match(/expectedManifestSHA256:\s*"([0-9a-f]{64})"/);
if (!manifestHashMatch) {
  fail(`${files.sdkReadme} is missing expectedManifestSHA256 in the downloaded-resource snippet`);
}
await compileDocumentedSDKConsumer(manifestHashMatch[1]);

console.log('SDK drift check passed');
