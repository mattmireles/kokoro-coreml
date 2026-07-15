#!/usr/bin/env node

import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const RepoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DefaultBotnetRoot = '/Users/mm/Documents/GitHub/botnet';
const RequiredFields = [
  'key',
  'text',
  'voice',
  'speed',
  'input_ids',
  'attention_mask',
  'ref_s',
  'canonical_duration_s'
];
const OptionalFields = ['num_tokens', 'hnsf_weights_sha256'];
const FloatTolerance = 1e-7;

function parseArgs(argv) {
  const args = new Map();
  const lists = new Map();
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    if (!item.startsWith('--')) {
      continue;
    }
    const key = item.slice(2);
    const values = [];
    while (index + 1 < argv.length && !argv[index + 1].startsWith('--')) {
      values.push(argv[index + 1]);
      index += 1;
    }
    if (values.length === 0) {
      args.set(key, 'true');
    } else if (values.length === 1) {
      args.set(key, values[0]);
    } else {
      lists.set(key, values);
    }
  }
  return { args, lists };
}

async function loadFixtures(paths) {
  const fixtures = [];
  for (const fixturePath of paths) {
    const parsed = JSON.parse(await readFile(fixturePath, 'utf8'));
    const entries = Array.isArray(parsed) ? parsed : [parsed];
    for (const entry of entries) {
      fixtures.push({ ...entry, fixturePath });
    }
  }
  return fixtures;
}

function runNodeScript({ script, cwd, args }) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [script, ...args], {
      cwd,
      env: process.env,
      stdio: ['ignore', 'pipe', 'pipe']
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });
    child.on('close', (status) => {
      resolve({ status, stdout, stderr });
    });
  });
}

async function prepareWithScript({ script, cwd, runtimeRoot, fixture, tmpdir }) {
  const textPath = path.join(tmpdir, `${fixture.key}.txt`);
  const outputPath = path.join(tmpdir, `${fixture.key}-${path.basename(cwd)}.json`);
  await writeFile(textPath, fixture.text, 'utf8');

  const args = [
    '--text-file',
    textPath,
    '--output',
    outputPath,
    '--key',
    fixture.key,
    '--voice',
    fixture.voice,
    '--speed',
    String(fixture.speed)
  ];
  if (runtimeRoot) {
    args.push('--runtime-root', runtimeRoot);
  }

  const result = await runNodeScript({ script, cwd, args });
  if (result.status !== 0) {
    return { ok: false, error: result.stderr.trim() || result.stdout.trim() };
  }
  return {
    ok: true,
    output: JSON.parse(await readFile(outputPath, 'utf8'))
  };
}

function assertArrayEqual(name, actual, expected) {
  if (!Array.isArray(actual) || !Array.isArray(expected)) {
    throw new Error(`${name} must be arrays`);
  }
  if (actual.length !== expected.length) {
    throw new Error(`${name} length differs: ${actual.length} !== ${expected.length}`);
  }
  for (let index = 0; index < actual.length; index += 1) {
    if (actual[index] !== expected[index]) {
      throw new Error(`${name}[${index}] differs: ${actual[index]} !== ${expected[index]}`);
    }
  }
}

function assertFloatArrayClose(name, actual, expected) {
  if (!Array.isArray(actual) || !Array.isArray(expected)) {
    throw new Error(`${name} must be arrays`);
  }
  if (actual.length !== expected.length) {
    throw new Error(`${name} length differs: ${actual.length} !== ${expected.length}`);
  }
  for (let index = 0; index < actual.length; index += 1) {
    if (Math.abs(actual[index] - expected[index]) > FloatTolerance) {
      throw new Error(`${name}[${index}] differs: ${actual[index]} !== ${expected[index]}`);
    }
  }
}

function compareOutputs(fixture, local, botnet) {
  for (const field of RequiredFields) {
    if (!(field in local)) {
      throw new Error(`${fixture.key}: local output missing required field ${field}`);
    }
    if (!(field in botnet)) {
      throw new Error(`${fixture.key}: botnet output missing required field ${field}`);
    }
  }

  for (const field of ['key', 'text', 'voice', 'speed', 'canonical_duration_s']) {
    if (local[field] !== botnet[field]) {
      throw new Error(`${fixture.key}: ${field} differs: ${local[field]} !== ${botnet[field]}`);
    }
  }
  assertArrayEqual(`${fixture.key}.input_ids`, local.input_ids, botnet.input_ids);
  assertArrayEqual(`${fixture.key}.attention_mask`, local.attention_mask, botnet.attention_mask);
  assertFloatArrayClose(`${fixture.key}.ref_s`, local.ref_s, botnet.ref_s);

  for (const field of OptionalFields) {
    if (field in local || field in botnet) {
      if (local[field] !== botnet[field]) {
        throw new Error(`${fixture.key}: optional field ${field} differs`);
      }
    }
  }
}

async function main() {
  const { args, lists } = parseArgs(process.argv.slice(2));
  const fixtures = lists.get('fixtures') ?? (args.has('fixtures') ? [args.get('fixtures')] : []);
  if (fixtures.length === 0) {
    throw new Error('provide --fixtures <fixture.json> [...]');
  }

  const botnetRoot = path.resolve(args.get('botnet-root') ?? DefaultBotnetRoot);
  const compareMode = args.get('compare') ?? 'full';
  if (compareMode !== 'full') {
    throw new Error(`unsupported --compare value: ${compareMode}`);
  }

  const localScript = path.join(RepoRoot, 'scripts/kokoro-prepare-input.mjs');
  const botnetScript = path.join(botnetRoot, 'scripts/kokoro-prepare-input.mjs');
  const loaded = await loadFixtures(fixtures);
  const tmpdir = await mkdtemp(path.join(os.tmpdir(), 'kokoro-prepare-compare-'));

  try {
    let passed = 0;
    for (const fixture of loaded) {
      const required = ['key', 'text', 'voice', 'speed'];
      for (const field of required) {
        if (!(field in fixture)) {
          throw new Error(`${fixture.fixturePath}: missing fixture field ${field}`);
        }
      }

      const local = await prepareWithScript({
        script: localScript,
        cwd: RepoRoot,
        runtimeRoot: RepoRoot,
        fixture,
        tmpdir
      });
      const botnet = await prepareWithScript({
        script: botnetScript,
        cwd: botnetRoot,
        runtimeRoot: null,
        fixture,
        tmpdir
      });

      if (fixture.expect_error) {
        if (local.ok || botnet.ok) {
          throw new Error(`${fixture.key}: expected both scripts to fail`);
        }
        passed += 1;
        console.log(`ok ${fixture.key} expected-error`);
        continue;
      }
      if (!local.ok || !botnet.ok) {
        throw new Error(
          `${fixture.key}: unexpected failure\nlocal: ${local.error ?? 'ok'}\nbotnet: ${botnet.error ?? 'ok'}`
        );
      }
      compareOutputs(fixture, local.output, botnet.output);
      passed += 1;
      console.log(`ok ${fixture.key}`);
    }
    console.log(`Compared ${passed} fixture(s) against Botnet`);
  } finally {
    await rm(tmpdir, { recursive: true, force: true });
  }
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});
