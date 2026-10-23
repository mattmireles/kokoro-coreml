#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { copyFile, lstat, mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const defaultRepoId = 'mattmireles/kokoro-coreml';
const runtimeAssetDir = path.join(repoRoot, 'swift-tts/Sources/KokoroTTS/Resources/KokoroRuntime');
const bundleMarkerName = '.kokoro-sdk-bundle';
// Every profile's contents, shared with scripts/download_models.py. Paths are
// bundle paths == HF paths == local checkout paths (coreml/..., g2p/...).
const sdkProfiles = JSON.parse(readFileSync(path.join(repoRoot, 'scripts/sdk_profiles.json'), 'utf8'));

/**
 * Returns HEAD only when the repository can honestly claim exact provenance.
 * `--allow-dirty 1` records `<HEAD>-dirty` instead, so a bundle built from
 * uncommitted sources never claims a commit it did not come from.
 */
function cleanSDKCommit(allowDirty) {
  const status = spawnSync(
    'git',
    ['status', '--porcelain', '--untracked-files=all'],
    { cwd: repoRoot, encoding: 'utf8' }
  );
  if (status.status !== 0) {
    throw new Error('failed to inspect SDK source provenance');
  }
  const dirty = Boolean(status.stdout.trim());
  if (dirty && !allowDirty) {
    throw new Error('refusing to build an SDK bundle from a dirty working tree (pass --allow-dirty 1 to record <HEAD>-dirty)');
  }
  const revision = spawnSync(
    'git',
    ['rev-parse', 'HEAD'],
    { cwd: repoRoot, encoding: 'utf8' }
  );
  if (revision.status !== 0 || !revision.stdout.trim()) {
    throw new Error('failed to resolve SDK source commit');
  }
  return dirty ? `${revision.stdout.trim()}-dirty` : revision.stdout.trim();
}

/** Parses command-line options into a map. */
function parseArgs(argv) {
  const args = new Map();
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    if (!item.startsWith('--')) {
      throw new Error(`unexpected positional argument: ${item}`);
    }
    const key = item.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) {
      throw new Error(`missing value for --${key}`);
    }
    args.set(key, value);
    index += 1;
  }
  return args;
}

/** Splits a comma-delimited option value. */
function splitList(value) {
  return value ? value.split(',').map((item) => item.trim()).filter(Boolean) : [];
}

/** Splits a comma-delimited integer option value. */
function splitIntegerList(value) {
  return splitList(value).map((item) => {
    const parsed = Number.parseInt(item, 10);
    if (!Number.isInteger(parsed) || parsed <= 0) {
      throw new Error(`invalid positive integer: ${item}`);
    }
    return parsed;
  });
}

/**
 * Returns { profile, voices, buckets, paths, durationTokenSizes } for the
 * requested profile. custom expands the per-bucket template; the hybrid
 * directory joins only for buckets >= 7 s, with that bucket's max row count.
 * Mirrors sdk_profile() in scripts/download_models.py (tests compare them).
 */
export function resolveProfile(args) {
  const profile = args.get('profile') || 'starter';
  const durationTokenSizes = sdkProfiles.duration_token_sizes;
  if (sdkProfiles.profiles[profile]) {
    return { profile, durationTokenSizes, ...sdkProfiles.profiles[profile] };
  }
  if (profile !== 'custom') {
    throw new Error(`unknown profile: ${profile}`);
  }
  const voices = splitList(args.get('voices'));
  const buckets = splitIntegerList(args.get('buckets'));
  if (voices.length === 0 || buckets.length === 0) {
    throw new Error('--profile custom requires --voices and --buckets');
  }
  const custom = sdkProfiles.custom;
  const paths = [...custom.paths];
  const rows = new Set();
  for (const bucket of buckets) {
    const tFrames = custom.t_frames[String(bucket)];
    if (tFrames === undefined) {
      throw new Error(`unsupported SDK bucket ${bucket}; choose from ${Object.keys(custom.t_frames).join(',')}`);
    }
    for (const template of custom.bucket_paths) {
      paths.push(template.replaceAll('{bucket}', String(bucket)).replaceAll('{t_frames}', String(tFrames)));
    }
    if (custom.hybrid_rows[String(bucket)] !== undefined) {
      rows.add(custom.hybrid_rows[String(bucket)]);
    }
  }
  if (rows.size > 0) {
    paths.push(...custom.hybrid_paths);
    for (const count of [...rows].sort((a, b) => a - b)) {
      paths.push(custom.hybrid_row_path.replaceAll('{rows}', String(count)));
    }
  }
  return { profile, voices, buckets, durationTokenSizes, paths: [...new Set(paths)] };
}

/** Returns the manifest model entry for a coreml/ path: coreml/<first segment>. */
function modelEntry(relativePath) {
  return relativePath.split('/').slice(0, 2).join('/');
}

/** Computes SHA-256 for one file. */
async function sha256File(filePath) {
  const data = await readFile(filePath);
  return createHash('sha256').update(data).digest('hex');
}

/** Refuses destructive output paths before the bundle builder deletes them. */
async function assertSafeOutputDirectory(outputDir) {
  const resolved = path.resolve(outputDir);
  const home = process.env.HOME ? path.resolve(process.env.HOME) : null;
  const dangerous = new Set([
    path.parse(resolved).root,
    repoRoot,
    path.dirname(repoRoot),
  ]);
  if (home) {
    dangerous.add(home);
  }
  if (dangerous.has(resolved)) {
    throw new Error(`refusing dangerous SDK bundle output path: ${resolved}`);
  }
  if (existsSync(resolved)) {
    const marker = path.join(resolved, bundleMarkerName);
    const runtimeManifest = path.join(resolved, 'KokoroRuntimeManifest.json');
    const hostedManifest = path.join(resolved, 'HostedManifest.json');
    if (!existsSync(marker) && !existsSync(runtimeManifest) && !existsSync(hostedManifest)) {
      throw new Error(`refusing to overwrite unmarked directory: ${resolved}`);
    }
  }
}

/** Loads and validates a download manifest generated by download_models.py. */
async function loadDownloadManifest(args, repoId, revision) {
  const manifestPath = args.get('download-manifest');
  if (!manifestPath) {
    if (args.get('allow-local-provenance') === '1') {
      return null;
    }
    throw new Error('--download-manifest is required for HF provenance; pass --allow-local-provenance 1 for local throwaway bundles');
  }
  const resolved = path.resolve(repoRoot, manifestPath);
  const manifest = JSON.parse(await readFile(resolved, 'utf8'));
  if (manifest.repo_id !== repoId) {
    throw new Error(`download manifest repo_id mismatch: expected ${repoId}, observed ${manifest.repo_id}`);
  }
  if ((manifest.revision || null) !== (revision || null)) {
    throw new Error(`download manifest revision mismatch: expected ${revision}, observed ${manifest.revision}`);
  }
  // A branch name moves; only a full commit SHA says which bytes were verified.
  if (!/^[0-9a-f]{40}$/.test(revision || '')) {
    throw new Error(`--revision must be a full 40-hex HF commit SHA, got ${revision}`);
  }
  return { path: resolved, manifest };
}

/** Verifies a source artifact file against the HF download manifest. */
async function verifyDownloadedFile(downloadManifest, relativePath) {
  if (!downloadManifest) {
    return;
  }
  const normalized = relativePath.split(path.sep).join('/');
  const entry = downloadManifest.manifest.files.find((item) => item.path === normalized);
  if (!entry) {
    throw new Error(`download manifest missing source file: ${normalized}`);
  }
  const filePath = path.join(repoRoot, relativePath);
  const fileStat = await stat(filePath);
  const fileHash = await sha256File(filePath);
  if (fileStat.size !== entry.bytes || fileHash !== entry.sha256) {
    throw new Error(`download manifest digest mismatch for ${normalized}`);
  }
}

/**
 * Verifies a copied model or voice source against the HF download manifest,
 * both ways: every local file must match a manifest entry, and every manifest
 * entry at or under `relativePath` must exist locally (a half-finished
 * download would otherwise ship a package missing files).
 */
async function verifySourceAgainstDownloadManifest(downloadManifest, relativePath) {
  if (!downloadManifest) {
    return;
  }
  const source = path.join(repoRoot, relativePath);
  const sourceStat = await stat(source);
  const local = new Set();
  if (sourceStat.isDirectory()) {
    for (const filePath of await listFiles(source)) {
      const rel = path.relative(repoRoot, filePath);
      await verifyDownloadedFile(downloadManifest, rel);
      local.add(rel.split(path.sep).join('/'));
    }
  } else if (sourceStat.isFile()) {
    await verifyDownloadedFile(downloadManifest, relativePath);
    local.add(relativePath);
  }
  for (const entry of downloadManifest.manifest.files) {
    if ((entry.path === relativePath || entry.path.startsWith(`${relativePath}/`)) && !local.has(entry.path)) {
      throw new Error(`download manifest file missing locally: ${entry.path}`);
    }
  }
}

/** Rejects symlinks and path escapes for source artifacts. */
async function assertRealPathInside(root, target) {
  if ((await lstat(target)).isSymbolicLink()) {
    throw new Error(`refusing symlink input: ${target}`);
  }
  const resolvedRoot = await realpath(root);
  const resolvedTarget = await realpath(target);
  if (!resolvedTarget.startsWith(`${resolvedRoot}${path.sep}`) && resolvedTarget !== resolvedRoot) {
    throw new Error(`path escapes ${root}: ${target}`);
  }
  return resolvedTarget;
}

/** Returns the real path for an existing file or directory. */
async function realpath(target) {
  const { realpath: fsRealpath } = await import('node:fs/promises');
  return fsRealpath(target);
}

/** Copies a directory tree while rejecting symlinks. */
async function copyDirectory(src, dest) {
  const srcStat = await stat(src);
  if (!srcStat.isDirectory()) {
    throw new Error(`source is not a directory: ${src}`);
  }
  await mkdir(dest, { recursive: true });
  for (const entry of await readdir(src, { withFileTypes: true })) {
    const from = path.join(src, entry.name);
    const to = path.join(dest, entry.name);
    if (entry.isSymbolicLink()) {
      throw new Error(`refusing to copy symlink: ${from}`);
    }
    if (entry.isDirectory()) {
      await copyDirectory(from, to);
    } else if (entry.isFile()) {
      await copyFile(from, to);
    }
  }
}

/** Lists files in a directory tree in stable order. */
async function listFiles(root) {
  const files = [];
  async function visit(dir) {
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      const current = path.join(dir, entry.name);
      if (entry.isSymbolicLink()) {
        throw new Error(`symlink not allowed in bundle inputs: ${current}`);
      }
      if (entry.isDirectory()) {
        await visit(current);
      } else if (entry.isFile()) {
        files.push(current);
      }
    }
  }
  await visit(root);
  return files.sort();
}

/** Computes a stable tree digest for a model directory (`.mlpackage` or coreml/hybrid). */
async function hashTree(packagePath, relativePath) {
  const digest = createHash('sha256');
  const files = [];
  for (const filePath of await listFiles(packagePath)) {
    const rel = path.relative(packagePath, filePath).split(path.sep).join('/');
    const fileStat = await stat(filePath);
    const fileHash = await sha256File(filePath);
    digest.update(rel);
    digest.update('\0');
    digest.update(String(fileStat.size));
    digest.update('\0');
    digest.update(fileHash);
    digest.update('\0');
    files.push({ path: rel, bytes: fileStat.size, sha256: fileHash });
  }
  return {
    path: relativePath,
    tree_sha256: digest.digest('hex'),
    file_count: files.length,
    bytes: files.reduce((sum, file) => sum + file.bytes, 0),
    files,
  };
}

/** Returns a file digest object for manifests. */
async function fileDigest(root, relativePath) {
  const filePath = path.join(root, relativePath);
  const fileStat = await stat(filePath);
  return {
    path: relativePath.split(path.sep).join('/'),
    bytes: fileStat.size,
    sha256: await sha256File(filePath),
  };
}

/** Builds one SDK bundle profile. */
async function buildBundle() {
  const args = parseArgs(process.argv.slice(2));
  const config = resolveProfile(args);
  const repoId = args.get('repo-id') || defaultRepoId;
  // Required: the HF commit the sources are verified against. A default here
  // went stale every release and silently recorded the wrong provenance.
  const revision = args.get('revision');
  if (!revision) {
    throw new Error('--revision <hf-commit-sha> is required');
  }
  const sdkCommit = cleanSDKCommit(args.get('allow-dirty') === '1');
  const outputDir = path.resolve(repoRoot, args.get('output') || `outputs/sdk-bundles/${config.profile}`);
  await assertSafeOutputDirectory(outputDir);
  const voiceSources = config.voices.map((voice) => `kokoro.js/voices/${voice}.bin`);
  const missing = [...config.paths, ...voiceSources].filter((item) => !existsSync(path.join(repoRoot, item)));
  if (missing.length > 0) {
    throw new Error(`missing bundle artifacts (run scripts/download_models.py first): ${missing.join(', ')}`);
  }
  const downloadManifest = await loadDownloadManifest(args, repoId, revision);

  await rm(outputDir, { recursive: true, force: true });
  await mkdir(path.join(outputDir, 'coreml'), { recursive: true });
  await mkdir(path.join(outputDir, 'voices'), { recursive: true });
  await mkdir(path.join(outputDir, 'runtime'), { recursive: true });
  await mkdir(path.join(outputDir, 'compiled'), { recursive: true });
  await writeFile(path.join(outputDir, bundleMarkerName), 'kokoro-sdk-bundle\n');

  for (const relativePath of config.paths) {
    const src = path.join(repoRoot, relativePath);
    await assertRealPathInside(path.join(repoRoot, relativePath.split('/')[0]), src);
    await verifySourceAgainstDownloadManifest(downloadManifest, relativePath);
    const dest = path.join(outputDir, relativePath);
    await mkdir(path.dirname(dest), { recursive: true });
    if ((await stat(src)).isDirectory()) {
      await copyDirectory(src, dest);
    } else {
      await copyFile(src, dest);
    }
  }
  const modelEntries = [...new Set(config.paths.filter((item) => item.startsWith('coreml/')).map(modelEntry))];
  const modelPackages = [];
  for (const entry of modelEntries) {
    modelPackages.push(await hashTree(path.join(outputDir, entry), entry));
  }
  if (args.get('compile-models') === '1') {
    // Only top-level .mlpackage entries. coreml/hybrid is skipped: the Swift
    // hybrid generator compiles its own rows into its own cache.
    for (const entry of modelEntries.filter((item) => item.endsWith('.mlpackage'))) {
      const result = spawnSync(
        'xcrun',
        ['coremlcompiler', 'compile', path.join(outputDir, entry), path.join(outputDir, 'compiled')],
        { stdio: 'inherit' }
      );
      if (result.status !== 0) {
        throw new Error(`coremlcompiler failed for ${entry}`);
      }
      const packageDigest = modelPackages.find((item) => item.path === entry);
      const compiledName = path.basename(entry).replace(/\.mlpackage$/, '.mlmodelc');
      await writeFile(
        path.join(outputDir, 'compiled', `${compiledName}.kokoro-source-tree-sha256`),
        `${packageDigest.tree_sha256}\n`
      );
    }
  }
  for (const [index, voice] of config.voices.entries()) {
    const src = path.join(repoRoot, voiceSources[index]);
    await assertRealPathInside(path.join(repoRoot, 'kokoro.js/voices'), src);
    await verifySourceAgainstDownloadManifest(downloadManifest, voiceSources[index]);
    await copyFile(src, path.join(outputDir, 'voices', `${voice}.bin`));
  }
  for (const name of ['kokoro-vocab.json', 'hnsf_weights.json']) {
    const src = path.join(runtimeAssetDir, name);
    await assertRealPathInside(runtimeAssetDir, src);
    await copyFile(src, path.join(outputDir, 'runtime', name));
  }

  const voiceDigests = [];
  for (const voice of config.voices) {
    voiceDigests.push(await fileDigest(outputDir, path.join('voices', `${voice}.bin`)));
  }
  const manifest = {
    // 2: model_packages may be directories that are not .mlpackage
    // (coreml/hybrid). SDK code that only knows 1 rejects it outright.
    schema_version: 2,
    generated_at: new Date().toISOString(),
    sdk_commit: sdkCommit,
    hf_repo_id: repoId,
    // Where models, voices, and G2P assets were verified byte-identical. Not a
    // hydratable pin: this manifest itself lands in a later HF commit.
    hf_artifact_revision: revision,
    hf_provenance_verified: downloadManifest !== null,
    hf_download_manifest_sha256: downloadManifest ? await sha256File(downloadManifest.path) : null,
    bundle_profile: config.profile,
    minimum_platforms: { iOS: '18.0', macOS: '15.0' },
    supported_languages: ['en-US'],
    buckets: config.buckets,
    duration_token_sizes: config.durationTokenSizes,
    model_packages: modelPackages,
    voices: voiceDigests,
    runtime_assets: {
      vocab: await fileDigest(outputDir, 'runtime/kokoro-vocab.json'),
      hnsf_weights: await fileDigest(outputDir, 'runtime/hnsf_weights.json'),
    },
  };
  if (config.paths.some((item) => item.startsWith('g2p/'))) {
    manifest.g2p_assets = [];
    for (const filePath of await listFiles(path.join(outputDir, 'g2p'))) {
      manifest.g2p_assets.push(await fileDigest(outputDir, path.relative(outputDir, filePath)));
    }
  }
  await writeFile(path.join(outputDir, 'KokoroRuntimeManifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);

  const hostedFiles = [];
  for (const filePath of await listFiles(outputDir)) {
    const rel = path.relative(outputDir, filePath).split(path.sep).join('/');
    if (rel === bundleMarkerName || rel.startsWith('compiled/')) {
      continue;
    }
    hostedFiles.push({
      path: rel,
      bytes: (await stat(filePath)).size,
      sha256: await sha256File(filePath),
    });
  }
  const hosted = {
    // Full sdkCommit, so a `-dirty` build never poses as a clean commit.
    version: `${config.profile}-${sdkCommit}`,
    files: hostedFiles,
  };
  await writeFile(path.join(outputDir, 'HostedManifest.json'), `${JSON.stringify(hosted, null, 2)}\n`);

  console.log(`built ${config.profile} SDK bundle at ${outputDir}`);
  console.log(`  models=${modelPackages.length} voices=${voiceDigests.length} files=${hostedFiles.length}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  buildBundle().catch((error) => {
    console.error(error instanceof Error ? error.stack || error.message : String(error));
    process.exitCode = 1;
  });
}
