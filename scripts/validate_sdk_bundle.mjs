#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { lstat, readFile, readdir, realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

/** Computes SHA-256 for one file. */
async function sha256File(filePath) {
  return createHash('sha256').update(await readFile(filePath)).digest('hex');
}

/** Resolves a manifest-relative path under a bundle root without escapes. */
async function containedPath(bundleRoot, relativePath) {
  const parts = relativePath.split('/');
  if (
    path.isAbsolute(relativePath) ||
    relativePath.includes('\\') ||
    parts.length === 0 ||
    parts.some((part) => part.length === 0 || part === '.' || part === '..')
  ) {
    throw new Error(`manifest path escapes bundle root: ${relativePath}`);
  }
  const rootReal = await realpath(bundleRoot);
  const target = path.join(bundleRoot, ...parts);
  let current = bundleRoot;
  for (const part of parts) {
    current = path.join(current, part);
    try {
      if ((await lstat(current)).isSymbolicLink()) {
        throw new Error(`symlink not allowed in manifest path: ${relativePath}`);
      }
    } catch (error) {
      if (error && error.code === 'ENOENT') {
        break;
      }
      throw error;
    }
  }
  const parentReal = await realpath(path.dirname(target));
  if (parentReal !== rootReal && !parentReal.startsWith(`${rootReal}${path.sep}`)) {
    throw new Error(`manifest path escapes bundle root: ${relativePath}`);
  }
  return target;
}

/** Lists all regular files in a bundle tree. */
async function listFiles(root) {
  const files = [];
  async function visit(dir) {
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      const current = path.join(dir, entry.name);
      if (entry.isSymbolicLink()) {
        throw new Error(`symlink not allowed in SDK bundle: ${current}`);
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

/** Computes a stable tree digest for one `.mlpackage`. */
async function hashPackage(packagePath) {
  const digest = createHash('sha256');
  let fileCount = 0;
  let bytes = 0;
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
    fileCount += 1;
    bytes += fileStat.size;
  }
  return { tree_sha256: digest.digest('hex'), file_count: fileCount, bytes };
}

/** Verifies one file digest entry relative to the bundle root. */
async function verifyFile(bundleRoot, entry) {
  const filePath = await containedPath(bundleRoot, entry.path);
  const fileStat = await stat(filePath);
  if (!fileStat.isFile()) {
    throw new Error(`manifest path is not a regular file: ${entry.path}`);
  }
  const actualHash = await sha256File(filePath);
  if (fileStat.size !== entry.bytes || actualHash !== entry.sha256) {
    throw new Error(`digest drift for ${entry.path}`);
  }
}

/** Validates the runtime and hosted manifests for one bundle root. */
async function validateBundle(bundleRoot) {
  const manifest = JSON.parse(await readFile(path.join(bundleRoot, 'KokoroRuntimeManifest.json'), 'utf8'));
  const hosted = JSON.parse(await readFile(path.join(bundleRoot, 'HostedManifest.json'), 'utf8'));
  for (const pkg of manifest.model_packages) {
    const actual = await hashPackage(await containedPath(bundleRoot, pkg.path));
    if (
      actual.tree_sha256 !== pkg.tree_sha256 ||
      actual.file_count !== pkg.file_count ||
      actual.bytes !== pkg.bytes
    ) {
      throw new Error(`package digest drift for ${pkg.path}`);
    }
  }
  for (const voice of manifest.voices) {
    await verifyFile(bundleRoot, voice);
  }
  await verifyFile(bundleRoot, manifest.runtime_assets.vocab);
  await verifyFile(bundleRoot, manifest.runtime_assets.hnsf_weights);
  for (const file of hosted.files) {
    await verifyFile(bundleRoot, file);
  }
  console.log(`SDK bundle verified: ${bundleRoot}`);
}

const bundleRoot = process.argv[2];
if (!bundleRoot) {
  console.error('usage: node scripts/validate_sdk_bundle.mjs <bundle-root>');
  process.exit(2);
}

validateBundle(path.resolve(bundleRoot)).catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
