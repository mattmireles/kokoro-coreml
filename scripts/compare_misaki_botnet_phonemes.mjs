#!/usr/bin/env node

import { spawnSync } from "node:child_process";

const DEFAULT_TEXTS = [
  "",
  "Hello world.",
  "Dr. Smith paid $12.50 for apples.",
  "Visit https://example.com, then email me@example.com.",
  "I live in Reading.",
];

/**
 * Prints command-line usage for the Misaki-vs-Botnet drift helper.
 *
 * Called by:
 * - `parseArgs(...)` when the user passes `--help` or `-h`.
 *
 * @returns {void}
 */
function usage() {
  console.error(
    [
      "Usage: node scripts/compare_misaki_botnet_phonemes.mjs --probe-bin <path> [--dyld-framework-path <path>] [--text <text>]...",
      "",
      "Compares the Swift Misaki probe against kokoro.js/Botnet eSpeak phonemization.",
    ].join("\n"),
  );
}

/**
 * Parses command-line options for the drift helper.
 *
 * Called by:
 * - `main()` before invoking the Swift probe executable.
 *
 * @param {string[]} argv Raw `process.argv` array.
 * @returns {{probeBin: string, dyldFrameworkPath: string | null, texts: string[]}}
 */
function parseArgs(argv) {
  const result = {
    probeBin: null,
    dyldFrameworkPath: null,
    texts: [],
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--probe-bin") {
      result.probeBin = requireValue(argv, ++i, arg);
    } else if (arg === "--dyld-framework-path") {
      result.dyldFrameworkPath = requireValue(argv, ++i, arg);
    } else if (arg === "--text") {
      result.texts.push(requireValue(argv, ++i, arg));
    } else if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!result.probeBin) {
    throw new Error("--probe-bin is required");
  }
  if (result.texts.length === 0) {
    result.texts = DEFAULT_TEXTS;
  }
  return result;
}

/**
 * Reads the value following a command-line flag.
 *
 * Called by:
 * - `parseArgs(...)` for all options that require a value.
 *
 * @param {string[]} argv Raw `process.argv` array.
 * @param {number} index Index expected to contain the option value.
 * @param {string} flag Flag name used in the error message.
 * @returns {string}
 */
function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

/**
 * Computes the number of matching UTF-16 code units at the start of two strings.
 *
 * Called by:
 * - `main()` while generating the Markdown drift table.
 *
 * @param {string} a First phoneme string.
 * @param {string} b Second phoneme string.
 * @returns {number}
 */
function commonPrefixLength(a, b) {
  const max = Math.min(a.length, b.length);
  let index = 0;
  while (index < max && a[index] === b[index]) {
    index += 1;
  }
  return index;
}

/**
 * Classifies the broad type of phoneme drift between Misaki and Botnet.
 *
 * Called by:
 * - `main()` for successful Swift probe rows.
 *
 * @param {string} misaki Phonemes from the Swift Misaki probe.
 * @param {string} botnet Phonemes from the Botnet JS/eSpeak oracle.
 * @returns {"exact" | "spacing-only" | "empty-output" | "phoneme-drift"}
 */
function classifyDrift(misaki, botnet) {
  if (misaki === botnet) {
    return "exact";
  }
  if (misaki.replaceAll(" ", "") === botnet.replaceAll(" ", "")) {
    return "spacing-only";
  }
  if (!misaki || !botnet) {
    return "empty-output";
  }
  return "phoneme-drift";
}

/**
 * Escapes a value for a single Markdown table cell.
 *
 * Called by:
 * - `main()` for every emitted Markdown cell that can contain arbitrary text.
 *
 * @param {unknown} value Value to render inside a Markdown table cell.
 * @returns {string}
 */
function escapeCell(value) {
  return String(value).replaceAll("|", "\\|").replaceAll("\n", " ");
}

/**
 * Runs the Swift probe and prints a Markdown comparison table against Botnet.
 *
 * Called by:
 * - The script entry point at the bottom of this file.
 *
 * @returns {Promise<void>}
 */
async function main() {
  const args = parseArgs(process.argv);
  const env = { ...process.env };
  if (args.dyldFrameworkPath) {
    env.DYLD_FRAMEWORK_PATH = args.dyldFrameworkPath;
  }
  const probe = spawnSync(args.probeBin, args.texts, {
    encoding: "utf8",
    env,
  });
  if (probe.status !== 0) {
    throw new Error(`Swift probe failed (${probe.status}): ${probe.stderr || probe.stdout}`);
  }

  const rows = probe.stdout
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  const { phonemize } = await import(new URL("../kokoro.js/src/phonemize.js", import.meta.url).href);

  console.log("| Text | Misaki Swift | Botnet JS/eSpeak | Drift | Shared prefix | Voice row consequence |");
  console.log("| --- | --- | --- | --- | ---: | --- |");
  for (const row of rows) {
    const botnet = await phonemize(row.text, "a", true);
    const misaki = row.phonemes ?? "";
    const drift = row.error ? "empty-output" : classifyDrift(misaki, botnet);
    const prefix = commonPrefixLength(misaki, botnet);
    const consequence = row.error
      ? `Misaki error: ${row.error}; Botnet length ${botnet.length}`
      : row.utf16Count === botnet.length
      ? `same UTF-16 count (${row.utf16Count})`
      : `voice row changes: Misaki ${row.utf16Count}, Botnet ${botnet.length}`;
    console.log(
      `| ${escapeCell(row.text)} | ${escapeCell(misaki)} | ${escapeCell(botnet)} | ${drift} | ${prefix} | ${escapeCell(consequence)} |`,
    );
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
