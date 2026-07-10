#!/usr/bin/env python3
"""Download CoreML models and voice files from Hugging Face Hub.

Models are stored on HF instead of Git LFS to avoid storage costs.
This script downloads them to the local paths the pipeline expects.

Usage::

    python scripts/download_models.py              # download everything
    python scripts/download_models.py --coreml     # only coreml/ packages
    python scripts/download_models.py --voices     # only kokoro.js/voices/
    python scripts/download_models.py --force      # re-download even if present

The HF token is read from (in order):
  1. HF_TOKEN environment variable
  2. .env file in the repo root
  3. huggingface-cli login cache (~/.huggingface/token)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Hugging Face repo that holds the model artifacts.
HF_REPO_ID = "mattmireles/kokoro-coreml"

# Patterns for each download group.  Matched against HF repo file paths.
# The HF repo hosts only the curated runtime set under coreml/ (DecoderPre +
# HAR-post buckets, F0Ntrain frame sizes, duration token sizes). Legacy
# coreml_fp32/ and KokoroVocoder.mlpackage were removed from HF in June 2026;
# the Swift runtime (swift/Sources/KokoroPipeline/KokoroPipeline.swift) never
# loads them.
COREML_PATTERNS = [
    "coreml/**",
]
VOICE_PATTERNS = [
    "kokoro.js/voices/*.bin",
]
# Files to always skip (not model artifacts).
# IMPORTANT: Do NOT add "*.json" here — it would exclude Manifest.json inside
# .mlpackage directories, producing corrupt/incomplete packages.
IGNORE_PATTERNS = [
    ".DS_Store",
    "*.py",
    "*.md",
    "*.txt",
    "*.yml",
    "*.yaml",
    "*.sh",
    "*.toml",
    # Repo-level JSON (not inside .mlpackage dirs).
    "config.json",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    ".claude/**",
    ".github/**",
    ".gitignore",
    ".gitattributes",
    ".venv*/**",
    "kokoro/**",
    "kokoro.js/src/**",
    "kokoro.js/demo/**",
    "kokoro.js/tests/**",
    "kokoro.js/*.json",
    "kokoro.js/*.js",
    "kokoro.js/*.ts",
    "kokoro.js/*.md",
    "tests/**",
    "scripts/**",
    "examples/**",
    "export_synth/**",
    "export_duration.py",
    "export_synthesizers.py",
    "convert_checkpoint.py",
    "README/**",
    "outputs/**",
]


def _load_token() -> str | None:
    """Load HF token from env, .env file, or huggingface-cli cache."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = _REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None  # Falls back to huggingface-cli login cache.


def download(
    allow_patterns: list[str],
    token: str | None,
    force: bool = False,
) -> str:
    """Download matching files from the HF repo to the repo root."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(_REPO_ROOT),
        allow_patterns=allow_patterns,
        ignore_patterns=IGNORE_PATTERNS,
        force_download=force,
        token=token,
    )


def _repair_missing_manifests(token: str | None, force: bool = False) -> int:
    """Download any Manifest.json files that snapshot_download silently skipped.

    huggingface_hub.snapshot_download() has a known quirk where it can skip
    Manifest.json files inside .mlpackage directories even when the
    allow_patterns should match them.  This function patches each affected
    package by downloading the file individually via hf_hub_download().

    Called by:
      - main(), immediately after download() completes.

    Returns:
        Number of packages that were repaired.
    """
    from huggingface_hub import hf_hub_download
    import glob as globmod

    repaired = 0
    # Search all .mlpackage dirs anywhere under the repo root.
    for pkg_dir in sorted(globmod.glob(str(_REPO_ROOT / "**" / "*.mlpackage"), recursive=True)):
        pkg = Path(pkg_dir)
        manifest = pkg / "Manifest.json"
        if manifest.exists() and not force:
            continue
        # Compute the HF-repo-relative path (e.g. "coreml/foo.mlpackage").
        rel_path = pkg.relative_to(_REPO_ROOT)
        hf_file = f"{rel_path}/Manifest.json"
        print(f"  Repairing {rel_path} — downloading missing Manifest.json ...")
        try:
            hf_hub_download(
                HF_REPO_ID,
                hf_file,
                token=token,
                local_dir=str(_REPO_ROOT),
                force_download=True,
            )
            repaired += 1
        except Exception as exc:
            print(f"    WARNING: could not download {hf_file}: {exc}")
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CoreML models and voices from Hugging Face Hub.",
    )
    parser.add_argument(
        "--coreml", action="store_true",
        help="Download only CoreML .mlpackage files",
    )
    parser.add_argument(
        "--voices", action="store_true",
        help="Download only kokoro.js voice .bin files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if files already exist locally",
    )
    args = parser.parse_args()

    token = _load_token()
    if token:
        print("Using HF token from configured credentials.")
    else:
        print("No HF_TOKEN found; using anonymous access (may fail for private repos).")

    # Determine which patterns to download.
    if args.coreml and not args.voices:
        patterns = COREML_PATTERNS
        label = "CoreML models"
    elif args.voices and not args.coreml:
        patterns = VOICE_PATTERNS
        label = "voice files"
    else:
        patterns = COREML_PATTERNS + VOICE_PATTERNS
        label = "all models and voices"

    print(f"\nDownloading {label} from {HF_REPO_ID}...")
    print(f"  Patterns: {patterns}")
    print(f"  Target:   {_REPO_ROOT}")
    print()

    try:
        result_dir = download(patterns, token, force=args.force)
        print(f"\nDownload complete: {result_dir}")
        # Repair any .mlpackage dirs where snapshot_download silently
        # skipped Manifest.json (known huggingface_hub quirk).
        n_repaired = _repair_missing_manifests(token, force=args.force)
        if n_repaired:
            print(f"Repaired {n_repaired} package(s) with missing Manifest.json.")
    except Exception as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Verify key artifacts exist and are complete (have Manifest.json).
    checks = [
        ("coreml/kokoro_duration.mlpackage", "Duration model (legacy fallback)"),
        ("coreml/kokoro_duration_t512.mlpackage", "Duration t512"),
        ("coreml/kokoro_f0ntrain_t120.mlpackage", "F0Ntrain t120"),
        ("coreml/kokoro_decoder_pre_3s.mlpackage", "DecoderPre 3s"),
        ("coreml/kokoro_decoder_har_post_3s.mlpackage", "HAR-post 3s"),
        ("coreml/kokoro_decoder_har_post_10s.mlpackage", "HAR-post 10s"),
    ]
    all_ok = True
    for rel_path, label in checks:
        p = _REPO_ROOT / rel_path
        manifest = p / "Manifest.json"
        weights = p / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
        if not p.exists():
            print(f"  MISSING: {label} ({rel_path})")
            all_ok = False
        elif not manifest.exists():
            print(f"  INCOMPLETE: {label} — missing Manifest.json")
            all_ok = False
        elif not weights.exists():
            print(f"  INCOMPLETE: {label} — missing weight.bin")
            all_ok = False
        else:
            print(f"  OK: {label}")

    # Also check all .mlpackage dirs under coreml/ for completeness.
    import glob as globmod
    print("\nPackage integrity check:")
    for pkg_dir in sorted(globmod.glob(str(_REPO_ROOT / "coreml" / "*.mlpackage"))):
        pkg = Path(pkg_dir)
        name = pkg.name
        manifest = pkg / "Manifest.json"
        if not manifest.exists():
            print(f"  INCOMPLETE: {name} — missing Manifest.json")
            all_ok = False
        else:
            print(f"  OK: {name}")

    if all_ok:
        print("\nAll models present and complete. Pipeline is ready.")
    else:
        print("\nSome models missing or incomplete. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
