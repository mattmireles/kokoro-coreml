#!/usr/bin/env python3
"""Download CoreML models and voice files from Hugging Face Hub.

Models are stored on Hugging Face, not in this git repo. This script downloads
them to the local paths the pipeline expects.

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
import fnmatch
import hashlib
import json
import os
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
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
ENGLISH_VOICE_PATTERNS = [
    "kokoro.js/voices/af_*.bin",
    "kokoro.js/voices/am_*.bin",
    "kokoro.js/voices/bf_*.bin",
    "kokoro.js/voices/bm_*.bin",
]
STARTER_BUCKET_SECONDS = [15]
SDK_DURATION_TOKEN_SIZES = [32, 64, 128, 256, 320, 384, 512]
STARTER_VOICES = ["af_heart"]
SDK_PROFILE_CONFIG = json.loads((_REPO_ROOT / "sdk_profiles.json").read_text(encoding="utf-8"))
SDK_DURATION_TOKEN_SIZES = list(SDK_PROFILE_CONFIG["duration_token_sizes"])
STARTER_BUCKET_SECONDS = list(SDK_PROFILE_CONFIG["profiles"]["starter"]["buckets"])
STARTER_VOICES = list(SDK_PROFILE_CONFIG["profiles"]["starter"]["voices"])
FULL_BUCKET_SECONDS = list(SDK_PROFILE_CONFIG["profiles"]["full"]["buckets"])
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
    repo_id: str = HF_REPO_ID,
    revision: str | None = None,
    force: bool = False,
) -> str:
    """Download matching files from the HF repo to the repo root."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(_REPO_ROOT),
        allow_patterns=allow_patterns,
        ignore_patterns=IGNORE_PATTERNS,
        force_download=force,
        token=token,
    )


def _repair_missing_manifests(
    token: str | None,
    repo_id: str = HF_REPO_ID,
    revision: str | None = None,
    force: bool = False,
) -> int:
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
                repo_id,
                hf_file,
                revision=revision,
                token=token,
                local_dir=str(_REPO_ROOT),
                force_download=True,
            )
            repaired += 1
        except Exception as exc:
            print(f"    WARNING: could not download {hf_file}: {exc}")
    return repaired


def _split_csv(raw: str | None) -> list[str]:
    """Split a comma-delimited CLI value into non-empty strings."""

    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _split_int_csv(raw: str | None) -> list[int]:
    """Split a comma-delimited CLI value into positive integers."""

    values: list[int] = []
    for item in _split_csv(raw):
        try:
            value = int(item)
        except ValueError as exc:
            raise SystemExit(f"invalid integer in comma list: {item}") from exc
        if value <= 0:
            raise SystemExit(f"integer values must be positive: {item}")
        values.append(value)
    return values


def _sdk_patterns(profile: str, voices: list[str], buckets: list[int]) -> list[str]:
    """Return HF allow patterns for an SDK download profile."""

    if profile == "full":
        return COREML_PATTERNS + ENGLISH_VOICE_PATTERNS
    if profile == "starter":
        voices = voices or STARTER_VOICES
        buckets = buckets or STARTER_BUCKET_SECONDS
        duration_sizes = SDK_DURATION_TOKEN_SIZES
    elif profile == "custom":
        if not voices:
            raise SystemExit("--sdk-profile custom requires --sdk-voices")
        if not buckets:
            raise SystemExit("--sdk-profile custom requires --sdk-buckets")
        duration_sizes = SDK_DURATION_TOKEN_SIZES
    else:
        raise SystemExit(f"unknown SDK profile: {profile}")

    patterns: list[str] = []
    for size in duration_sizes:
        patterns.append(f"coreml/kokoro_duration_t{size}.mlpackage/**")
    for bucket in buckets:
        t_frames = bucket * 40
        patterns.extend([
            f"coreml/kokoro_f0ntrain_t{t_frames}.mlpackage/**",
            f"coreml/kokoro_decoder_pre_{bucket}s.mlpackage/**",
            f"coreml/kokoro_decoder_har_post_{bucket}s.mlpackage/**",
        ])
    for voice in voices:
        patterns.append(f"kokoro.js/voices/{voice}.bin")
    return patterns


def _sdk_required_packages(profile: str, voices: list[str], buckets: list[int]) -> list[str]:
    """Return model package directories required by an SDK profile."""

    if profile == "full":
        buckets = FULL_BUCKET_SECONDS
        duration_sizes = SDK_DURATION_TOKEN_SIZES
    elif profile == "starter":
        buckets = buckets or STARTER_BUCKET_SECONDS
        duration_sizes = SDK_DURATION_TOKEN_SIZES
    elif profile == "custom":
        duration_sizes = SDK_DURATION_TOKEN_SIZES
    else:
        raise SystemExit(f"unknown SDK profile: {profile}")

    packages: list[str] = []
    for size in duration_sizes:
        packages.append(f"coreml/kokoro_duration_t{size}.mlpackage")
    for bucket in buckets:
        packages.extend([
            f"coreml/kokoro_f0ntrain_t{bucket * 40}.mlpackage",
            f"coreml/kokoro_decoder_pre_{bucket}s.mlpackage",
            f"coreml/kokoro_decoder_har_post_{bucket}s.mlpackage",
        ])
    return packages


def _sha256_file(path: Path) -> str:
    """Compute a SHA-256 digest for a regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches_any(path: str, patterns: list[str]) -> bool:
    """Return whether a repository-relative path matches any allow pattern."""

    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _repo_files_for_patterns(repo_id: str, revision: str | None, patterns: list[str], token: str | None) -> list[dict]:
    """Return HF file metadata for the exact repo files matched by allow patterns."""

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = []
    for item in api.list_repo_tree(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        recursive=True,
        expand=True,
    ):
        path = getattr(item, "path", None)
        if not isinstance(path, str):
            continue
        if not _matches_any(path, patterns) or _matches_any(path, IGNORE_PATTERNS):
            continue
        size = getattr(item, "size", None)
        lfs = getattr(item, "lfs", None)
        local = _REPO_ROOT / path
        if size is None and lfs is None and local.is_dir():
            continue
        sha256 = None
        if isinstance(lfs, dict):
            sha256 = lfs.get("sha256")
            size = size if size is not None else lfs.get("size")
        elif lfs is not None:
            sha256 = getattr(lfs, "sha256", None)
            size = size if size is not None else getattr(lfs, "size", None)
        if not sha256:
            if local.is_file():
                sha256 = _sha256_file(local)
        if size is None:
            if local.is_file():
                size = local.stat().st_size
        if size is None or not sha256:
            raise SystemExit(f"could not determine HF digest metadata for {path}")
        files.append({
            "path": path,
            "bytes": int(size),
            "sha256": sha256,
        })
    return sorted(files, key=lambda item: item["path"])


def _write_download_manifest(
    manifest_path: Path,
    repo_id: str,
    revision: str | None,
    patterns: list[str],
    files: Iterable[dict],
) -> None:
    """Write a manifest for the exact HF files resolved by this download run."""

    records = sorted(files, key=lambda item: item["path"])
    manifest = {
        "schema_version": 1,
        "repo_id": repo_id,
        "revision": revision,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "allow_patterns": patterns,
        "files": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    parser.add_argument(
        "--repo-id",
        default=HF_REPO_ID,
        help=f"Hugging Face repo ID (default: {HF_REPO_ID})",
    )
    parser.add_argument(
        "--revision",
        help="Pinned Hugging Face revision/commit to download",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="Write a local file manifest for downloaded SDK artifacts",
    )
    parser.add_argument(
        "--sdk-profile",
        choices=["starter", "custom", "full"],
        help="Download SDK-oriented artifact profile",
    )
    parser.add_argument(
        "--sdk-voices",
        help="Comma-delimited custom SDK voices, for example af_heart,af_bella",
    )
    parser.add_argument(
        "--sdk-buckets",
        help="Comma-delimited custom SDK bucket seconds, for example 3,15",
    )
    args = parser.parse_args()

    token = _load_token()
    if token:
        print("Using HF token from configured credentials.")
    else:
        print("No HF_TOKEN found; using anonymous access (may fail for private repos).")

    # Determine which patterns to download.
    sdk_voices = _split_csv(args.sdk_voices)
    sdk_buckets = _split_int_csv(args.sdk_buckets)

    if args.sdk_profile:
        patterns = _sdk_patterns(
            args.sdk_profile,
            voices=sdk_voices,
            buckets=sdk_buckets,
        )
        label = f"SDK {args.sdk_profile} profile"
    elif args.coreml and not args.voices:
        patterns = COREML_PATTERNS
        label = "CoreML models"
    elif args.voices and not args.coreml:
        patterns = VOICE_PATTERNS
        label = "voice files"
    else:
        patterns = COREML_PATTERNS + VOICE_PATTERNS
        label = "all models and voices"

    print(f"\nDownloading {label} from {args.repo_id}...")
    if args.revision:
        print(f"  Revision: {args.revision}")
    print(f"  Patterns: {patterns}")
    print(f"  Target:   {_REPO_ROOT}")
    print()

    try:
        result_dir = download(
            patterns,
            token,
            repo_id=args.repo_id,
            revision=args.revision,
            force=args.force,
        )
        print(f"\nDownload complete: {result_dir}")
        # Repair any .mlpackage dirs where snapshot_download silently
        # skipped Manifest.json (known huggingface_hub quirk).
        n_repaired = _repair_missing_manifests(
            token,
            repo_id=args.repo_id,
            revision=args.revision,
            force=args.force,
        )
        if n_repaired:
            print(f"Repaired {n_repaired} package(s) with missing Manifest.json.")
        if args.manifest_out:
            files = _repo_files_for_patterns(args.repo_id, args.revision, patterns, token)
            _write_download_manifest(args.manifest_out, args.repo_id, args.revision, patterns, files)
            print(f"Wrote download manifest: {args.manifest_out}")
    except Exception as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Verify key artifacts exist and are complete (have Manifest.json).
    if args.sdk_profile:
        checks = [
            (rel_path, rel_path.removeprefix("coreml/").removesuffix(".mlpackage"))
            for rel_path in _sdk_required_packages(args.sdk_profile, sdk_voices, sdk_buckets)
        ]
    elif args.voices and not args.coreml:
        checks = []
    else:
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

    if checks:
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
    else:
        print("\nVoice-only download: skipped Core ML package integrity check.")

    if all_ok:
        print("\nAll requested artifacts are present and complete. Pipeline is ready.")
    else:
        print("\nSome models missing or incomplete. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
