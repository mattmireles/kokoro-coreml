#!/usr/bin/env python3
"""Download CoreML models and voice files from Hugging Face Hub.

Models are stored on Hugging Face, not in this git repo. This script downloads
them to the local paths the pipeline expects.

Usage::

    python scripts/download_models.py              # download everything
    python scripts/download_models.py --coreml     # only coreml/ packages
    python scripts/download_models.py --voices     # only kokoro.js/voices/
    python scripts/download_models.py --force      # re-download even if present
    python scripts/download_models.py --sdk-profile starter --revision <sha>
                                                   # one SDK profile (scripts/sdk_profiles.json)

The HF token is read from (in order):
  1. HF_TOKEN environment variable
  2. .env file in the repo root
  3. huggingface-cli login cache (~/.huggingface/token)
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from hf_common import DEFAULT_REPO_ID as HF_REPO_ID
from hf_common import REPO_ROOT as _REPO_ROOT
from hf_common import git_blob_sha1, load_hf_token, load_sdk_profiles, sha256_file

# Patterns for each download group.  Matched against HF repo file paths.
# The HF repo hosts the runtime set under coreml/; the paper's frozen record
# also sits in sibling coreml-*/ prefixes, which these patterns never fetch.
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
    # Only coreml/ holds downloaded packages (including coreml/hybrid/). The
    # rest of the checkout has local experiment dirs (coreml-fold/, ...) whose
    # paths do not exist on HF, so --force must not try to "repair" them.
    for pkg_dir in sorted(globmod.glob(str(_REPO_ROOT / "coreml" / "**" / "*.mlpackage"), recursive=True)):
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


def sdk_profile(profile: str, voices: list[str], buckets: list[int]) -> dict:
    """Return {voices, buckets, paths} for an SDK profile from sdk_profiles.json.

    `paths` are repo-relative files or directories (bundle path == HF path).
    custom expands the per-bucket template; the hybrid generator directory is
    added only for buckets that use it (>= 7 s), with that bucket's row count.
    Mirrors resolveProfile() in scripts/build_sdk_bundle.mjs.
    """

    spec = load_sdk_profiles()
    if profile in spec["profiles"]:
        return dict(spec["profiles"][profile])
    if profile != "custom":
        raise SystemExit(f"unknown SDK profile: {profile}")
    if not voices or not buckets:
        raise SystemExit("--sdk-profile custom requires --sdk-voices and --sdk-buckets")
    custom = spec["custom"]
    paths = list(custom["paths"])
    rows = set()
    for bucket in buckets:
        t_frames = custom["t_frames"].get(str(bucket))
        if t_frames is None:
            raise SystemExit(f"unsupported SDK bucket {bucket}; choose from {sorted(map(int, custom['t_frames']))}")
        paths += [p.format(bucket=bucket, t_frames=t_frames) for p in custom["bucket_paths"]]
        if str(bucket) in custom["hybrid_rows"]:
            rows.add(custom["hybrid_rows"][str(bucket)])
    if rows:
        paths += custom["hybrid_paths"]
        paths += [custom["hybrid_row_path"].format(rows=n) for n in sorted(rows)]
    return {"voices": list(voices), "buckets": list(buckets), "paths": list(dict.fromkeys(paths))}


def sdk_patterns(config: dict) -> list[str]:
    """Return HF allow patterns for a resolved SDK profile."""

    patterns = []
    for path in config["paths"]:
        patterns += [path, f"{path}/**"]
    return patterns + [f"kokoro.js/voices/{voice}.bin" for voice in config["voices"]]


def _matches_any(path: str, patterns: list[str]) -> bool:
    """Return whether a repository-relative path matches any allow pattern."""

    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _repo_files_for_patterns(repo_id: str, revision: str | None, patterns: list[str], token: str | None) -> list[dict]:
    """Return HF file metadata for the exact repo files matched by allow patterns."""

    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    api = HfApi(token=token)
    files = []
    for item in api.list_repo_tree(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        recursive=True,
        expand=True,
    ):
        # `list_repo_tree(recursive=True)` yields both file (blob) and folder
        # (tree) nodes for nested paths like `*.mlpackage/Data/...`. Folder
        # nodes carry no size/sha256 and are never themselves a download
        # target, so they must be skipped here rather than treated as a
        # missing-digest error below.
        if not isinstance(item, RepoFile):
            continue
        path = getattr(item, "path", None)
        if not isinstance(path, str):
            continue
        if not _matches_any(path, patterns) or _matches_any(path, IGNORE_PATTERNS):
            continue
        files.append(_verified_file_record(item, _REPO_ROOT / path))
    return sorted(files, key=lambda item: item["path"])


def _verified_file_record(item, local: Path) -> dict:
    """Return {path, bytes, sha256} for one HF file.

    LFS files carry an HF-side sha256, recorded as is (build_sdk_bundle.mjs
    later checks local bytes against it). Small non-LFS files carry only a git
    blob id, so the local file must match that blob id before its local sha256
    is recorded; otherwise the manifest would vouch for whatever is on disk.
    """

    lfs = item.lfs
    if lfs is not None:
        sha256 = lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256
        return {"path": item.path, "bytes": int(item.size), "sha256": sha256}
    if not local.is_file() or git_blob_sha1(local) != item.blob_id:
        raise SystemExit(f"local file does not match HF blob id for {item.path}")
    return {"path": item.path, "bytes": local.stat().st_size, "sha256": sha256_file(local)}


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
        choices=[*load_sdk_profiles()["profiles"], "custom"],
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

    token = load_hf_token()
    if token:
        print("Using HF token from configured credentials.")
    else:
        print("No HF_TOKEN found; using anonymous access (may fail for private repos).")

    # Determine which patterns to download.
    sdk_voices = _split_csv(args.sdk_voices)
    sdk_buckets = _split_int_csv(args.sdk_buckets)

    if args.sdk_profile:
        sdk_config = sdk_profile(args.sdk_profile, sdk_voices, sdk_buckets)
        patterns = sdk_patterns(sdk_config)
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

    # Verify the profile's artifacts exist and packages are complete.
    all_ok = True
    required = []
    if args.sdk_profile:
        required = sdk_config["paths"] + [f"kokoro.js/voices/{v}.bin" for v in sdk_config["voices"]]
    for rel_path in required:
        p = _REPO_ROOT / rel_path
        if not p.exists():
            print(f"  MISSING: {rel_path}")
            all_ok = False
        elif p.suffix == ".mlpackage" and not (p / "Manifest.json").exists():
            print(f"  INCOMPLETE: {rel_path} — missing Manifest.json")
            all_ok = False
        elif p.suffix == ".mlpackage" and not (p / "Data" / "com.apple.CoreML" / "weights" / "weight.bin").exists():
            print(f"  INCOMPLETE: {rel_path} — missing weight.bin")
            all_ok = False
        else:
            print(f"  OK: {rel_path}")

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
