#!/usr/bin/env python3
"""Compute stable digests for Core ML `.mlpackage` directory trees."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the tree hasher."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", nargs="+", type=Path, help=".mlpackage directories to hash")
    parser.add_argument("--output", type=Path, help="Write JSON output to this path")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Compute SHA-256 for a regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_mlpackage_tree(path: Path) -> dict[str, Any]:
    """Hash every file in one `.mlpackage` and return a stable summary."""

    if path.is_symlink():
        raise SystemExit(f"{path} must not be a symlink")
    if not path.is_dir() or path.suffix != ".mlpackage":
        raise SystemExit(f"{path} is not an .mlpackage directory")

    files = []
    tree_digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        if file_path.is_symlink():
            raise SystemExit(f"{file_path} must not be a symlink")
        rel = file_path.relative_to(path).as_posix()
        size = file_path.stat().st_size
        file_hash = sha256_file(file_path)
        tree_digest.update(rel.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(str(size).encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(file_hash.encode("ascii"))
        tree_digest.update(b"\0")
        files.append({
            "path": rel,
            "bytes": size,
            "sha256": file_hash,
        })

    return {
        "path": path.as_posix(),
        "tree_sha256": tree_digest.hexdigest(),
        "file_count": len(files),
        "bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def main() -> None:
    """CLI entry point for the `.mlpackage` tree hasher."""

    args = parse_args()
    result = {
        "schema_version": 1,
        "packages": [hash_mlpackage_tree(path) for path in args.packages],
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
