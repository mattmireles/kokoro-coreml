"""Helpers shared by the Hugging Face release scripts.

One copy of the token lookup, file digests, repo constants and the SDK profile
file, imported by download_models.py, prepare_hf_sdk_metadata.py,
upload_hf_artifacts.py and inspect_hf_artifacts.py. ``python3 scripts/x.py``
puts scripts/ on sys.path, so a plain ``from hf_common import ...`` works.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = "mattmireles/kokoro-coreml"
# Paths hosted by upload_hf_artifacts.py, never by the metadata commit.
ARTIFACT_PREFIXES = ("coreml/", "g2p/")
# Every SDK profile's contents, shared with scripts/build_sdk_bundle.mjs.
SDK_PROFILES_PATH = REPO_ROOT / "scripts" / "sdk_profiles.json"
# Manifests whose file inventories must stay resolvable on HF, by profile.
# starter is hydrated by the Swift store from the one hosted manifest at the HF
# root; full is fetched by download_models.py, so its runtime manifest is the
# inventory.
INVENTORY_MANIFESTS = {"starter": "HostedManifest.json", "full": "sdk/full/KokoroRuntimeManifest.json"}


def load_sdk_profiles() -> dict[str, Any]:
    """Return scripts/sdk_profiles.json."""

    return json.loads(SDK_PROFILES_PATH.read_text(encoding="utf-8"))


def load_hf_token() -> str | None:
    """Return an HF token from HF_TOKEN, the repo .env, or the huggingface-cli cache."""

    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("HF_TOKEN="):
                return stripped.split("=", 1)[1].strip().strip("\"'")
    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token()
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    """Compute a SHA-256 digest for one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """Return the git blob id HF reports as `blob_id` for a non-LFS file."""

    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
