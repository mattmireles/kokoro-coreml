import json
import subprocess
from pathlib import Path


def sha256_hex(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def test_validate_sdk_bundle_rejects_hosted_manifest_path_escape(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    bundle = tmp_path / "bundle"
    runtime = bundle / "runtime"
    runtime.mkdir(parents=True)
    vocab = b"{}"
    hnsf = b"{}"
    (runtime / "kokoro-vocab.json").write_bytes(vocab)
    (runtime / "hnsf_weights.json").write_bytes(hnsf)

    runtime_manifest = {
        "model_packages": [],
        "voices": [],
        "runtime_assets": {
            "vocab": {
                "path": "runtime/kokoro-vocab.json",
                "bytes": len(vocab),
                "sha256": sha256_hex(vocab),
            },
            "hnsf_weights": {
                "path": "runtime/hnsf_weights.json",
                "bytes": len(hnsf),
                "sha256": sha256_hex(hnsf),
            },
        },
    }
    hosted = {
        "version": "bad",
        "files": [{
            "path": "../evil",
            "bytes": 0,
            "sha256": sha256_hex(b""),
        }],
    }
    (bundle / "KokoroRuntimeManifest.json").write_text(json.dumps(runtime_manifest), encoding="utf-8")
    (bundle / "HostedManifest.json").write_text(json.dumps(hosted), encoding="utf-8")

    result = subprocess.run(
        ["node", "scripts/validate_sdk_bundle.mjs", str(bundle)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "manifest path escapes bundle root" in result.stderr
