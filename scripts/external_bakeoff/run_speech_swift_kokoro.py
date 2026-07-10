#!/usr/bin/env python3
"""Run Soniqo speech-swift KokoroTTS through a generated Swift CLI."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external_bakeoff.schema import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VOICE,
    result_file_payload,
    result_record,
    sha256_bytes,
    validate_manifest,
    validate_result_payload,
    load_json,
    write_json,
)

PACKAGE_SWIFT = """// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "SoniqoKokoroBench",
    platforms: [.macOS("15.0"), .iOS("18.0")],
    dependencies: [
        .package(path: "__SPEECH_SWIFT_PATH__"),
    ],
    targets: [
        .executableTarget(
            name: "SoniqoKokoroBench",
            dependencies: [
                .product(name: "KokoroTTS", package: "speech-swift"),
            ]
        )
    ]
)
"""

MAIN_SWIFT = r"""import Foundation
import KokoroTTS
import CoreML
import CryptoKit

struct Result: Codable {
    let coldWallTimeS: Double
    let warmWallTimesS: [Double]
    let observedAudioDurationS: Double
    let sampleCount: Int
    let sampleRate: Int
    let audioSHA256: String
    let availableVoiceCount: Int
}

func writeWavMono16(path: String, samples: [Float], sampleRate: Int) throws {
    var peak: Float = 1e-7
    for sample in samples {
        peak = max(peak, abs(sample))
    }
    var pcm = [Int16]()
    pcm.reserveCapacity(samples.count)
    for sample in samples {
        let scaled = max(-1.0, min(1.0, sample / peak))
        pcm.append(Int16((scaled * 32767.0).rounded()))
    }

    var data = Data()
    data.append(contentsOf: "RIFF".utf8)
    var riffChunkSize = UInt32(36 + pcm.count * 2).littleEndian
    withUnsafeBytes(of: &riffChunkSize) { data.append(contentsOf: $0) }
    data.append(contentsOf: "WAVE".utf8)
    data.append(contentsOf: "fmt ".utf8)
    var subchunk1Size = UInt32(16).littleEndian
    withUnsafeBytes(of: &subchunk1Size) { data.append(contentsOf: $0) }
    var audioFormat = UInt16(1).littleEndian
    withUnsafeBytes(of: &audioFormat) { data.append(contentsOf: $0) }
    var numChannels = UInt16(1).littleEndian
    withUnsafeBytes(of: &numChannels) { data.append(contentsOf: $0) }
    var sr = UInt32(sampleRate).littleEndian
    withUnsafeBytes(of: &sr) { data.append(contentsOf: $0) }
    var byteRate = UInt32(sampleRate * 2).littleEndian
    withUnsafeBytes(of: &byteRate) { data.append(contentsOf: $0) }
    var blockAlign = UInt16(2).littleEndian
    withUnsafeBytes(of: &blockAlign) { data.append(contentsOf: $0) }
    var bitsPerSample = UInt16(16).littleEndian
    withUnsafeBytes(of: &bitsPerSample) { data.append(contentsOf: $0) }
    data.append(contentsOf: "data".utf8)
    var dataSize = UInt32(pcm.count * 2).littleEndian
    withUnsafeBytes(of: &dataSize) { data.append(contentsOf: $0) }
    pcm.withUnsafeBytes { data.append(contentsOf: $0) }
    try data.write(to: URL(fileURLWithPath: path))
}

func sha256Hex(_ floats: [Float]) -> String {
    var bytes = floats
    let digest = bytes.withUnsafeMutableBytes { raw -> SHA256.Digest in
        SHA256.hash(data: Data(raw))
    }
    return digest.map { String(format: "%02x", $0) }.joined()
}

let args = CommandLine.arguments
guard args.count >= 6 else {
    fputs("usage: SoniqoKokoroBench TEXT VOICE ITERATIONS COMPUTE_UNITS WAV_PATH\n", stderr)
    exit(2)
}

let text = args[1]
let voice = args[2]
let iterations = Int(args[3]) ?? 5
let computeUnits: MLComputeUnits
switch args[4].lowercased() {
case "all": computeUnits = .all
case "cpuandgpu": computeUnits = .cpuAndGPU
case "cpuonly": computeUnits = .cpuOnly
case "cpuandneuralengine": computeUnits = .cpuAndNeuralEngine
default: computeUnits = .all
}
let wavPath = args[5]

let model = try await KokoroTTSModel.fromPretrained(computeUnits: computeUnits)

let coldStart = CFAbsoluteTimeGetCurrent()
_ = try model.synthesize(text: text, voice: voice, language: "en", speed: 1.0)
let cold = CFAbsoluteTimeGetCurrent() - coldStart

var warm: [Double] = []
var lastAudio: [Float] = []
for _ in 0..<iterations {
    let start = CFAbsoluteTimeGetCurrent()
    lastAudio = try model.synthesize(text: text, voice: voice, language: "en", speed: 1.0)
    warm.append(CFAbsoluteTimeGetCurrent() - start)
}
try FileManager.default.createDirectory(
    at: URL(fileURLWithPath: wavPath).deletingLastPathComponent(),
    withIntermediateDirectories: true
)
try writeWavMono16(path: wavPath, samples: lastAudio, sampleRate: KokoroTTSModel.outputSampleRate)

let result = Result(
    coldWallTimeS: cold,
    warmWallTimesS: warm,
    observedAudioDurationS: Double(lastAudio.count) / Double(KokoroTTSModel.outputSampleRate),
    sampleCount: lastAudio.count,
    sampleRate: KokoroTTSModel.outputSampleRate,
    audioSHA256: sha256Hex(lastAudio),
    availableVoiceCount: model.availableVoices.count
)
let data = try JSONEncoder().encode(result)
print(String(data: data, encoding: .utf8)!)
"""


def _ensure_cli(work_dir: Path, speech_swift: Path) -> Path:
    src = work_dir / "Sources" / "SoniqoKokoroBench"
    src.mkdir(parents=True, exist_ok=True)
    (work_dir / "Package.swift").write_text(
        PACKAGE_SWIFT.replace("__SPEECH_SWIFT_PATH__", str(speech_swift))
    )
    (src / "main.swift").write_text(MAIN_SWIFT)
    return work_dir


def _run_cli(
    package_dir: Path,
    text: str,
    voice: str,
    iterations: int,
    compute_units: str,
    wav_path: Path,
) -> dict:
    cmd = [
        "swift", "run", "-c", "release", "SoniqoKokoroBench",
        text, voice, str(iterations), compute_units, str(wav_path),
    ]
    proc = subprocess.run(cmd, cwd=package_dir, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "swift SoniqoKokoroBench failed\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "runtime_input_manifest.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--speech-swift", type=Path, required=True)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--compute-units", default="all")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--input-key", action="append", default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--spotcheck-dir", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    validate_manifest(manifest)
    keys = args.input_key or list(manifest["inputs"].keys())
    spotcheck_dir = args.spotcheck_dir or (
        DEFAULT_OUTPUT_DIR / "spotcheck_wavs" / f"soniqo_speech_swift_kokoro_{args.machine_id}"
    )
    spotcheck_dir = spotcheck_dir.resolve()

    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="soniqo-kokoro-bench-"))
    _ensure_cli(work_dir, args.speech_swift.resolve())

    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.speech_swift, text=True
    ).strip()

    records = []
    for key in keys:
        item = manifest["inputs"][key]
        spotcheck_wav = spotcheck_dir / f"{key}.wav"
        raw = _run_cli(
            work_dir,
            item["text"],
            args.voice,
            args.iterations,
            args.compute_units,
            spotcheck_wav,
        )
        records.append(
            result_record(
                impl="soniqo-speech-swift-kokoro",
                framework="Swift + Core ML",
                hardware_target="ANE/Core ML",
                version=sha,
                machine_id=args.machine_id,
                input_key=key,
                text=item["text"],
                voice=args.voice,
                cold_wall_time_s=float(raw["coldWallTimeS"]),
                warm_wall_times_s=[float(x) for x in raw["warmWallTimesS"]],
                canonical_audio_duration_s=float(item["canonical_duration_s"]),
                observed_audio_duration_s=float(raw["observedAudioDurationS"]),
                output_sha256=str(raw["audioSHA256"]),
                provenance={
                    "speech_swift_path": str(args.speech_swift),
                    "compute_units": args.compute_units,
                    "sample_rate": raw["sampleRate"],
                    "sample_count": raw["sampleCount"],
                    "available_voice_count": raw["availableVoiceCount"],
                    "spotcheck_wav": str(spotcheck_wav),
                },
            )
        )

    payload = result_file_payload(
        impl="soniqo-speech-swift-kokoro",
        machine_id=args.machine_id,
        records=records,
        provenance={
            "speech_swift_sha": sha,
            "compute_units": args.compute_units,
            "spotcheck_dir": str(spotcheck_dir),
        },
    )
    validate_result_payload(payload)
    output = args.output or (DEFAULT_OUTPUT_DIR / f"results_soniqo_speech_swift_kokoro_{args.machine_id}.json")
    write_json(output, payload)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
