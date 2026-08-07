import CryptoKit
import Foundation
@testable import KokoroTTS

/// Builds minimal generated-bundle trees for provider and validation tests.
///
/// The `.mlpackage` directories here are not real Core ML packages: they carry
/// the manifest shape, digests, and file layout the SDK validates, which is all
/// any test that stops short of Core ML compilation needs.
enum KokoroBundleFixture {
    /// Creates a minimal generated-bundle shape for provider validation tests.
    ///
    /// - Parameters:
    ///   - removeVoiceFile: Deletes the voice payload after writing the manifest.
    ///   - schemaVersion: Manifest schema version to declare.
    ///   - voiceHashOverride: Replaces the voice digest with a known-bad value.
    ///   - voicePath: Manifest-relative voice path, including escape attempts.
    ///   - modelPackages: Replacement model-package manifest entries.
    ///   - durationTokenSizes: Declared duration token sizes.
    ///   - hnsfPayload: Replacement hn-NSF weights JSON.
    /// - Returns: Bundle root directory.
    static func makeBundleRoot(
        removeVoiceFile: Bool = false,
        schemaVersion: Int = 1,
        voiceHashOverride: String? = nil,
        voicePath: String = "voices/af_heart.bin",
        modelPackages: [[String: Any]]? = nil,
        durationTokenSizes: [Int] = [128],
        hnsfPayload: String? = nil
    ) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let runtime = root.appendingPathComponent("runtime", isDirectory: true)
        let voices = root.appendingPathComponent("voices", isDirectory: true)
        try FileManager.default.createDirectory(at: runtime, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: voices, withIntermediateDirectories: true)

        let vocabURL = try KokoroRuntimeAssets.url(for: .vocab)
        let hnsfURL = try KokoroRuntimeAssets.url(for: .hnsfWeights)
        let bundledVocab = runtime.appendingPathComponent("kokoro-vocab.json")
        let bundledHnsf = runtime.appendingPathComponent("hnsf_weights.json")
        try FileManager.default.copyItem(at: vocabURL, to: bundledVocab)
        if let hnsfPayload {
            try Data(hnsfPayload.utf8).write(to: bundledHnsf)
        } else {
            try FileManager.default.copyItem(at: hnsfURL, to: bundledHnsf)
        }

        let voiceURL = voices.appendingPathComponent("af_heart.bin")
        let voiceData = Data(count: 256 * 4)
        try voiceData.write(to: voiceURL)
        if removeVoiceFile {
            try FileManager.default.removeItem(at: voiceURL)
        }
        try writeRequiredPackages(root: root)

        let manifest: [String: Any] = [
            "schema_version": schemaVersion,
            "sdk_commit": "test",
            "hf_repo_id": "test/repo",
            "hf_revision": "testrev",
            "hf_provenance_verified": true,
            "hf_download_manifest_sha256": String(repeating: "a", count: 64),
            "minimum_platforms": ["iOS": "18.0", "macOS": "15.0"],
            "supported_languages": ["en-US"],
            "bundle_profile": "starter",
            "buckets": [15],
            "duration_token_sizes": durationTokenSizes,
            "model_packages": modelPackages ?? requiredPackageEntries(),
            "voices": [[
                "path": voicePath,
                "bytes": voiceData.count,
                "sha256": voiceHashOverride ?? sha256(voiceData),
            ]],
            "runtime_assets": [
                "vocab": digest(path: "runtime/kokoro-vocab.json", url: bundledVocab),
                "hnsf_weights": digest(path: "runtime/hnsf_weights.json", url: bundledHnsf),
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: root.appendingPathComponent("KokoroRuntimeManifest.json"))
        return root
    }

    /// Creates a manifest digest object for a file.
    ///
    /// - Parameters:
    ///   - path: Manifest-relative path.
    ///   - url: File to digest.
    /// - Returns: Manifest digest dictionary.
    static func digest(path: String, url: URL) -> [String: Any] {
        let data = try! Data(contentsOf: url)
        return [
            "path": path,
            "bytes": data.count,
            "sha256": sha256(data),
        ]
    }

    /// Writes a minimal one-file model package fixture.
    ///
    /// - Parameters:
    ///   - root: Bundle root.
    ///   - path: Manifest-relative package path.
    ///   - data: Payload bytes for the single package file.
    static func writeOneFilePackage(root: URL, path: String, data: Data) throws {
        let package = root.appendingPathComponent(path, isDirectory: true)
        if FileManager.default.fileExists(atPath: package.path) {
            try FileManager.default.removeItem(at: package)
        }
        let payload = root.appendingPathComponent(path, isDirectory: true)
            .appendingPathComponent("Data/com.apple.CoreML", isDirectory: true)
        try FileManager.default.createDirectory(at: payload, withIntermediateDirectories: true)
        try data.write(to: payload.appendingPathComponent("model.mlmodel"))
    }

    /// Manifest-relative path of the single file inside a fixture package.
    static let packagePayloadRelativePath = "Data/com.apple.CoreML/model.mlmodel"

    /// Creates the matching manifest entry for ``writeOneFilePackage(root:path:data:)``.
    ///
    /// - Parameters:
    ///   - path: Manifest-relative package path.
    ///   - data: Payload bytes for the single package file.
    /// - Returns: Model-package manifest dictionary.
    static func modelPackageEntry(path: String, data: Data) -> [String: Any] {
        let rel = packagePayloadRelativePath
        let fileHash = sha256(data)
        var digest = SHA256()
        digest.update(data: Data(rel.utf8))
        digest.update(data: Data([0]))
        digest.update(data: Data(String(data.count).utf8))
        digest.update(data: Data([0]))
        digest.update(data: Data(fileHash.utf8))
        digest.update(data: Data([0]))
        return [
            "path": path,
            "tree_sha256": digest.finalize().map { String(format: "%02x", $0) }.joined(),
            "file_count": 1,
            "bytes": data.count,
            "files": [[
                "path": rel,
                "bytes": data.count,
                "sha256": fileHash,
            ]],
        ]
    }

    /// Creates the minimal model package set required for a 15s starter bundle.
    ///
    /// - Returns: Model-package manifest entries.
    static func requiredPackageEntries() -> [[String: Any]] {
        [
            modelPackageEntry(path: "coreml/kokoro_duration_t128.mlpackage", data: Data("duration-128".utf8)),
            modelPackageEntry(path: "coreml/kokoro_f0ntrain_t600.mlpackage", data: Data("f0-600".utf8)),
            modelPackageEntry(path: "coreml/kokoro_decoder_pre_15s.mlpackage", data: Data("decoder-pre-15".utf8)),
            modelPackageEntry(path: "coreml/kokoro_decoder_har_post_15s.mlpackage", data: Data("har-post-15".utf8)),
        ]
    }

    /// Creates default model package directories for the generated-bundle fixture.
    ///
    /// - Parameter root: Bundle root.
    static func writeRequiredPackages(root: URL) throws {
        try writeOneFilePackage(root: root, path: "coreml/kokoro_duration_t128.mlpackage", data: Data("duration-128".utf8))
        try writeOneFilePackage(root: root, path: "coreml/kokoro_f0ntrain_t600.mlpackage", data: Data("f0-600".utf8))
        try writeOneFilePackage(root: root, path: "coreml/kokoro_decoder_pre_15s.mlpackage", data: Data("decoder-pre-15".utf8))
        try writeOneFilePackage(root: root, path: "coreml/kokoro_decoder_har_post_15s.mlpackage", data: Data("har-post-15".utf8))
    }

    /// Computes a SHA-256 digest string.
    ///
    /// - Parameter data: Bytes to digest.
    /// - Returns: Lowercase hex digest.
    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
