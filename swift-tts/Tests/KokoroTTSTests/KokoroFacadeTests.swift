import CryptoKit
import XCTest
@testable import KokoroTTS

final class KokoroFacadeTests: XCTestCase {
    /// Verifies the SDK model provider validates runtime assets and voice hashes.
    func testModelProviderLoadsMinimalManifestAndHnsfWeights() throws {
        let root = try makeBundleRoot()

        let provider = try KokoroSDKModelProvider(resources: .directory(root))
        let hnsf = try provider.hnsfWeights()

        XCTAssertEqual(provider.manifest.bundleProfile, "starter")
        XCTAssertEqual(provider.availableBucketSeconds(), [15])
        XCTAssertEqual(hnsf.linearWeights.count, 9)
    }

    /// Verifies missing voice files surface as public SDK errors.
    func testModelProviderRejectsMissingVoiceFile() throws {
        let root = try makeBundleRoot(removeVoiceFile: true)

        XCTAssertThrowsError(try KokoroSDKModelProvider(resources: .directory(root))) { error in
            XCTAssertEqual(error as? KokoroError, .missingVoice("af_heart"))
        }
    }

    /// Verifies unsupported manifest schema versions fail before load.
    func testModelProviderRejectsUnsupportedManifestSchema() throws {
        let root = try makeBundleRoot(schemaVersion: 99)

        XCTAssertThrowsError(try KokoroSDKModelProvider(resources: .directory(root))) { error in
            XCTAssertEqual(error as? KokoroError, .unsupportedManifestSchema(99))
        }
    }

    /// Verifies bad file hashes are rejected before model loading.
    func testModelProviderRejectsBadVoiceHash() throws {
        let root = try makeBundleRoot(voiceHashOverride: String(repeating: "0", count: 64))

        XCTAssertThrowsError(try KokoroSDKModelProvider(resources: .directory(root))) { error in
            XCTAssertEqual(error as? KokoroError, .badHash(path: "voices/af_heart.bin"))
        }
    }

    /// Verifies runtime manifest digest paths cannot escape the generated bundle.
    func testModelProviderRejectsEscapedDigestPath() throws {
        let root = try makeBundleRoot(voicePath: "voices/../runtime/hnsf_weights.json")

        XCTAssertThrowsError(try KokoroSDKModelProvider(resources: .directory(root))) { error in
            XCTAssertEqual(error as? KokoroError, .pathEscape("voices/../runtime/hnsf_weights.json"))
        }
    }

    /// Verifies runtime manifest digest paths use the same slash-only grammar as hosted manifests.
    func testModelProviderRejectsBackslashDigestPath() throws {
        let root = try makeBundleRoot(voicePath: "voices\\af_heart.bin")

        XCTAssertThrowsError(try KokoroSDKModelProvider(resources: .directory(root))) { error in
            XCTAssertEqual(error as? KokoroError, .pathEscape("voices\\af_heart.bin"))
        }
    }

    /// Verifies downloaded resources carry an explicit compiled-model cache.
    func testDownloadedResourceProviderUsesCompiledCacheDirectory() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let compiled = root.appendingPathComponent("compiled-cache", isDirectory: true)
        let provider = KokoroResourceProvider.downloadedDirectory(root: root, compiledModelsDirectory: compiled)

        XCTAssertEqual(try provider.rootURL(), root)
        XCTAssertEqual(try provider.compiledModelsDirectoryURL(), compiled)
    }

    /// Verifies facade load defers Core ML compilation and Misaki/MLX setup.
    func testFacadeLoadDefersModelCompilationAndMisakiSetup() async throws {
        let root = try makeBundleRoot()

        _ = try await loadFacadeFromMainActor(resources: .directory(root))
    }

    /// Verifies hosted manifest paths cannot escape the downloaded cache.
    func testDownloadedStoreRejectsPathEscapes() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)

        XCTAssertThrowsError(try KokoroDownloadedModelStore.containedURL(rootURL: root, relativePath: "../evil")) { error in
            XCTAssertEqual(error as? KokoroError, .pathEscape("../evil"))
        }
        XCTAssertThrowsError(try KokoroDownloadedModelStore.containedURL(rootURL: root, relativePath: "voices\\evil.bin")) { error in
            XCTAssertEqual(error as? KokoroError, .pathEscape("voices\\evil.bin"))
        }
    }

    /// Verifies hosted manifest paths are sanitized before building remote URLs.
    func testDownloadedStoreRejectsRemotePathEscapes() throws {
        let manifestURL = URL(string: "https://models.example.test/coreml/v1/HostedManifest.json")!

        XCTAssertThrowsError(try KokoroDownloadedModelStore.remoteURL(
            manifestURL: manifestURL,
            relativePath: "coreml/../evil.mlpackage"
        )) { error in
            XCTAssertEqual(error as? KokoroError, .pathEscape("coreml/../evil.mlpackage"))
        }
        XCTAssertEqual(
            try KokoroDownloadedModelStore.remoteURL(
                manifestURL: manifestURL,
                relativePath: "voices/af_heart.bin"
            ).absoluteString,
            "https://models.example.test/coreml/v1/voices/af_heart.bin"
        )
    }

    /// Verifies package bundle resources resolve through the provided bundle.
    func testPackageBundleProviderResolvesBundleResourceRoot() throws {
        let bundle = Bundle(for: KokoroFacadeTests.self)
        let provider = KokoroResourceProvider.packageBundle(bundle)

        XCTAssertEqual(try provider.rootURL(), bundle.resourceURL)
    }

    /// Verifies model package corruption is rejected before Core ML compiles it.
    func testModelProviderRejectsBadModelPackageTreeHash() throws {
        let packagePath = "coreml/kokoro_duration_t32.mlpackage"
        var entries = requiredPackageEntries()
        entries[0] = [
            "path": packagePath,
            "tree_sha256": String(repeating: "0", count: 64),
            "file_count": 1,
            "bytes": 5,
            "files": [[
                "path": "Data/com.apple.CoreML/model.mlmodel",
                "bytes": 5,
                "sha256": sha256(Data("hello".utf8)),
            ]],
        ]
        let root = try makeBundleRoot(modelPackages: entries)
        let package = root.appendingPathComponent(packagePath, isDirectory: true)
        let payload = package.appendingPathComponent("Data/com.apple.CoreML", isDirectory: true)
        try FileManager.default.removeItem(at: package)
        try FileManager.default.createDirectory(at: payload, withIntermediateDirectories: true)
        try Data("hello".utf8).write(to: payload.appendingPathComponent("model.mlmodel"))

        let provider = try KokoroSDKModelProvider(resources: .directory(root))
        let choice = try XCTUnwrap(provider.durationModelChoices().first)

        XCTAssertThrowsError(try provider.durationModel(choice: choice)) { error in
            XCTAssertEqual(error as? KokoroError, .badHash(path: packagePath))
        }
    }

    /// Verifies stale duration packages on disk are ignored unless the manifest lists them.
    func testModelProviderFiltersDurationChoicesToManifestPackages() throws {
        let root = try makeBundleRoot()
        try writeOneFilePackage(
            root: root,
            path: "coreml/kokoro_duration_exact_t44.mlpackage",
            data: Data("stale-exact-duration".utf8)
        )

        let provider = try KokoroSDKModelProvider(resources: .directory(root))

        XCTAssertEqual(provider.durationModelChoices().map(\.tokenLength), [32, 64, 128, 256, 320, 384, 512])
    }

    /// Verifies bundles fail fast when a manifest bucket has no matching stage model.
    func testModelProviderRejectsMissingBucketStagePackage() throws {
        let packages = requiredPackageEntries().filter {
            ($0["path"] as? String) != "coreml/kokoro_f0ntrain_t600.mlpackage"
        }
        let root = try makeBundleRoot(modelPackages: packages)

        XCTAssertThrowsError(try KokoroSDKModelProvider(resources: .directory(root))) { error in
            XCTAssertEqual(error as? KokoroError, .missingModel("kokoro_f0ntrain_t600.mlpackage"))
        }
    }

    /// Verifies every manifest duration size must be backed by a package.
    func testModelProviderRejectsMissingManifestDurationPackage() throws {
        let root = try makeBundleRoot(modelPackages: [
            modelPackageEntry(path: "coreml/kokoro_duration_t32.mlpackage", data: Data("duration-32".utf8)),
            modelPackageEntry(path: "coreml/kokoro_f0ntrain_t600.mlpackage", data: Data("f0-600".utf8)),
            modelPackageEntry(path: "coreml/kokoro_decoder_pre_15s.mlpackage", data: Data("decoder-pre-15".utf8)),
            modelPackageEntry(path: "coreml/kokoro_decoder_har_post_15s.mlpackage", data: Data("har-post-15".utf8)),
        ])

        XCTAssertThrowsError(try KokoroSDKModelProvider(resources: .directory(root))) { error in
            XCTAssertEqual(error as? KokoroError, .missingModel("kokoro_duration_t64.mlpackage"))
        }
    }


    /// Verifies runtime asset parent symlinks are rejected even if target bytes match.
    func testModelProviderRejectsSymlinkedRuntimeParent() throws {
        let root = try makeBundleRoot()
        let runtime = root.appendingPathComponent("runtime", isDirectory: true)
        let outside = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try FileManager.default.removeItem(at: runtime)
        try FileManager.default.createSymbolicLink(at: runtime, withDestinationURL: outside)

        XCTAssertThrowsError(try KokoroSDKModelProvider(resources: .directory(root))) { error in
            guard case .pathEscape = error as? KokoroError else {
                XCTFail("expected pathEscape, got \(error)")
                return
            }
        }
    }

    /// Verifies a generated bundle root cannot be supplied through a symlink.
    func testModelProviderRejectsSymlinkedBundleRoot() throws {
        let root = try makeBundleRoot()
        let link = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: root)

        XCTAssertThrowsError(try KokoroSDKModelProvider(resources: .directory(link))) { error in
            guard case .pathEscape = error as? KokoroError else {
                XCTFail("expected pathEscape, got \(error)")
                return
            }
        }
    }

    /// Verifies cached compiled models cannot be supplied through symlinks.
    func testModelProviderRejectsSymlinkedCompiledCache() throws {
        let root = try makeBundleRoot()
        let compiled = root.appendingPathComponent("compiled", isDirectory: true)
        let outside = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let linkedModel = compiled.appendingPathComponent("kokoro_duration_t32.mlmodelc", isDirectory: true)
        try FileManager.default.createDirectory(at: compiled, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: linkedModel, withDestinationURL: outside)
        try Data("\(modelPackageEntry(path: "coreml/kokoro_duration_t32.mlpackage", data: Data("duration-32".utf8))["tree_sha256"]!)\n".utf8)
            .write(to: compiled.appendingPathComponent("kokoro_duration_t32.mlmodelc.kokoro-source-tree-sha256"))
        let provider = try KokoroSDKModelProvider(resources: .directory(root))
        let choice = try XCTUnwrap(provider.durationModelChoices().first)

        XCTAssertThrowsError(try provider.durationModel(choice: choice)) { error in
            guard case .pathEscape = error as? KokoroError else {
                XCTFail("expected pathEscape, got \(error)")
                return
            }
        }
    }

    /// Verifies downloaded cache validation rejects symlinked parent directories.
    func testDownloadedStoreRejectsSymlinkedCacheParent() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let outside = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let voices = root.appendingPathComponent("voices", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: voices, withDestinationURL: outside)

        XCTAssertThrowsError(try KokoroDownloadedModelStore.rejectExistingSymlinkComponents(
            rootURL: root,
            targetURL: voices.appendingPathComponent("af_heart.bin")
        )) { error in
            guard case .pathEscape = error as? KokoroError else {
                XCTFail("expected pathEscape, got \(error)")
                return
            }
        }
    }

    /// Verifies the downloaded cache root itself cannot be a symlink.
    func testDownloadedStoreRejectsSymlinkedCacheRoot() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let link = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: root)

        XCTAssertThrowsError(try KokoroDownloadedModelStore.rejectRootSymlink(rootURL: link)) { error in
            guard case .pathEscape = error as? KokoroError else {
                XCTFail("expected pathEscape, got \(error)")
                return
            }
        }
    }

    /// Verifies public starter constants match the starter bundle profile.
    func testStarterVoiceConstantsMatchStarterBundle() {
        XCTAssertEqual(KokoroVoiceID.starterVoices, [.afHeart])
        XCTAssertEqual(VoiceTable.defaultVoiceID, .afHeart)
        XCTAssertEqual(KokoroVoiceID("af_bella").rawValue, "af_bella")
        XCTAssertEqual(KokoroVoiceID("am_michael").rawValue, "am_michael")
    }

    /// Verifies V1 raw-text synthesis rejects non-English Kokoro voices even if
    /// a custom bundle accidentally includes their embedding files.
    func testPrepareRejectsNonEnglishVoicePrefixes() async throws {
        let root = try makeBundleRoot()
        let tts = try await loadFacadeFromMainActor(resources: .directory(root))

        XCTAssertFalse(KokoroVoiceID("jf_alpha").isSupportedRawTextLanguage)
        do {
            _ = try await tts.prepare("Hello world.", voice: KokoroVoiceID("jf_alpha"))
            XCTFail("expected unsupportedVoice")
        } catch {
            XCTAssertEqual(error as? KokoroError, .unsupportedVoice("jf_alpha"))
        }
    }

    /// Creates a minimal generated-bundle shape for provider validation tests.
    private func makeBundleRoot(
        removeVoiceFile: Bool = false,
        schemaVersion: Int = 1,
        voiceHashOverride: String? = nil,
        voicePath: String = "voices/af_heart.bin",
        modelPackages: [[String: Any]]? = nil
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
        try FileManager.default.copyItem(at: hnsfURL, to: bundledHnsf)

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
            "minimum_platforms": ["iOS": "18.0", "macOS": "15.0"],
            "supported_languages": ["en-US"],
            "bundle_profile": "starter",
            "buckets": [15],
            "duration_token_sizes": [32, 64, 128, 256, 320, 384, 512],
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
    private func digest(path: String, url: URL) -> [String: Any] {
        let data = try! Data(contentsOf: url)
        return [
            "path": path,
            "bytes": data.count,
            "sha256": sha256(data),
        ]
    }

    /// Writes a minimal one-file model package fixture.
    private func writeOneFilePackage(root: URL, path: String, data: Data) throws {
        let package = root.appendingPathComponent(path, isDirectory: true)
        if FileManager.default.fileExists(atPath: package.path) {
            try FileManager.default.removeItem(at: package)
        }
        let payload = root.appendingPathComponent(path, isDirectory: true)
            .appendingPathComponent("Data/com.apple.CoreML", isDirectory: true)
        try FileManager.default.createDirectory(at: payload, withIntermediateDirectories: true)
        try data.write(to: payload.appendingPathComponent("model.mlmodel"))
    }

    /// Creates the matching manifest entry for `writeOneFilePackage`.
    private func modelPackageEntry(path: String, data: Data) -> [String: Any] {
        let rel = "Data/com.apple.CoreML/model.mlmodel"
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

    /// Creates and registers the minimal model package set required for a 15s starter bundle.
    private func requiredPackageEntries() -> [[String: Any]] {
        [
            modelPackageEntry(path: "coreml/kokoro_duration_t32.mlpackage", data: Data("duration-32".utf8)),
            modelPackageEntry(path: "coreml/kokoro_duration_t64.mlpackage", data: Data("duration-64".utf8)),
            modelPackageEntry(path: "coreml/kokoro_duration_t128.mlpackage", data: Data("duration-128".utf8)),
            modelPackageEntry(path: "coreml/kokoro_duration_t256.mlpackage", data: Data("duration-256".utf8)),
            modelPackageEntry(path: "coreml/kokoro_duration_t320.mlpackage", data: Data("duration-320".utf8)),
            modelPackageEntry(path: "coreml/kokoro_duration_t384.mlpackage", data: Data("duration-384".utf8)),
            modelPackageEntry(path: "coreml/kokoro_duration_t512.mlpackage", data: Data("duration-512".utf8)),
            modelPackageEntry(path: "coreml/kokoro_f0ntrain_t600.mlpackage", data: Data("f0-600".utf8)),
            modelPackageEntry(path: "coreml/kokoro_decoder_pre_15s.mlpackage", data: Data("decoder-pre-15".utf8)),
            modelPackageEntry(path: "coreml/kokoro_decoder_har_post_15s.mlpackage", data: Data("har-post-15".utf8)),
        ]
    }

    /// Creates default model package directories for the generated-bundle fixture.
    private func writeRequiredPackages(root: URL) throws {
        try writeOneFilePackage(root: root, path: "coreml/kokoro_duration_t32.mlpackage", data: Data("duration-32".utf8))
        try writeOneFilePackage(root: root, path: "coreml/kokoro_duration_t64.mlpackage", data: Data("duration-64".utf8))
        try writeOneFilePackage(root: root, path: "coreml/kokoro_duration_t128.mlpackage", data: Data("duration-128".utf8))
        try writeOneFilePackage(root: root, path: "coreml/kokoro_duration_t256.mlpackage", data: Data("duration-256".utf8))
        try writeOneFilePackage(root: root, path: "coreml/kokoro_duration_t320.mlpackage", data: Data("duration-320".utf8))
        try writeOneFilePackage(root: root, path: "coreml/kokoro_duration_t384.mlpackage", data: Data("duration-384".utf8))
        try writeOneFilePackage(root: root, path: "coreml/kokoro_duration_t512.mlpackage", data: Data("duration-512".utf8))
        try writeOneFilePackage(root: root, path: "coreml/kokoro_f0ntrain_t600.mlpackage", data: Data("f0-600".utf8))
        try writeOneFilePackage(root: root, path: "coreml/kokoro_decoder_pre_15s.mlpackage", data: Data("decoder-pre-15".utf8))
        try writeOneFilePackage(root: root, path: "coreml/kokoro_decoder_har_post_15s.mlpackage", data: Data("har-post-15".utf8))
    }

    /// Computes a SHA-256 digest string.
    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Calls facade load while isolated to `MainActor`.
    ///
    /// The fake `.mlpackage` directories in `makeBundleRoot()` are not valid
    /// Core ML packages. If `KokoroTTS.load` compiles models, this helper throws
    /// before returning. The SwiftPM test environment also lacks Misaki's MLX
    /// runtime bundle, so eager Misaki setup would fail here. Both costs belong
    /// to `prewarm`, `prepare`, or synthesis.
    @MainActor
    private func loadFacadeFromMainActor(resources: KokoroResourceProvider) async throws -> KokoroTTS {
        try await KokoroTTS.load(resources: resources)
    }
}
