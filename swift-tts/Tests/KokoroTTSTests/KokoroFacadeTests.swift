import CryptoKit
import KokoroPipeline
import XCTest
@testable import KokoroTTS

private struct CharacterCountPhonemizer: KokoroPhonemizer {
    /// Phoneme scalar emitted once per normalized input character.
    let phoneme: Character

    /// Returns a known Kokoro vocab phoneme repeated to match input character count.
    ///
    /// - Parameter text: Raw chunk text.
    /// - Returns: Deterministic phonemes sized to trigger token-budget fallback.
    func phonemize(_ text: String) throws -> KokoroPhonemeResult {
        let count = KokoroTextProcessor.normalizeWhitespace(text).count
        return KokoroPhonemeResult(phonemes: String(repeating: String(phoneme), count: count))
    }
}

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
        XCTAssertEqual(provider.explicitCompiledModelsDirectoryURL(), compiled)
    }

    /// Verifies default compiled caches are keyed by manifest identity, not bundle path.
    func testDefaultCompiledCacheIsStableAcrossBundlePaths() throws {
        let first = try KokoroSDKModelProvider(resources: .directory(makeBundleRoot()))
        let second = try KokoroSDKModelProvider(resources: .directory(makeBundleRoot()))

        XCTAssertEqual(first.compiledModelsDirectory, second.compiledModelsDirectory)
        XCTAssertFalse(first.compiledModelsDirectory.path.contains("KokoroRuntimeManifest.json"))
    }

    /// Verifies app callers may still override the default compiled-model cache.
    func testDirectoryResourceProviderUsesExplicitCompiledCacheDirectory() throws {
        let root = try makeBundleRoot()
        let compiled = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)

        let provider = try KokoroSDKModelProvider(resources: .directory(root, compiledModelsDirectory: compiled))

        XCTAssertEqual(provider.compiledModelsDirectory, compiled)
    }

    /// Verifies facade load defers Core ML compilation and Misaki/MLX setup.
    func testFacadeLoadDefersModelCompilationAndMisakiSetup() async throws {
        let root = try makeBundleRoot()

        _ = try await loadFacadeFromMainActor(resources: .directory(root))
    }

    /// Verifies preparation keeps every chunk inside the one prewarmed duration shape.
    func testFacadePrepareRecursivelySplitsChunksBeyondRuntimeDurationShape() async throws {
        let root = try makeBundleRoot()
        let modelProvider = try KokoroSDKModelProvider(resources: .directory(root))
        let vocab = try modelProvider.vocab()
        let processor = KokoroTextProcessor(
            phonemizer: CharacterCountPhonemizer(phoneme: "h"),
            vocab: vocab
        )
        let tts = KokoroTTS(
            chunker: TextChunker(maxChunkSeconds: 10_000),
            americanTextProcessor: processor,
            britishTextProcessor: processor,
            voiceTable: VoiceTable(voicesDirectory: root.appendingPathComponent("voices", isDirectory: true)),
            modelProvider: modelProvider,
            hnsf: try modelProvider.hnsfWeights()
        )
        let text = String(repeating: "a", count: PipelineConstants.maxCallerChunkTokens + 100)

        let prepared = try await tts.prepare(text, voice: .afHeart)

        XCTAssertGreaterThan(prepared.count, 1)
        XCTAssertTrue(
            prepared.allSatisfy {
                ($0.numTokens ?? 0) <= KokoroTTS.runtimeDurationTokenLength
            }
        )
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

    /// Verifies downloaded manifests require a caller-pinned SHA-256 digest.
    func testDownloadedStoreRejectsManifestHashMismatch() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let manifestURL = root.appendingPathComponent("HostedManifest.json")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data(#"{"version":"test","files":[]}"#.utf8).write(to: manifestURL)

        do {
            _ = try await KokoroDownloadedModelStore(
                manifestURL: manifestURL,
                expectedManifestSHA256: String(repeating: "0", count: 64),
                cacheDirectory: root.appendingPathComponent("cache", isDirectory: true)
            ).hydrate()
            XCTFail("expected badHash")
        } catch {
            XCTAssertEqual(error as? KokoroError, .badHash(path: "HostedManifest.json"))
        }
    }

    /// Verifies insecure remote manifests require explicit local-development opt-in.
    func testDownloadedStoreRejectsInsecureRemoteManifestByDefault() async throws {
        let manifestURL = URL(string: "http://models.example.test/HostedManifest.json")!

        do {
            _ = try await KokoroDownloadedModelStore(
                manifestURL: manifestURL,
                expectedManifestSHA256: String(repeating: "0", count: 64),
                cacheDirectory: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            ).hydrate()
            XCTFail("expected insecureManifestURL")
        } catch {
            XCTAssertEqual(error as? KokoroError, .insecureManifestURL(manifestURL))
        }
    }

    /// Verifies streamed download bytes reach the public hydration progress
    /// path instead of waiting for an entire large model file to finish.
    func testDownloadedStoreReportsIncrementalFileProgress() throws {
        let recorder = LockedIntRecorder()
        let expectation = expectation(description: "streamed byte progress")
        let target = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
            .appendingPathComponent("fixture.bin")
        let url = URL(string: "https://models.example.test/fixture.bin")!
        let downloader = KokoroDownloadedModelStore.CappedFileDownloader(
            target: target,
            expectedBytes: 1_024,
            expectedSHA256: String(repeating: "0", count: 64),
            maxBytes: 1_024,
            label: "fixture.bin"
        ) { bytes in
            recorder.append(bytes)
            expectation.fulfill()
        }
        let task = URLSession.shared.dataTask(with: url)
        let response = try XCTUnwrap(HTTPURLResponse(
            url: url,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Length": "1024"]
        ))

        downloader.urlSession(URLSession.shared, dataTask: task, didReceive: response) { _ in }
        downloader.urlSession(
            URLSession.shared,
            dataTask: task,
            didReceive: Data(repeating: 0, count: 128)
        )

        wait(for: [expectation], timeout: 1)
        XCTAssertEqual(recorder.values, [128])
        task.cancel()
    }

    /// Verifies the hosted manifest itself emits streamed progress before its
    /// complete response arrives.
    func testDownloadedStoreReportsIncrementalManifestProgress() {
        let recorder = LockedIntRecorder()
        let expectation = expectation(description: "manifest byte progress")
        let downloader = KokoroDownloadedModelStore.CappedDataDownloader(
            maxBytes: 1_024,
            label: "HostedManifest.json"
        ) { bytes in
            recorder.append(bytes)
            expectation.fulfill()
        }
        let task = URLSession.shared.dataTask(
            with: URL(string: "https://models.example.test/HostedManifest.json")!
        )

        downloader.urlSession(
            URLSession.shared,
            dataTask: task,
            didReceive: Data(repeating: 0x7b, count: 64)
        )

        wait(for: [expectation], timeout: 1)
        XCTAssertEqual(recorder.values, [64])
        task.cancel()
    }

    /// Verifies cancellation that wins before URLSession task installation is
    /// not lost in the cancellation-handler setup window.
    func testDownloadedFileHonorsPreStartCancellation() async {
        let downloader = KokoroDownloadedModelStore.CappedFileDownloader(
            target: FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString),
            expectedBytes: 1_024,
            expectedSHA256: String(repeating: "0", count: 64),
            maxBytes: 1_024,
            label: "fixture.bin",
            progress: nil
        )
        let task = Task {
            try await downloader.download(
                from: URL(string: "https://models.example.test/fixture.bin")!
            )
        }
        task.cancel()

        do {
            try await task.value
            XCTFail("expected CancellationError")
        } catch is CancellationError {
            // Expected.
        } catch {
            XCTFail("expected CancellationError, got \(error)")
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
        let packagePath = "coreml/kokoro_duration_t128.mlpackage"
        var entries = requiredPackageEntries()
        let packageIndex = try XCTUnwrap(entries.firstIndex {
            ($0["path"] as? String) == packagePath
        })
        entries[packageIndex] = [
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

    /// Verifies the facade exposes one padded runtime shape and ignores stale packages.
    func testModelProviderSelectsSingleRuntimeDurationShape() throws {
        let root = try makeBundleRoot()
        try writeOneFilePackage(
            root: root,
            path: "coreml/kokoro_duration_exact_t44.mlpackage",
            data: Data("stale-exact-duration".utf8)
        )

        let provider = try KokoroSDKModelProvider(resources: .directory(root))

        XCTAssertEqual(
            provider.durationModelChoices().map(\.tokenLength),
            [KokoroTTS.runtimeDurationTokenLength]
        )
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
            XCTAssertEqual(error as? KokoroError, .missingModel("duration"))
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
        let linkedModel = compiled.appendingPathComponent("kokoro_duration_t128.mlmodelc", isDirectory: true)
        try FileManager.default.createDirectory(at: compiled, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: linkedModel, withDestinationURL: outside)
        try Data("\(modelPackageEntry(path: "coreml/kokoro_duration_t128.mlpackage", data: Data("duration-128".utf8))["tree_sha256"]!)\n".utf8)
            .write(to: compiled.appendingPathComponent("kokoro_duration_t128.mlmodelc.kokoro-source-tree-sha256"))
        let provider = try KokoroSDKModelProvider(resources: .directory(root, compiledModelsDirectory: compiled))
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

    /// Verifies the hosted-version sidecar is protected by cache symlink checks.
    func testDownloadedStoreRejectsSymlinkedHostedVersionSidecar() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let outside = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let sidecar = root.appendingPathComponent(".kokoro-hosted-version")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: sidecar,
            withDestinationURL: outside.appendingPathComponent("version")
        )

        XCTAssertThrowsError(try KokoroDownloadedModelStore.rejectExistingSymlinkComponents(
            rootURL: root,
            targetURL: sidecar
        )) { error in
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

    /// Verifies malformed hn-NSF payloads fail before release-build DSP code.
    func testModelProviderRejectsMalformedHnsfWeights() throws {
        let root = try makeBundleRoot(hnsfPayload: #"{"linear_weights":[1.0],"linear_bias":0.0}"#)

        let provider = try KokoroSDKModelProvider(resources: .directory(root))

        XCTAssertThrowsError(try provider.hnsfWeights()) { error in
            XCTAssertEqual(error as? KokoroError, .badHash(path: "runtime/hnsf_weights.json"))
        }
    }

    /// Verifies manifests must declare the SDK's one fixed duration shape.
    func testModelProviderRejectsNonSDKDurationTokenSet() throws {
        let root = try makeBundleRoot(durationTokenSizes: [32])

        XCTAssertThrowsError(try KokoroSDKModelProvider(resources: .directory(root))) { error in
            XCTAssertEqual(error as? KokoroError, .missingModel("duration_token_sizes"))
        }
    }

    /// Creates a minimal generated-bundle shape for provider validation tests.
    ///
    /// Forwards to ``KokoroBundleFixture`` so other suites can build the same
    /// tree without duplicating the manifest shape.
    private func makeBundleRoot(
        removeVoiceFile: Bool = false,
        schemaVersion: Int = 1,
        voiceHashOverride: String? = nil,
        voicePath: String = "voices/af_heart.bin",
        modelPackages: [[String: Any]]? = nil,
        durationTokenSizes: [Int] = [128],
        hnsfPayload: String? = nil
    ) throws -> URL {
        try KokoroBundleFixture.makeBundleRoot(
            removeVoiceFile: removeVoiceFile,
            schemaVersion: schemaVersion,
            voiceHashOverride: voiceHashOverride,
            voicePath: voicePath,
            modelPackages: modelPackages,
            durationTokenSizes: durationTokenSizes,
            hnsfPayload: hnsfPayload
        )
    }

    /// Creates a manifest digest object for a file.
    private func digest(path: String, url: URL) -> [String: Any] {
        KokoroBundleFixture.digest(path: path, url: url)
    }

    /// Writes a minimal one-file model package fixture.
    private func writeOneFilePackage(root: URL, path: String, data: Data) throws {
        try KokoroBundleFixture.writeOneFilePackage(root: root, path: path, data: data)
    }

    /// Creates the matching manifest entry for `writeOneFilePackage`.
    private func modelPackageEntry(path: String, data: Data) -> [String: Any] {
        KokoroBundleFixture.modelPackageEntry(path: path, data: data)
    }

    /// Creates the minimal model package set required for a 15s starter bundle.
    private func requiredPackageEntries() -> [[String: Any]] {
        KokoroBundleFixture.requiredPackageEntries()
    }

    /// Creates default model package directories for the generated-bundle fixture.
    private func writeRequiredPackages(root: URL) throws {
        try KokoroBundleFixture.writeRequiredPackages(root: root)
    }

    /// Computes a SHA-256 digest string.
    private func sha256(_ data: Data) -> String {
        KokoroBundleFixture.sha256(data)
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

private final class LockedIntRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [Int] = []

    func append(_ value: Int) {
        lock.lock()
        storage.append(value)
        lock.unlock()
    }

    var values: [Int] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}
