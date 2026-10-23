import Foundation
@testable import KokoroPipeline
import XCTest
@testable import KokoroTTS

/// Regression tests for the schema-2 release bundle (release contract): split
/// duration/F0Ntrain pairs, fold_poly, and the `coreml/hybrid` tree, with no
/// whole duration/F0Ntrain, har_post or range package.
final class SchemaTwoManifestTests: XCTestCase {
    // The hybrid's failure memory is process-wide: start every test with it clear.
    override func setUp() {
        if #available(macOS 15.0, iOS 18.0, *) { HybridGenerator.resetFailureForTesting() }
    }

    override func tearDown() {
        if #available(macOS 15.0, iOS 18.0, *) { HybridGenerator.resetFailureForTesting() }
    }

    private func contractRoot(
        trees: [String: [(String, Data)]] = KokoroBundleFixture.contractStarterTrees(),
        schemaVersion: Int = KokoroSDKModelProvider.supportedSchemaVersion
    ) throws -> URL {
        try KokoroBundleFixture.makeBundleRoot(schemaVersion: schemaVersion, packageTrees: trees)
    }

    /// Regression: init demanded kokoro_duration_t128, kokoro_f0ntrain_t600
    /// and a har_post package, none of which a schema-2 starter ships.
    func testContractStarterInitSucceeds() throws {
        let provider = try KokoroSDKModelProvider(resources: .directory(try contractRoot()))

        XCTAssertEqual(provider.availableBucketSeconds(), [15])
        XCTAssertEqual(provider.durationModelChoices().map(\.tokenLength), [KokoroTTS.runtimeDurationTokenLength])
        XCTAssertEqual(KokoroSDKModelProvider.foldedPackageName(provider.manifest, bucket: 15),
                       "kokoro_generator_fold_poly_15s.mlpackage")
    }

    /// The `coreml/hybrid` directory entry validates with the package digest.
    func testHybridDirectoryEntryValidates() throws {
        let root = try contractRoot()
        let provider = try KokoroSDKModelProvider(resources: .directory(root))

        let entry = try provider.validateModelPackageIfNeeded(root.appendingPathComponent("coreml/hybrid", isDirectory: true))

        XCTAssertEqual(entry.path, "coreml/hybrid")
        XCTAssertEqual(entry.fileCount, 3)
    }

    /// Regression: plan.json and adain.bin were read with no digest check. A
    /// tampered file now fails before HybridGenerator reads it.
    func testTamperedHybridFilesAreRejected() throws {
        guard #available(macOS 15.0, iOS 18.0, *) else { throw XCTSkip("hybrid needs macOS 15") }
        for file in ["adain.bin", "plan.json"] {
            let root = try contractRoot()
            try Data("tampered".utf8).write(to: root.appendingPathComponent("coreml/hybrid/\(file)"))
            let provider = try KokoroSDKModelProvider(resources: .directory(root))

            XCTAssertThrowsError(try provider.hybridGenerator(bucketSec: 15, xPreFrames: 800), file) { error in
                XCTAssertEqual(error as? KokoroError, .badHash(path: "coreml/hybrid"))
            }
        }
    }

    /// The folded generator is the hybrid's fallback, so `coreml/hybrid`
    /// without a fold package for every bucket is rejected at load.
    func testHybridWithoutFoldIsRejected() throws {
        var trees = KokoroBundleFixture.contractStarterTrees()
        trees["coreml/kokoro_generator_fold_poly_15s.mlpackage"] = nil

        XCTAssertThrowsError(try KokoroSDKModelProvider(resources: .directory(try contractRoot(trees: trees)))) { error in
            XCTAssertEqual(error as? KokoroError, .missingModel("kokoro_generator_fold_poly_15s.mlpackage"))
        }
    }

    /// Schema 1 bundles (whole models, GPU generators) are no longer read.
    func testSchemaOneIsRejected() throws {
        XCTAssertThrowsError(try KokoroSDKModelProvider(resources: .directory(try contractRoot(schemaVersion: 1)))) { error in
            XCTAssertEqual(error as? KokoroError, .unsupportedManifestSchema(1))
        }
    }

    /// The cpuOnly policy turns the hybrid (ANE + Metal) off without touching it.
    func testCpuOnlyPolicyTurnsHybridOff() throws {
        guard #available(macOS 15.0, iOS 18.0, *) else { throw XCTSkip("hybrid needs macOS 15") }
        XCTAssertTrue(HybridGenerator.allowed, "otherwise the nil below would not be the policy's doing")
        let root = try contractRoot()
        try Data("tampered".utf8).write(to: root.appendingPathComponent("coreml/hybrid/plan.json"))
        let provider = try KokoroSDKModelProvider(resources: .directory(root), computePolicy: .cpuOnly)

        XCTAssertNil(try provider.hybridGenerator(bucketSec: 15, xPreFrames: 800))
    }

    /// Regression: a corrupt cached hybrid compile was never evicted, so every
    /// launch failed the hybrid and fell back to folded. Needs the real
    /// starter bundle (scripts/build_sdk_bundle.mjs) at /tmp/kokoro-sdk-contract-starter.
    func testCorruptHybridCompileCacheIsEvictedAndRebuilt() throws {
        guard #available(macOS 15.0, iOS 18.0, *) else { throw XCTSkip("hybrid needs macOS 15") }
        let root = URL(fileURLWithPath: "/tmp/kokoro-sdk-contract-starter", isDirectory: true)
        guard FileManager.default.fileExists(atPath: root.appendingPathComponent("coreml/hybrid/plan.json").path) else {
            throw XCTSkip("no real starter bundle at \(root.path)")
        }
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("kokoro-hybrid-cache-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: cache) }
        let provider = try KokoroSDKModelProvider(resources: .directory(root, compiledModelsDirectory: cache))
        let tree = try provider.validateModelPackageIfNeeded(root.appendingPathComponent("coreml/hybrid", isDirectory: true))
        // A cache entry whose sidecar matches but whose bundle does not load.
        let compiled = cache.appendingPathComponent("hybrid_generator_r6.mlmodelc", isDirectory: true)
        try FileManager.default.createDirectory(at: compiled, withIntermediateDirectories: true)
        try Data("corrupt".utf8).write(to: compiled.appendingPathComponent("coremldata.bin"))
        try Data("\(tree.treeSHA256)\n".utf8).write(to: cache.appendingPathComponent("hybrid_generator_r6.mlmodelc.kokoro-source-tree-sha256"))

        XCTAssertNotNil(try provider.hybridGenerator(bucketSec: 15, xPreFrames: 1200))
        XCTAssertNotEqual(try Data(contentsOf: compiled.appendingPathComponent("coremldata.bin")), Data("corrupt".utf8))
    }

    /// Regression: with an environment switch off, a schema-2 bundle failed
    /// with a bare missing package name that did not say why it was wanted.
    func testSwitchedOffFastPathNamesTheSwitch() {
        setenv(SplitDuration.envKey, "0", 1)
        defer { unsetenv(SplitDuration.envKey) }

        XCTAssertEqual(KokoroSDKModelProvider.missingPackageDetail("kokoro_duration_t128.mlpackage"),
                       "KOKORO_SPLIT_DURATION=0 needs a bundle with kokoro_duration_t128.mlpackage")
        XCTAssertEqual(KokoroSDKModelProvider.missingPackageDetail("kokoro_f0ntrain_t600.mlpackage"), "kokoro_f0ntrain_t600.mlpackage")
    }

    /// Regression: the placeholder pin built a URL with "TODO_REV" and failed
    /// later with a 404 or hash mismatch. Now it fails first, and clearly.
    func testUnpinnedHostedRevisionFailsBeforeAnyNetworkCall() throws {
        XCTAssertThrowsError(try KokoroHostedModels.requirePinnedRevision("TODO_REV")) { error in
            XCTAssertEqual(error as? KokoroError, .unpinnedHostedRevision("TODO_REV"))
            XCTAssertTrue(error.localizedDescription.contains("no pinned Hugging Face revision"))
        }
        XCTAssertThrowsError(try KokoroHostedModels.requirePinnedRevision(String(repeating: "g", count: 40)))
        XCTAssertNoThrow(try KokoroHostedModels.requirePinnedRevision(String(repeating: "a1", count: 20)))
    }

    /// Regression: a relative URL with a leading `..` standardized to a path
    /// with the `..` dropped, so containment compared the wrong directory.
    func testPathGuardResolvesLeadingDotDotAgainstWorkingDirectory() {
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        let relative = URL(fileURLWithPath: "../kokoro-up", relativeTo: cwd)

        XCTAssertEqual(KokoroPathGuard.lexicalPath(relative), cwd.deletingLastPathComponent().appendingPathComponent("kokoro-up").path)
        XCTAssertFalse(KokoroPathGuard.isInside(relative, root: cwd))
    }

    /// Regression: a cache root under /private/tmp threw
    /// pathEscape(".../compiled"): `standardizedFileURL` dropped `/private`
    /// from the existing root but not from the not-yet-created child.
    func testPathGuardAcceptsNewChildOfPrivatePrefixedRoot() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        // /var and /tmp are symlinks into /private on macOS and iOS.
        guard tmp.path.hasPrefix("/var/") || tmp.path.hasPrefix("/tmp/") else { throw XCTSkip("temp dir \(tmp.path)") }
        let root = URL(fileURLWithPath: "/private" + tmp.path, isDirectory: true)
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.path))
        let child = root.appendingPathComponent("compiled", isDirectory: true)

        XCTAssertNoThrow(try KokoroPathGuard.rejectExistingSymlinkComponents(rootURL: root, targetURL: child))
        XCTAssertNoThrow(try KokoroPathGuard.containedURL(rootURL: root, relativePath: "compiled/x.mlmodelc"))
        XCTAssertThrowsError(try KokoroPathGuard.rejectExistingSymlinkComponents(
            rootURL: root, targetURL: root.appendingPathComponent("../escape")))
    }
}
