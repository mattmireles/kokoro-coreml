import XCTest
@testable import KokoroPipeline

final class DurationChoiceTests: XCTestCase {
    func testExactDurationPackagesAreIgnoredByDefault() throws {
        let dir = try makeDurationPackageDirectory()

        let choices = KokoroPipeline.discoverDurationChoices(
            modelsDirectory: dir,
            useExactDurationModels: false
        )

        XCTAssertEqual(choices.map(\.cacheKey), ["padded_t32", "padded_t64"])
        XCTAssertEqual(
            try KokoroPipeline.selectDurationChoice(choices, actualTokens: 44).cacheKey,
            "padded_t64"
        )
    }

    func testExactDurationPackagesAreOptIn() throws {
        let dir = try makeDurationPackageDirectory()

        let choices = KokoroPipeline.discoverDurationChoices(
            modelsDirectory: dir,
            useExactDurationModels: true
        )

        XCTAssertEqual(choices.map(\.cacheKey), ["padded_t32", "exact_t44", "padded_t64"])
        let exact = try KokoroPipeline.selectDurationChoice(choices, actualTokens: 44)
        XCTAssertEqual(exact.cacheKey, "exact_t44")
        XCTAssertFalse(exact.allowsPadding)
        XCTAssertFalse(exact.requiresAttentionMask)
    }

    func testExactDurationPackagesAreDiscoveredThroughModelsDirectorySymlink() throws {
        let dir = try makeDurationPackageDirectory()
        let link = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createSymbolicLink(
            at: link,
            withDestinationURL: dir
        )

        let choices = KokoroPipeline.discoverDurationChoices(
            modelsDirectory: link,
            useExactDurationModels: true
        )

        XCTAssertEqual(choices.map(\.cacheKey), ["padded_t32", "exact_t44", "padded_t64"])
        XCTAssertEqual(
            try KokoroPipeline.selectDurationChoice(choices, actualTokens: 44).cacheKey,
            "exact_t44"
        )
    }

    func testDurationChoicesCanBeCappedForProductionWorkers() throws {
        let dir = try makeDurationPackageDirectory()
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("kokoro_duration_t512.mlpackage", isDirectory: true),
            withIntermediateDirectories: true
        )

        let choices = KokoroPipeline.discoverDurationChoices(
            modelsDirectory: dir,
            maxDurationTokenLength: 256
        )

        XCTAssertFalse(choices.contains { $0.cacheKey == "padded_t512" })
    }

    private func makeDurationPackageDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("kokoro_duration_t32.mlpackage", isDirectory: true),
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("kokoro_duration_exact_t44.mlpackage", isDirectory: true),
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("kokoro_duration_t64.mlpackage", isDirectory: true),
            withIntermediateDirectories: true
        )
        return dir
    }
}
