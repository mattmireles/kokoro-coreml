import Foundation
import XCTest
@testable import KokoroSay

/// Regression tests for kokoro-say's argument parsing.
final class SayOptionsTests: XCTestCase {
    private func reason(_ arguments: [String]) -> String? {
        do {
            _ = try SayOptions(arguments: arguments)
            return nil
        } catch SayError.invalid(let reason) {
            return reason
        } catch {
            return "\(error)"
        }
    }

    /// Regression: a mistyped bundle path was spoken as text.
    func testPathWithoutManifestIsAUsageErrorNotText() {
        XCTAssertEqual(reason(["/tmp/no-such-kokoro-bundle"]),
                       "/tmp/no-such-kokoro-bundle is not a bundle root: it has no KokoroRuntimeManifest.json")
        XCTAssertNotNil(reason([FileManager.default.temporaryDirectory.path]), "an existing directory is a path too")
    }

    /// Text, even with a slash, stays text when it has spaces.
    func testTextWithSpacesStaysText() throws {
        XCTAssertEqual(try SayOptions(arguments: ["Hello and/or goodbye."]).text, "Hello and/or goodbye.")
    }

    /// Regression: --bundle with --manifest-url silently used the bundle.
    func testBundleWithManifestURLIsAUsageError() {
        XCTAssertEqual(reason(["--bundle", "/tmp/x", "--manifest-url", "https://example.com/m.json", "--manifest-sha256", "ab"]),
                       "use either a bundle root (--bundle) or --manifest-url, not both")
    }

    /// Regression: --manifest-sha256 without --manifest-url was silently ignored.
    func testManifestSHAWithoutURLIsAUsageError() {
        XCTAssertThrowsError(try SayOptions(arguments: ["--manifest-sha256", "ab", "Hi."]))
    }
}
