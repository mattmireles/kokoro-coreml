/// The SDK picks the folded ANE generator from the manifest when one is
/// listed for a bucket, preferring the variant the device ANE runs
/// (README/Notes/ane-folded-generator.md).
import KokoroPipeline
import XCTest
@testable import KokoroTTS

final class FoldedGeneratorSelectionTests: XCTestCase {
    func testNoFoldedPackageMeansNil() {
        let paths: Set<String> = ["coreml/kokoro_decoder_har_post_15s.mlpackage"]
        XCTAssertNil(KokoroSDKModelProvider.foldedPackageName(paths: paths, bucket: 15))
    }

    func testPicksTheOnlyListedVariant() {
        let paths: Set<String> = ["coreml/kokoro_generator_fold_15s.mlpackage"]
        XCTAssertEqual(KokoroSDKModelProvider.foldedPackageName(paths: paths, bucket: 15), "kokoro_generator_fold_15s.mlpackage")
        XCTAssertNil(KokoroSDKModelProvider.foldedPackageName(paths: paths, bucket: 30))
    }

    /// The split duration model is used only when both of its packages are listed.
    func testSplitDurationNeedsFrontAndBack() {
        let both: Set<String> = ["coreml/kokoro_duration_ane_front_t512.mlpackage", "coreml/kokoro_duration_back_t512.mlpackage"]
        XCTAssertEqual(KokoroSDKModelProvider.splitDurationPackages(paths: both, tokenLength: 512)?.front,
                       "kokoro_duration_ane_front_t512.mlpackage")
        XCTAssertNil(KokoroSDKModelProvider.splitDurationPackages(paths: both, tokenLength: 256))
        XCTAssertNil(KokoroSDKModelProvider.splitDurationPackages(paths: ["coreml/kokoro_duration_ane_front_t512.mlpackage"], tokenLength: 512))
    }

    func testSplitF0NNeedsFrontAndBack() {
        let both: Set<String> = ["coreml/kokoro_f0ntrain_front_t1200.mlpackage", "coreml/kokoro_f0ntrain_ane_back_t1200.mlpackage"]
        XCTAssertEqual(KokoroSDKModelProvider.splitF0NPackages(paths: both, tFrames: 1200)?.back, "kokoro_f0ntrain_ane_back_t1200.mlpackage")
        XCTAssertNil(KokoroSDKModelProvider.splitF0NPackages(paths: both, tFrames: 600))
        XCTAssertNil(KokoroSDKModelProvider.splitF0NPackages(paths: ["coreml/kokoro_f0ntrain_front_t1200.mlpackage"], tFrames: 1200))
    }

    func testMacPrefersPolynomialVariant() {
        let paths: Set<String> = ["coreml/kokoro_generator_fold_15s.mlpackage", "coreml/kokoro_generator_fold_poly_15s.mlpackage"]
        #if os(macOS)
        XCTAssertEqual(KokoroSDKModelProvider.foldedPackageName(paths: paths, bucket: 15), "kokoro_generator_fold_poly_15s.mlpackage")
        #endif
    }
}
