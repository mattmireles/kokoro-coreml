/// Multifunction Core ML packages for the bucketed stages.
///
/// A multifunction `mlprogram` (coremltools 8+, macOS 15) holds one fixed-shape
/// graph per bucket behind a function name and stores the shared weights once,
/// so the duration, f0ntrain and decoder-pre bucket sets shrink from 217, 98 and
/// 193 MB to 55, 20 and 64 MB on disk. Loading is by `functionName`; the weights
/// are deduplicated on disk only (each loaded function is resident on its own),
/// and a GPU function loads 0.2-0.4 s slower than a separate package. The
/// loaders prefer a multifunction package when its file is present and fall
/// back to the separate per-bucket packages otherwise, so an older model set
/// keeps working unchanged. Built by `scripts/build_multifunction_packages.py`.

import CoreML
import Foundation

public extension PipelineConstants {
    static let durationMultifunctionPackage = "kokoro_duration_multifunction.mlpackage"
    static let f0ntrainMultifunctionPackage = "kokoro_f0ntrain_multifunction.mlpackage"
    static let decoderPreMultifunctionPackage = "kokoro_decoder_pre_multifunction.mlpackage"

    /// Function names inside the multifunction packages; the build script uses the same forms.
    static func durationFunctionName(tokenLength: Int) -> String { "t\(tokenLength)" }
    static func f0ntrainFunctionName(tFrames: Int) -> String { "t\(tFrames)" }
    static func decoderPreFunctionName(bucketSec: Int) -> String { "bucket_\(bucketSec)s" }
}

/// A compiled multifunction package and the functions it exposes.
public struct MultifunctionPackage {
    public let packageURL: URL
    public let compiledURL: URL
    public let functionNames: Set<String>

    /// Compiles the package at `url` and lists its functions; nil when no file is there.
    /// Compilation is synchronous, like the rest of model loading.
    public static func open(at url: URL) throws -> MultifunctionPackage? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        // Function selection needs the macOS 15 / iOS 18 runtime; an older OS
        // reports no functions and the loaders fall back to separate packages.
        guard #available(macOS 15.0, iOS 18.0, *) else { return nil }
        let compiled = try MLModel.compileModel(at: url)
        let asset = try MLModelAsset(url: compiled)
        let semaphore = DispatchSemaphore(value: 0)
        var names: [String] = []
        var failure: Error?
        asset.functionNames { result, error in
            if let result { names = result }
            failure = error
            semaphore.signal()
        }
        semaphore.wait()
        if let failure { throw failure }
        return MultifunctionPackage(packageURL: url, compiledURL: compiled, functionNames: Set(names))
    }

    public func load(function name: String, computeUnits: MLComputeUnits) throws -> MLModel {
        guard functionNames.contains(name) else {
            throw PipelineError.modelNotLoaded("\(packageURL.lastPathComponent):\(name)")
        }
        guard #available(macOS 15.0, iOS 18.0, *) else {
            throw PipelineError.modelNotLoaded("\(packageURL.lastPathComponent):\(name) needs macOS 15 / iOS 18")
        }
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        config.functionName = name
        return try MLModel.withGPUFallback(config) { try MLModel(contentsOf: compiledURL, configuration: $0) }
    }
}
