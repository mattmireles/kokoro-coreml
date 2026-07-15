import CoreML
import CryptoKit
import Foundation
import KokoroPipeline

/// Lazy Core ML model provider for generated Kokoro SDK bundles.
final class KokoroSDKModelProvider: KokoroModelProvider {
    /// Bundle root containing `coreml/`, `voices/`, `runtime/`, and manifest.
    let rootURL: URL

    /// Decoded runtime manifest.
    let manifest: KokoroRuntimeManifest

    /// Model directory under the bundle root.
    private let modelsDirectory: URL

    /// Directory containing or caching compiled `.mlmodelc` models.
    private let compiledModelsDirectory: URL

    /// Lazy duration choices discovered from available model packages.
    private let durationChoices: [DurationModelChoice]

    /// Loaded Core ML model cache.
    private var models: [String: MLModel] = [:]

    /// Manifest model-package paths already verified in this process.
    private var validatedModelPackages: Set<String> = []

    /// Active compute-unit policy.
    private var computePolicy: KokoroComputePolicy

    /// Sidecar suffix tying reusable `.mlmodelc` output to a source tree hash.
    private static let compiledSourceHashSuffix = ".kokoro-source-tree-sha256"

    /// Creates a model provider from a resource provider.
    ///
    /// - Parameters:
    ///   - resources: Runtime bundle location.
    ///   - computePolicy: Core ML compute-unit policy for model stages.
    init(resources: KokoroResourceProvider, computePolicy: KokoroComputePolicy = .gistDefault) throws {
        let root = try resources.rootURL()
        let manifestURL = root.appendingPathComponent("KokoroRuntimeManifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw KokoroError.missingManifest(manifestURL)
        }
        self.rootURL = root
        try Self.rejectRootSymlink(rootURL: root)
        let manifest = try JSONDecoder().decode(KokoroRuntimeManifest.self, from: Data(contentsOf: manifestURL))
        guard manifest.schemaVersion == 1 else {
            throw KokoroError.unsupportedManifestSchema(manifest.schemaVersion)
        }
        let manifestModelPaths = Set(manifest.modelPackages.map(\.path))
        self.modelsDirectory = root.appendingPathComponent("coreml", isDirectory: true)
        let compiledModelsDirectory = try resources.compiledModelsDirectoryURL()
        try Self.validateCompiledModelsDirectory(rootURL: root, compiledModelsDirectory: compiledModelsDirectory)
        self.compiledModelsDirectory = compiledModelsDirectory
        let durationChoices = KokoroPipeline.discoverDurationChoices(modelsDirectory: modelsDirectory)
            .filter { manifestModelPaths.contains("coreml/\($0.packageURL.lastPathComponent)") }
        try Self.validateModelSet(
            rootURL: root,
            modelsDirectory: modelsDirectory,
            manifest: manifest,
            durationChoices: durationChoices
        )
        self.durationChoices = durationChoices
        self.manifest = manifest
        self.computePolicy = computePolicy
        try Self.validateFileDigests(rootURL: root, manifest: manifest)
    }

    /// Returns supported duration model choices.
    func durationModelChoices() -> [DurationModelChoice] {
        durationChoices
    }

    /// Returns supported bucket seconds.
    func availableBucketSeconds() -> [Int] {
        manifest.buckets
    }

    /// Loads a duration model.
    func durationModel(choice: DurationModelChoice) throws -> MLModel {
        try model(cacheKey: "duration.\(choice.cacheKey)", url: choice.packageURL, units: computePolicy.duration)
    }

    /// Loads an F0Ntrain model.
    func f0ntrainModel(tFrames: Int) throws -> MLModel {
        try model(
            cacheKey: "f0ntrain.\(tFrames)",
            url: modelsDirectory.appendingPathComponent("kokoro_f0ntrain_t\(tFrames).mlpackage"),
            units: computePolicy.f0ntrain
        )
    }

    /// Loads a decoder-pre model.
    func decoderPreModel(bucketSec: Int) throws -> MLModel {
        try model(
            cacheKey: "decoder_pre.\(bucketSec)",
            url: modelsDirectory.appendingPathComponent("kokoro_decoder_pre_\(bucketSec)s.mlpackage"),
            units: computePolicy.decoderPre
        )
    }

    /// Loads a generator/HAR-post model.
    func generatorModel(bucketSec: Int) throws -> MLModel {
        try model(
            cacheKey: "generator.\(bucketSec)",
            url: modelsDirectory.appendingPathComponent("kokoro_decoder_har_post_\(bucketSec)s.mlpackage"),
            units: computePolicy.generator
        )
    }

    /// Loads the selected duration model and all models for selected buckets.
    ///
    /// - Parameters:
    ///   - actualTokens: Optional unpadded token count used to choose one
    ///     duration bucket. If omitted, every discovered duration model loads.
    ///   - bucketSeconds: Bucket seconds to load. Defaults to manifest buckets.
    func prewarm(actualTokens: Int? = nil, bucketSeconds: [Int]? = nil) throws {
        let selectedBuckets = bucketSeconds ?? availableBucketSeconds()
        if let actualTokens {
            let choice = try KokoroPipeline.selectDurationChoice(durationChoices, actualTokens: actualTokens)
            _ = try durationModel(choice: choice)
        } else {
            for choice in durationChoices {
                _ = try durationModel(choice: choice)
            }
        }
        for bucket in selectedBuckets {
            guard let tFrames = PipelineConstants.tFramesForBucket[bucket] else {
                throw KokoroError.missingModel("bucket \(bucket)s")
            }
            _ = try f0ntrainModel(tFrames: tFrames)
            _ = try decoderPreModel(bucketSec: bucket)
            _ = try generatorModel(bucketSec: bucket)
        }
    }

    /// Switches all future model loads to CPU-only and clears loaded models.
    func degradeToCPUOnly() {
        computePolicy = .cpuOnly
        models.removeAll()
    }

    /// Returns the bundle voice directory.
    func voicesDirectory() -> URL {
        rootURL.appendingPathComponent("voices", isDirectory: true)
    }

    /// Loads verified hn-NSF weights from the bundle.
    func hnsfWeights() throws -> (linearWeights: [Float], linearBias: Float) {
        struct Payload: Decodable {
            let linear_weights: [Float]
            let linear_bias: Float
        }
        let url = try Self.containedURL(rootURL: rootURL, relativePath: manifest.runtimeAssets.hnsfWeights.path)
        let payload = try JSONDecoder().decode(Payload.self, from: Data(contentsOf: url))
        return (payload.linear_weights, payload.linear_bias)
    }

    /// Loads the verified vocab declared by the runtime manifest.
    func vocab() throws -> [String: Int32] {
        let url = try Self.containedURL(rootURL: rootURL, relativePath: manifest.runtimeAssets.vocab.path)
        return try KokoroTextProcessor.loadVocab(from: url)
    }

    /// Loads or returns a cached Core ML model.
    private func model(cacheKey: String, url: URL, units: MLComputeUnits) throws -> MLModel {
        if let cached = models[cacheKey] {
            return cached
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw KokoroError.missingModel(url.lastPathComponent)
        }
        let package = try validateModelPackageIfNeeded(url)
        let config = MLModelConfiguration()
        config.computeUnits = units
        let compiledName = url.deletingPathExtension().lastPathComponent + ".mlmodelc"
        let precompiled = compiledModelsDirectory.appendingPathComponent(compiledName, isDirectory: true)
        let compiled = try compiledModelURL(
            sourceURL: url,
            destinationURL: precompiled,
            sourceTreeSHA256: package.treeSHA256
        )
        let loaded: MLModel
        do {
            loaded = try MLModel(contentsOf: compiled, configuration: config)
        } catch {
            throw KokoroError.coreMLLoadFailed(url.lastPathComponent)
        }
        models[cacheKey] = loaded
        return loaded
    }

    /// Returns a reusable compiled model URL, compiling and caching if needed.
    private func compiledModelURL(sourceURL: URL, destinationURL: URL, sourceTreeSHA256: String) throws -> URL {
        if try Self.compiledSidecarMatches(
            rootURL: rootURL,
            destinationURL: destinationURL,
            sourceTreeSHA256: sourceTreeSHA256
        ) {
            return destinationURL
        }
        let compiled: URL
        do {
            compiled = try MLModel.compileModel(at: sourceURL)
        } catch {
            throw KokoroError.coreMLLoadFailed(sourceURL.lastPathComponent)
        }
        do {
            try Self.rejectExistingSymlinkComponents(
                rootURL: rootURL,
                targetURL: destinationURL.deletingLastPathComponent()
            )
            try FileManager.default.createDirectory(
                at: destinationURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try Self.rejectExistingSymlinkComponents(
                rootURL: rootURL,
                targetURL: destinationURL.deletingLastPathComponent()
            )
            if FileManager.default.fileExists(atPath: destinationURL.path) {
                try FileManager.default.removeItem(at: destinationURL)
            }
            try FileManager.default.copyItem(at: compiled, to: destinationURL)
            try Data("\(sourceTreeSHA256)\n".utf8).write(
                to: Self.compiledSourceHashURL(for: destinationURL),
                options: .atomic
            )
            return destinationURL
        } catch {
            return compiled
        }
    }

    /// Returns whether a cached compiled model was built from the expected source tree.
    private static func compiledSidecarMatches(
        rootURL: URL,
        destinationURL: URL,
        sourceTreeSHA256: String
    ) throws -> Bool {
        guard FileManager.default.fileExists(atPath: destinationURL.path) else {
            return false
        }
        try rejectExistingSymlinkComponents(rootURL: rootURL, targetURL: destinationURL)
        let values = try destinationURL.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard values.isSymbolicLink != true, values.isDirectory == true else {
            return false
        }
        let sidecarURL = compiledSourceHashURL(for: destinationURL)
        guard FileManager.default.fileExists(atPath: sidecarURL.path) else {
            return false
        }
        try rejectExistingSymlinkComponents(rootURL: rootURL, targetURL: sidecarURL)
        let sidecarValues = try sidecarURL.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard sidecarValues.isSymbolicLink != true,
              sidecarValues.isRegularFile == true,
              let sidecar = try? String(contentsOf: sidecarURL, encoding: .utf8)
        else {
            return false
        }
        return sidecar.trimmingCharacters(in: .whitespacesAndNewlines) == sourceTreeSHA256
    }

    /// Sidecar file recording the source package tree hash for one `.mlmodelc`.
    private static func compiledSourceHashURL(for destinationURL: URL) -> URL {
        destinationURL.deletingLastPathComponent()
            .appendingPathComponent("\(destinationURL.lastPathComponent)\(compiledSourceHashSuffix)")
    }

    /// Validates one model package tree before Core ML compiles or loads it.
    private func validateModelPackageIfNeeded(_ packageURL: URL) throws -> KokoroRuntimeManifest.ModelPackage {
        let relativePath = "coreml/\(packageURL.lastPathComponent)"
        guard let expected = manifest.modelPackages.first(where: { $0.path == relativePath }) else {
            throw KokoroError.missingModel(packageURL.lastPathComponent)
        }
        guard !validatedModelPackages.contains(relativePath) else {
            return expected
        }
        try Self.rejectExistingSymlinkComponents(rootURL: rootURL, targetURL: packageURL)
        try Self.validatePackageTree(packageURL: packageURL, relativePath: relativePath, expected: expected)
        validatedModelPackages.insert(relativePath)
        return expected
    }

    /// Validates file digests for runtime assets and voices.
    private static func validateFileDigests(rootURL: URL, manifest: KokoroRuntimeManifest) throws {
        try validate(digest: manifest.runtimeAssets.vocab, rootURL: rootURL)
        try validate(digest: manifest.runtimeAssets.hnsfWeights, rootURL: rootURL)
        for voice in manifest.voices {
            try validate(digest: voice, rootURL: rootURL)
        }
    }

    /// Validates one file digest entry.
    private static func validate(digest: KokoroRuntimeManifest.FileDigest, rootURL: URL) throws {
        let url = try containedURL(rootURL: rootURL, relativePath: digest.path)
        let path = url.standardizedFileURL.path
        try rejectExistingSymlinkComponents(rootURL: rootURL, targetURL: url)
        guard FileManager.default.fileExists(atPath: path) else {
            if digest.path.hasPrefix("voices/") {
                throw KokoroError.missingVoice(url.deletingPathExtension().lastPathComponent)
            }
            throw KokoroError.missingRuntimeAsset(digest.path)
        }
        let values = try url.resourceValues(forKeys: [.isSymbolicLinkKey, .isRegularFileKey])
        guard values.isSymbolicLink != true, values.isRegularFile == true else {
            throw KokoroError.pathEscape(digest.path)
        }
        let data = try Data(contentsOf: url)
        guard data.count == digest.bytes else {
            throw KokoroError.badHash(path: digest.path)
        }
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard hash == digest.sha256 else {
            throw KokoroError.badHash(path: digest.path)
        }
    }

    /// Resolves a manifest path under a bundle root without accepting lexical escapes.
    private static func containedURL(rootURL: URL, relativePath: String) throws -> URL {
        let components = relativePath.split(separator: "/", omittingEmptySubsequences: false)
        guard !relativePath.hasPrefix("/"),
              !relativePath.contains("\\"),
              !components.isEmpty,
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw KokoroError.pathEscape(relativePath)
        }
        let url = components.reduce(rootURL) { partial, component in
            partial.appendingPathComponent(String(component))
        }
        let root = rootURL.standardizedFileURL.path
        let target = url.standardizedFileURL.path
        guard target == root || target.hasPrefix("\(root)/") else {
            throw KokoroError.pathEscape(relativePath)
        }
        return url
    }

    /// Verifies the compiled cache root is under the bundle root when reused.
    private static func validateCompiledModelsDirectory(rootURL: URL, compiledModelsDirectory: URL) throws {
        let root = rootURL.standardizedFileURL.path
        let target = compiledModelsDirectory.standardizedFileURL.path
        guard target == root || target.hasPrefix("\(root)/") else {
            throw KokoroError.pathEscape(target)
        }
        try rejectExistingSymlinkComponents(rootURL: rootURL, targetURL: compiledModelsDirectory)
    }

    /// Verifies the manifest describes a model set that can satisfy synthesis.
    private static func validateModelSet(
        rootURL: URL,
        modelsDirectory: URL,
        manifest: KokoroRuntimeManifest,
        durationChoices: [DurationModelChoice]
    ) throws {
        guard !durationChoices.isEmpty else {
            throw KokoroError.missingModel("duration")
        }
        let paths = Set(manifest.modelPackages.map(\.path))
        for tokenLength in manifest.durationTokenSizes {
            let name = "kokoro_duration_t\(tokenLength).mlpackage"
            let path = "coreml/\(name)"
            guard paths.contains(path),
                  durationChoices.contains(where: { $0.tokenLength == tokenLength && $0.packageURL.lastPathComponent == name }) else {
                throw KokoroError.missingModel(name)
            }
        }
        for package in manifest.modelPackages {
            let url = try containedURL(rootURL: rootURL, relativePath: package.path)
            try rejectExistingSymlinkComponents(rootURL: rootURL, targetURL: url)
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard FileManager.default.fileExists(atPath: url.path),
                  values?.isDirectory == true,
                  values?.isSymbolicLink != true else {
                throw KokoroError.missingModel(url.lastPathComponent)
            }
        }
        for bucket in manifest.buckets {
            guard let tFrames = PipelineConstants.tFramesForBucket[bucket] else {
                throw KokoroError.missingModel("bucket \(bucket)s")
            }
            for name in [
                "kokoro_f0ntrain_t\(tFrames).mlpackage",
                "kokoro_decoder_pre_\(bucket)s.mlpackage",
                "kokoro_decoder_har_post_\(bucket)s.mlpackage",
            ] {
                let path = "coreml/\(name)"
                guard paths.contains(path) else {
                    throw KokoroError.missingModel(name)
                }
                let url = modelsDirectory.appendingPathComponent(name, isDirectory: true)
                guard FileManager.default.fileExists(atPath: url.path) else {
                    throw KokoroError.missingModel(name)
                }
            }
        }
    }

    /// Rejects symlinked bundle components before reads or writes can follow them.
    private static func rejectExistingSymlinkComponents(rootURL: URL, targetURL: URL) throws {
        let root = rootURL.standardizedFileURL.path
        let target = targetURL.standardizedFileURL.path
        guard target == root || target.hasPrefix("\(root)/") else {
            throw KokoroError.pathEscape(target)
        }
        try rejectRootSymlink(rootURL: rootURL)
        let suffix = String(target.dropFirst(root.count)).split(separator: "/")
        var current = rootURL
        for component in suffix {
            current = current.appendingPathComponent(String(component))
            guard FileManager.default.fileExists(atPath: current.path) else {
                continue
            }
            let values = try current.resourceValues(forKeys: [.isSymbolicLinkKey])
            if values.isSymbolicLink == true {
                throw KokoroError.pathEscape(current.path)
            }
        }
    }

    /// Rejects a generated bundle root that is itself a symlink.
    private static func rejectRootSymlink(rootURL: URL) throws {
        guard FileManager.default.fileExists(atPath: rootURL.path) else {
            return
        }
        let values = try rootURL.resourceValues(forKeys: [.isSymbolicLinkKey])
        if values.isSymbolicLink == true {
            throw KokoroError.pathEscape(rootURL.path)
        }
    }

    /// Validates a model package against the manifest tree digest.
    private static func validatePackageTree(
        packageURL: URL,
        relativePath: String,
        expected: KokoroRuntimeManifest.ModelPackage
    ) throws {
        // Parent symlink checks happen before this method. This traversal still
        // rejects symlinks inside the package tree itself.
        let files = try packageFiles(packageURL: packageURL)
        guard files.count == expected.fileCount else {
            throw KokoroError.badHash(path: relativePath)
        }
        var totalBytes = 0
        var digest = SHA256()
        for file in files {
            let rel = file.relativePath
            let data = try Data(contentsOf: file.url)
            totalBytes += data.count
            let fileHash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            digest.update(data: Data(rel.utf8))
            digest.update(data: Data([0]))
            digest.update(data: Data(String(data.count).utf8))
            digest.update(data: Data([0]))
            digest.update(data: Data(fileHash.utf8))
            digest.update(data: Data([0]))
        }
        let treeHash = digest.finalize().map { String(format: "%02x", $0) }.joined()
        guard totalBytes == expected.bytes, treeHash == expected.treeSHA256 else {
            throw KokoroError.badHash(path: relativePath)
        }
    }

    /// Lists regular package files in the same stable order as the bundle script.
    private static func packageFiles(packageURL: URL) throws -> [(relativePath: String, url: URL)] {
        let rootPath = packageURL.standardizedFileURL.path
        var files: [(relativePath: String, url: URL)] = []
        let keys: [URLResourceKey] = [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey]
        guard let enumerator = FileManager.default.enumerator(
            at: packageURL,
            includingPropertiesForKeys: keys,
            options: []
        ) else {
            throw KokoroError.missingModel(packageURL.lastPathComponent)
        }
        for case let fileURL as URL in enumerator {
            let values = try fileURL.resourceValues(forKeys: Set(keys))
            if values.isSymbolicLink == true {
                throw KokoroError.pathEscape(fileURL.path)
            }
            guard values.isRegularFile == true else {
                continue
            }
            let path = fileURL.standardizedFileURL.path
            guard path.hasPrefix("\(rootPath)/") else {
                throw KokoroError.pathEscape(path)
            }
            files.append((String(path.dropFirst(rootPath.count + 1)), fileURL))
        }
        return files.sorted { $0.relativePath < $1.relativePath }
    }
}
