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
    let compiledModelsDirectory: URL

    /// Single padded duration shape used by the starter runtime.
    private let durationChoices: [DurationModelChoice]

    /// Loaded Core ML model cache.
    private var models: [String: MLModel] = [:]

    /// Manifest model-package paths already verified in this process.
    private var validatedModelPackages: Set<String> = []

    /// Compute-unit policy selected for this facade.
    private let computePolicy: KokoroComputePolicy

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
        guard manifest.hfProvenanceVerified else {
            throw KokoroError.badHash(path: "hf_provenance_verified")
        }
        let manifestModelPaths = Set(manifest.modelPackages.map(\.path))
        self.modelsDirectory = root.appendingPathComponent("coreml", isDirectory: true)
        let compiledModelsDirectory = resources.explicitCompiledModelsDirectoryURL()
            ?? Self.defaultCompiledModelsDirectory(for: manifest)
        try Self.validateCompiledModelsDirectory(bundleRootURL: root, compiledModelsDirectory: compiledModelsDirectory)
        self.compiledModelsDirectory = compiledModelsDirectory
        let durationChoices = KokoroPipeline.discoverDurationChoices(modelsDirectory: modelsDirectory)
            .filter { manifestModelPaths.contains("coreml/\($0.packageURL.lastPathComponent)") }
        try Self.validateModelSet(
            rootURL: root,
            modelsDirectory: modelsDirectory,
            manifest: manifest,
            durationChoices: durationChoices
        )
        guard let runtimeDuration = durationChoices.first(where: {
            $0.allowsPadding && $0.tokenLength == KokoroTTS.runtimeDurationTokenLength
        }) else {
            throw KokoroError.missingModel(
                "kokoro_duration_t\(KokoroTTS.runtimeDurationTokenLength).mlpackage"
            )
        }
        // The starter bundle has one 15-second acoustic bucket. Matching it
        // with one padded duration shape avoids seven independent first-load
        // and specialization paths during playback.
        self.durationChoices = [runtimeDuration]
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
    ///   - progress: Called immediately before each potentially blocking load.
    func prewarm(
        actualTokens: Int? = nil,
        bucketSeconds: [Int]? = nil,
        progress: ((String) -> Void)? = nil
    ) throws {
        let selectedBuckets = bucketSeconds ?? availableBucketSeconds()
        if let actualTokens {
            let choice = try KokoroPipeline.selectDurationChoice(durationChoices, actualTokens: actualTokens)
            _ = try model(
                cacheKey: "duration.\(choice.cacheKey)",
                url: choice.packageURL,
                units: computePolicy.duration,
                progress: progress
            )
        } else {
            for choice in durationChoices {
                _ = try model(
                    cacheKey: "duration.\(choice.cacheKey)",
                    url: choice.packageURL,
                    units: computePolicy.duration,
                    progress: progress
                )
            }
        }
        for bucket in selectedBuckets {
            guard let tFrames = PipelineConstants.tFramesForBucket[bucket] else {
                throw KokoroError.missingModel("bucket \(bucket)s")
            }
            _ = try model(
                cacheKey: "f0ntrain.\(tFrames)",
                url: modelsDirectory.appendingPathComponent("kokoro_f0ntrain_t\(tFrames).mlpackage"),
                units: computePolicy.f0ntrain,
                progress: progress
            )
            _ = try model(
                cacheKey: "decoder_pre.\(bucket)",
                url: modelsDirectory.appendingPathComponent("kokoro_decoder_pre_\(bucket)s.mlpackage"),
                units: computePolicy.decoderPre,
                progress: progress
            )
            _ = try model(
                cacheKey: "generator.\(bucket)",
                url: modelsDirectory.appendingPathComponent("kokoro_decoder_har_post_\(bucket)s.mlpackage"),
                units: computePolicy.generator,
                progress: progress
            )
        }
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
        guard payload.linear_weights.count == 9, payload.linear_weights.allSatisfy(\.isFinite), payload.linear_bias.isFinite else {
            throw KokoroError.badHash(path: manifest.runtimeAssets.hnsfWeights.path)
        }
        return (payload.linear_weights, payload.linear_bias)
    }

    /// Loads the verified vocab declared by the runtime manifest.
    func vocab() throws -> [String: Int32] {
        let url = try Self.containedURL(rootURL: rootURL, relativePath: manifest.runtimeAssets.vocab.path)
        return try KokoroTextProcessor.loadVocab(from: url)
    }

    /// Loads or returns a cached Core ML model.
    private func model(
        cacheKey: String,
        url: URL,
        units: MLComputeUnits,
        progress: ((String) -> Void)? = nil
    ) throws -> MLModel {
        if let cached = models[cacheKey] {
            return cached
        }
        progress?("resolve.\(cacheKey)")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw KokoroError.missingModel(url.lastPathComponent)
        }
        let package = try validateModelPackageIfNeeded(url)
        let config = MLModelConfiguration()
        config.computeUnits = units
        let compiledName = url.deletingPathExtension().lastPathComponent + ".mlmodelc"
        let precompiled = compiledModelsDirectory.appendingPathComponent(compiledName, isDirectory: true)
        let bundled = Bundle.main.url(
            forResource: url.deletingPathExtension().lastPathComponent,
            withExtension: "mlmodelc"
        )
        var compiled: URL
        if let bundled {
            progress?("bundled.\(cacheKey)")
            compiled = bundled
        } else {
            progress?("compile.\(cacheKey)")
            compiled = try compiledModelURL(
                sourceURL: url,
                destinationURL: precompiled,
                sourceTreeSHA256: package.treeSHA256
            )
        }
        progress?("instantiate.\(cacheKey)")
        let loaded: MLModel
        do {
            loaded = try MLModel(contentsOf: compiled, configuration: config)
        } catch {
            if compiled.standardizedFileURL.path == precompiled.standardizedFileURL.path {
                try? Self.removeCompiledCache(cacheRootURL: compiledModelsDirectory, destinationURL: precompiled)
                compiled = try compiledModelURL(
                    sourceURL: url,
                    destinationURL: precompiled,
                    sourceTreeSHA256: package.treeSHA256
                )
                do {
                    loaded = try MLModel(contentsOf: compiled, configuration: config)
                } catch {
                    throw KokoroError.coreMLLoadFailed(url.lastPathComponent)
                }
            } else {
                throw KokoroError.coreMLLoadFailed(url.lastPathComponent)
            }
        }
        models[cacheKey] = loaded
        return loaded
    }

    /// Returns a reusable compiled model URL, compiling and caching if needed.
    private func compiledModelURL(sourceURL: URL, destinationURL: URL, sourceTreeSHA256: String) throws -> URL {
        if try Self.compiledSidecarMatches(
            cacheRootURL: compiledModelsDirectory,
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
                rootURL: compiledModelsDirectory,
                targetURL: destinationURL.deletingLastPathComponent()
            )
            try FileManager.default.createDirectory(
                at: destinationURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try Self.rejectExistingSymlinkComponents(
                rootURL: compiledModelsDirectory,
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
        cacheRootURL: URL,
        destinationURL: URL,
        sourceTreeSHA256: String
    ) throws -> Bool {
        guard FileManager.default.fileExists(atPath: destinationURL.path) else {
            return false
        }
        try rejectExistingSymlinkComponents(rootURL: cacheRootURL, targetURL: destinationURL)
        let values = try destinationURL.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard values.isSymbolicLink != true, values.isDirectory == true else {
            return false
        }
        let sidecarURL = compiledSourceHashURL(for: destinationURL)
        guard FileManager.default.fileExists(atPath: sidecarURL.path) else {
            return false
        }
        try rejectExistingSymlinkComponents(rootURL: cacheRootURL, targetURL: sidecarURL)
        let sidecarValues = try sidecarURL.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard sidecarValues.isSymbolicLink != true,
              sidecarValues.isRegularFile == true,
              let sidecar = try? String(contentsOf: sidecarURL, encoding: .utf8)
        else {
            return false
        }
        return sidecar.trimmingCharacters(in: .whitespacesAndNewlines) == sourceTreeSHA256
    }

    /// Removes a corrupt compiled cache entry and its source-hash sidecar.
    private static func removeCompiledCache(cacheRootURL: URL, destinationURL: URL) throws {
        try rejectExistingSymlinkComponents(rootURL: cacheRootURL, targetURL: destinationURL)
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }
        let sidecar = compiledSourceHashURL(for: destinationURL)
        try rejectExistingSymlinkComponents(rootURL: cacheRootURL, targetURL: sidecar)
        if FileManager.default.fileExists(atPath: sidecar.path) {
            try FileManager.default.removeItem(at: sidecar)
        }
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

    /// Verifies the compiled cache root can be used without following symlinks.
    private static func validateCompiledModelsDirectory(bundleRootURL rootURL: URL, compiledModelsDirectory: URL) throws {
        let root = rootURL.standardizedFileURL.path
        let target = compiledModelsDirectory.standardizedFileURL.path
        if target == root || target.hasPrefix("\(root)/") {
            try rejectExistingSymlinkComponents(rootURL: rootURL, targetURL: compiledModelsDirectory)
        } else {
            try rejectRootSymlink(rootURL: compiledModelsDirectory)
        }
    }

    /// Returns a stable writable cache directory for compiled Core ML models.
    ///
    /// The cache key deliberately avoids the runtime bundle path because iOS app
    /// update installs can change bundle URLs while keeping the exact same SDK
    /// resource manifest. Manifest identity gives app-bundled resources stable
    /// cache reuse while still invalidating when the HF revision, SDK commit,
    /// profile, runtime assets, model packages, or voice embeddings change.
    private static func defaultCompiledModelsDirectory(for manifest: KokoroRuntimeManifest) -> URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let digest = compiledCacheDigest(for: manifest).prefix(16)
        return caches
            .appendingPathComponent("KokoroTTS", isDirectory: true)
            .appendingPathComponent("compiled-\(digest)", isDirectory: true)
    }

    /// Builds the manifest identity digest used by the default compiled cache.
    private static func compiledCacheDigest(for manifest: KokoroRuntimeManifest) -> String {
        var lines: [String] = [
            "schema=\(manifest.schemaVersion)",
            "sdk=\(manifest.sdkCommit)",
            "repo=\(manifest.hfRepoID)",
            "revision=\(manifest.hfRevision)",
            "profile=\(manifest.bundleProfile)",
            "download=\(manifest.hfDownloadManifestSHA256)",
            "vocab=\(manifest.runtimeAssets.vocab.sha256)",
            "hnsf=\(manifest.runtimeAssets.hnsfWeights.sha256)",
        ]
        lines.append(contentsOf: manifest.modelPackages
            .sorted { $0.path < $1.path }
            .map { "\($0.path)=\($0.treeSHA256)" })
        lines.append(contentsOf: manifest.voices
            .sorted { $0.path < $1.path }
            .map { "\($0.path)=\($0.sha256)" })
        let data = Data(lines.joined(separator: "\n").utf8)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
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
        let sdkDurationTokenSizes = [KokoroTTS.runtimeDurationTokenLength]
        guard manifest.durationTokenSizes == sdkDurationTokenSizes else {
            throw KokoroError.missingModel("duration_token_sizes")
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
            if Self.isSymbolicLink(current) {
                throw KokoroError.pathEscape(current.path)
            }
            guard FileManager.default.fileExists(atPath: current.path) else {
                continue
            }
        }
    }

    /// Rejects a generated bundle root that is itself a symlink.
    private static func rejectRootSymlink(rootURL: URL) throws {
        if Self.isSymbolicLink(rootURL) {
            throw KokoroError.pathEscape(rootURL.path)
        }
        guard FileManager.default.fileExists(atPath: rootURL.path) else {
            return
        }
    }

    /// Returns true when a URL is itself a symbolic link.
    private static func isSymbolicLink(_ url: URL) -> Bool {
        (try? FileManager.default.destinationOfSymbolicLink(atPath: url.path)) != nil
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
