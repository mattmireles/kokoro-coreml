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

    /// Compute-unit policy selected for this facade.
    private let computePolicy: KokoroComputePolicy

    /// Whether to use the flexible (RangeDim) generator. Then that one
    /// program serves every bucket at the utterance's real length instead of
    /// the bucket's; otherwise the per-bucket generators serve (older
    /// bundles, and OS 27 when the bundle ships any bucket generator: macOS
    /// 27's MPSGraph can abort on the flexible program, and iOS 27 is gated
    /// too as a precaution).
    private let usesFlexibleGenerator: Bool

    /// Whether the bundle's `coreml/hybrid` generator may run: listed in the
    /// manifest and the policy is not ``KokoroComputePolicy/cpuOnly``.
    private let usesHybridGenerator: Bool

    /// The only manifest schema this SDK reads. Schema 2 bundles ship the
    /// split duration/F0Ntrain pairs, ANE decoder-pre, folded generators and
    /// the `coreml/hybrid` tree; older SDKs reject them with this error too.
    static let supportedSchemaVersion = 2

    /// Manifest entry of the hybrid generator tree (plan.json, adain.bin,
    /// hybrid_generator_r{N}.mlpackage), digested like a package.
    static let hybridDirectoryPath = "coreml/hybrid"

    /// Sidecar suffix tying reusable `.mlmodelc` output to a source tree hash.
    private static let compiledSourceHashSuffix = ".kokoro-source-tree-sha256"

    /// Creates a model provider from a resource provider.
    ///
    /// - Parameters:
    ///   - resources: Runtime bundle location.
    ///   - computePolicy: Core ML compute-unit policy for model stages.
    ///   - flexibleGeneratorAllowed: Whether this OS may run the flexible
    ///     generator; injected by tests.
    init(
        resources: KokoroResourceProvider,
        computePolicy: KokoroComputePolicy = .gistDefault,
        flexibleGeneratorAllowed: Bool = PipelineConstants.flexibleGeneratorAllowed()
    ) throws {
        let root = try resources.rootURL()
        let manifestURL = root.appendingPathComponent("KokoroRuntimeManifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw KokoroError.missingManifest(manifestURL)
        }
        self.rootURL = root
        try KokoroPathGuard.rejectRootSymlink(rootURL: root)
        let manifest = try JSONDecoder().decode(KokoroRuntimeManifest.self, from: Data(contentsOf: manifestURL))
        guard manifest.schemaVersion == Self.supportedSchemaVersion else {
            throw KokoroError.unsupportedManifestSchema(manifest.schemaVersion)
        }
        guard manifest.hfProvenanceVerified else {
            throw KokoroError.badHash(path: "hf_provenance_verified")
        }
        let manifestModelPaths = Set(manifest.modelPackages.map(\.path))
        let modelsDirectory = root.appendingPathComponent("coreml", isDirectory: true)
        self.modelsDirectory = modelsDirectory
        let compiledModelsDirectory = resources.explicitCompiledModelsDirectoryURL()
            ?? Self.defaultCompiledModelsDirectory(for: manifest)
        try Self.validateCompiledModelsDirectory(bundleRootURL: root, compiledModelsDirectory: compiledModelsDirectory)
        self.compiledModelsDirectory = compiledModelsDirectory
        // Re-rooted under `modelsDirectory`: discovery resolves symlinks
        // (/tmp -> /private/tmp), which would fail the lexical containment checks.
        let durationChoices = KokoroPipeline.discoverDurationChoices(modelsDirectory: modelsDirectory)
            .filter { $0.functionName == nil }
            .filter {
                manifestModelPaths.contains("coreml/\($0.packageURL.lastPathComponent)")
                    || Self.splitDurationPackages(paths: manifestModelPaths, tokenLength: $0.tokenLength) != nil
            }
            .map {
                DurationModelChoice(cacheKey: $0.cacheKey, tokenLength: $0.tokenLength,
                                    packageURL: modelsDirectory.appendingPathComponent($0.packageURL.lastPathComponent),
                                    requiresAttentionMask: $0.requiresAttentionMask, allowsPadding: $0.allowsPadding)
            }
        let usesFlexibleGenerator = PipelineConstants.usesFlexibleGenerator(
            hasFlexibleGenerator: Self.listsFlexibleGenerator(manifest),
            hasAnyBucketGenerator: Self.listsBucketGenerators(manifest),
            allowed: flexibleGeneratorAllowed
        )
        try Self.validateModelSet(
            rootURL: root,
            manifest: manifest,
            durationChoices: durationChoices,
            usesFlexibleGenerator: usesFlexibleGenerator
        )
        guard let runtimeDuration = durationChoices.first(where: {
            $0.allowsPadding && $0.tokenLength == KokoroTTS.runtimeDurationTokenLength
        }) else {
            throw KokoroError.missingModel(Self.durationPackagesDetail(tokenLength: KokoroTTS.runtimeDurationTokenLength))
        }
        // The starter bundle has one 15-second acoustic bucket. Matching it
        // with one padded duration shape avoids seven independent first-load
        // and specialization paths during playback.
        self.durationChoices = [runtimeDuration]
        self.manifest = manifest
        self.computePolicy = computePolicy
        self.usesFlexibleGenerator = usesFlexibleGenerator
        self.usesHybridGenerator = manifestModelPaths.contains(Self.hybridDirectoryPath) && computePolicy != .cpuOnly
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

    /// Compute units for a fast-path package (split pairs, folded generator).
    ///
    /// Those packages keep their measured placement under every policy except
    /// ``KokoroComputePolicy/cpuOnly``, which runs them on the CPU instead (and
    /// turns the hybrid generator off): a schema-2 bundle has no whole-model
    /// fallback, so "CPU only" must still be able to run the fast-path files.
    private func fastPathUnits(_ measured: MLComputeUnits) -> MLComputeUnits {
        computePolicy == .cpuOnly ? .cpuOnly : measured
    }

    /// Two-part duration model (ANE-layout BERT front + CPU back) when the
    /// manifest lists both packages for this size; see SplitDuration.swift.
    func splitDuration(choice: DurationModelChoice) throws -> SplitDuration? {
        try loadSplitDuration(choice: choice, progress: nil)
    }

    private func loadSplitDuration(choice: DurationModelChoice, progress: ((String) -> Void)?) throws -> SplitDuration? {
        guard SplitDuration.allowed, choice.allowsPadding,
              let names = Self.splitDurationPackages(paths: Set(manifest.modelPackages.map(\.path)), tokenLength: choice.tokenLength)
        else { return nil }
        let front = try model(cacheKey: "duration.front.t\(choice.tokenLength)", url: modelsDirectory.appendingPathComponent(names.front),
                              units: fastPathUnits(SplitDuration.frontComputeUnits), progress: progress)
        let back = try model(cacheKey: "duration.back.t\(choice.tokenLength)", url: modelsDirectory.appendingPathComponent(names.back),
                             units: fastPathUnits(SplitDuration.backComputeUnits), progress: progress)
        return SplitDuration(front: front, back: back)
    }

    /// Two-part F0Ntrain when the manifest lists both packages for this size.
    func splitF0N(tFrames: Int) throws -> SplitF0N? {
        try loadSplitF0N(tFrames: tFrames, progress: nil)
    }

    private func loadSplitF0N(tFrames: Int, progress: ((String) -> Void)?) throws -> SplitF0N? {
        guard SplitF0N.allowed,
              let names = Self.splitF0NPackages(paths: Set(manifest.modelPackages.map(\.path)), tFrames: tFrames)
        else { return nil }
        let front = try model(cacheKey: "f0ntrain.front.t\(tFrames)", url: modelsDirectory.appendingPathComponent(names.front),
                              units: fastPathUnits(SplitF0N.frontComputeUnits), progress: progress)
        let back = try model(cacheKey: "f0ntrain.back.t\(tFrames)", url: modelsDirectory.appendingPathComponent(names.back),
                             units: fastPathUnits(SplitF0N.backComputeUnits), progress: progress)
        return SplitF0N(front: front, back: back)
    }

    /// The split F0Ntrain packages for `tFrames` when the manifest lists both.
    static func splitF0NPackages(paths: Set<String>, tFrames: Int) -> (front: String, back: String)? {
        let names = SplitF0N.packageNames(tFrames: tFrames)
        guard paths.contains("coreml/\(names.front)"), paths.contains("coreml/\(names.back)") else { return nil }
        return names
    }

    /// The split duration packages for `tokenLength` when the manifest lists both.
    static func splitDurationPackages(paths: Set<String>, tokenLength: Int) -> (front: String, back: String)? {
        let names = SplitDuration.packageNames(tokenLength: tokenLength)
        guard paths.contains("coreml/\(names.front)"), paths.contains("coreml/\(names.back)") else { return nil }
        return names
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

    /// The row-folded Neural Engine generator package the manifest lists for
    /// `bucket`, preferring the variant this device's ANE runs best
    /// (``PipelineConstants/foldedGeneratorCandidates(bucketSec:)``), else nil.
    static func foldedPackageName(_ manifest: KokoroRuntimeManifest, bucket: Int) -> String? {
        foldedPackageName(paths: Set(manifest.modelPackages.map(\.path)), bucket: bucket)
    }

    /// Same, over manifest package paths (`coreml/<name>`).
    static func foldedPackageName(paths: Set<String>, bucket: Int) -> String? {
        guard PipelineConstants.foldedGeneratorAllowed else { return nil }
        return PipelineConstants.foldedGeneratorCandidates(bucketSec: bucket).first { paths.contains("coreml/\($0)") }
    }

    /// Loads the folded ANE generator for `bucketSec` when the bundle ships one.
    /// It replaces the GPU generators for that bucket: on an M1 the GPU
    /// generator alone held ~1 GB of footprint at 30 s, the folded one ~145 MB.
    func foldedGeneratorModel(bucketSec: Int) throws -> MLModel? {
        guard let name = Self.foldedPackageName(manifest, bucket: bucketSec) else { return nil }
        return try model(
            cacheKey: "generator.fold.\(bucketSec)",
            url: modelsDirectory.appendingPathComponent(name),
            units: fastPathUnits(.cpuAndNeuralEngine)
        )
    }

    /// Hybrid ANE-conv + Metal generator, when the manifest lists the
    /// `coreml/hybrid` tree (README/Notes/ane-folded-generator.md). The tree
    /// digest is checked before plan.json or adain.bin is read, and the row
    /// package compiles into the compiled-model cache like every other model.
    /// One is kept resident. Any error here or in its predict turns the hybrid
    /// off for the process and the folded generator serves
    /// (`resolveHybridGenerator` in KokoroSynthesisExecutor.swift).
    private var hybrid: (rows: Int, generator: AnyObject)?
    private var hybridTree: KokoroRuntimeManifest.ModelPackage?
    private let hybridLock = NSLock()
    func hybridGenerator(bucketSec: Int, xPreFrames: Int) throws -> AnyObject? {
        guard #available(macOS 15.0, iOS 18.0, *), usesHybridGenerator else { return nil }
        hybridLock.lock()
        defer { hybridLock.unlock() }
        // After a failure the folded generator serves; free the hybrid's ANE
        // functions and IOSurfaces instead of holding both (jetsam on iOS).
        guard HybridGenerator.allowed else { hybrid = nil; return nil }
        let dir = modelsDirectory.appendingPathComponent("hybrid", isDirectory: true)
        let tree = try hybridTree ?? validateModelPackageIfNeeded(dir)
        hybridTree = tree
        guard let rows = HybridGenerator.rows(bucketSec: bucketSec, xPreFrames: xPreFrames, directory: dir) else { return nil }
        if let hybrid, hybrid.rows == rows { return hybrid.generator }
        hybrid = nil
        let name = "hybrid_generator_r\(rows)"
        let source = dir.appendingPathComponent("\(name).mlpackage", isDirectory: true)
        let cached = compiledModelsDirectory.appendingPathComponent("\(name).mlmodelc", isDirectory: true)
        let compiled = try compiledModelURL(sourceURL: source, destinationURL: cached, sourceTreeSHA256: tree.treeSHA256)
        let generator: HybridGenerator
        do {
            generator = try HybridGenerator(directory: dir, rows: rows, compiledModel: compiled)
        } catch HybridGeneratorError.modelLoad where compiled == cached {
            // A cached compile Core ML can no longer load: evict it and recompile
            // once, as `model(cacheKey:)` does. Other errors (plan, Metal,
            // KOKORO_HYBRID_E5) are not the cache's fault and are not retried.
            try Self.removeCompiledCache(cacheRootURL: compiledModelsDirectory, destinationURL: cached)
            let recompiled = try compiledModelURL(sourceURL: source, destinationURL: cached, sourceTreeSHA256: tree.treeSHA256)
            generator = try HybridGenerator(directory: dir, rows: rows, compiledModel: recompiled)
        }
        hybrid = (rows, generator)
        return generator
    }

    /// Loads the flexible generator when the bundle ships it, else nil.
    func flexibleGeneratorModel() throws -> MLModel? {
        guard usesFlexibleGenerator else { return nil }
        return try model(
            cacheKey: "generator.range",
            url: modelsDirectory.appendingPathComponent(PipelineConstants.flexibleGeneratorPackage),
            units: computePolicy.generator
        )
    }

    /// Whether `manifest` lists the flexible generator package.
    static func listsFlexibleGenerator(_ manifest: KokoroRuntimeManifest) -> Bool {
        manifest.modelPackages.contains { $0.path == "coreml/\(PipelineConstants.flexibleGeneratorPackage)" }
    }

    /// Whether `manifest` lists a per-bucket generator for any bucket. With the
    /// gate off, a bundle missing one for some bucket then fails validation
    /// instead of running the flexible generator.
    static func listsBucketGenerators(_ manifest: KokoroRuntimeManifest) -> Bool {
        manifest.buckets.contains { bucket in
            manifest.modelPackages.contains { $0.path == "coreml/kokoro_decoder_har_post_\(bucket)s.mlpackage" }
        }
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
        let choices = try actualTokens.map { [try KokoroPipeline.selectDurationChoice(durationChoices, actualTokens: $0)] }
            ?? durationChoices
        // The same program the executor will run: the split pair when it ships.
        for choice in choices where try loadSplitDuration(choice: choice, progress: progress) == nil {
            _ = try model(
                cacheKey: "duration.\(choice.cacheKey)",
                url: choice.packageURL,
                units: computePolicy.duration,
                progress: progress
            )
        }
        for bucket in selectedBuckets {
            guard let tFrames = PipelineConstants.tFramesForBucket[bucket] else {
                throw KokoroError.missingModel("bucket \(bucket)s")
            }
            if try loadSplitF0N(tFrames: tFrames, progress: progress) == nil {
                _ = try model(
                    cacheKey: "f0ntrain.\(tFrames)",
                    url: modelsDirectory.appendingPathComponent("kokoro_f0ntrain_t\(tFrames).mlpackage"),
                    units: computePolicy.f0ntrain,
                    progress: progress
                )
            }
            _ = try model(
                cacheKey: "decoder_pre.\(bucket)",
                url: modelsDirectory.appendingPathComponent("kokoro_decoder_pre_\(bucket)s.mlpackage"),
                units: computePolicy.decoderPre,
                progress: progress
            )
            if let folded = Self.foldedPackageName(manifest, bucket: bucket) {
                _ = try model(
                    cacheKey: "generator.fold.\(bucket)",
                    url: modelsDirectory.appendingPathComponent(folded),
                    units: fastPathUnits(.cpuAndNeuralEngine),
                    progress: progress
                )
            } else if !usesFlexibleGenerator {
                _ = try model(
                    cacheKey: "generator.\(bucket)",
                    url: modelsDirectory.appendingPathComponent("kokoro_decoder_har_post_\(bucket)s.mlpackage"),
                    units: computePolicy.generator,
                    progress: progress
                )
            }
        }
        if usesFlexibleGenerator, selectedBuckets.contains(where: { Self.foldedPackageName(manifest, bucket: $0) == nil }) {
            _ = try model(
                cacheKey: "generator.range",
                url: modelsDirectory.appendingPathComponent(PipelineConstants.flexibleGeneratorPackage),
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
        let url = try KokoroPathGuard.containedURL(rootURL: rootURL, relativePath: manifest.runtimeAssets.hnsfWeights.path)
        let payload = try JSONDecoder().decode(Payload.self, from: Data(contentsOf: url))
        guard payload.linear_weights.count == 9, payload.linear_weights.allSatisfy(\.isFinite), payload.linear_bias.isFinite else {
            throw KokoroError.badHash(path: manifest.runtimeAssets.hnsfWeights.path)
        }
        return (payload.linear_weights, payload.linear_bias)
    }

    /// Loads the verified vocab declared by the runtime manifest.
    func vocab() throws -> [String: Int32] {
        let url = try KokoroPathGuard.containedURL(rootURL: rootURL, relativePath: manifest.runtimeAssets.vocab.path)
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
            throw KokoroError.missingModel(Self.missingPackageDetail(url.lastPathComponent))
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

    /// The missing package's name, prefixed with the environment switch that
    /// asked for it. A schema-2 bundle ships only the fast-path packages, so
    /// KOKORO_SPLIT_DURATION=0, KOKORO_SPLIT_F0N=0 or KOKORO_FOLDED_GENERATOR=0
    /// sends the stage to a whole or GPU package the bundle does not have.
    static func missingPackageDetail(_ name: String) -> String {
        let switches = [
            ("kokoro_duration_t", SplitDuration.envKey, SplitDuration.allowed),
            ("kokoro_f0ntrain_t", SplitF0N.envKey, SplitF0N.allowed),
            ("kokoro_decoder_har_post_", PipelineConstants.foldedGeneratorEnvKey, PipelineConstants.foldedGeneratorAllowed),
        ]
        guard let (_, key, _) = switches.first(where: { name.hasPrefix($0.0) && !$0.2 }) else { return name }
        return "\(key)=0 needs a bundle with \(name)"
    }

    /// Names a missing duration program by the split pair the SDK runs.
    static func durationPackagesDetail(tokenLength: Int) -> String {
        let names = SplitDuration.packageNames(tokenLength: tokenLength)
        return "\(names.front) + \(names.back)"
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
            try KokoroPathGuard.rejectExistingSymlinkComponents(
                rootURL: compiledModelsDirectory,
                targetURL: destinationURL.deletingLastPathComponent()
            )
            try FileManager.default.createDirectory(
                at: destinationURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try KokoroPathGuard.rejectExistingSymlinkComponents(
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
        try KokoroPathGuard.rejectExistingSymlinkComponents(rootURL: cacheRootURL, targetURL: destinationURL)
        let values = try destinationURL.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard values.isSymbolicLink != true, values.isDirectory == true else {
            return false
        }
        let sidecarURL = compiledSourceHashURL(for: destinationURL)
        guard FileManager.default.fileExists(atPath: sidecarURL.path) else {
            return false
        }
        try KokoroPathGuard.rejectExistingSymlinkComponents(rootURL: cacheRootURL, targetURL: sidecarURL)
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
        try KokoroPathGuard.rejectExistingSymlinkComponents(rootURL: cacheRootURL, targetURL: destinationURL)
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }
        let sidecar = compiledSourceHashURL(for: destinationURL)
        try KokoroPathGuard.rejectExistingSymlinkComponents(rootURL: cacheRootURL, targetURL: sidecar)
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
    ///
    /// Repeat validation is absorbed by ``KokoroFileDigest``'s process-wide memo
    /// rather than by per-provider bookkeeping, so a facade rebuilt for a second
    /// playback re-walks the package directory but re-hashes nothing.
    ///
    /// Module-internal so regression tests can assert that second pass without
    /// compiling a real Core ML model.
    ///
    /// - Parameter packageURL: `.mlpackage` directory to validate.
    /// - Returns: Matching manifest entry.
    func validateModelPackageIfNeeded(_ packageURL: URL) throws -> KokoroRuntimeManifest.ModelPackage {
        let relativePath = "coreml/\(packageURL.lastPathComponent)"
        guard let expected = manifest.modelPackages.first(where: { $0.path == relativePath }) else {
            throw KokoroError.missingModel(packageURL.lastPathComponent)
        }
        try KokoroPathGuard.rejectExistingSymlinkComponents(rootURL: rootURL, targetURL: packageURL)
        try Self.validatePackageTree(packageURL: packageURL, relativePath: relativePath, expected: expected)
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
        let url = try KokoroPathGuard.containedURL(rootURL: rootURL, relativePath: digest.path)
        let path = url.standardizedFileURL.path
        try KokoroPathGuard.rejectExistingSymlinkComponents(rootURL: rootURL, targetURL: url)
        guard FileManager.default.fileExists(atPath: path) else {
            if digest.path.hasPrefix("voices/") {
                throw KokoroError.missingVoice(url.deletingPathExtension().lastPathComponent)
            }
            throw KokoroError.missingRuntimeAsset(digest.path)
        }
        let values = try url.resourceValues(forKeys: [.isSymbolicLinkKey, .isRegularFileKey, .fileSizeKey])
        guard values.isSymbolicLink != true, values.isRegularFile == true else {
            throw KokoroError.pathEscape(digest.path)
        }
        guard values.fileSize == digest.bytes else {
            throw KokoroError.badHash(path: digest.path)
        }
        guard try KokoroFileDigest.memoizedSHA256(ofFileAt: url) == digest.sha256 else {
            throw KokoroError.badHash(path: digest.path)
        }
    }

    /// Verifies the compiled cache root can be used without following symlinks.
    private static func validateCompiledModelsDirectory(bundleRootURL rootURL: URL, compiledModelsDirectory: URL) throws {
        if KokoroPathGuard.isInside(compiledModelsDirectory, root: rootURL) {
            try KokoroPathGuard.rejectExistingSymlinkComponents(rootURL: rootURL, targetURL: compiledModelsDirectory)
        } else {
            try KokoroPathGuard.rejectRootSymlink(rootURL: compiledModelsDirectory)
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
            "revision=\(manifest.hfArtifactRevision)",
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
    ///
    /// Per stage, either program form satisfies it: duration and F0Ntrain by
    /// the whole package or the split pair; a bucket's generator by a folded
    /// package (then no har_post or range package is needed), else the
    /// flexible or per-bucket GPU generator. `coreml/hybrid` needs a folded
    /// package for every bucket: it is the hybrid's fallback, and serves the
    /// buckets below ``HybridGenerator/minBucketSec``. Checks the bundle only;
    /// the KOKORO_* environment switches do not change what it must contain.
    static func validateModelSet(
        rootURL: URL,
        manifest: KokoroRuntimeManifest,
        durationChoices: [DurationModelChoice],
        usesFlexibleGenerator: Bool
    ) throws {
        guard !durationChoices.isEmpty else {
            // KOKORO_SPLIT_DURATION=0 hides the split pair from discovery.
            throw KokoroError.missingModel(SplitDuration.allowed ? "duration"
                : missingPackageDetail("kokoro_duration_t\(KokoroTTS.runtimeDurationTokenLength).mlpackage"))
        }
        let sdkDurationTokenSizes = [KokoroTTS.runtimeDurationTokenLength]
        guard manifest.durationTokenSizes == sdkDurationTokenSizes else {
            throw KokoroError.missingModel("duration_token_sizes")
        }
        let paths = Set(manifest.modelPackages.map(\.path))
        for tokenLength in manifest.durationTokenSizes {
            guard paths.contains("coreml/kokoro_duration_t\(tokenLength).mlpackage")
                    || splitDurationPackages(paths: paths, tokenLength: tokenLength) != nil,
                  durationChoices.contains(where: { $0.tokenLength == tokenLength }) else {
                throw KokoroError.missingModel(durationPackagesDetail(tokenLength: tokenLength))
            }
        }
        for package in manifest.modelPackages {
            let url = try KokoroPathGuard.containedURL(rootURL: rootURL, relativePath: package.path)
            try KokoroPathGuard.rejectExistingSymlinkComponents(rootURL: rootURL, targetURL: url)
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard FileManager.default.fileExists(atPath: url.path),
                  values?.isDirectory == true,
                  values?.isSymbolicLink != true else {
                throw KokoroError.missingModel(url.lastPathComponent)
            }
        }
        let listsHybrid = paths.contains(hybridDirectoryPath)
        var required = usesFlexibleGenerator ? [PipelineConstants.flexibleGeneratorPackage] : []
        for bucket in manifest.buckets {
            guard let tFrames = PipelineConstants.tFramesForBucket[bucket] else {
                throw KokoroError.missingModel("bucket \(bucket)s")
            }
            if splitF0NPackages(paths: paths, tFrames: tFrames) == nil {
                required.append("kokoro_f0ntrain_t\(tFrames).mlpackage")
            }
            required.append("kokoro_decoder_pre_\(bucket)s.mlpackage")
            if let folded = PipelineConstants.foldedGeneratorCandidates(bucketSec: bucket).first(where: { paths.contains("coreml/\($0)") }) {
                required.append(folded)
            } else if listsHybrid {
                throw KokoroError.missingModel(PipelineConstants.foldedGeneratorPackage(bucketSec: bucket, poly: true))
            } else if !usesFlexibleGenerator {
                required.append("kokoro_decoder_har_post_\(bucket)s.mlpackage")
            }
        }
        for name in required {
            guard paths.contains("coreml/\(name)") else {
                throw KokoroError.missingModel(name)
            }
        }
    }

    /// Validates a model package against the manifest tree digest.
    ///
    /// File bytes are streamed, never loaded whole: a `Data(contentsOf:)` read of
    /// the 67 MB `weight.bin` in the acoustic package produced a matching
    /// resident spike on every validation pass.
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
            let values = try file.url.resourceValues(forKeys: [.fileSizeKey])
            let byteCount = values.fileSize ?? 0
            totalBytes += byteCount
            let fileHash = try KokoroFileDigest.memoizedSHA256(ofFileAt: file.url)
            digest.update(data: Data(rel.utf8))
            digest.update(data: Data([0]))
            digest.update(data: Data(String(byteCount).utf8))
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
