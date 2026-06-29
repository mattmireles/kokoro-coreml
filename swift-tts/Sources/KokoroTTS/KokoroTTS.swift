import Foundation
import KokoroPipeline

/// High-level SDK facade for local Kokoro TTS.
public actor KokoroTTS {
    /// SDK module name used in diagnostics and probe output.
    public static let sdkName = "KokoroTTS"

    /// The low-level token cap inherited from ``KokoroPipeline``.
    public static let maxCallerChunkTokens = PipelineConstants.maxCallerChunkTokens

    /// Text chunker used before per-chunk tokenization.
    private let chunker: TextChunker

    /// Text processor used for American English raw text preparation.
    private let americanTextProcessor: KokoroTextProcessor

    /// Text processor used for British English raw text preparation.
    private let britishTextProcessor: KokoroTextProcessor

    /// Voice table loader for selected bundle voices.
    private var voiceTable: VoiceTable

    /// Core ML model provider for synthesis.
    private let modelProvider: KokoroSDKModelProvider

    /// hn-NSF linear weights used by the Swift harmonic source.
    private let hnsf: (linearWeights: [Float], linearBias: Float)

    /// Creates a loaded facade.
    ///
    /// - Parameters:
    ///   - chunker: Text chunker.
    ///   - americanTextProcessor: American English raw text processor.
    ///   - britishTextProcessor: British English raw text processor.
    ///   - voiceTable: Voice table loader.
    ///   - modelProvider: Core ML model provider.
    ///   - hnsf: hn-NSF linear weights and bias.
    private init(
        chunker: TextChunker,
        americanTextProcessor: KokoroTextProcessor,
        britishTextProcessor: KokoroTextProcessor,
        voiceTable: VoiceTable,
        modelProvider: KokoroSDKModelProvider,
        hnsf: (linearWeights: [Float], linearBias: Float)
    ) {
        self.chunker = chunker
        self.americanTextProcessor = americanTextProcessor
        self.britishTextProcessor = britishTextProcessor
        self.voiceTable = voiceTable
        self.modelProvider = modelProvider
        self.hnsf = hnsf
    }

    /// Loads a KokoroTTS facade from a generated runtime bundle.
    ///
    /// - Parameters:
    ///   - resources: Bundle resource location.
    ///   - computePolicy: Core ML compute-unit policy for model stages.
    /// - Returns: Loaded actor ready for `prepare` and `synthesize`.
    public static func load(
        resources: KokoroResourceProvider,
        computePolicy: KokoroComputePolicy = .gistDefault
    ) async throws -> KokoroTTS {
        let loadTask: Task<KokoroTTS, Error> = Task.detached(priority: .userInitiated) {
            try Task.checkCancellation()
            return try loadSynchronously(resources: resources, computePolicy: computePolicy)
        }
        return try await withTaskCancellationHandler {
            try await loadTask.value
        } onCancel: {
            loadTask.cancel()
        }
    }

    /// Performs synchronous resource validation and facade construction.
    ///
    /// Called only from ``load(resources:computePolicy:)`` inside a detached
    /// task so manifest reads, digest checks, vocab loading, and hn-NSF loading
    /// cannot run on a caller's main-actor executor. Core ML compilation is not
    /// performed here; callers use ``prewarm(text:voice:options:)`` to trigger
    /// lazy model compilation before the first user-visible synthesis.
    private static func loadSynchronously(
        resources: KokoroResourceProvider,
        computePolicy: KokoroComputePolicy
    ) throws -> KokoroTTS {
        let modelProvider = try KokoroSDKModelProvider(resources: resources, computePolicy: computePolicy)
        let vocab = try modelProvider.vocab()
        let americanTextProcessor = KokoroTextProcessor(
            phonemizer: KokoroMisakiPhonemizer(british: false),
            vocab: vocab
        )
        let britishTextProcessor = KokoroTextProcessor(
            phonemizer: KokoroMisakiPhonemizer(british: true),
            vocab: vocab
        )
        let voiceTable = VoiceTable(voicesDirectory: modelProvider.voicesDirectory())
        let hnsf = try modelProvider.hnsfWeights()
        return KokoroTTS(
            chunker: TextChunker(),
            americanTextProcessor: americanTextProcessor,
            britishTextProcessor: britishTextProcessor,
            voiceTable: voiceTable,
            modelProvider: modelProvider,
            hnsf: hnsf
        )
    }

    /// Prepares raw text into one or more model-ready inputs.
    ///
    /// - Parameters:
    ///   - text: Raw caller text.
    ///   - voice: Kokoro voice ID.
    ///   - options: Synthesis options.
    /// - Returns: Prepared inputs, one per chunk.
    public func prepare(
        _ text: String,
        voice: KokoroVoiceID = .afHeart,
        options: KokoroSynthesisOptions = KokoroSynthesisOptions()
    ) throws -> [KokoroPreparedInput] {
        guard voice.isSupportedRawTextLanguage else {
            throw KokoroError.unsupportedVoice(voice.rawValue)
        }
        guard modelProvider.manifest.voices.contains(where: { $0.path == "voices/\(voice.rawValue).bin" }) else {
            throw KokoroError.unsupportedVoice(voice.rawValue)
        }
        let activeChunker = options.maxChunkSeconds == chunker.maxChunkSeconds
            ? chunker
            : TextChunker(maxChunkSeconds: options.maxChunkSeconds)
        let chunks = activeChunker.chunks(
            for: text,
            speed: Double(options.speed),
            maxCharacters: options.maxCharacters
        )
        guard !chunks.isEmpty else {
            throw KokoroError.emptyText
        }
        var prepared: [KokoroPreparedInput] = []
        let processor = textProcessor(for: voice)
        for (index, chunk) in chunks.enumerated() {
            do {
                let phonemes = try processor.phonemize(chunk)
                let refS = try voiceTable.refS(voiceID: voice, phonemeCount: phonemes.utf16Count)
                prepared.append(try processor.prepare(
                    text: chunk,
                    voice: voice,
                    refS: refS,
                    options: options,
                    key: chunks.count == 1 ? nil : "chunk-\(String(format: "%03d", index))",
                    phonemeResult: phonemes
                ))
            } catch {
                throw Self.mapPreparationError(error)
            }
        }
        return prepared
    }

    /// Returns the locale-appropriate text processor for a Kokoro voice.
    ///
    /// - Parameter voice: Kokoro voice ID whose prefix selects the Misaki path.
    /// - Returns: American or British English text processor.
    private func textProcessor(for voice: KokoroVoiceID) -> KokoroTextProcessor {
        voice.usesBritishEnglish ? britishTextProcessor : americanTextProcessor
    }

    /// Synthesizes raw text to mono 24 kHz PCM audio.
    ///
    /// - Parameters:
    ///   - text: Raw caller text.
    ///   - voice: Kokoro voice ID.
    ///   - options: Synthesis options.
    /// - Returns: Raw PCM audio plus sample rate.
    public func synthesize(
        _ text: String,
        voice: KokoroVoiceID = .afHeart,
        options: KokoroSynthesisOptions = KokoroSynthesisOptions()
    ) async throws -> KokoroAudio {
        try Task.checkCancellation()
        let inputs = try prepare(text, voice: voice, options: options)
        do {
            return try synthesizePrepared(inputs, modelProvider: modelProvider, hnsf: hnsf)
        } catch is CancellationError {
            throw KokoroError.synthesisCancelled
        } catch let error as KokoroError {
            switch error {
            case .coreMLLoadFailed:
                return try retrySynthesisCPUOnly(inputs, modelProvider: modelProvider, hnsf: hnsf)
            default:
                throw error
            }
        } catch let error as PipelineError {
            throw Self.mapSynthesisError(error)
        } catch {
            return try retrySynthesisCPUOnly(inputs, modelProvider: modelProvider, hnsf: hnsf)
        }
    }

    /// Loads models likely needed by a synthesis call without producing audio.
    ///
    /// - Parameters:
    ///   - text: Raw caller text used to select the duration bucket.
    ///   - voice: Kokoro voice ID.
    ///   - options: Synthesis options.
    public func prewarm(
        text: String = "Hello world.",
        voice: KokoroVoiceID = .afHeart,
        options: KokoroSynthesisOptions = KokoroSynthesisOptions()
    ) async throws {
        try Task.checkCancellation()
        let inputs = try prepare(text, voice: voice, options: options)
        for input in inputs {
            try Task.checkCancellation()
            try modelProvider.prewarm(actualTokens: input.numTokens)
        }
    }

    /// Synthesizes already prepared chunks with a model provider.
    private func synthesizePrepared(
        _ inputs: [KokoroPreparedInput],
        modelProvider: KokoroSDKModelProvider,
        hnsf: (linearWeights: [Float], linearBias: Float)
    ) throws -> KokoroAudio {
        var segments: [[Float]] = []
        for input in inputs {
            try Task.checkCancellation()
            var dump: TensorDumpWriter? = nil
            let result = try executeKokoroSynthesis(
                request: input.synthesisRequest(),
                modelProvider: modelProvider,
                linearWeights: hnsf.linearWeights,
                linearBias: hnsf.linearBias,
                tensorDump: &dump
            )
            segments.append(result.audio)
        }
        let samples = PcmJoiner.join(segments: segments, sampleRate: PipelineConstants.sampleRate)
        guard !samples.isEmpty, samples.allSatisfy({ $0.isFinite }) else {
            throw KokoroError.invalidAudioOutput
        }
        return KokoroAudio(samples: samples)
    }

    /// Retries synthesis after switching future Core ML loads to CPU-only.
    private func retrySynthesisCPUOnly(
        _ inputs: [KokoroPreparedInput],
        modelProvider: KokoroSDKModelProvider,
        hnsf: (linearWeights: [Float], linearBias: Float)
    ) throws -> KokoroAudio {
        modelProvider.degradeToCPUOnly()
        do {
            return try synthesizePrepared(inputs, modelProvider: modelProvider, hnsf: hnsf)
        } catch {
            throw Self.mapSynthesisError(error)
        }
    }

    /// Maps lower-level runtime failures onto public SDK errors.
    private static func mapSynthesisError(_ error: Error) -> Error {
        if error is CancellationError {
            return KokoroError.synthesisCancelled
        }
        if let error = error as? KokoroError {
            return error
        }
        if let error = error as? PipelineError {
            switch error {
            case .inputTooLong(let tokens, let maxTokens):
                return KokoroError.inputTooLong(tokens: tokens, maxTokens: maxTokens)
            case .modelNotLoaded(let name):
                return KokoroError.missingModel(name)
            case .noBucketAvailable:
                return KokoroError.missingModel("bucket")
            }
        }
        return KokoroError.coreMLPredictionFailed(String(describing: error))
    }

    /// Maps lower-level preparation failures onto public SDK errors.
    private static func mapPreparationError(_ error: Error) -> Error {
        if let error = error as? KokoroError {
            return error
        }
        if let error = error as? KokoroVoiceTableError {
            switch error {
            case .missingVoice(let voice):
                return KokoroError.missingVoice(voice)
            case .malformedVoice(let voice):
                return KokoroError.malformedVoice(voice)
            }
        }
        if let error = error as? KokoroTextProcessingError {
            switch error {
            case .emptyText:
                return KokoroError.emptyText
            case .emptyTokenization:
                return KokoroError.emptyPhonemizerOutput
            case .invalidSpeed(let speed):
                return KokoroError.invalidSpeed(speed)
            case .tokenBudgetExceeded(let actual, let maximum):
                return KokoroError.inputTooLong(tokens: actual, maxTokens: maximum)
            case .vocabUnavailable:
                return KokoroError.missingRuntimeAsset("runtime/kokoro-vocab.json")
            case .invalidVoiceEmbedding:
                return KokoroError.invalidAudioOutput
            }
        }
        return error
    }
}
