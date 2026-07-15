import Foundation
import KokoroPipeline

/// Errors raised while loading Kokoro voice embedding tables.
enum KokoroVoiceTableError: Error, Equatable, LocalizedError {
    /// The requested `.bin` file was absent.
    case missingVoice(String)

    /// The `.bin` file length was not a non-empty multiple of 256 float32 rows.
    case malformedVoice(String)

    /// Human-readable explanation for app logs and tests.
    var errorDescription: String? {
        switch self {
        case .missingVoice(let voice):
            return "Kokoro voice embedding file is missing for \(voice)."
        case .malformedVoice(let voice):
            return "Kokoro voice embedding file is malformed for \(voice)."
        }
    }
}

/// Loader for Kokoro voice embedding `.bin` tables.
///
/// Each voice file is a little-endian float32 matrix with 256 columns. The row
/// selected for a text chunk uses the Gist/Botnet fleet rule:
/// `rowIndex = clamp(phonemeUTF16Count - 1, 0, rowCount - 1)`.
struct VoiceTable {
    /// Number of floats per Kokoro voice embedding row.
    static let embeddingDim = PipelineConstants.voiceEmbeddingDim

    /// Default voice bundled by the starter SDK profile.
    static let defaultVoiceID = KokoroVoiceID.afHeart

    /// Voice IDs bundled by the starter SDK profile.
    static let supportedVoiceIDs = KokoroVoiceID.starterVoices

    /// Directory containing `<voice>.bin` files.
    private let voicesDirectory: URL

    /// In-memory cache of decoded voice tables keyed by raw voice ID.
    private var tables: [KokoroVoiceID: [Float]] = [:]

    /// Creates a voice table loader.
    ///
    /// - Parameter voicesDirectory: Directory containing Kokoro voice `.bin`
    ///   files.
    init(voicesDirectory: URL) {
        self.voicesDirectory = voicesDirectory
    }

    /// Returns the selected 256-float `ref_s` row for a voice and phoneme count.
    ///
    /// - Parameters:
    ///   - voiceID: Kokoro voice identifier.
    ///   - phonemeCount: UTF-16 phoneme string length before BOS/EOS framing.
    /// - Returns: Selected voice embedding row.
    mutating func refS(voiceID: KokoroVoiceID, phonemeCount: Int) throws -> [Float] {
        let rows = try table(for: voiceID)
        let rowCount = rows.count / Self.embeddingDim
        let rowIndex = max(0, min(rowCount - 1, phonemeCount - 1))
        let start = rowIndex * Self.embeddingDim
        return Array(rows[start..<(start + Self.embeddingDim)])
    }

    /// Loads or returns a cached full voice table.
    ///
    /// - Parameter voiceID: Kokoro voice identifier.
    /// - Returns: Flat row-major float matrix.
    mutating func table(for voiceID: KokoroVoiceID) throws -> [Float] {
        if let cached = tables[voiceID] {
            return cached
        }
        guard Self.isSafeVoiceID(voiceID.rawValue) else {
            throw KokoroVoiceTableError.missingVoice(voiceID.rawValue)
        }
        let url = voicesDirectory.appendingPathComponent("\(voiceID.rawValue).bin")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw KokoroVoiceTableError.missingVoice(voiceID.rawValue)
        }
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true else {
            throw KokoroVoiceTableError.missingVoice(voiceID.rawValue)
        }
        let data = try Data(contentsOf: url)
        guard data.count % (Self.embeddingDim * 4) == 0, !data.isEmpty else {
            throw KokoroVoiceTableError.malformedVoice(voiceID.rawValue)
        }

        var floats = [Float](repeating: 0, count: data.count / 4)
        data.withUnsafeBytes { raw in
            for index in 0..<floats.count {
                let bits = raw.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)
                floats[index] = Float(bitPattern: UInt32(littleEndian: bits))
            }
        }
        tables[voiceID] = floats
        return floats
    }

    /// Returns whether a public voice ID can be used as a single file stem.
    private static func isSafeVoiceID(_ rawValue: String) -> Bool {
        !rawValue.isEmpty
            && !rawValue.contains("/")
            && !rawValue.contains("\\")
            && !rawValue.contains("..")
    }
}
