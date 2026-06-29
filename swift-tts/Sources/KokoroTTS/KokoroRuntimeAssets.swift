import Foundation

/// Identifies checked runtime assets bundled with the high-level SDK package.
///
/// These files are intentionally small and checked into source control. Larger
/// Core ML packages and voice embeddings are added by later bundle/download
/// phases, but text prep needs these assets to be canonical before synthesis is
/// exposed.
enum KokoroRuntimeAsset: String, CaseIterable, Sendable {
    /// Kokoro phoneme vocabulary JSON used to convert phonemes into token IDs.
    case vocab = "kokoro-vocab.json"

    /// hn-NSF linear projection weights used by the Swift harmonic source path.
    case hnsfWeights = "hnsf_weights.json"

    /// Hash manifest fragment for the small checked SDK runtime assets.
    case manifest = "KokoroRuntimeAssets.json"

    /// File name stored inside the package resource directory.
    var fileName: String {
        rawValue
    }
}

/// Resource lookup failure for bundled SDK runtime assets.
enum KokoroRuntimeAssetError: Error, Equatable, LocalizedError {
    /// The Swift package resource bundle did not contain the requested asset.
    case missingResource(String)

    /// Human-readable error text for app logs and tests.
    var errorDescription: String? {
        switch self {
        case .missingResource(let name):
            return "KokoroTTS runtime asset is missing from the package bundle: \(name)"
        }
    }
}

/// Accessor for small checked runtime assets in the `KokoroTTS` Swift package.
enum KokoroRuntimeAssets {
    /// Resource subdirectory containing Kokoro SDK runtime inputs.
    static let directoryName = "KokoroRuntime"

    /// Returns the package-bundled URL for a checked runtime asset.
    ///
    /// - Parameter asset: Asset identifier declared by ``KokoroRuntimeAsset``.
    /// - Returns: File URL inside the SwiftPM resource bundle.
    /// - Throws: ``KokoroRuntimeAssetError/missingResource(_:)`` when the asset
    ///   was not copied into the package bundle.
    static func url(for asset: KokoroRuntimeAsset) throws -> URL {
        let fileURL = URL(fileURLWithPath: asset.fileName)
        let stem = fileURL.deletingPathExtension().lastPathComponent
        let ext = fileURL.pathExtension.isEmpty ? nil : fileURL.pathExtension
        if let url = Bundle.module.url(
            forResource: stem,
            withExtension: ext,
            subdirectory: directoryName
        ) {
            return url
        }
        if let url = Bundle.module.url(forResource: stem, withExtension: ext) {
            return url
        }
        throw KokoroRuntimeAssetError.missingResource(asset.fileName)
    }
}
