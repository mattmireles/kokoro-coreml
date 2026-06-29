import CryptoKit
import Foundation

/// Locates a generated Kokoro SDK runtime bundle.
public enum KokoroResourceProvider: Sendable {
    /// Explicit directory containing `KokoroRuntimeManifest.json`.
    case directory(URL, compiledModelsDirectory: URL? = nil)

    /// App bundle resource directory containing `KokoroRuntimeManifest.json`.
    case appBundle(Bundle, subdirectory: String? = nil, compiledModelsDirectory: URL? = nil)

    /// Swift package resource directory containing `KokoroRuntimeManifest.json`.
    case packageBundle(Bundle, subdirectory: String? = nil, compiledModelsDirectory: URL? = nil)

    /// Downloaded bundle root with a writable compiled-model cache directory.
    case downloadedDirectory(root: URL, compiledModelsDirectory: URL)

    /// Resolves the runtime bundle root URL.
    ///
    /// - Returns: Directory containing `KokoroRuntimeManifest.json`.
    public func rootURL() throws -> URL {
        switch self {
        case .directory(let url, _):
            return url
        case .downloadedDirectory(let root, _):
            return root
        case .appBundle(let bundle, let subdirectory, _),
             .packageBundle(let bundle, let subdirectory, _):
            if let subdirectory {
                guard let url = bundle.resourceURL?.appendingPathComponent(subdirectory, isDirectory: true) else {
                    throw KokoroError.missingManifest(bundle.bundleURL)
                }
                return url
            }
            guard let url = bundle.resourceURL else {
                throw KokoroError.missingManifest(bundle.bundleURL)
            }
            return url
        }
    }

    /// Resolves the writable or bundled compiled-model directory.
    ///
    /// - Returns: Directory where `.mlmodelc` models may be loaded or cached.
    func compiledModelsDirectoryURL() throws -> URL {
        switch self {
        case .downloadedDirectory(_, let compiledModelsDirectory):
            return compiledModelsDirectory
        case .directory(let root, let compiledModelsDirectory):
            return compiledModelsDirectory ?? Self.defaultCompiledModelsDirectory(for: root)
        case .appBundle(_, _, let compiledModelsDirectory),
             .packageBundle(_, _, let compiledModelsDirectory):
            return try compiledModelsDirectory ?? Self.defaultCompiledModelsDirectory(for: rootURL())
        }
    }

    /// Returns a stable writable cache directory for compiled Core ML models.
    private static func defaultCompiledModelsDirectory(for root: URL) -> URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let digest = SHA256.hash(data: Data(root.standardizedFileURL.path.utf8))
            .prefix(8)
            .map { String(format: "%02x", $0) }
            .joined()
        return caches
            .appendingPathComponent("KokoroTTS", isDirectory: true)
            .appendingPathComponent("compiled-\(digest)", isDirectory: true)
    }
}
