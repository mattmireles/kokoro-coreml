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

    /// Resolves the caller-supplied compiled-model cache directory.
    ///
    /// The SDK computes a stable manifest-keyed cache when this returns `nil`.
    ///
    /// - Returns: Explicit directory where `.mlmodelc` models may be loaded or cached.
    func explicitCompiledModelsDirectoryURL() -> URL? {
        switch self {
        case .downloadedDirectory(_, let compiledModelsDirectory):
            return compiledModelsDirectory
        case .directory(_, let compiledModelsDirectory):
            return compiledModelsDirectory
        case .appBundle(_, _, let compiledModelsDirectory),
             .packageBundle(_, _, let compiledModelsDirectory):
            return compiledModelsDirectory
        }
    }
}
