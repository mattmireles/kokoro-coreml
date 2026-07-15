import Foundation

/// Locates a generated Kokoro SDK runtime bundle.
public enum KokoroResourceProvider: Sendable {
    /// Explicit directory containing `KokoroRuntimeManifest.json`.
    case directory(URL)

    /// App bundle resource directory containing `KokoroRuntimeManifest.json`.
    case appBundle(Bundle, subdirectory: String? = nil)

    /// Swift package resource directory containing `KokoroRuntimeManifest.json`.
    case packageBundle(Bundle, subdirectory: String? = nil)

    /// Downloaded bundle root with a writable compiled-model cache directory.
    case downloadedDirectory(root: URL, compiledModelsDirectory: URL)

    /// Resolves the runtime bundle root URL.
    ///
    /// - Returns: Directory containing `KokoroRuntimeManifest.json`.
    public func rootURL() throws -> URL {
        switch self {
        case .directory(let url):
            return url
        case .downloadedDirectory(let root, _):
            return root
        case .appBundle(let bundle, let subdirectory),
             .packageBundle(let bundle, let subdirectory):
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
        case .directory(let root):
            return root.appendingPathComponent("compiled", isDirectory: true)
        case .appBundle(_, _), .packageBundle(_, _):
            return try rootURL().appendingPathComponent("compiled", isDirectory: true)
        }
    }
}
