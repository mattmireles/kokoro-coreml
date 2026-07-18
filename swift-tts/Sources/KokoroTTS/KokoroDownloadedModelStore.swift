import CryptoKit
import Foundation

/// Downloads a hosted-manifest SDK bundle into a local cache directory.
public struct KokoroDownloadedModelStore: Sendable {
    /// Remote manifest URL.
    public let manifestURL: URL

    /// Local cache directory.
    public let cacheDirectory: URL

    /// Creates a downloaded model store.
    ///
    /// - Parameters:
    ///   - manifestURL: URL for a `{ version, files }` hosted manifest.
    ///   - cacheDirectory: Local cache root for downloaded files.
    public init(manifestURL: URL, cacheDirectory: URL) {
        self.manifestURL = manifestURL
        self.cacheDirectory = cacheDirectory
    }

    /// Downloads missing or hash-mismatched files and returns a directory provider.
    public func hydrate() async throws -> KokoroResourceProvider {
        struct Manifest: Decodable {
            struct File: Decodable {
                let path: String
                let bytes: Int
                let sha256: String
            }
            let version: String?
            let files: [File]
        }
        try Task.checkCancellation()
        let data = try await downloadWithRetry(manifestURL)
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        try Self.rejectRootSymlink(rootURL: cacheDirectory)
        try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        try Self.rejectRootSymlink(rootURL: cacheDirectory)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var mutableCache = cacheDirectory
        try mutableCache.setResourceValues(values)

        let compiledCache = cacheDirectory.appendingPathComponent("compiled", isDirectory: true)
        if let version = manifest.version,
           cachedVersion() != version,
           FileManager.default.fileExists(atPath: compiledCache.path) {
            try FileManager.default.removeItem(at: compiledCache)
        }

        for file in manifest.files {
            try Task.checkCancellation()
            let target = try Self.containedURL(rootURL: cacheDirectory, relativePath: file.path)
            try Self.rejectExistingSymlinkComponents(rootURL: cacheDirectory, targetURL: target)
            if try await isValidFile(url: target, bytes: file.bytes, sha256: file.sha256) {
                continue
            }
            try Self.rejectExistingSymlinkComponents(rootURL: cacheDirectory, targetURL: target.deletingLastPathComponent())
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Self.rejectExistingSymlinkComponents(rootURL: cacheDirectory, targetURL: target.deletingLastPathComponent())
            let fileURL = try Self.remoteURL(manifestURL: manifestURL, relativePath: file.path)
            let payload = try await downloadWithRetry(fileURL)
            try Task.checkCancellation()
            try payload.write(to: target, options: .atomic)
            guard try await isValidFile(url: target, bytes: file.bytes, sha256: file.sha256) else {
                throw KokoroError.badHash(path: file.path)
            }
        }
        if let version = manifest.version {
            try Data(version.utf8).write(
                to: cacheDirectory.appendingPathComponent(".kokoro-hosted-version"),
                options: .atomic
            )
        }
        return .downloadedDirectory(
            root: cacheDirectory,
            compiledModelsDirectory: compiledCache
        )
    }

    /// Downloads one file with a small fixed retry budget.
    private func downloadWithRetry(_ url: URL) async throws -> Data {
        if url.isFileURL {
            try Task.checkCancellation()
            return try Data(contentsOf: url)
        }
        var lastError: Error?
        for _ in 0..<3 {
            try Task.checkCancellation()
            do {
                let (data, response) = try await URLSession.shared.data(from: url)
                if let http = response as? HTTPURLResponse,
                   !(200..<300).contains(http.statusCode) {
                    throw URLError(.badServerResponse)
                }
                return data
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                if Task.isCancelled {
                    throw CancellationError()
                }
                lastError = error
            }
        }
        throw lastError ?? URLError(.cannotLoadFromNetwork)
    }

    /// Checks whether a cached file matches expected size and hash.
    private func isValidFile(url: URL, bytes: Int, sha256: String) async throws -> Bool {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return false
        }
        let values = try url.resourceValues(forKeys: [.isSymbolicLinkKey, .isRegularFileKey])
        guard values.isSymbolicLink != true, values.isRegularFile == true else {
            throw KokoroError.pathEscape(url.path)
        }
        let data = try Data(contentsOf: url)
        guard data.count == bytes else {
            return false
        }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == sha256
    }

    /// Returns the locally recorded hosted bundle version, if present.
    private func cachedVersion() -> String? {
        let url = cacheDirectory.appendingPathComponent(".kokoro-hosted-version")
        return try? String(contentsOf: url, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Resolves a hosted-manifest path under the cache root.
    static func containedURL(rootURL: URL, relativePath: String) throws -> URL {
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

    /// Resolves a hosted-manifest path against the manifest URL.
    static func remoteURL(manifestURL: URL, relativePath: String) throws -> URL {
        let components = relativePath.split(separator: "/", omittingEmptySubsequences: false)
        guard !relativePath.hasPrefix("/"),
              !relativePath.contains("\\"),
              !components.isEmpty,
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw KokoroError.pathEscape(relativePath)
        }
        return components.reduce(manifestURL.deletingLastPathComponent()) { partial, component in
            partial.appendingPathComponent(String(component))
        }
    }

    /// Rejects symlinked cache components before reads or writes can follow them.
    static func rejectExistingSymlinkComponents(rootURL: URL, targetURL: URL) throws {
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

    /// Rejects a cache root that is itself a symlink.
    static func rejectRootSymlink(rootURL: URL) throws {
        guard FileManager.default.fileExists(atPath: rootURL.path) else {
            return
        }
        let values = try rootURL.resourceValues(forKeys: [.isSymbolicLinkKey])
        if values.isSymbolicLink == true {
            throw KokoroError.pathEscape(rootURL.path)
        }
    }
}
