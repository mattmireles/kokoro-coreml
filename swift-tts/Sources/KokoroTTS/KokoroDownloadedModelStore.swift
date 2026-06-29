import CryptoKit
import Foundation

/// Safety limits for hosted-manifest SDK downloads.
public struct KokoroDownloadLimits: Sendable {
    /// Maximum hosted manifest byte count.
    public let maxManifestBytes: Int

    /// Maximum single hosted file byte count.
    public let maxFileBytes: Int

    /// Maximum total byte count declared by one hosted manifest.
    public let maxTotalBytes: Int

    /// Maximum hosted file count declared by one hosted manifest.
    public let maxFileCount: Int

    /// Creates hosted download safety limits.
    public init(
        maxManifestBytes: Int = 2 * 1024 * 1024,
        maxFileBytes: Int = 1024 * 1024 * 1024,
        maxTotalBytes: Int = 5 * 1024 * 1024 * 1024,
        maxFileCount: Int = 512
    ) {
        self.maxManifestBytes = maxManifestBytes
        self.maxFileBytes = maxFileBytes
        self.maxTotalBytes = maxTotalBytes
        self.maxFileCount = maxFileCount
    }

    /// Production-oriented default limits for starter and full SDK bundles.
    public static let `default` = KokoroDownloadLimits()
}

/// Decoded hosted-manifest payload used by downloaded-resource mode.
private struct KokoroHostedManifest: Decodable {
    /// One hosted file entry.
    struct File: Decodable {
        /// Manifest-relative hosted path.
        let path: String

        /// Expected byte count.
        let bytes: Int

        /// Expected SHA-256 digest.
        let sha256: String
    }

    /// Optional hosted bundle version.
    let version: String?

    /// Hosted files to hydrate.
    let files: [File]
}

/// Downloads a hosted-manifest SDK bundle into a local cache directory.
public struct KokoroDownloadedModelStore: Sendable {
    /// Remote manifest URL.
    public let manifestURL: URL

    /// Expected SHA-256 digest of the hosted manifest JSON.
    public let expectedManifestSHA256: String

    /// Local cache directory.
    public let cacheDirectory: URL

    /// Whether to allow non-HTTPS manifest URLs for local development.
    public let allowInsecureLocalDevelopment: Bool

    /// Download safety limits.
    public let limits: KokoroDownloadLimits

    /// Creates a downloaded model store.
    ///
    /// - Parameters:
    ///   - manifestURL: URL for a `{ version, files }` hosted manifest.
    ///   - expectedManifestSHA256: SHA-256 digest of the exact manifest JSON.
    ///   - cacheDirectory: Local cache root for downloaded files.
    ///   - allowInsecureLocalDevelopment: Allows `http://` manifests only for
    ///     local development and tests. Production callers should leave this
    ///     false and serve manifests over HTTPS.
    ///   - limits: File count and byte-count safety limits.
    public init(
        manifestURL: URL,
        expectedManifestSHA256: String,
        cacheDirectory: URL,
        allowInsecureLocalDevelopment: Bool = false,
        limits: KokoroDownloadLimits = .default
    ) {
        self.manifestURL = manifestURL
        self.expectedManifestSHA256 = expectedManifestSHA256
        self.cacheDirectory = cacheDirectory
        self.allowInsecureLocalDevelopment = allowInsecureLocalDevelopment
        self.limits = limits
    }

    /// Downloads missing or hash-mismatched files and returns a directory provider.
    public func hydrate() async throws -> KokoroResourceProvider {
        try Task.checkCancellation()
        try validateManifestURL()
        let data = try await downloadWithRetry(
            manifestURL,
            maxBytes: limits.maxManifestBytes,
            label: "HostedManifest.json"
        )
        guard Self.sha256(data) == expectedManifestSHA256 else {
            throw KokoroError.badHash(path: "HostedManifest.json")
        }
        let manifest = try JSONDecoder().decode(KokoroHostedManifest.self, from: data)
        try validateManifest(manifest)
        try Self.rejectRootSymlink(rootURL: cacheDirectory)
        try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        try Self.rejectRootSymlink(rootURL: cacheDirectory)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var mutableCache = cacheDirectory
        try mutableCache.setResourceValues(values)

        let compiledCache = cacheDirectory.appendingPathComponent("compiled", isDirectory: true)
        let versionURL = cacheDirectory.appendingPathComponent(".kokoro-hosted-version")
        try Self.rejectExistingSymlinkComponents(rootURL: cacheDirectory, targetURL: versionURL)
        if let version = manifest.version,
           cachedVersion(versionURL: versionURL) != version,
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
            let payload = try await downloadWithRetry(
                fileURL,
                maxBytes: min(file.bytes, limits.maxFileBytes),
                label: file.path
            )
            try Task.checkCancellation()
            try payload.write(to: target, options: .atomic)
            guard try await isValidFile(url: target, bytes: file.bytes, sha256: file.sha256) else {
                throw KokoroError.badHash(path: file.path)
            }
        }
        if let version = manifest.version {
            try Self.rejectExistingSymlinkComponents(rootURL: cacheDirectory, targetURL: versionURL)
            try Data(version.utf8).write(
                to: versionURL,
                options: .atomic
            )
        }
        return .downloadedDirectory(
            root: cacheDirectory,
            compiledModelsDirectory: compiledCache
        )
    }

    /// Downloads one file with a small fixed retry budget.
    private func downloadWithRetry(_ url: URL, maxBytes: Int, label: String) async throws -> Data {
        if url.isFileURL {
            try Task.checkCancellation()
            let values = try url.resourceValues(forKeys: [.fileSizeKey])
            let byteCount = values.fileSize ?? 0
            if byteCount > maxBytes {
                throw KokoroError.downloadTooLarge(path: label, bytes: byteCount, maxBytes: maxBytes)
            }
            let data = try Data(contentsOf: url)
            if data.count > maxBytes {
                throw KokoroError.downloadTooLarge(path: label, bytes: data.count, maxBytes: maxBytes)
            }
            return data
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
                if data.count > maxBytes {
                    throw KokoroError.downloadTooLarge(path: label, bytes: data.count, maxBytes: maxBytes)
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
        if bytes > limits.maxFileBytes {
            throw KokoroError.downloadTooLarge(path: url.lastPathComponent, bytes: bytes, maxBytes: limits.maxFileBytes)
        }
        let data = try Data(contentsOf: url)
        guard data.count == bytes else {
            return false
        }
        return Self.sha256(data) == sha256
    }

    /// Returns the locally recorded hosted bundle version, if present.
    private func cachedVersion(versionURL url: URL) -> String? {
        return try? String(contentsOf: url, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Validates the remote manifest URL before any download.
    private func validateManifestURL() throws {
        if manifestURL.isFileURL {
            return
        }
        guard manifestURL.scheme == "https" || allowInsecureLocalDevelopment else {
            throw KokoroError.insecureManifestURL(manifestURL)
        }
    }

    /// Validates manifest-level count, byte, and hash-shape constraints.
    private func validateManifest(_ manifest: KokoroHostedManifest) throws {
        guard manifest.files.count <= limits.maxFileCount else {
            throw KokoroError.downloadTooLarge(
                path: "HostedManifest.json file count",
                bytes: manifest.files.count,
                maxBytes: limits.maxFileCount
            )
        }
        var totalBytes = 0
        for file in manifest.files {
            guard file.bytes >= 0, file.bytes <= limits.maxFileBytes else {
                throw KokoroError.downloadTooLarge(path: file.path, bytes: file.bytes, maxBytes: limits.maxFileBytes)
            }
            guard file.sha256.range(of: "^[0-9a-f]{64}$", options: .regularExpression) != nil else {
                throw KokoroError.badHash(path: file.path)
            }
            totalBytes += file.bytes
            if totalBytes > limits.maxTotalBytes {
                throw KokoroError.downloadTooLarge(path: "HostedManifest.json", bytes: totalBytes, maxBytes: limits.maxTotalBytes)
            }
        }
    }

    /// Computes a lowercase SHA-256 hex digest.
    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
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
            if Self.isSymbolicLink(current) {
                throw KokoroError.pathEscape(current.path)
            }
            guard FileManager.default.fileExists(atPath: current.path) else {
                continue
            }
        }
    }

    /// Rejects a cache root that is itself a symlink.
    static func rejectRootSymlink(rootURL: URL) throws {
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
}
