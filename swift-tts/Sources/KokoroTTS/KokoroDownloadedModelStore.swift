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
    /// Bump when the shipped runtime's compiled-model contract changes.
    private static let compiledCacheDirectoryName = "compiled-v2"

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
    public func hydrate(
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> KokoroResourceProvider {
        try Task.checkCancellation()
        try validateManifestURL()
        let data = try await downloadWithRetry(
            manifestURL,
            maxBytes: limits.maxManifestBytes,
            label: "HostedManifest.json"
        ) { _ in
            // The manifest is intentionally tiny relative to model assets, but
            // a slow active transfer must still keep app-level stall watchdogs
            // alive before its final byte arrives.
            progress?(0)
        }
        guard Self.sha256(data) == expectedManifestSHA256 else {
            throw KokoroError.badHash(path: "HostedManifest.json")
        }
        let manifest = try JSONDecoder().decode(KokoroHostedManifest.self, from: data)
        try validateManifest(manifest)
        let totalBytes = max(1, manifest.files.reduce(0) { $0 + $1.bytes })
        var completedBytes = 0
        progress?(0)
        try Self.rejectRootSymlink(rootURL: cacheDirectory)
        try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        try Self.rejectRootSymlink(rootURL: cacheDirectory)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var mutableCache = cacheDirectory
        try mutableCache.setResourceValues(values)

        let legacyCompiledCache = cacheDirectory.appendingPathComponent("compiled", isDirectory: true)
        let compiledCache = cacheDirectory.appendingPathComponent(
            Self.compiledCacheDirectoryName,
            isDirectory: true
        )
        try Self.rejectExistingSymlinkComponents(
            rootURL: cacheDirectory,
            targetURL: legacyCompiledCache
        )
        if FileManager.default.fileExists(atPath: legacyCompiledCache.path) {
            try FileManager.default.removeItem(at: legacyCompiledCache)
        }
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
                completedBytes += file.bytes
                progress?(min(1, Double(completedBytes) / Double(totalBytes)))
                continue
            }
            try Self.rejectExistingSymlinkComponents(rootURL: cacheDirectory, targetURL: target.deletingLastPathComponent())
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Self.rejectExistingSymlinkComponents(rootURL: cacheDirectory, targetURL: target.deletingLastPathComponent())
            let fileURL = try Self.remoteURL(manifestURL: manifestURL, relativePath: file.path)
            let completedBeforeFile = completedBytes
            try await downloadFileWithRetry(
                fileURL,
                target: target,
                expectedBytes: file.bytes,
                expectedSHA256: file.sha256,
                label: file.path
            ) { downloadedBytes in
                let hydratedBytes = completedBeforeFile + min(downloadedBytes, file.bytes)
                progress?(min(1, Double(hydratedBytes) / Double(totalBytes)))
            }
            guard try await isValidFile(url: target, bytes: file.bytes, sha256: file.sha256) else {
                throw KokoroError.badHash(path: file.path)
            }
            completedBytes += file.bytes
            progress?(min(1, Double(completedBytes) / Double(totalBytes)))
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
    private func downloadWithRetry(
        _ url: URL,
        maxBytes: Int,
        label: String,
        progress: (@Sendable (Int) -> Void)? = nil
    ) async throws -> Data {
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
                return try await Self.downloadCappedData(
                    from: url,
                    maxBytes: maxBytes,
                    label: label,
                    progress: progress
                )
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

    /// Downloads one hosted file directly to disk with a small fixed retry budget.
    private func downloadFileWithRetry(
        _ url: URL,
        target: URL,
        expectedBytes: Int,
        expectedSHA256: String,
        label: String,
        progress: (@Sendable (Int) -> Void)? = nil
    ) async throws {
        let maxBytes = min(expectedBytes, limits.maxFileBytes)
        if url.isFileURL {
            try Task.checkCancellation()
            let values = try url.resourceValues(forKeys: [.fileSizeKey])
            let byteCount = values.fileSize ?? 0
            guard byteCount <= maxBytes else {
                throw KokoroError.downloadTooLarge(path: label, bytes: byteCount, maxBytes: maxBytes)
            }
            guard byteCount == expectedBytes,
                  try Self.fileSHA256(url) == expectedSHA256 else {
                throw KokoroError.badHash(path: label)
            }
            if FileManager.default.fileExists(atPath: target.path) {
                try FileManager.default.removeItem(at: target)
            }
            try FileManager.default.copyItem(at: url, to: target)
            return
        }
        var lastError: Error?
        for _ in 0..<3 {
            try Task.checkCancellation()
            do {
                try await Self.downloadCappedFile(
                    from: url,
                    to: target,
                    expectedBytes: expectedBytes,
                    expectedSHA256: expectedSHA256,
                    maxBytes: maxBytes,
                    label: label,
                    progress: progress
                )
                return
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

    /// Downloads an HTTP(S) response while enforcing the byte cap incrementally.
    private static func downloadCappedData(
        from url: URL,
        maxBytes: Int,
        label: String,
        progress: (@Sendable (Int) -> Void)? = nil
    ) async throws -> Data {
        let downloader = CappedDataDownloader(
            maxBytes: maxBytes,
            label: label,
            progress: progress
        )
        return try await downloader.download(from: url)
    }

    /// Downloads an HTTP(S) file to disk while enforcing the byte cap incrementally.
    private static func downloadCappedFile(
        from url: URL,
        to target: URL,
        expectedBytes: Int,
        expectedSHA256: String,
        maxBytes: Int,
        label: String,
        progress: (@Sendable (Int) -> Void)? = nil
    ) async throws {
        let downloader = CappedFileDownloader(
            target: target,
            expectedBytes: expectedBytes,
            expectedSHA256: expectedSHA256,
            maxBytes: maxBytes,
            label: label,
            progress: progress
        )
        try await downloader.download(from: url)
    }

    /// URLSession delegate that rejects oversized responses before buffering them fully.
    final class CappedDataDownloader: NSObject, URLSessionDataDelegate, @unchecked Sendable {
        /// Maximum accepted response bytes.
        private let maxBytes: Int

        /// Human-readable download label used in errors.
        private let label: String

        /// Optional streamed byte-progress observer.
        private let progress: (@Sendable (Int) -> Void)?

        /// Accumulated response data, bounded by `maxBytes`.
        private var data = Data()

        /// Active async continuation.
        private var continuation: CheckedContinuation<Data, Error>?

        /// Active URLSession, retained until completion.
        private var session: URLSession?

        /// Active data task, cancelled on byte-limit failures.
        private var task: URLSessionDataTask?

        /// Error detected before URLSession's completion callback.
        private var terminalError: Error?

        /// Creates a capped downloader.
        ///
        /// - Parameters:
        ///   - maxBytes: Maximum accepted response byte count.
        ///   - label: Human-readable path or manifest label for errors.
        init(
            maxBytes: Int,
            label: String,
            progress: (@Sendable (Int) -> Void)? = nil
        ) {
            self.maxBytes = maxBytes
            self.label = label
            self.progress = progress
        }

        /// Starts a capped data download.
        ///
        /// - Parameter url: HTTP(S) URL to download.
        /// - Returns: Response bytes when status and size are valid.
        func download(from url: URL) async throws -> Data {
            try Task.checkCancellation()
            return try await withTaskCancellationHandler {
                try Task.checkCancellation()
                return try await withCheckedThrowingContinuation {
                    (continuation: CheckedContinuation<Data, Error>) in
                    guard !Task.isCancelled else {
                        continuation.resume(throwing: CancellationError())
                        return
                    }
                    self.continuation = continuation
                    let session = URLSession(configuration: .ephemeral, delegate: self, delegateQueue: nil)
                    self.session = session
                    let task = session.dataTask(with: url)
                    self.task = task
                    if Task.isCancelled {
                        task.cancel()
                    } else {
                        task.resume()
                    }
                }
            } onCancel: {
                self.task?.cancel()
            }
        }

        /// Validates status and declared content length before accepting bytes.
        func urlSession(
            _ session: URLSession,
            dataTask: URLSessionDataTask,
            didReceive response: URLResponse,
            completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
        ) {
            if let http = response as? HTTPURLResponse,
               !(200..<300).contains(http.statusCode) {
                terminalError = URLError(.badServerResponse)
                completionHandler(.cancel)
                return
            }
            if response.expectedContentLength > Int64(maxBytes) {
                terminalError = KokoroError.downloadTooLarge(
                    path: label,
                    bytes: Int(response.expectedContentLength),
                    maxBytes: maxBytes
                )
                completionHandler(.cancel)
                return
            }
            completionHandler(.allow)
        }

        /// Appends one streamed response chunk if it remains inside the byte cap.
        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive chunk: Data) {
            let nextCount = data.count + chunk.count
            guard nextCount <= maxBytes else {
                terminalError = KokoroError.downloadTooLarge(path: label, bytes: nextCount, maxBytes: maxBytes)
                dataTask.cancel()
                return
            }
            data.append(chunk)
            progress?(data.count)
        }

        /// Resumes the async caller with either the bounded data or the first terminal error.
        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
            session.invalidateAndCancel()
            defer {
                continuation = nil
                self.session = nil
                self.task = nil
            }
            if let terminalError {
                continuation?.resume(throwing: terminalError)
            } else if let error {
                continuation?.resume(throwing: error)
            } else {
                continuation?.resume(returning: data)
            }
        }
    }

    /// URLSession delegate that downloads a file to disk without buffering it in memory.
    final class CappedFileDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
        /// Final target URL under the SDK cache root.
        private let target: URL

        /// Expected byte count from the hosted manifest.
        private let expectedBytes: Int

        /// Expected SHA-256 digest from the hosted manifest.
        private let expectedSHA256: String

        /// Maximum accepted response bytes.
        private let maxBytes: Int

        /// Human-readable download label used in errors.
        private let label: String

        /// Optional streamed byte-progress observer.
        private let progress: (@Sendable (Int) -> Void)?

        /// Active async continuation.
        private var continuation: CheckedContinuation<Void, Error>?

        /// Active URLSession, retained until completion.
        private var session: URLSession?

        /// Active download task, cancelled on byte-limit failures.
        private var task: URLSessionDownloadTask?

        /// Error detected before URLSession's completion callback.
        private var terminalError: Error?

        /// Whether `didFinishDownloadingTo` moved a verified file into place.
        private var finished = false

        /// Creates a capped file downloader.
        ///
        /// - Parameters:
        ///   - target: Final cache file URL.
        ///   - expectedBytes: Expected final byte count.
        ///   - expectedSHA256: Expected final SHA-256 digest.
        ///   - maxBytes: Maximum accepted response byte count.
        ///   - label: Human-readable path or manifest label for errors.
        init(
            target: URL,
            expectedBytes: Int,
            expectedSHA256: String,
            maxBytes: Int,
            label: String,
            progress: (@Sendable (Int) -> Void)?
        ) {
            self.target = target
            self.expectedBytes = expectedBytes
            self.expectedSHA256 = expectedSHA256
            self.maxBytes = maxBytes
            self.label = label
            self.progress = progress
        }

        /// Starts a capped disk-backed download.
        ///
        /// - Parameter url: HTTP(S) URL to download.
        func download(from url: URL) async throws {
            try Task.checkCancellation()
            try await withTaskCancellationHandler {
                try Task.checkCancellation()
                try await withCheckedThrowingContinuation {
                    (continuation: CheckedContinuation<Void, Error>) in
                    guard !Task.isCancelled else {
                        continuation.resume(throwing: CancellationError())
                        return
                    }
                    self.continuation = continuation
                    let session = URLSession(configuration: .ephemeral, delegate: self, delegateQueue: nil)
                    self.session = session
                    let task = session.downloadTask(with: url)
                    self.task = task
                    if Task.isCancelled {
                        task.cancel()
                    } else {
                        task.resume()
                    }
                }
            } onCancel: {
                self.task?.cancel()
            }
        }

        /// Cancels the task as soon as URLSession reports the byte cap is exceeded.
        func urlSession(
            _ session: URLSession,
            downloadTask: URLSessionDownloadTask,
            didWriteData bytesWritten: Int64,
            totalBytesWritten: Int64,
            totalBytesExpectedToWrite: Int64
        ) {
            guard totalBytesWritten <= Int64(maxBytes) else {
                terminalError = KokoroError.downloadTooLarge(
                    path: label,
                    bytes: Int(totalBytesWritten),
                    maxBytes: maxBytes
                )
                downloadTask.cancel()
                return
            }
            progress?(Int(totalBytesWritten))
        }

        /// Verifies the temporary file and moves it into the app cache.
        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
            do {
                let values = try location.resourceValues(forKeys: [.fileSizeKey])
                let byteCount = values.fileSize ?? 0
                guard byteCount == expectedBytes else {
                    throw KokoroError.badHash(path: label)
                }
                guard byteCount <= maxBytes else {
                    throw KokoroError.downloadTooLarge(path: label, bytes: byteCount, maxBytes: maxBytes)
                }
                guard try KokoroDownloadedModelStore.fileSHA256(location) == expectedSHA256 else {
                    throw KokoroError.badHash(path: label)
                }
                if FileManager.default.fileExists(atPath: target.path) {
                    try FileManager.default.removeItem(at: target)
                }
                try FileManager.default.moveItem(at: location, to: target)
                finished = true
            } catch {
                terminalError = error
            }
        }

        /// Resumes the async caller after URLSession completes or fails.
        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
            session.invalidateAndCancel()
            defer {
                continuation = nil
                self.session = nil
                self.task = nil
            }
            if let http = task.response as? HTTPURLResponse,
               !(200..<300).contains(http.statusCode),
               terminalError == nil {
                continuation?.resume(throwing: URLError(.badServerResponse))
            } else if let terminalError {
                continuation?.resume(throwing: terminalError)
            } else if let error {
                continuation?.resume(throwing: error)
            } else if !finished {
                continuation?.resume(throwing: URLError(.cannotDecodeContentData))
            } else {
                continuation?.resume()
            }
        }
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
        let fileValues = try url.resourceValues(forKeys: [.fileSizeKey])
        guard fileValues.fileSize == bytes else {
            return false
        }
        return try Self.fileSHA256(url) == sha256
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

    /// Computes a lowercase SHA-256 hex digest by streaming file bytes.
    private static func fileSHA256(_ url: URL) throws -> String {
        guard let stream = InputStream(url: url) else {
            throw URLError(.cannotOpenFile)
        }
        stream.open()
        defer { stream.close() }
        var digest = SHA256()
        let bufferSize = 1024 * 1024
        var buffer = [UInt8](repeating: 0, count: bufferSize)
        while stream.hasBytesAvailable {
            let count = stream.read(&buffer, maxLength: bufferSize)
            if count < 0 {
                throw stream.streamError ?? URLError(.cannotDecodeContentData)
            }
            if count == 0 {
                break
            }
            digest.update(data: Data(buffer.prefix(count)))
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
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
