import CryptoKit
import Foundation

/// Streaming SHA-256 helpers plus a process-wide memo of already-verified files.
///
/// The hosted-bundle hydrator and the Core ML model provider verify the exact
/// same bytes on disk. Before this memo existed, one listen pass hashed the
/// runtime bundle twice — once in ``KokoroDownloadedModelStore/hydrate(phaseProgress:)``
/// and again in every freshly constructed `KokoroSDKModelProvider` — roughly
/// 350 MB of SHA-256 for a 165 MB bundle, repeated on every preparation. Apps
/// that drop and rebuild the facade between playbacks paid that cost each time.
///
/// The memo is keyed on a cheap file identity (absolute path, byte size,
/// modification date), so an untouched file is hashed once per launch while any
/// on-disk change forces a real rehash. It is deliberately process-scoped: it
/// never persists across launches, so a corrupted bundle is always caught by the
/// first verification pass after a cold start.
enum KokoroFileDigest {
    /// Read granularity for streaming digests.
    ///
    /// One megabyte keeps peak allocation flat regardless of model weight size.
    /// Reading a 67 MB `weight.bin` with `Data(contentsOf:)` instead produced a
    /// 67 MB resident spike per validation pass on iPhone.
    private static let bufferSize = 1024 * 1024

    /// Maximum memoized entries before the whole memo is dropped.
    ///
    /// A hosted manifest is capped at ``KokoroDownloadLimits/maxFileCount``
    /// files, so this ceiling is only reachable if bundle files are rewritten
    /// many times inside one launch. Clearing wholesale keeps the memo from
    /// growing without bound while staying allocation-free in the normal case.
    private static let memoCapacity = 4096

    /// Guards every mutable member below.
    private static let lock = NSLock()

    /// File identity to verified lowercase SHA-256 hex digest.
    private static var digests: [String: String] = [:]

    /// Count of full-file digests actually computed this launch.
    private static var computedDigests = 0

    /// Number of full-file digests computed since launch.
    ///
    /// Exposed for regression tests that assert repeated preparation stops
    /// re-hashing an unchanged bundle. Production code never reads this.
    static var computedDigestCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return computedDigests
    }

    /// Returns a file's SHA-256, reusing this launch's value when it is unchanged.
    ///
    /// - Parameter url: Regular file to digest.
    /// - Returns: Lowercase SHA-256 hex digest.
    static func memoizedSHA256(ofFileAt url: URL) throws -> String {
        let identity = try identity(for: url)
        lock.lock()
        let cached = digests[identity]
        lock.unlock()
        if let cached {
            return cached
        }
        let digest = try sha256(ofFileAt: url)
        lock.lock()
        if digests.count >= memoCapacity {
            digests.removeAll(keepingCapacity: true)
        }
        digests[identity] = digest
        lock.unlock()
        return digest
    }

    /// Computes a lowercase SHA-256 hex digest by streaming file bytes.
    ///
    /// - Parameter url: Regular file to digest.
    /// - Returns: Lowercase SHA-256 hex digest.
    static func sha256(ofFileAt url: URL) throws -> String {
        guard let stream = InputStream(url: url) else {
            throw URLError(.cannotOpenFile)
        }
        stream.open()
        defer { stream.close() }
        var digest = SHA256()
        var buffer = [UInt8](repeating: 0, count: bufferSize)
        while stream.hasBytesAvailable {
            let count = stream.read(&buffer, maxLength: bufferSize)
            if count < 0 {
                throw stream.streamError ?? URLError(.cannotDecodeContentData)
            }
            if count == 0 {
                break
            }
            buffer.withUnsafeBytes { raw in
                digest.update(bufferPointer: UnsafeRawBufferPointer(rebasing: raw[0..<count]))
            }
        }
        lock.lock()
        computedDigests += 1
        lock.unlock()
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Builds a cheap change-detection identity for a regular file.
    ///
    /// Size plus modification date is what the filesystem can answer with a
    /// single `stat`. It is not a cryptographic identity, which is why the memo
    /// it feeds is process-scoped rather than persisted.
    ///
    /// - Parameter url: Regular file to describe.
    /// - Returns: Stable identity string for this path, size, and mtime.
    static func identity(for url: URL) throws -> String {
        let values = try url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
        let bytes = values.fileSize ?? -1
        let modified = values.contentModificationDate?.timeIntervalSince1970 ?? -1
        return "\(url.standardizedFileURL.path)|\(bytes)|\(modified)"
    }
}
