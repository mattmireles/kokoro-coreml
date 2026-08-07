import Foundation

extension KokoroDownloadedModelStore {
    /// Filename suffix for a durable partial transfer.
    static let partialSuffix = "part"

    /// Filename suffix for the ETag recorded beside a partial transfer.
    static let partialETagSuffix = "part-etag"

    /// Durable partial-transfer file for a final cache path.
    ///
    /// - Parameter target: Final cache file URL.
    /// - Returns: Sibling `.part` URL.
    static func partialURL(for target: URL) -> URL {
        target.appendingPathExtension(partialSuffix)
    }

    /// Sidecar recording which server object a partial transfer came from.
    ///
    /// - Parameter target: Final cache file URL.
    /// - Returns: Sibling `.part-etag` URL.
    static func partialETagURL(for target: URL) -> URL {
        target.appendingPathExtension(partialETagSuffix)
    }

    /// Removes any partial transfer recorded for a cache path.
    ///
    /// Existence is probed with `attributesOfItem`, which uses `lstat`: a
    /// dangling or hostile symlink is therefore removed rather than followed,
    /// which `fileExists(atPath:)` would do.
    ///
    /// - Parameter target: Final cache file URL.
    static func discardPartialTransfer(for target: URL) {
        let manager = FileManager.default
        for url in [partialURL(for: target), partialETagURL(for: target)] {
            guard (try? manager.attributesOfItem(atPath: url.path)) != nil else {
                continue
            }
            try? manager.removeItem(at: url)
        }
    }

    /// Returns whether a path is itself a symbolic link.
    ///
    /// - Parameter url: Path to inspect without following links.
    /// - Returns: True when the item at `url` is a symlink.
    static func isPartialSymbolicLink(_ url: URL) -> Bool {
        (try? FileManager.default.destinationOfSymbolicLink(atPath: url.path)) != nil
    }

    /// Streams one hosted file to `<target>.part` and resumes interrupted transfers.
    ///
    /// `URLSessionDownloadTask` keeps its temporary file private to URLSession
    /// and discards it whenever the task fails, so an interrupted 67 MB weight
    /// transfer restarted from byte zero on the next attempt — the failure mode
    /// users saw as "it downloads the model again". Writing into a `.part`
    /// sibling of the final cache path instead makes progress durable across
    /// in-process retries and across app launches.
    ///
    /// The ETag recorded next to the part file is the resume key. `models.gist.is`
    /// serves immutable ETags with `accept-ranges: bytes`, so an unchanged ETag
    /// proves the remaining bytes belong to the same object. The request carries
    /// both `Range` and `If-Range`, which lets a compliant server answer a stale
    /// resume with a plain `200` full body; that response truncates the part file
    /// and starts over. Anything inconsistent is discarded rather than repaired,
    /// because a wrongly stitched model file would only surface as a hash failure
    /// after the whole transfer completed.
    ///
    /// Deliberately out of scope: background `URLSession`. Resume here covers
    /// retry and relaunch, not transfers that continue while the app is
    /// suspended.
    final class CappedFileDownloader: NSObject, URLSessionDataDelegate, @unchecked Sendable {
        /// Final target URL under the SDK cache root.
        private let target: URL

        /// Expected byte count from the hosted manifest.
        private let expectedBytes: Int

        /// Expected SHA-256 digest from the hosted manifest.
        private let expectedSHA256: String

        /// Maximum accepted total bytes.
        private let maxBytes: Int

        /// Human-readable download label used in errors.
        private let label: String

        /// Optional streamed byte-progress observer.
        private let progress: (@Sendable (Int) -> Void)?

        /// Session configuration used for the transfer.
        ///
        /// Injectable so regression tests can install a stub `URLProtocol`
        /// instead of reaching the network.
        private let sessionConfiguration: URLSessionConfiguration

        /// Active async continuation.
        private var continuation: CheckedContinuation<Void, Error>?

        /// Active URLSession, retained until completion.
        private var session: URLSession?

        /// Active data task, cancelled on byte-limit failures.
        private var task: URLSessionDataTask?

        /// Error detected before URLSession's completion callback.
        private var terminalError: Error?

        /// Open handle on the part file while bytes stream in.
        private var handle: FileHandle?

        /// Byte offset the current response continues from.
        private var resumeOffset = 0

        /// Total bytes on disk in the part file, including the resumed prefix.
        private var writtenBytes = 0

        /// ETag recorded beside an existing part file, if one is resumable.
        private var recordedETag: String?

        /// Synchronizes installation/cancellation of the active task.
        private let stateLock = NSLock()

        /// Remembers cancellation that arrives before the task is installed.
        private var cancellationRequested = false

        /// Creates a capped, resumable file downloader.
        ///
        /// - Parameters:
        ///   - target: Final cache file URL.
        ///   - expectedBytes: Expected final byte count.
        ///   - expectedSHA256: Expected final SHA-256 digest.
        ///   - maxBytes: Maximum accepted total byte count.
        ///   - label: Human-readable path or manifest label for errors.
        ///   - sessionConfiguration: URLSession configuration for the transfer.
        ///   - progress: Optional streamed byte-progress observer.
        init(
            target: URL,
            expectedBytes: Int,
            expectedSHA256: String,
            maxBytes: Int,
            label: String,
            sessionConfiguration: URLSessionConfiguration = .ephemeral,
            progress: (@Sendable (Int) -> Void)? = nil
        ) {
            self.target = target
            self.expectedBytes = expectedBytes
            self.expectedSHA256 = expectedSHA256
            self.maxBytes = maxBytes
            self.label = label
            self.sessionConfiguration = sessionConfiguration
            self.progress = progress
        }

        /// Starts a capped disk-backed download, resuming when possible.
        ///
        /// - Parameter url: HTTP(S) URL to download.
        func download(from url: URL) async throws {
            try Task.checkCancellation()
            let request = makeRequest(url: url)
            try await withTaskCancellationHandler {
                try Task.checkCancellation()
                try await withCheckedThrowingContinuation {
                    (continuation: CheckedContinuation<Void, Error>) in
                    guard !Task.isCancelled else {
                        continuation.resume(throwing: CancellationError())
                        return
                    }
                    let session = URLSession(
                        configuration: sessionConfiguration,
                        delegate: self,
                        delegateQueue: nil
                    )
                    let task = session.dataTask(with: request)
                    self.stateLock.lock()
                    self.continuation = continuation
                    self.session = session
                    self.task = task
                    let shouldCancel = self.cancellationRequested || Task.isCancelled
                    self.stateLock.unlock()
                    if shouldCancel {
                        task.cancel()
                    } else {
                        task.resume()
                    }
                }
            } onCancel: {
                self.stateLock.lock()
                self.cancellationRequested = true
                let task = self.task
                self.stateLock.unlock()
                task?.cancel()
            }
        }

        /// Builds the transfer request, resuming from `<target>.part` when usable.
        ///
        /// - Parameter url: HTTP(S) URL to download.
        /// - Returns: Request carrying `Range`/`If-Range` when a resume is possible.
        func makeRequest(url: URL) -> URLRequest {
            var request = URLRequest(url: url)
            guard let resume = resumableState() else {
                KokoroDownloadedModelStore.discardPartialTransfer(for: target)
                resumeOffset = 0
                recordedETag = nil
                return request
            }
            resumeOffset = resume.bytes
            recordedETag = resume.etag
            request.setValue("bytes=\(resume.bytes)-", forHTTPHeaderField: "Range")
            request.setValue(resume.etag, forHTTPHeaderField: "If-Range")
            return request
        }

        /// Reads a resumable part file and its recorded ETag.
        ///
        /// A part file at or beyond the expected size cannot be extended by a
        /// range request, so it is treated as unusable rather than repaired.
        ///
        /// - Returns: Resumable byte count and ETag, or `nil` to start over.
        private func resumableState() -> (bytes: Int, etag: String)? {
            let partURL = KokoroDownloadedModelStore.partialURL(for: target)
            let etagURL = KokoroDownloadedModelStore.partialETagURL(for: target)
            guard !KokoroDownloadedModelStore.isPartialSymbolicLink(partURL),
                  !KokoroDownloadedModelStore.isPartialSymbolicLink(etagURL),
                  let values = try? partURL.resourceValues(
                      forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey]
                  ),
                values.isSymbolicLink != true,
                values.isRegularFile == true,
                let bytes = values.fileSize,
                bytes > 0,
                bytes < expectedBytes,
                let etag = try? String(contentsOf: etagURL, encoding: .utf8)
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                !etag.isEmpty
            else {
                return nil
            }
            return (bytes, etag)
        }

        /// Validates the response, then opens the part file for the incoming body.
        func urlSession(
            _ session: URLSession,
            dataTask: URLSessionDataTask,
            didReceive response: URLResponse,
            completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
        ) {
            do {
                try acceptResponse(response)
                completionHandler(.allow)
            } catch {
                terminalError = error
                completionHandler(.cancel)
            }
        }

        /// Decides whether a response continues, restarts, or fails the transfer.
        ///
        /// - Parameter response: Response received before any body bytes.
        private func acceptResponse(_ response: URLResponse) throws {
            guard let http = response as? HTTPURLResponse else {
                throw URLError(.badServerResponse)
            }
            let etag = http.value(forHTTPHeaderField: "ETag")
            switch http.statusCode {
            case 206:
                // A partial reply is only usable when it continues the exact
                // object the part file came from.
                guard resumeOffset > 0, etag == nil || etag == recordedETag else {
                    KokoroDownloadedModelStore.discardPartialTransfer(for: target)
                    resumeOffset = 0
                    throw URLError(.badServerResponse)
                }
            case 200..<300:
                // Full body: either there was nothing to resume, or the server
                // rejected `If-Range` because the object changed. Either way the
                // recorded bytes are stale and must not be prepended.
                KokoroDownloadedModelStore.discardPartialTransfer(for: target)
                resumeOffset = 0
            default:
                throw URLError(.badServerResponse)
            }
            let declared = response.expectedContentLength
            if declared > 0, resumeOffset + Int(declared) > maxBytes {
                throw KokoroError.downloadTooLarge(
                    path: label,
                    bytes: resumeOffset + Int(declared),
                    maxBytes: maxBytes
                )
            }
            try openPartFile()
            if let etag {
                try? Data("\(etag)\n".utf8).write(
                    to: KokoroDownloadedModelStore.partialETagURL(for: target),
                    options: .atomic
                )
            }
        }

        /// Opens the part file positioned exactly at the resume offset.
        private func openPartFile() throws {
            let manager = FileManager.default
            let partURL = KokoroDownloadedModelStore.partialURL(for: target)
            try manager.createDirectory(
                at: partURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            // Never write through a symlink planted at the part path.
            guard !KokoroDownloadedModelStore.isPartialSymbolicLink(partURL) else {
                throw KokoroError.pathEscape(partURL.path)
            }
            if !manager.fileExists(atPath: partURL.path) {
                manager.createFile(atPath: partURL.path, contents: nil)
            }
            let handle = try FileHandle(forWritingTo: partURL)
            // Truncating first covers both a fresh start and a part file that is
            // somehow longer than the offset the server agreed to continue from.
            try handle.truncate(atOffset: UInt64(resumeOffset))
            try handle.seek(toOffset: UInt64(resumeOffset))
            self.handle = handle
            writtenBytes = resumeOffset
        }

        /// Appends one streamed response chunk if it remains inside the byte cap.
        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive chunk: Data) {
            guard terminalError == nil, let handle else {
                return
            }
            let nextCount = writtenBytes + chunk.count
            guard nextCount <= maxBytes else {
                terminalError = KokoroError.downloadTooLarge(
                    path: label,
                    bytes: nextCount,
                    maxBytes: maxBytes
                )
                dataTask.cancel()
                return
            }
            do {
                try handle.write(contentsOf: chunk)
            } catch {
                terminalError = error
                dataTask.cancel()
                return
            }
            writtenBytes = nextCount
            progress?(writtenBytes)
        }

        /// Resumes the async caller after URLSession completes or fails.
        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
            session.invalidateAndCancel()
            try? handle?.close()
            handle = nil
            let outcome = resolveOutcome(task: task, transportError: error)
            stateLock.lock()
            let continuation = continuation
            self.continuation = nil
            self.session = nil
            self.task = nil
            stateLock.unlock()
            if let outcome {
                continuation?.resume(throwing: outcome)
            } else {
                continuation?.resume()
            }
        }

        /// Finalizes the transfer, keeping the part file when a retry can resume it.
        ///
        /// - Parameters:
        ///   - task: Completed URLSession task.
        ///   - transportError: URLSession's own error, if any.
        /// - Returns: Error to surface, or `nil` when the file is now in place.
        private func resolveOutcome(task: URLSessionTask, transportError: Error?) -> Error? {
            if let terminalError {
                return terminalError
            }
            if let transportError {
                // Interrupted mid-flight. The part file and its ETag stay on
                // disk so the next attempt resumes instead of restarting.
                return transportError
            }
            if let http = task.response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                return URLError(.badServerResponse)
            }
            do {
                try promotePartFile()
                return nil
            } catch {
                return error
            }
        }

        /// Verifies the finished part file and moves it onto the final cache path.
        private func promotePartFile() throws {
            let partURL = KokoroDownloadedModelStore.partialURL(for: target)
            let values = try partURL.resourceValues(forKeys: [.fileSizeKey])
            let byteCount = values.fileSize ?? 0
            if byteCount < expectedBytes {
                // A short body is still a valid prefix, so keep it for the next
                // attempt rather than throwing away real progress.
                throw URLError(.cannotDecodeContentData)
            }
            guard byteCount == expectedBytes,
                  try KokoroFileDigest.sha256(ofFileAt: partURL) == expectedSHA256 else {
                KokoroDownloadedModelStore.discardPartialTransfer(for: target)
                throw KokoroError.badHash(path: label)
            }
            let manager = FileManager.default
            if manager.fileExists(atPath: target.path) {
                try manager.removeItem(at: target)
            }
            try manager.moveItem(at: partURL, to: target)
            try? manager.removeItem(at: KokoroDownloadedModelStore.partialETagURL(for: target))
        }
    }
}
