import CryptoKit
import Foundation
import XCTest
@testable import KokoroTTS

/// Guards durable resume for interrupted hosted-file transfers.
///
/// A failed transfer used to throw away every byte it had received, so a user on
/// a flaky connection could never finish the 67 MB acoustic weights: each retry
/// restarted at zero. These tests pin both halves of the fix — resume when the
/// server object is unchanged, restart when it is not — against a stub
/// `URLProtocol` so no network is involved.
final class DownloadResumeTests: XCTestCase {
    /// Full hosted-file body used by both fixtures.
    private let body = Data((0..<4_096).map { UInt8($0 % 253) })

    /// Byte count already on disk before the resumed attempt.
    private let prefixBytes = 1_024

    override func tearDown() {
        StubTransport.shared.reset()
        super.tearDown()
    }

    /// Verifies an interrupted transfer continues from the part file with a range request.
    func testInterruptedDownloadResumesFromPartFileWithRangeHeader() async throws {
        let target = try makeTargetURL()
        try writePartialTransfer(target: target, bytes: body.prefix(prefixBytes), etag: "\"v1\"")
        let fullBody = body
        let resumeOffset = prefixBytes
        StubTransport.shared.handler = { request in
            guard request.value(forHTTPHeaderField: "Range") == "bytes=\(resumeOffset)-",
                  request.value(forHTTPHeaderField: "If-Range") == "\"v1\"" else {
                return StubHTTPResponse(
                    statusCode: 200,
                    headers: [
                        "ETag": "\"v1\"",
                        "Content-Length": "\(fullBody.count)",
                    ],
                    body: fullBody
                )
            }
            let remainder = fullBody.suffix(from: resumeOffset)
            return StubHTTPResponse(
                statusCode: 206,
                headers: [
                    "ETag": "\"v1\"",
                    "Accept-Ranges": "bytes",
                    "Content-Range": "bytes \(resumeOffset)-\(fullBody.count - 1)/\(fullBody.count)",
                    "Content-Length": "\(remainder.count)",
                ],
                body: Data(remainder)
            )
        }

        try await makeDownloader(target: target).download(from: Self.remoteURL)

        XCTAssertEqual(try Data(contentsOf: target), body)
        XCTAssertEqual(
            StubTransport.shared.requests.first?.value(forHTTPHeaderField: "Range"),
            "bytes=\(prefixBytes)-"
        )
        assertNoPartialTransferRemains(for: target)
    }

    /// Verifies stale partial bytes are discarded when the server object changed.
    func testResumeDiscardsPartFileOnETagMismatch() async throws {
        let target = try makeTargetURL()
        // Deliberately wrong bytes: if they were ever prepended to the new body
        // the promoted file would fail its manifest digest.
        try writePartialTransfer(
            target: target,
            bytes: Data(repeating: 0xFF, count: prefixBytes),
            etag: "\"v1\""
        )
        let fullBody = body
        StubTransport.shared.handler = { _ in
            // `If-Range` no longer matches, so a compliant server ignores the
            // range and replies with the whole current object.
            StubHTTPResponse(
                statusCode: 200,
                headers: [
                    "ETag": "\"v2\"",
                    "Accept-Ranges": "bytes",
                    "Content-Length": "\(fullBody.count)",
                ],
                body: fullBody
            )
        }

        try await makeDownloader(target: target).download(from: Self.remoteURL)

        XCTAssertEqual(try Data(contentsOf: target), body)
        XCTAssertEqual(
            StubTransport.shared.requests.first?.value(forHTTPHeaderField: "If-Range"),
            "\"v1\"",
            "the attempt should still have offered the stale ETag"
        )
        assertNoPartialTransferRemains(for: target)
    }

    /// Remote URL used by every fixture.
    private static let remoteURL = URL(string: "https://models.example.test/voices/af_heart.bin")!

    /// Creates a stub-backed downloader for one target path.
    ///
    /// - Parameter target: Final cache file URL.
    /// - Returns: Downloader pinned to ``body``'s size and digest.
    private func makeDownloader(target: URL) -> KokoroDownloadedModelStore.CappedFileDownloader {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        return KokoroDownloadedModelStore.CappedFileDownloader(
            target: target,
            expectedBytes: body.count,
            expectedSHA256: SHA256.hash(data: body).map { String(format: "%02x", $0) }.joined(),
            maxBytes: body.count,
            label: "voices/af_heart.bin",
            sessionConfiguration: configuration
        )
    }

    /// Creates an empty cache directory and returns the file path inside it.
    ///
    /// - Returns: Final cache file URL.
    private func makeTargetURL() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent("af_heart.bin")
    }

    /// Seeds a durable partial transfer and its recorded ETag.
    ///
    /// - Parameters:
    ///   - target: Final cache file URL.
    ///   - bytes: Partial prefix already on disk.
    ///   - etag: ETag the prefix was fetched under.
    private func writePartialTransfer(target: URL, bytes: some DataProtocol, etag: String) throws {
        try Data(bytes).write(to: KokoroDownloadedModelStore.partialURL(for: target))
        try Data("\(etag)\n".utf8).write(to: KokoroDownloadedModelStore.partialETagURL(for: target))
    }

    /// Asserts a promoted transfer left no part file or ETag sidecar behind.
    ///
    /// - Parameter target: Final cache file URL.
    private func assertNoPartialTransferRemains(for target: URL) {
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: KokoroDownloadedModelStore.partialURL(for: target).path
            )
        )
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: KokoroDownloadedModelStore.partialETagURL(for: target).path
            )
        )
    }
}

/// One canned HTTP reply served by ``StubURLProtocol``.
struct StubHTTPResponse {
    /// HTTP status code.
    let statusCode: Int

    /// Response header fields.
    let headers: [String: String]

    /// Response body bytes.
    let body: Data
}

/// Thread-safe state shared with ``StubURLProtocol``.
///
/// `URLProtocol` subclasses are instantiated by URLSession, so the handler and
/// the request log have to live outside the instance.
final class StubTransport: @unchecked Sendable {
    /// Process-wide stub state.
    static let shared = StubTransport()

    /// Guards every mutable member.
    private let lock = NSLock()

    /// Reply builder for the next requests.
    private var storedHandler: (@Sendable (URLRequest) -> StubHTTPResponse)?

    /// Requests observed in arrival order.
    private var storedRequests: [URLRequest] = []

    /// Reply builder for incoming requests.
    var handler: (@Sendable (URLRequest) -> StubHTTPResponse)? {
        get {
            lock.lock()
            defer { lock.unlock() }
            return storedHandler
        }
        set {
            lock.lock()
            storedHandler = newValue
            lock.unlock()
        }
    }

    /// Requests observed so far.
    var requests: [URLRequest] {
        lock.lock()
        defer { lock.unlock() }
        return storedRequests
    }

    /// Records one observed request.
    ///
    /// - Parameter request: Request URLSession is about to serve from the stub.
    func record(_ request: URLRequest) {
        lock.lock()
        storedRequests.append(request)
        lock.unlock()
    }

    /// Clears the handler and request log between tests.
    func reset() {
        lock.lock()
        storedHandler = nil
        storedRequests = []
        lock.unlock()
    }
}

/// Serves ``StubTransport`` replies in place of a real network stack.
final class StubURLProtocol: URLProtocol {
    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        StubTransport.shared.record(request)
        guard let handler = StubTransport.shared.handler,
              let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.unsupportedURL))
            return
        }
        let stub = handler(request)
        guard let response = HTTPURLResponse(
            url: url,
            statusCode: stub.statusCode,
            httpVersion: "HTTP/1.1",
            headerFields: stub.headers
        ) else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        if !stub.body.isEmpty {
            client?.urlProtocol(self, didLoad: stub.body)
        }
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {
    }
}
