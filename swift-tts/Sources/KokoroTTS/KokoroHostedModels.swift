import Foundation

/// The one checked-in pin of the hosted starter bundle on Hugging Face.
///
/// `kokoro-say` and ``KokoroResourceProvider/huggingFace(cacheDirectory:progress:)``
/// read only these values. A release updates both together: the manifest URL
/// names an immutable HF revision, and the SHA-256 is of that exact
/// `HostedManifest.json`, so a moved branch or edited file fails closed
/// with ``KokoroError/badHash(path:)`` instead of loading other models.
public enum KokoroHostedModels {
    /// Hugging Face model repository.
    public static let repoID = "mattmireles/kokoro-coreml"

    /// HF commit holding the schema-2 starter bundle (published 2026-10-23
    /// from kokoro-coreml e8bf3d4e by `prepare_hf_sdk_metadata.py --upload`).
    public static let revision = "034f7741bf938fd6b4b9c75c22bc72cd795f908a"

    /// Hosted manifest of the starter bundle at ``revision``.
    public static let manifestURL = URL(string: "https://huggingface.co/\(repoID)/resolve/\(revision)/HostedManifest.json")!

    /// SHA-256 of the exact `HostedManifest.json` bytes at ``revision``.
    public static let manifestSHA256 = "f3abd5811585b8c89f1c15e647409311731b4262da4315242c69f88675bced9d"

    /// Throws ``KokoroError/unpinnedHostedRevision`` unless `revision` is a
    /// 40-hex Hugging Face commit, so an unreleased build fails before any
    /// network call instead of with a 404 or hash mismatch.
    static func requirePinnedRevision(_ revision: String = revision) throws {
        guard revision.count == 40, revision.allSatisfy(\.isHexDigit) else {
            throw KokoroError.unpinnedHostedRevision(revision)
        }
    }

    /// Default download cache: `~/Library/Caches/kokoro-coreml` (the app's
    /// Caches directory on iOS). The system may purge it; the next call
    /// downloads again.
    public static var defaultCacheDirectory: URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return caches.appendingPathComponent("kokoro-coreml", isDirectory: true)
    }
}

public extension KokoroResourceProvider {
    /// Downloads (or re-verifies) the pinned starter bundle from Hugging Face
    /// into `cacheDirectory` and returns it as a resource provider.
    ///
    /// ```swift
    /// let tts = try await KokoroTTS.load(resources: .huggingFace())
    /// ```
    static func huggingFace(
        cacheDirectory: URL = KokoroHostedModels.defaultCacheDirectory,
        progress: (@Sendable (KokoroHydrationProgress) -> Void)? = nil
    ) async throws -> KokoroResourceProvider {
        try KokoroHostedModels.requirePinnedRevision()
        return try await KokoroDownloadedModelStore(
            manifestURL: KokoroHostedModels.manifestURL,
            expectedManifestSHA256: KokoroHostedModels.manifestSHA256,
            cacheDirectory: cacheDirectory
        ).hydrate(phaseProgress: progress)
    }
}
