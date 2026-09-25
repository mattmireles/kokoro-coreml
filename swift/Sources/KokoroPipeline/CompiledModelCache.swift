/// Compiles `.mlpackage` files, optionally keeping the compiled bundles across launches.
///
/// `MLModel.compileModel` writes into a temporary directory the system may
/// reclaim, so a compile that is not moved somewhere durable is paid again on
/// every launch. On iPadOS that is the whole model set: compiling this
/// pipeline's packages takes minutes, and the botnet worker is unreachable for
/// all of it (`botnet/apps/ios-worker`, `KokoroNativeRuntime.swift`).
///
/// The cache is opt-in: a nil directory compiles into a temporary directory
/// every time, which is the historical behaviour. The caller owns the
/// directory's identity and should derive it from the model set's digest,
/// since a cached bundle is keyed only by its package's file name and cannot
/// tell a stale compile from a current one.

import CoreML
import Foundation

enum CompiledModelCache {
    /// Loads `package`, reusing a cached compile when `cache` is set.
    ///
    /// A cached bundle that fails to load is deleted and recompiled rather
    /// than thrown: the common causes (an interrupted move, an OS upgrade that
    /// changed the compiled format) are all repaired by compiling again.
    static func load(package: URL, configuration: MLModelConfiguration, cache: URL?) throws -> MLModel {
        guard let cache else {
            return try MLModel(contentsOf: MLModel.compileModel(at: package), configuration: configuration)
        }
        let cached = cachedURL(for: package, in: cache)
        if FileManager.default.fileExists(atPath: cached.path) {
            if let model = try? MLModel(contentsOf: cached, configuration: configuration) {
                return model
            }
            try? FileManager.default.removeItem(at: cached)
        }
        return try MLModel(contentsOf: compile(package, into: cache), configuration: configuration)
    }

    /// The compiled bundle for `package`: the cached one when present, else a fresh compile.
    ///
    /// For callers that open the bundle themselves (a multifunction package is
    /// opened once and then loaded function by function). `isUsable` rejects a
    /// cached bundle that no longer opens, which is then recompiled.
    static func compiledURL(for package: URL, cache: URL?, isUsable: (URL) -> Bool) throws -> URL {
        guard let cache else { return try MLModel.compileModel(at: package) }
        let cached = cachedURL(for: package, in: cache)
        if FileManager.default.fileExists(atPath: cached.path) {
            if isUsable(cached) { return cached }
            try? FileManager.default.removeItem(at: cached)
        }
        return try compile(package, into: cache)
    }

    private static func cachedURL(for package: URL, in cache: URL) -> URL {
        cache.appendingPathComponent(package.deletingPathExtension().lastPathComponent)
            .appendingPathExtension("mlmodelc")
    }

    /// Compiles `package` and moves the bundle into `cache`. A cache that
    /// cannot be written is a slow pipeline, not a broken one: the temporary
    /// compile is returned and the next launch retries the move.
    private static func compile(_ package: URL, into cache: URL) throws -> URL {
        let compiled = try MLModel.compileModel(at: package)
        let cached = cachedURL(for: package, in: cache)
        do {
            try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: cached)
            try FileManager.default.moveItem(at: compiled, to: cached)
            return cached
        } catch {
            return compiled
        }
    }
}
