import Foundation

/// Path containment and symlink checks shared by the bundle provider and the
/// download store.
///
/// Containment is lexical (`URL.standardized`), never `standardizedFileURL`:
/// the latter drops a leading `/private` only for paths that exist, so a root
/// `/private/tmp/x` became `/tmp/x` while a not-yet-created child
/// `/private/tmp/x/compiled` kept its prefix and failed the check
/// (`kokoro-say --cache-dir /private/tmp/...` threw pathEscape).
enum KokoroPathGuard {
    /// Lexical absolute path: `.` and `..` removed, no filesystem lookups.
    /// `absoluteURL` first: a relative file URL standardizes against its base
    /// only then, so a leading `..` resolves against the working directory
    /// instead of being dropped.
    static func lexicalPath(_ url: URL) -> String {
        url.absoluteURL.standardized.path
    }

    /// Whether `target` is `root` or lies under it, compared lexically.
    static func isInside(_ target: URL, root: URL) -> Bool {
        let rootPath = lexicalPath(root)
        let targetPath = lexicalPath(target)
        return targetPath == rootPath || targetPath.hasPrefix(rootPath.hasSuffix("/") ? rootPath : "\(rootPath)/")
    }

    /// Resolves a manifest path under a root without accepting lexical escapes.
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
        guard isInside(url, root: rootURL) else {
            throw KokoroError.pathEscape(relativePath)
        }
        return url
    }

    /// Rejects symlinked components between `rootURL` and `targetURL` before
    /// reads or writes can follow them.
    static func rejectExistingSymlinkComponents(rootURL: URL, targetURL: URL) throws {
        guard isInside(targetURL, root: rootURL) else {
            throw KokoroError.pathEscape(lexicalPath(targetURL))
        }
        try rejectRootSymlink(rootURL: rootURL)
        let suffix = String(lexicalPath(targetURL).dropFirst(lexicalPath(rootURL).count)).split(separator: "/")
        var current = rootURL
        for component in suffix {
            current = current.appendingPathComponent(String(component))
            if isSymbolicLink(current) {
                throw KokoroError.pathEscape(current.path)
            }
        }
    }

    /// Rejects a root that is itself a symlink.
    static func rejectRootSymlink(rootURL: URL) throws {
        if isSymbolicLink(rootURL) {
            throw KokoroError.pathEscape(rootURL.path)
        }
    }

    /// Returns true when a URL is itself a symbolic link.
    static func isSymbolicLink(_ url: URL) -> Bool {
        (try? FileManager.default.destinationOfSymbolicLink(atPath: url.path)) != nil
    }
}
