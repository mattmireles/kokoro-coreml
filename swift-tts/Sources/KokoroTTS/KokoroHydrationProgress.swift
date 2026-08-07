import Foundation

/// One hydration progress update, tagged with the work that produced it.
///
/// Hydration does two very different jobs behind one loop: it re-verifies bytes
/// that are already on disk, and it transfers bytes that are not. Both used to
/// arrive through the same `Double` callback, so an app could not tell a warm
/// cache sweep from a real transfer and captioned every preparation
/// "Downloading… 165 MB". ``Phase`` makes that distinction explicit.
public struct KokoroHydrationProgress: Sendable, Equatable {
    /// What hydration is doing while it reports a fraction.
    public enum Phase: Sendable, Equatable {
        /// Checking bytes already present in the cache, or fetching the tiny
        /// hosted manifest that says which bytes are expected. No payload byte
        /// leaves the network in this phase.
        case verifying

        /// Transferring at least one hosted payload file into the cache.
        case downloading
    }

    /// Work that produced this update.
    public let phase: Phase

    /// Hydration completion in `0...1`, measured in manifest bytes.
    public let fraction: Double

    /// Creates a hydration progress update.
    ///
    /// - Parameters:
    ///   - phase: Work that produced this update.
    ///   - fraction: Hydration completion in `0...1`.
    public init(phase: Phase, fraction: Double) {
        self.phase = phase
        self.fraction = fraction
    }
}

/// Serializes hydration progress and latches the downloading phase.
///
/// Hydration walks the manifest file by file, so a bundle that is half cached
/// would otherwise flip the caption between "verifying" and "downloading" many
/// times. Once any payload transfer starts, the rest of that hydration call
/// stays ``KokoroHydrationProgress/Phase/downloading``: a hydration that moves
/// bytes is a download from the user's point of view, however much of it was
/// already cached.
final class KokoroHydrationProgressReporter: @unchecked Sendable {
    /// Caller-supplied progress sink, or `nil` when the caller wants no updates.
    private let sink: (@Sendable (KokoroHydrationProgress) -> Void)?

    /// Guards ``downloading`` across URLSession delegate queues.
    private let lock = NSLock()

    /// Whether this hydration has begun transferring payload bytes.
    private var downloading = false

    /// Creates a reporter over an optional progress sink.
    ///
    /// - Parameter sink: Caller progress callback.
    init(sink: (@Sendable (KokoroHydrationProgress) -> Void)?) {
        self.sink = sink
    }

    /// Latches the downloading phase for the rest of this hydration.
    ///
    /// Called when hydration commits to fetching a file, not when the first
    /// response byte arrives: a stalled connect is still a download attempt, and
    /// the app needs its caption before bytes move.
    func beginDownloading() {
        lock.lock()
        downloading = true
        lock.unlock()
    }

    /// Emits one clamped progress update in the current phase.
    ///
    /// - Parameter fraction: Hydration completion, clamped into `0...1`.
    func report(fraction: Double) {
        guard let sink else {
            return
        }
        lock.lock()
        let phase: KokoroHydrationProgress.Phase = downloading ? .downloading : .verifying
        lock.unlock()
        sink(KokoroHydrationProgress(phase: phase, fraction: min(1, max(0, fraction))))
    }
}
