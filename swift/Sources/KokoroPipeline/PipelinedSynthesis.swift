import CoreML
import Foundation

/// Runs several chunks of one document with two chunks in flight: while chunk
/// i runs its bucket stages (F0Ntrain, DecoderPre, hn-NSF, generator), chunk
/// i+1 runs its duration stage. Results arrive in order.
///
/// Why: on an iPhone 12 Pro the hybrid generator runs ~420 ms back to back but
/// ~560-600 ms after 100+ ms without its own work (README/Notes/
/// ane-folded-generator.md, "Cold generator after a break"). Running the next
/// chunk's duration stage during this chunk's generator shortens that break.
///
/// Requests must have `warmModelsBeforeTiming == false` (the warm-up touches
/// bucket models before the gate).
///
/// - Parameters:
///   - onResult: called on the caller's thread with (index, result), in order.
public func executeKokoroSynthesisPipelined(
    requests: [KokoroSynthesisRequest],
    modelProvider: KokoroModelProvider,
    linearWeights: [Float],
    linearBias: Float,
    onResult: (Int, SynthesisResult) throws -> Void
) throws {
    precondition(requests.allSatisfy { !$0.warmModelsBeforeTiming }, "pipelined chunks cannot warm models")
    let provider = LockedModelProvider(modelProvider)
    try runPipelinedChunks(count: requests.count, work: { i, gate in
        var dump: TensorDumpWriter? = nil
        return try executeKokoroSynthesis(request: requests[i], modelProvider: provider,
                                          linearWeights: linearWeights, linearBias: linearBias,
                                          tensorDump: &dump, beforeBucketStages: gate)
    }, onResult: onResult)
}

/// Two-stage pipeline over `count` work items. `work(i, gate)` runs on a
/// background thread and must call `gate()` at most once, where its second
/// stage starts. At most one item runs its first stage at a time, second
/// stages run one at a time in index order, and item i+1's first stage
/// overlaps item i's second stage. `onResult` gets results in index order on
/// the caller's thread as they finish. After a failure no new item starts;
/// every started item finishes before this returns or throws.
func runPipelinedChunks<T>(
    count n: Int,
    work: @escaping (Int, @escaping () -> Void) throws -> T,
    onResult: (Int, T) throws -> Void
) throws {
    // Semaphores start at 0 and are signalled once: libdispatch traps when a
    // semaphore is freed with a count below its initial value.
    let frontFree = DispatchSemaphore(value: 0)   // one item in its first stage at a time
    let turn = (0...n).map { _ in DispatchSemaphore(value: 0) }   // second-stage order
    frontFree.signal()
    turn[0].signal()
    let ready = (0..<n).map { _ in DispatchSemaphore(value: 0) }   // result stored, or item skipped
    let lock = NSLock()
    var results = [Result<T, Error>?](repeating: nil, count: n)
    var stopped = false
    let items = DispatchGroup()
    let spawner = DispatchGroup()

    func runItem(_ i: Int) {
        var passedGate = false
        let result = Result { try work(i) {
            guard !passedGate else { return }
            turn[i].wait()
            passedGate = true
            frontFree.signal()
        } }
        if !passedGate {   // never reached the gate (failed early, or no second stage)
            turn[i].wait()
            frontFree.signal()
        }
        lock.lock(); results[i] = result; if case .failure = result { stopped = true }; lock.unlock()
        turn[i + 1].signal()
        ready[i].signal()
    }

    DispatchQueue.global(qos: .userInitiated).async(group: spawner) {
        for i in 0..<n {
            frontFree.wait()
            lock.lock(); let stop = stopped; lock.unlock()
            if stop {
                frontFree.signal()
                for j in i..<n { ready[j].signal() }   // skipped
                return
            }
            DispatchQueue.global(qos: .userInitiated).async(group: items) { runItem(i) }
        }
    }

    func finish() {
        lock.lock(); stopped = true; lock.unlock()
        spawner.wait()
        items.wait()
    }
    do {
        for i in 0..<n {
            ready[i].wait()
            lock.lock(); let r = results[i]; lock.unlock()
            guard let r else { break }   // skipped after an earlier failure
            try onResult(i, try r.get())
        }
    } catch {
        finish()
        throw error
    }
    finish()
}

/// Serializes model vending (the providers' caches are plain dictionaries);
/// predictions on the vended models still run concurrently.
final class LockedModelProvider: KokoroModelProvider {
    private let base: KokoroModelProvider
    private let lock = NSRecursiveLock()
    init(_ base: KokoroModelProvider) { self.base = base }
    private func locked<T>(_ f: () throws -> T) rethrows -> T { lock.lock(); defer { lock.unlock() }; return try f() }

    func durationModelChoices() -> [DurationModelChoice] { locked { base.durationModelChoices() } }
    func availableBucketSeconds() -> [Int] { locked { base.availableBucketSeconds() } }
    func durationModel(choice: DurationModelChoice) throws -> MLModel { try locked { try base.durationModel(choice: choice) } }
    func f0ntrainModel(tFrames: Int) throws -> MLModel { try locked { try base.f0ntrainModel(tFrames: tFrames) } }
    func decoderPreModel(bucketSec: Int) throws -> MLModel { try locked { try base.decoderPreModel(bucketSec: bucketSec) } }
    func generatorModel(bucketSec: Int) throws -> MLModel { try locked { try base.generatorModel(bucketSec: bucketSec) } }
    func prepareForBucket(bucketSec: Int, tFrames: Int) throws { try locked { try base.prepareForBucket(bucketSec: bucketSec, tFrames: tFrames) } }
    func flexibleGeneratorModel() throws -> MLModel? { try locked { try base.flexibleGeneratorModel() } }
    func foldedGeneratorModel(bucketSec: Int) throws -> MLModel? { try locked { try base.foldedGeneratorModel(bucketSec: bucketSec) } }
    func hybridGenerator(bucketSec: Int, xPreFrames: Int) throws -> AnyObject? { try locked { try base.hybridGenerator(bucketSec: bucketSec, xPreFrames: xPreFrames) } }
    func splitDuration(choice: DurationModelChoice) throws -> SplitDuration? { try locked { try base.splitDuration(choice: choice) } }
    func splitF0N(tFrames: Int) throws -> SplitF0N? { try locked { try base.splitF0N(tFrames: tFrames) } }
}
