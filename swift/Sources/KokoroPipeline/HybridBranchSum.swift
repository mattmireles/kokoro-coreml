import Foundation

@available(macOS 15.0, iOS 18.0, *)
extension HybridGenerator {
    /// Runs three branches on up to `chains.count` concurrent chains and sums
    /// their outputs. Each finished branch is added into the accumulator right
    /// away (`add(y, acc, mode)` under a lock, mode 0 copy, 1 add, 2 last add)
    /// and its chain goes back to the free list for the next branch.
    ///
    /// Without an `accumulator` (one chain per branch), the first finished
    /// branch's output is the accumulator: no extra tensor, no copy. That
    /// branch's chain never goes back to the free list: `parallel` may run the
    /// branches one after another, and a reused chain would overwrite the sum.
    /// `claim(y)` runs (under the lock) when y becomes the accumulator.
    static func sumBranches<C, T: AnyObject>(chains: [C], accumulator: T?,
                                             parallel: ([() throws -> Void]) throws -> Void,
                                             body: @escaping (Int, C) throws -> T,
                                             claim: @escaping (T) -> Void = { _ in },
                                             add: @escaping (T, T, Int) -> Void) throws -> T {
        let lock = NSLock()
        var acc = accumulator
        var added = 0
        var free = chains
        let slots = DispatchSemaphore(value: chains.count)
        var work: [() throws -> Void] = []
        for j in 0..<3 {
            work.append {
                slots.wait()
                lock.lock(); let ch = free.removeLast(); lock.unlock()
                var holdsSum = false
                defer { lock.lock(); if !holdsSum { free.append(ch) }; lock.unlock(); slots.signal() }
                let y = try body(j, ch)
                lock.lock(); defer { lock.unlock() }
                if let a = acc {
                    add(y, a, added == 0 ? 0 : (added == 1 ? 1 : 2))
                } else {
                    claim(y)   // y becomes the accumulator
                    acc = y
                    holdsSum = true
                }
                added += 1
            }
        }
        try parallel(work)
        return acc!
    }
}
