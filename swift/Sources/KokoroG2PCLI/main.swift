// kokoro-g2p: read text lines from stdin, print Kokoro IPA phonemes one per line.
//
// Usage: kokoro-g2p [assets-dir] < input.txt
// Default assets dir: ~/.cache/fluidaudio/Models/kokoro

import Foundation
import KokoroG2P

let assets = CommandLine.arguments.count > 1
    ? URL(fileURLWithPath: CommandLine.arguments[1])
    : FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/fluidaudio/Models/kokoro")

do {
    let frontend = try KokoroEnglishFrontend(assetsDirectory: assets)
    while let line = readLine() {
        print(try await frontend.phonemes(for: line))
    }
} catch {
    FileHandle.standardError.write("kokoro-g2p: \(error)\n".data(using: .utf8)!)
    exit(1)
}
