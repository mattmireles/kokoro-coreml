import Foundation
import KokoroPipeline
#if canImport(AVFoundation)
import AVFoundation
#endif

/// In-memory mono PCM audio returned by the public SDK.
public struct KokoroAudio: Equatable, Sendable {
    /// Mono floating-point PCM samples in the range produced by KokoroPipeline.
    public let samples: [Float]

    /// Sample rate in Hz.
    public let sampleRate: Int

    /// Audio duration in seconds.
    public var durationSeconds: Double {
        samples.isEmpty ? 0 : Double(samples.count) / Double(sampleRate)
    }

    /// Creates an audio value.
    ///
    /// - Parameters:
    ///   - samples: Mono floating-point PCM samples.
    ///   - sampleRate: Sample rate in Hz.
    public init(samples: [Float], sampleRate: Int = PipelineConstants.sampleRate) {
        self.samples = samples
        self.sampleRate = sampleRate
    }

    #if canImport(AVFoundation)
    /// Creates an `AVAudioPCMBuffer` from this mono PCM audio.
    ///
    /// - Returns: Float32 mono PCM buffer.
    public func makePCMBuffer() throws -> AVAudioPCMBuffer {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ), let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            throw KokoroError.invalidAudioOutput
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        guard let channel = buffer.floatChannelData?[0] else {
            throw KokoroError.invalidAudioOutput
        }
        for (index, sample) in samples.enumerated() {
            channel[index] = sample
        }
        return buffer
    }
    #endif
}
