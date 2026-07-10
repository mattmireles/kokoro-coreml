import SwiftUI
import KokoroTTS
import CoreML
import UIKit

struct BenchmarkPayload: Codable, Sendable {
    let impl: String
    let framework: String
    let hardwareTarget: String
    let computeUnits: String
    let warmIterations: Int
    let records: [BenchmarkRecord]

    enum CodingKeys: String, CodingKey {
        case impl
        case framework
        case hardwareTarget = "hardware_target"
        case computeUnits = "compute_units"
        case warmIterations = "warm_iterations"
        case records
    }
}

struct BenchmarkRecord: Codable, Sendable {
    let inputKey: String
    let textSHA256: String
    let voice: String
    let canonicalAudioDurationS: Double
    let expectedBucketS: Int
    let coldWallTimeS: Double
    let warmWallTimesS: [Double]
    let sampleCount: Int
    let sampleRate: Int
    let observedAudioDurationS: Double
    let rtfObserved: [Double]

    enum CodingKeys: String, CodingKey {
        case inputKey = "input_key"
        case textSHA256 = "text_sha256"
        case voice
        case canonicalAudioDurationS = "canonical_audio_duration_s"
        case expectedBucketS = "expected_bucket_s"
        case coldWallTimeS = "cold_wall_time_s"
        case warmWallTimesS = "warm_wall_times_s"
        case sampleCount = "sample_count"
        case sampleRate = "sample_rate"
        case observedAudioDurationS = "observed_audio_duration_s"
        case rtfObserved = "rtf_observed"
    }
}

@main
struct SoniqoKokoroIOSRunnerApp: App {
    var body: some Scene {
        WindowGroup {
            BenchmarkView()
        }
    }
}

@MainActor
final class BenchmarkViewModel: ObservableObject {
    @Published var status = "Ready"
    @Published var result = ""

    private let warmIterations = 5

    func run() {
        status = "Running"
        result = ""
        let inputs = runtimeInputs
        let iterationCount = warmIterations
        Task {
            do {
                let payload = try await Task.detached(priority: .userInitiated) { () -> BenchmarkPayload in
                    let model = try await KokoroTTSModel.fromPretrained(computeUnits: .all)
                    var records: [BenchmarkRecord] = []

                    for input in inputs {
                        let coldStart = CFAbsoluteTimeGetCurrent()
                        _ = try model.synthesize(
                            text: input.text,
                            voice: input.voice,
                            language: "en",
                            speed: Float(input.speed)
                        )
                        let cold = CFAbsoluteTimeGetCurrent() - coldStart

                        var warmTimes: [Double] = []
                        var lastAudio: [Float] = []
                        for _ in 0..<iterationCount {
                            let warmStart = CFAbsoluteTimeGetCurrent()
                            lastAudio = try model.synthesize(
                                text: input.text,
                                voice: input.voice,
                                language: "en",
                                speed: Float(input.speed)
                            )
                            warmTimes.append(CFAbsoluteTimeGetCurrent() - warmStart)
                        }

                        let sampleRate = KokoroTTSModel.outputSampleRate
                        let observedDuration = Double(lastAudio.count) / Double(sampleRate)
                        records.append(BenchmarkRecord(
                            inputKey: input.key,
                            textSHA256: input.textSHA256,
                            voice: input.voice,
                            canonicalAudioDurationS: input.canonicalDurationS,
                            expectedBucketS: input.expectedBucketS,
                            coldWallTimeS: cold,
                            warmWallTimesS: warmTimes,
                            sampleCount: lastAudio.count,
                            sampleRate: sampleRate,
                            observedAudioDurationS: observedDuration,
                            rtfObserved: warmTimes.map { $0 / observedDuration }
                        ))
                    }

                    return BenchmarkPayload(
                        impl: "soniqo-speech-swift-kokoro-ios",
                        framework: "Swift + Core ML",
                        hardwareTarget: "ANE/Core ML",
                        computeUnits: "all",
                        warmIterations: iterationCount,
                        records: records
                    )
                }.value
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let data = try encoder.encode(payload)
                let rendered = String(data: data, encoding: .utf8) ?? ""
                result = rendered
                UIPasteboard.general.string = rendered
                status = "Done; JSON copied"
            } catch {
                status = "Failed"
                result = String(describing: error)
            }
        }
    }
}

struct BenchmarkView: View {
    @StateObject private var model = BenchmarkViewModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Soniqo Kokoro")
                .font(.title2)
                .bold()
            Text(model.status)
                .font(.headline)
            Button("Run") {
                model.run()
            }
            .buttonStyle(.borderedProminent)
            Button("Copy JSON") {
                UIPasteboard.general.string = model.result
            }
            .disabled(model.result.isEmpty)
            List(runtimeInputs) { input in
                HStack {
                    Text(input.key)
                        .font(.headline)
                    Spacer()
                    Text(String(format: "%.3fs", input.canonicalDurationS))
                        .font(.caption)
                }
            }
            .frame(height: 180)
            ScrollView {
                Text(model.result)
                    .font(.system(.body, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding()
    }
}
