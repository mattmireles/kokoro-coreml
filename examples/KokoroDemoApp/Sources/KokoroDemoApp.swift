import AVFoundation
import Darwin
import SwiftUI
import KokoroTTS
import UIKit

@main
struct KokoroDemoApp: App {
    var body: some Scene {
        WindowGroup {
            DemoView(model: DemoModel())
        }
    }
}

struct DemoView: View {
    @StateObject var model: DemoModel
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        NavigationStack {
            Form {
                Section("Resources") {
                    Picker("Mode", selection: $model.resourceMode) {
                        ForEach(DemoResourceMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    if model.resourceMode == .downloaded {
                        TextField("URL", text: $model.manifestURLString, axis: .vertical)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                    } else {
                        TextField("Bundle subdirectory", text: $model.bundleSubdirectory)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                    }
                }
                Section("Text") {
                    TextEditor(text: $model.text)
                        .frame(minHeight: 120)
                    Picker("Voice", selection: $model.voice) {
                        Text("af_heart").tag("af_heart")
                    }
                    Button(model.isRunning ? "Synthesizing..." : "Synthesize") {
                        Task { await model.synthesize() }
                    }
                    .disabled(model.isRunning)
                }
                Section("Status") {
                    Text(model.status)
                        .font(.footnote.monospaced())
                }
            }
            .navigationTitle("Kokoro")
        }
        .task {
            await model.runLaunchAutomationIfRequested()
        }
        .onAppear {
            model.logScenePhase(scenePhase, reason: "appear")
        }
        .onChange(of: scenePhase) { _, phase in
            model.logScenePhase(phase, reason: "change")
        }
    }
}

enum DemoResourceMode: String, CaseIterable, Identifiable {
    case downloaded
    case bundled

    var id: String { rawValue }

    var title: String {
        switch self {
        case .downloaded:
            return "Download"
        case .bundled:
            return "Bundled"
        }
    }

    init?(argument: String) {
        switch argument {
        case "downloaded", "download":
            self = .downloaded
        case "bundled", "bundle":
            self = .bundled
        default:
            return nil
        }
    }
}

@MainActor
final class DemoModel: ObservableObject {
    @Published var resourceMode: DemoResourceMode = .downloaded
    @Published var manifestURLString = ProcessInfo.processInfo.environment["KOKORO_MANIFEST_URL"] ?? ""
    @Published var bundleSubdirectory = ProcessInfo.processInfo.environment["KOKORO_BUNDLE_SUBDIRECTORY"] ?? "KokoroRuntime"
    @Published var text = "Hello world."
    @Published var voice = "af_heart"
    @Published var status = "Idle"
    @Published var isRunning = false

    private let scenario: String
    private let memoryPressureMegabytes: Int
    private let invalidResourceMode: String?
    private let invalidMemoryPressureMegabytes: String?
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var didAutoRun = false
    private var memoryWarningObserver: NSObjectProtocol?
    private static let validScenarios: Set<String> = ["smoke", "warm", "long", "cancel", "memory-pressure", "all"]
    private static let defaultMemoryPressureMegabytes = 384
    private static let memoryPressureChunkMegabytes = 16
    private static let minimumMemoryPressureMegabytes = 64
    private static let maximumMemoryPressureMegabytes = 768
    private static let bytesPerMegabyte = 1_048_576
    private static let memoryPageBytes = 4_096

    init() {
        let args = CommandLine.arguments
        if let manifest = Self.value(after: "--manifest-url", in: args) {
            manifestURLString = manifest
        }
        if let mode = Self.value(after: "--resource-mode", in: args) {
            if let parsed = DemoResourceMode(argument: mode) {
                resourceMode = parsed
                invalidResourceMode = nil
            } else {
                invalidResourceMode = mode
            }
        } else {
            invalidResourceMode = nil
        }
        if let subdirectory = Self.value(after: "--bundle-subdirectory", in: args) {
            bundleSubdirectory = subdirectory
        }
        if let text = Self.value(after: "--text", in: args) {
            self.text = text
        }
        scenario = Self.value(after: "--scenario", in: args) ?? "smoke"
        let pressureString = Self.value(after: "--memory-pressure-megabytes", in: args)
            ?? ProcessInfo.processInfo.environment["KOKORO_MEMORY_PRESSURE_MEGABYTES"]
        if let pressureString {
            if let pressure = Int(pressureString) {
                memoryPressureMegabytes = pressure
                invalidMemoryPressureMegabytes = nil
            } else {
                memoryPressureMegabytes = Self.defaultMemoryPressureMegabytes
                invalidMemoryPressureMegabytes = pressureString
            }
        } else {
            memoryPressureMegabytes = Self.defaultMemoryPressureMegabytes
            invalidMemoryPressureMegabytes = nil
        }
        memoryWarningObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { _ in
            print("KOKORO_DEMO_MEMORY_WARNING")
        }
    }

    deinit {
        if let memoryWarningObserver {
            NotificationCenter.default.removeObserver(memoryWarningObserver)
        }
    }

    func runLaunchAutomationIfRequested() async {
        guard !didAutoRun, CommandLine.arguments.contains("--auto-run") else {
            return
        }
        didAutoRun = true
        await runScenario(named: scenario)
    }

    func logScenePhase(_ phase: ScenePhase, reason: String) {
        print("KOKORO_DEMO_SCENE_PHASE reason=\(reason) phase=\(Self.label(for: phase))")
    }

    func synthesize() async {
        guard !isRunning else {
            return
        }
        if let invalidResourceMode {
            print("KOKORO_DEMO_ERROR resource-mode=\(invalidResourceMode) \(DemoScenarioError.unsupportedResourceMode(invalidResourceMode))")
            return
        }
        isRunning = true
        status = "Loading resources..."
        do {
            let resources = try await loadResources()
            status = "Loading SDK..."
            let tts = try await KokoroTTS.load(resources: resources)
            status = "Synthesizing..."
            let audio = try await tts.synthesize(text, voice: KokoroVoiceID(voice))
            try play(audio)
            status = "Done: \(audio.samples.count) samples, \(audio.durationSeconds)s"
            print("KOKORO_DEMO_DONE samples=\(audio.samples.count) sampleRate=\(audio.sampleRate) duration=\(audio.durationSeconds)")
        } catch {
            status = "Error: \(error.localizedDescription)"
            print("KOKORO_DEMO_ERROR \(error)")
        }
        isRunning = false
    }

    private func runScenario(named scenario: String) async {
        guard !isRunning else {
            return
        }
        guard Self.validScenarios.contains(scenario) else {
            print("KOKORO_DEMO_ERROR scenario=\(scenario) \(DemoScenarioError.unsupportedScenario(scenario))")
            return
        }
        if let invalidResourceMode {
            print("KOKORO_DEMO_ERROR scenario=\(scenario) resource-mode=\(invalidResourceMode) \(DemoScenarioError.unsupportedResourceMode(invalidResourceMode))")
            return
        }
        if scenario == "memory-pressure", let invalidMemoryPressureMegabytes {
            print("KOKORO_DEMO_ERROR scenario=\(scenario) memory-pressure-megabytes=\(invalidMemoryPressureMegabytes) \(DemoScenarioError.invalidMemoryPressureMegabytes(invalidMemoryPressureMegabytes))")
            return
        }
        isRunning = true
        do {
            print("KOKORO_DEMO_PID pid=\(getpid()) scenario=\(scenario) resourceMode=\(resourceMode.rawValue)")
            status = "Loading resources..."
            let resources = try await loadResources()
            status = "Loading SDK..."
            let tts = try await KokoroTTS.load(resources: resources)
            switch scenario {
            case "smoke":
                let audio = try await runSynthesis(tts: tts, text: text, label: "smoke")
                try play(audio)
            case "warm":
                _ = try await runSynthesis(tts: tts, text: text, label: "warm-first")
                let audio = try await runSynthesis(tts: tts, text: text, label: "warm-second")
                try play(audio)
            case "long":
                let audio = try await runSynthesis(tts: tts, text: Self.longText, label: "long")
                try play(audio)
            case "cancel":
                try await runCancellation(tts: tts)
            case "memory-pressure":
                try await runMemoryPressureRecovery(tts: tts)
            case "all":
                _ = try await runSynthesis(tts: tts, text: text, label: "all-first")
                _ = try await runSynthesis(tts: tts, text: text, label: "all-warm")
                _ = try await runSynthesis(tts: tts, text: Self.longText, label: "all-long")
                try await runCancellation(tts: tts)
                print("KOKORO_DEMO_MEMORY physicalFootprintBytes=\(try Self.physicalFootprintBytes())")
            default:
                preconditionFailure("Scenario validation drifted for \(scenario)")
            }
            status = "Scenario \(scenario) complete"
            print("KOKORO_DEMO_SCENARIO_DONE scenario=\(scenario)")
        } catch {
            status = "Error: \(error.localizedDescription)"
            print("KOKORO_DEMO_ERROR scenario=\(scenario) \(error)")
        }
        isRunning = false
    }

    private func loadResources() async throws -> KokoroResourceProvider {
        if let invalidResourceMode {
            throw DemoScenarioError.unsupportedResourceMode(invalidResourceMode)
        }
        print("KOKORO_DEMO_RESOURCE_MODE mode=\(resourceMode.rawValue)")
        switch resourceMode {
        case .downloaded:
            guard let manifestURL = URL(string: manifestURLString), !manifestURLString.isEmpty else {
                throw DemoScenarioError.missingManifestURL
            }
            return try await KokoroDownloadedModelStore(
                manifestURL: manifestURL,
                cacheDirectory: try Self.cacheDirectory()
            ).hydrate()
        case .bundled:
            let subdirectory = bundleSubdirectory.trimmingCharacters(in: .whitespacesAndNewlines)
            return .appBundle(.main, subdirectory: subdirectory.isEmpty ? nil : subdirectory)
        }
    }

    private func runSynthesis(tts: KokoroTTS, text: String, label: String) async throws -> KokoroAudio {
        let start = Date()
        let audio = try await tts.synthesize(text, voice: KokoroVoiceID(voice))
        let elapsed = Date().timeIntervalSince(start)
        print("KOKORO_DEMO_DONE label=\(label) samples=\(audio.samples.count) sampleRate=\(audio.sampleRate) duration=\(audio.durationSeconds) elapsedSeconds=\(elapsed)")
        return audio
    }

    private func runCancellation(tts: KokoroTTS) async throws {
        let task = Task {
            try await tts.synthesize(Self.longText, voice: KokoroVoiceID(voice))
        }
        try await Task.sleep(nanoseconds: 500_000_000)
        task.cancel()
        do {
            _ = try await task.value
            throw DemoScenarioError.cancellationDidNotThrow
        } catch is CancellationError {
            print("KOKORO_DEMO_CANCELLED error=CancellationError()")
        } catch KokoroError.synthesisCancelled {
            print("KOKORO_DEMO_CANCELLED error=KokoroError.synthesisCancelled")
        } catch {
            throw error
        }
    }

    private func runMemoryPressureRecovery(tts: KokoroTTS) async throws {
        guard memoryPressureMegabytes >= Self.minimumMemoryPressureMegabytes,
              memoryPressureMegabytes <= Self.maximumMemoryPressureMegabytes
        else {
            throw DemoScenarioError.memoryPressureMegabytesOutOfRange(
                memoryPressureMegabytes,
                minimum: Self.minimumMemoryPressureMegabytes,
                maximum: Self.maximumMemoryPressureMegabytes
            )
        }

        let baseline = try Self.physicalFootprintBytes()
        print("KOKORO_DEMO_MEMORY_PRESSURE_BEGIN requestedMegabytes=\(memoryPressureMegabytes) baselinePhysicalFootprintBytes=\(baseline)")

        var allocations: [[UInt8]] = []
        allocations.reserveCapacity((memoryPressureMegabytes + Self.memoryPressureChunkMegabytes - 1) / Self.memoryPressureChunkMegabytes)
        var allocatedMegabytes = 0
        var peak = baseline

        while allocatedMegabytes < memoryPressureMegabytes {
            let chunkMegabytes = min(Self.memoryPressureChunkMegabytes, memoryPressureMegabytes - allocatedMegabytes)
            var chunk = [UInt8](repeating: 0, count: chunkMegabytes * Self.bytesPerMegabyte)
            for offset in stride(from: 0, to: chunk.count, by: Self.memoryPageBytes) {
                chunk[offset] = UInt8(truncatingIfNeeded: allocatedMegabytes + offset)
            }
            allocations.append(chunk)
            allocatedMegabytes += chunkMegabytes
            let current = try Self.physicalFootprintBytes()
            peak = max(peak, current)
            print("KOKORO_DEMO_MEMORY_PRESSURE_STEP allocatedMegabytes=\(allocatedMegabytes) physicalFootprintBytes=\(current)")
            try await Task.sleep(nanoseconds: 50_000_000)
        }

        print("KOKORO_DEMO_MEMORY_PRESSURE_PEAK allocatedMegabytes=\(allocatedMegabytes) physicalFootprintBytes=\(peak)")
        allocations.removeAll(keepingCapacity: false)
        try await Task.sleep(nanoseconds: 500_000_000)
        let afterRelease = try Self.physicalFootprintBytes()
        print("KOKORO_DEMO_MEMORY_PRESSURE_RELEASED physicalFootprintBytes=\(afterRelease)")

        let audio = try await runSynthesis(tts: tts, text: text, label: "memory-pressure-recovery")
        print("KOKORO_DEMO_MEMORY_PRESSURE_DONE requestedMegabytes=\(memoryPressureMegabytes) baselinePhysicalFootprintBytes=\(baseline) peakPhysicalFootprintBytes=\(peak) afterReleasePhysicalFootprintBytes=\(afterRelease) recoverySamples=\(audio.samples.count)")
    }

    private func play(_ audio: KokoroAudio) throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .spokenAudio, options: [])
        try session.setActive(true)
        let buffer = try audio.makePCMBuffer()
        if !engine.attachedNodes.contains(player) {
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: buffer.format)
        }
        if !engine.isRunning {
            try engine.start()
        }
        player.stop()
        player.scheduleBuffer(buffer, at: nil, options: [])
        player.play()
    }

    private static func cacheDirectory() throws -> URL {
        let documents = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        return documents.appendingPathComponent("KokoroModelCache", isDirectory: true)
    }

    private static var longText: String {
        Array(repeating: "Local text to speech should keep working on device, even when the caller sends several sentences at once.", count: 12)
            .joined(separator: " ")
    }

    private static func physicalFootprintBytes() throws -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            throw DemoScenarioError.memoryFootprintUnavailable(result)
        }
        return info.phys_footprint
    }

    private static func value(after flag: String, in args: [String]) -> String? {
        guard let index = args.firstIndex(of: flag), args.indices.contains(index + 1) else {
            return nil
        }
        return args[index + 1]
    }

    private static func label(for phase: ScenePhase) -> String {
        switch phase {
        case .active:
            return "active"
        case .inactive:
            return "inactive"
        case .background:
            return "background"
        @unknown default:
            return "unknown"
        }
    }
}

enum DemoScenarioError: Error, LocalizedError {
    case unsupportedScenario(String)
    case unsupportedResourceMode(String)
    case invalidMemoryPressureMegabytes(String)
    case memoryPressureMegabytesOutOfRange(Int, minimum: Int, maximum: Int)
    case missingManifestURL
    case cancellationDidNotThrow
    case memoryFootprintUnavailable(kern_return_t)

    var errorDescription: String? {
        switch self {
        case .unsupportedScenario(let scenario):
            return "Unsupported scenario: \(scenario)"
        case .unsupportedResourceMode(let mode):
            return "Unsupported resource mode: \(mode)"
        case .invalidMemoryPressureMegabytes(let value):
            return "Invalid memory pressure megabytes: \(value)"
        case .memoryPressureMegabytesOutOfRange(let value, let minimum, let maximum):
            return "Memory pressure megabytes \(value) is outside the supported range \(minimum)...\(maximum)."
        case .missingManifestURL:
            return "Missing manifest URL for downloaded resource mode."
        case .cancellationDidNotThrow:
            return "Cancellation scenario completed without throwing a known cancellation error."
        case .memoryFootprintUnavailable(let result):
            return "Unable to read task_vm_info.phys_footprint: \(result)"
        }
    }
}
