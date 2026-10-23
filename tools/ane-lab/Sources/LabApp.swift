/// ANE lab runner: times arbitrary Core ML models on a tethered iPhone.
///
/// Launch arguments:
///   --job job.json   job file in Documents (default job.json)
///
/// Job JSON:
///   { "model": "gen.mlpackage",            // in Documents (.mlpackage or .mlmodelc)
///     "units": ["ne", "gpu", "cpu", "all"],
///     "warmup": 3, "iterations": 10,
///     "inputs_dir": "in",                   // optional: <name>.bin raw float32 per input
///     "save_outputs": true,                 // write out_<units>_<name>.bin (float32)
///     "function": "rb3_c1_0",               // optional: function of a multifunction package
///     "rotate": ["rb3_c1_1", "rb3_c1_2"],   // optional: also load these same-shape functions
///                                           //   and call all of them round-robin
///     "surface_sets": 8,                    // optional (iosurface): rotate through this many
///                                           //   distinct input/output surface sets
///     "io": "iosurface",                    // optional: fp16 inputs/outputs as IOSurface-backed
///                                           //   arrays (output backings), as HybridGenerator does
///     "between_ms": 5,                      // optional: sleep between predictions
///     "between_gpu": true,                  // optional (iosurface): a GPU pass writes every
///                                           //   input surface before each prediction
///     "out": "result.json" }
///
/// Inputs without a .bin file get uniform random values in [-1, 1).
/// For every compute-unit setting the result records load time, warm predict
/// times, the MLComputePlan op count per device, and the process
/// phys_footprint before load, after load and at peak (sampled after every
/// predict). Console line "LABDONE" marks completion.
import SwiftUI
import CoreML
import CoreVideo
import Darwin
import Metal

@main
struct LabApp: App {
    init() { Task.detached { await Lab.run() } }
    var body: some Scene { WindowGroup { Text("ANE lab") } }
}

struct Job: Decodable {
    let model: String
    let units: [String]?
    let warmup: Int?
    let iterations: Int?
    let inputs_dir: String?
    let save_outputs: Bool?
    let function: String?
    let io: String?
    let between_ms: Double?
    let between_gpu: Bool?
    let rotate: [String]?
    let surface_sets: Int?
    let e5rt: Bool?
    let e5chain_steps: Int?
    let out: String?
}

enum Lab {
    static func footprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? info.phys_footprint : 0
    }

    static func units(_ s: String) -> MLComputeUnits {
        switch s {
        case "ne": return .cpuAndNeuralEngine
        case "gpu": return .cpuAndGPU
        case "cpu": return .cpuOnly
        default: return .all
        }
    }

    static func makeInput(_ desc: MLFeatureDescription, dir: URL?) throws -> MLFeatureValue {
        guard let c = desc.multiArrayConstraint else { throw NSError(domain: "lab", code: 1) }
        let shape = c.shape
        let n = shape.reduce(1) { $0 * $1.intValue }
        var values = [Float](repeating: 0, count: n)
        if let dir, let data = try? Data(contentsOf: dir.appendingPathComponent(desc.name + ".bin")) {
            precondition(data.count == n * 4, "\(desc.name).bin has \(data.count) bytes, want \(n * 4)")
            values = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        } else {
            for i in 0..<n { values[i] = Float.random(in: -1..<1) }
        }
        let arr = try MLMultiArray(shape: shape, dataType: c.dataType == .float16 ? .float16 : .float32)
        for i in 0..<n { arr[i] = NSNumber(value: values[i]) }
        return MLFeatureValue(multiArray: arr)
    }

    /// fp16 array in an IOSurface (width = last dim), like HybridGenerator's SharedTensor.
    static func surfaceArray(_ shape: [NSNumber], random: Bool) -> MLMultiArray? {
        let w = shape.last!.intValue, h = shape.dropLast().reduce(1) { $0 * $1.intValue }
        var pb: CVPixelBuffer?
        var attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]]
        if ProcessInfo.processInfo.arguments.contains("--metal-compat") { attrs[kCVPixelBufferMetalCompatibilityKey as String] = true }
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_OneComponent16Half, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let pb, CVPixelBufferGetBytesPerRow(pb) == w * 2 else { return nil }
        if random {
            CVPixelBufferLockBaseAddress(pb, [])
            let p = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: Float16.self)
            for i in 0..<(w * h) { p[i] = Float16(Float.random(in: -1..<1)) }
            CVPixelBufferUnlockBaseAddress(pb, [])
        }
        return MLMultiArray(pixelBuffer: pb, shape: shape)
    }

    /// A GPU pass that reads and rewrites a half buffer (stands in for AdaIN+Snake).
    static func scalePipeline(_ dev: MTLDevice) -> MTLComputePipelineState? {
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void scale(device half4 *x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
            half4 v = x[i]; x[i] = v * half(0.999) + half(0.001) * sin(v);
        }
        """
        guard let lib = try? dev.makeLibrary(source: src, options: nil), let f = lib.makeFunction(name: "scale") else { return nil }
        return try? dev.makeComputePipelineState(function: f)
    }

    static func saveOutput(_ arr: MLMultiArray, to url: URL) throws {
        var out = [Float](repeating: 0, count: arr.count)
        for i in 0..<arr.count { out[i] = arr[i].floatValue }
        try out.withUnsafeBytes { try Data($0).write(to: url) }
    }

    static func planCounts(_ url: URL, _ cfg: MLModelConfiguration) async -> [String: Int] {
        guard let plan = try? await MLComputePlan.load(contentsOf: url, configuration: cfg),
              case let .program(program) = plan.modelStructure else { return [:] }
        var counts: [String: Int] = [:]
        for (_, fn) in program.functions {
            for op in fn.block.operations {
                let key: String
                switch plan.deviceUsage(for: op)?.preferred {
                case .neuralEngine: key = "ne"
                case .gpu: key = "gpu"
                case .cpu: key = "cpu"
                default: key = "none"
                }
                counts[key, default: 0] += 1
            }
        }
        return counts
    }

    static func run() async {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let args = CommandLine.arguments
        let jobName = args.firstIndex(of: "--job").map { args[$0 + 1] } ?? "job.json"
        var result: [String: Any] = ["started_footprint": footprint()]
        do {
            let job = try JSONDecoder().decode(Job.self, from: Data(contentsOf: docs.appendingPathComponent(jobName)))
            var modelURL = docs.appendingPathComponent(job.model)
            if modelURL.pathExtension == "mlpackage" {
                let t0 = Date()
                modelURL = try await MLModel.compileModel(at: modelURL)
                result["compile_s"] = Date().timeIntervalSince(t0)
            }
            let inDir = job.inputs_dir.map { docs.appendingPathComponent($0) }
            var rows: [[String: Any]] = []
            for u in job.units ?? ["ne", "gpu"] {
                var row: [String: Any] = ["units": u]
                do {
                    let cfg = MLModelConfiguration()
                    cfg.computeUnits = units(u)
                    if let fn = job.function { cfg.functionName = fn }
                    row["plan"] = await planCounts(modelURL, cfg)
                    let before = footprint()
                    let t0 = Date()
                    let model = try MLModel(contentsOf: modelURL, configuration: cfg)
                    var models = [model]
                    for fn in job.rotate ?? [] {
                        let c = MLModelConfiguration(); c.computeUnits = cfg.computeUnits; c.functionName = fn
                        models.append(try MLModel(contentsOf: modelURL, configuration: c))
                    }
                    row["load_s"] = Date().timeIntervalSince(t0)
                    row["footprint_after_load"] = footprint()
                    let surfaces = job.io == "iosurface"
                    var feats: [String: MLFeatureValue] = [:]
                    for (name, desc) in model.modelDescription.inputDescriptionsByName {
                        if surfaces, let c = desc.multiArrayConstraint, let a = surfaceArray(c.shape, random: true) {
                            feats[name] = MLFeatureValue(multiArray: a)
                        } else {
                            feats[name] = try makeInput(desc, dir: inDir)
                        }
                    }
                    var providers = [try MLDictionaryFeatureProvider(dictionary: feats)]
                    var backingSets: [[String: Any]] = []
                    if surfaces, let n = job.surface_sets, n > 1 {
                        for _ in 1..<n {
                            var f2: [String: MLFeatureValue] = [:]
                            for (name, desc) in model.modelDescription.inputDescriptionsByName {
                                if let c = desc.multiArrayConstraint, let a = surfaceArray(c.shape, random: true) { f2[name] = MLFeatureValue(multiArray: a) }
                            }
                            providers.append(try MLDictionaryFeatureProvider(dictionary: f2))
                            var b2: [String: Any] = [:]
                            for (name, desc) in model.modelDescription.outputDescriptionsByName {
                                if let c = desc.multiArrayConstraint, let a = surfaceArray(c.shape, random: false) { b2[name] = a }
                            }
                            backingSets.append(b2)
                        }
                    }
                    let provider = providers[0]
                    // GPU writers for the input surfaces (between_gpu), as in the hybrid pipeline.
                    let dev = MTLCreateSystemDefaultDevice()!, queue = dev.makeCommandQueue()!
                    var gpuBuffers: [MTLBuffer] = []
                    let scalePipe = scalePipeline(dev)
                    if surfaces, job.between_gpu ?? false {
                        for v in feats.values {
                            guard let pb = v.multiArrayValue?.pixelBuffer else { continue }
                            CVPixelBufferLockBaseAddress(pb, [])
                            let base = CVPixelBufferGetBaseAddress(pb)!, len = CVPixelBufferGetDataSize(pb)
                            CVPixelBufferUnlockBaseAddress(pb, [])
                            if let b = dev.makeBuffer(bytesNoCopy: base, length: (len + 16383) / 16384 * 16384,
                                                      options: .storageModeShared, deallocator: nil) { gpuBuffers.append(b) }
                        }
                        row["gpu_buffers"] = gpuBuffers.count
                    }
                    let gpuOutputs = surfaces && ProcessInfo.processInfo.arguments.contains("--gpu-outputs")
                    let opts = MLPredictionOptions()
                    var firstBackings: [String: Any] = [:]
                    if surfaces {
                        var backings: [String: Any] = [:]
                        for (name, desc) in model.modelDescription.outputDescriptionsByName {
                            if let c = desc.multiArrayConstraint, let a = surfaceArray(c.shape, random: false) { backings[name] = a }
                        }
                        opts.outputBackings = backings
                        firstBackings = backings
                        if gpuOutputs {   // the GPU also touches the output surfaces before each call
                            for case let a as MLMultiArray in backings.values {
                                guard let pb = a.pixelBuffer else { continue }
                                CVPixelBufferLockBaseAddress(pb, [])
                                let base = CVPixelBufferGetBaseAddress(pb)!, len = CVPixelBufferGetDataSize(pb)
                                CVPixelBufferUnlockBaseAddress(pb, [])
                                if let b = dev.makeBuffer(bytesNoCopy: base, length: (len + 16383) / 16384 * 16384,
                                                          options: .storageModeShared, deallocator: nil) { gpuBuffers.append(b) }
                            }
                            row["gpu_buffers"] = gpuBuffers.count
                        }
                        row["io"] = "iosurface (\(feats.count) in, \(backings.count) out backings)"
                    }
                    var peak = footprint()
                    var times: [Double] = []
                    var last: MLFeatureProvider?
                    for i in 0..<((job.warmup ?? 3) + (job.iterations ?? 10)) {
                        if !gpuBuffers.isEmpty {   // a compute pass reads and rewrites every input, like AdaIN+Snake
                            let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
                            e.setComputePipelineState(scalePipe!)
                            for b in gpuBuffers {
                                e.setBuffer(b, offset: 0, index: 0)
                                e.dispatchThreads(MTLSize(width: b.length / 8, height: 1, depth: 1),
                                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                            }
                            e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                        }
                        let t = Date()
                        let set = i % providers.count
                        if set > 0 { opts.outputBackings = backingSets[set - 1] } else if !backingSets.isEmpty { opts.outputBackings = firstBackings }
                        last = try autoreleasepool { try models[i % models.count].prediction(from: providers[set], options: opts) }
                        let dt = Date().timeIntervalSince(t)
                        if let ms = job.between_ms { usleep(useconds_t(ms * 1000)) }
                        if i >= (job.warmup ?? 3) { times.append(dt) }
                        peak = max(peak, footprint())
                    }
                    times.sort()
                    row["predict_ms"] = times.map { $0 * 1000 }
                    row["median_ms"] = times[times.count / 2] * 1000
                    row["min_ms"] = times[0] * 1000
                    row["footprint_before"] = before
                    row["footprint_peak"] = peak
                    if job.save_outputs ?? false, let last {
                        var saved: [String] = []
                        for name in last.featureNames {
                            if let arr = last.featureValue(for: name)?.multiArrayValue {
                                let file = "out_\(u)_\(name).bin"
                                try saveOutput(arr, to: docs.appendingPathComponent(file))
                                saved.append(file)
                            }
                        }
                        row["saved"] = saved
                    }
                    print("LAB: \(u) median \(row["median_ms"]!) ms")
                } catch {
                    row["error"] = "\(error)"
                }
                rows.append(row)
            }
            result["rows"] = rows
            if let steps = job.e5chain_steps {   // GPU <-> ANE event chaining vs sync (E5Chain.swift)
                do {
                    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
                    let model = try MLModel(contentsOf: modelURL, configuration: cfg)
                    let r = try E5Chain.bench(compiledModel: modelURL, model: model, steps: steps, rounds: job.iterations ?? 10)
                    result["e5chain"] = r
                    print("LAB: e5chain sync \(r["sync_median_ms"]!) ms chained \(r["chain_median_ms"]!) ms equal \(r["outputs_equal"]!)")
                } catch {
                    result["e5chain_error"] = "\(error)"
                    print("LAB: e5chain failed: \(error)")
                }
            }
            if job.e5rt ?? false {   // private e5rt dispatch of the same function (E5Direct.swift)
                do {
                    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
                    if let fn = job.function { cfg.functionName = fn }
                    let model = try MLModel(contentsOf: modelURL, configuration: cfg)
                    let e5 = try E5Direct.bench(compiledModel: modelURL, function: job.function, model: model,
                                                warmup: job.warmup ?? 3, iterations: job.iterations ?? 10)
                    result["e5rt"] = e5
                    print("LAB: e5rt median \(e5["e5_median_ms"]!) ms, max diff \(e5["e5_max_diff_vs_coreml"]!)")
                } catch {
                    result["e5rt_error"] = "\(error)"
                    print("LAB: e5rt failed: \(error)")
                }
            }
        } catch {
            result["error"] = "\(error)"
        }
        result["thermal"] = ProcessInfo.processInfo.thermalState.rawValue
        let data = try? JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
        try? data?.write(to: docs.appendingPathComponent("result.json"))
        print("LABDONE")
        exit(0)
    }
}
