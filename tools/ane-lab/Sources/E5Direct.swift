import CoreML
import Darwin
import Foundation
import IOSurface

/// Experiment: dispatch a compiled Core ML function straight through Espresso's
/// private e5rt C API (the layer Core ML itself calls), bypassing MLModel.
/// Call sequence traced from Core ML with lldb on macOS 26 (README/Notes/
/// ane-folded-generator.md). Inputs are all-ones fp16 IOSurfaces; the result
/// holds warm timings and the largest difference to Core ML's output.
enum E5Direct {
    typealias P = UnsafeMutableRawPointer
    typealias F1 = @convention(c) (P?) -> Int64
    typealias F2 = @convention(c) (P?, P?) -> Int64
    typealias F3 = @convention(c) (P?, P?, P?) -> Int64
    typealias F4 = @convention(c) (P?, P?, P?, P?) -> Int64

    struct Failure: Error { let step: String; let code: Int64; let message: String }

    static func sym<T>(_ name: String, _: T.Type) throws -> T {
        guard let p = dlsym(UnsafeMutableRawPointer(bitPattern: -2), name) else { throw Failure(step: name, code: -1, message: "missing symbol") }
        return unsafeBitCast(p, to: T.self)
    }

    static func lastError() -> String {
        typealias E = @convention(c) () -> UnsafePointer<CChar>?
        guard let f = try? sym("e5rt_get_last_error_message", E.self), let s = f() else { return "" }
        return String(cString: s)
    }

    static func check(_ step: String, _ code: Int64) throws {
        if code != 0 { throw Failure(step: step, code: code, message: lastError()) }
    }

    static func cstr(_ s: String) -> P { P(strdup(s)!) }

    static func surface(bytes: Int) -> IOSurfaceRef {
        IOSurfaceCreate([kIOSurfaceWidth: bytes / 2, kIOSurfaceHeight: 1, kIOSurfaceBytesPerElement: 2,
                         kIOSurfaceAllocSize: bytes] as CFDictionary)!
    }

    static func bench(compiledModel: URL, function: String?, model: MLModel, warmup: Int, iterations: Int) throws -> [String: Any] {
        _ = dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW)
        var compiler: P?, opts: P?, lib: P?, fn: P?, opOpts: P?, op: P?, stream: P?
        let t0 = Date()
        try check("compiler_create", try sym("e5rt_e5_compiler_create", F1.self)(&compiler))
        try check("options_create", try sym("e5rt_e5_compiler_options_create", F1.self)(&opts))
        try check("set_mask", try sym("e5rt_e5_compiler_options_set_compute_device_types_mask", F2.self)(opts, P(bitPattern: 4)))
        let entry = function ?? "main"
        var entries: [P?] = [cstr(entry)]
        try check("set_entry_points", try sym("e5rt_e5_compiler_options_set_mil_entry_points", F3.self)(opts, &entries, P(bitPattern: 1)))
        let mil = compiledModel.appendingPathComponent("model.mil").path
        try check("compile", try sym("e5rt_e5_compiler_compile", F4.self)(compiler, cstr(mil), opts, &lib))
        try check("retain_function", try sym("e5rt_program_library_retain_program_function", F3.self)(lib, cstr(entry), &fn))
        try check("op_options", try sym("e5rt_precompiled_compute_op_create_options_create_with_program_function", F2.self)(&opOpts, fn))
        try check("create_op", try sym("e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options", F2.self)(&op, opOpts))
        try check("stream_create", try sym("e5rt_execution_stream_create", F1.self)(&stream))
        let compileS = Date().timeIntervalSince(t0)

        let retainIn = try sym("e5rt_execution_stream_operation_retain_input_port", F3.self)
        let retainOut = try sym("e5rt_execution_stream_operation_retain_output_port", F3.self)
        let fromSurface = try sym("e5rt_buffer_object_create_from_iosurface", F3.self)
        let bind = try sym("e5rt_io_port_bind_buffer_object", F2.self)
        var refInputs: [String: MLFeatureValue] = [:]
        for (name, desc) in model.modelDescription.inputDescriptionsByName {
            let shape = desc.multiArrayConstraint!.shape
            let count = shape.reduce(1) { $0 * $1.intValue }
            let s = surface(bytes: count * 2)
            IOSurfaceLock(s, [], nil)
            let p = IOSurfaceGetBaseAddress(s).assumingMemoryBound(to: UInt16.self)
            for k in 0..<count { p[k] = 0x3C00 }
            IOSurfaceUnlock(s, [], nil)
            var port: P?, buf: P?
            try check("in_port \(name)", retainIn(op, cstr(name), &port))
            try check("in_buffer \(name)", fromSurface(&buf, P(Unmanaged.passRetained(s).toOpaque()), P(bitPattern: 0x103)))
            try check("in_bind \(name)", bind(port, buf))
            let a = try MLMultiArray(shape: shape, dataType: .float16)
            let q = a.dataPointer.assumingMemoryBound(to: UInt16.self)
            for k in 0..<count { q[k] = 0x3C00 }
            refInputs[name] = MLFeatureValue(multiArray: a)
        }
        var outputs: [(String, IOSurfaceRef, Int)] = []
        for (name, desc) in model.modelDescription.outputDescriptionsByName {
            let count = desc.multiArrayConstraint!.shape.reduce(1) { $0 * $1.intValue }
            let s = surface(bytes: count * 2)
            var port: P?, buf: P?
            try check("out_port \(name)", retainOut(op, cstr(name), &port))
            try check("out_buffer \(name)", fromSurface(&buf, P(Unmanaged.passRetained(s).toOpaque()), P(bitPattern: 0x103)))
            try check("out_bind \(name)", bind(port, buf))
            outputs.append((name, s, count))
        }
        let prep = try sym("e5rt_execution_stream_operation_prepare_op_for_encode", F1.self)
        let encode = try sym("e5rt_execution_stream_encode_operation", F2.self)
        let run = try sym("e5rt_execution_stream_execute_sync", F1.self)
        let reset = try sym("e5rt_execution_stream_reset", F1.self)
        let qos = try sym("e5rt_execution_stream_set_quality_of_service", F2.self)
        try check("qos", qos(stream, P(bitPattern: 0x21)))
        var times: [Double] = []
        for i in 0..<(warmup + iterations) {
            let t = Date()
            try check("prepare", prep(op))
            try check("encode", encode(stream, op))
            try check("execute", run(stream))
            try check("reset", reset(stream))
            if i >= warmup { times.append(Date().timeIntervalSince(t) * 1000) }
        }
        times.sort()

        // Compare with Core ML on the same all-ones inputs.
        let ref = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: refInputs))
        var maxDiff: Float = 0
        for (name, s, count) in outputs {
            guard let r = ref.featureValue(for: name)?.multiArrayValue else { continue }
            IOSurfaceLock(s, .readOnly, nil)
            let p = IOSurfaceGetBaseAddress(s).assumingMemoryBound(to: Float16.self)
            for k in stride(from: 0, to: count, by: 7) { maxDiff = max(maxDiff, abs(Float(p[k]) - r[k].floatValue)) }
            IOSurfaceUnlock(s, .readOnly, nil)
        }
        return ["e5_compile_s": compileS, "e5_median_ms": times[times.count / 2], "e5_min_ms": times[0],
                "e5_ms": times, "e5_max_diff_vs_coreml": Double(maxDiff)]
    }
}
