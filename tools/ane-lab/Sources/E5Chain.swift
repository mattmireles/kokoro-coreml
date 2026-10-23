import CoreML
import Darwin
import Foundation
import IOSurface
import Metal

/// Experiment: does chaining GPU and ANE work with shared events (no host
/// round trip per hand-off) beat host-synchronous alternation?
///
/// `steps` times: a GPU pass reads y and writes t (like AdaIN+Snake), then the
/// ANE conv reads t (and r) and writes y. "sync" commits each command buffer,
/// waits, then runs the ANE op with execute_sync. "chained" encodes everything
/// up front: each command buffer waits on a MTLSharedEvent value and signals
/// the next; each ANE op (its own e5rt operation and stream) depends on the
/// GPU's value and signals completion (e5rt async events made from the same
/// MTLSharedEvent, which is an IOSurfaceSharedEvent subclass), then
/// `submit_async`. Results must be bit-identical.
enum E5Chain {
    typealias P = UnsafeMutableRawPointer

    static func bench(compiledModel: URL, model: MLModel, steps: Int, rounds: Int) throws -> [String: Any] {
        _ = dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW)
        let s = E5Direct.self
        var compiler: P?, opts: P?, lib: P?
        try s.check("compiler_create", try s.sym("e5rt_e5_compiler_create", s.F1.self)(&compiler))
        try s.check("options_create", try s.sym("e5rt_e5_compiler_options_create", s.F1.self)(&opts))
        try s.check("set_mask", try s.sym("e5rt_e5_compiler_options_set_compute_device_types_mask", s.F2.self)(opts, P(bitPattern: 4)))
        var entries: [P?] = [s.cstr("main")]
        try s.check("entries", try s.sym("e5rt_e5_compiler_options_set_mil_entry_points", s.F3.self)(opts, &entries, P(bitPattern: 1)))
        try s.check("compile", try s.sym("e5rt_e5_compiler_compile", s.F4.self)(compiler, s.cstr(compiledModel.appendingPathComponent("model.mil").path), opts, &lib))

        let shape = model.modelDescription.inputDescriptionsByName["t"]!.multiArrayConstraint!.shape
        let count = shape.reduce(1) { $0 * $1.intValue }
        let bytes = count * 2
        let dev = MTLCreateSystemDefaultDevice()!, queue = dev.makeCommandQueue()!
        func surface() -> (IOSurfaceRef, MTLBuffer) {
            let sf = E5Direct.surface(bytes: bytes)
            IOSurfaceLock(sf, [], nil)
            let base = IOSurfaceGetBaseAddress(sf)
            let p = base.assumingMemoryBound(to: UInt16.self)
            for k in 0..<count { p[k] = 0x3C00 }
            IOSurfaceUnlock(sf, [], nil)
            let mb = dev.makeBuffer(bytesNoCopy: base, length: (bytes + 16383) / 16384 * 16384,
                                    options: [.storageModeShared, .hazardTrackingModeUntracked], deallocator: nil)!
            return (sf, mb)
        }
        let (t, tm) = surface(), (r, _) = surface(), (y, ym) = surface()
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void mix(device const half4 *y [[buffer(0)]], device half4 *t [[buffer(1)]], uint i [[thread_position_in_grid]]) {
            t[i] = y[i] * half(0.999) + half(0.001);
        }
        """
        let pipe = try dev.makeComputePipelineState(function: try dev.makeLibrary(source: src, options: nil).makeFunction(name: "mix")!)

        let fromSurface = try s.sym("e5rt_buffer_object_create_from_iosurface", s.F3.self)
        func buffer(_ sf: IOSurfaceRef) throws -> P? {
            var b: P?
            try s.check("buffer", fromSurface(&b, P(Unmanaged.passUnretained(sf).toOpaque()), P(bitPattern: 0x103)))
            return b
        }
        let bt = try buffer(t), br = try buffer(r), by = try buffer(y)
        let retainFn = try s.sym("e5rt_program_library_retain_program_function", s.F3.self)
        let optsWithFn = try s.sym("e5rt_precompiled_compute_op_create_options_create_with_program_function", s.F2.self)
        let createOp = try s.sym("e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options", s.F2.self)
        let createStream = try s.sym("e5rt_execution_stream_create", s.F1.self)
        let retainIn = try s.sym("e5rt_execution_stream_operation_retain_input_port", s.F3.self)
        let retainOut = try s.sym("e5rt_execution_stream_operation_retain_output_port", s.F3.self)
        let bind = try s.sym("e5rt_io_port_bind_buffer_object", s.F2.self)
        var ops: [P?] = [], streams: [P?] = []
        for _ in 0..<steps {
            var fn: P?, oo: P?, op: P?, st: P?, pt: P?, pr: P?, py: P?
            try s.check("fn", retainFn(lib, s.cstr("main"), &fn))
            try s.check("oo", optsWithFn(&oo, fn))
            try s.check("op", createOp(&op, oo))
            try s.check("stream", createStream(&st))
            try s.check("pt", retainIn(op, s.cstr("t"), &pt)); try s.check("bt", bind(pt, bt))
            try s.check("pr", retainIn(op, s.cstr("r"), &pr)); try s.check("br", bind(pr, br))
            try s.check("py", retainOut(op, s.cstr("y"), &py)); try s.check("by", bind(py, by))
            ops.append(op); streams.append(st)
        }
        let prep = try s.sym("e5rt_execution_stream_operation_prepare_op_for_encode", s.F1.self)
        let encode = try s.sym("e5rt_execution_stream_encode_operation", s.F2.self)
        let execute = try s.sym("e5rt_execution_stream_execute_sync", s.F1.self)
        let reset = try s.sym("e5rt_execution_stream_reset", s.F1.self)
        let submit = try s.sym("e5rt_execution_stream_submit_async", s.F2.self)
        let makeEvent = try s.sym("e5rt_async_event_create_from_iosurface_shared_event", s.F3.self)
        let setFuture = try s.sym("e5rt_async_event_set_active_future_value", s.F2.self)
        let bindDeps = try s.sym("e5rt_execution_stream_operation_bind_dependent_events", s.F3.self)
        let bindDone = try s.sym("e5rt_execution_stream_operation_bind_completion_event", s.F2.self)
        let releaseEvent = try s.sym("e5rt_async_event_release", s.F1.self)

        func gpuPass(_ cb: MTLCommandBuffer) {
            let e = cb.makeComputeCommandEncoder()!
            e.setComputePipelineState(pipe)
            e.setBuffer(ym, offset: 0, index: 0)
            e.setBuffer(tm, offset: 0, index: 1)
            e.dispatchThreads(MTLSize(width: count / 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            e.endEncoding()
        }
        func readY() -> [UInt16] {
            IOSurfaceLock(y, .readOnly, nil); defer { IOSurfaceUnlock(y, .readOnly, nil) }
            let p = IOSurfaceGetBaseAddress(y).assumingMemoryBound(to: UInt16.self)
            return [p[0], p[count / 2], p[count - 1]]
        }
        func resetInputs() {
            for sf in [t, y] {
                IOSurfaceLock(sf, [], nil)
                let p = IOSurfaceGetBaseAddress(sf).assumingMemoryBound(to: UInt16.self)
                for k in 0..<count { p[k] = 0x3C00 }
                IOSurfaceUnlock(sf, [], nil)
            }
        }

        var syncMs: [Double] = [], chainMs: [Double] = []
        var syncOut: [UInt16] = [], chainOut: [UInt16] = []
        var blocks: [AnyObject] = []
        for round in 0..<(rounds + 1) {
            resetInputs()
            var t0 = Date()
            for i in 0..<steps {
                let cb = queue.makeCommandBuffer()!
                gpuPass(cb); cb.commit(); cb.waitUntilCompleted()
                try s.check("prep", prep(ops[i])); try s.check("enc", encode(streams[i], ops[i]))
                try s.check("exec", execute(streams[i])); try s.check("reset", reset(streams[i]))
            }
            if round > 0 { syncMs.append(Date().timeIntervalSince(t0) * 1000) }
            syncOut = readY()

            resetInputs()
            let ev = dev.makeSharedEvent()!
            var events: [P?] = []
            t0 = Date()
            for i in 0..<steps {
                let cb = queue.makeCommandBuffer()!
                if i > 0 { cb.encodeWaitForEvent(ev, value: UInt64(2 * i)) }
                gpuPass(cb)
                cb.encodeSignalEvent(ev, value: UInt64(2 * i + 1))
                cb.commit()
                var waitEv: P?, doneEv: P?
                try s.check("wait_ev", makeEvent(&waitEv, s.cstr("w"), P(Unmanaged.passUnretained(ev).toOpaque())))
                try s.check("done_ev", makeEvent(&doneEv, s.cstr("d"), P(Unmanaged.passUnretained(ev).toOpaque())))
                try s.check("fw", setFuture(waitEv, P(bitPattern: UInt(2 * i + 1))))
                try s.check("fd", setFuture(doneEv, P(bitPattern: UInt(2 * i + 2))))
                var deps: [P?] = [waitEv]
                try s.check("deps", bindDeps(ops[i], &deps, P(bitPattern: 1)))
                try s.check("done", bindDone(ops[i], doneEv))
                try s.check("prep", prep(ops[i])); try s.check("enc", encode(streams[i], ops[i]))
                let blk: @convention(block) (UnsafeMutableRawPointer?) -> Void = { _ in }
                blocks.append(blk as AnyObject)
                try s.check("submit", submit(streams[i], Unmanaged.passUnretained(blk as AnyObject).toOpaque()))
                events += [waitEv, doneEv]
            }
            let deadline = Date().addingTimeInterval(5)
            while ev.signaledValue < UInt64(2 * steps) && Date() < deadline {}
            if round > 0 { chainMs.append(Date().timeIntervalSince(t0) * 1000) }
            chainOut = readY()
            if ev.signaledValue < UInt64(2 * steps) { throw E5Direct.Failure(step: "chain timeout at \(ev.signaledValue)", code: -2, message: "") }
            for i in 0..<steps {
                var deps: [P?] = []
                _ = bindDeps(ops[i], &deps, nil)
                _ = bindDone(ops[i], nil)
                _ = reset(streams[i])
            }
            for var e in events { withUnsafeMutablePointer(to: &e) { _ = releaseEvent(P($0)) } }
        }
        syncMs.sort(); chainMs.sort()
        return ["steps": steps, "sync_median_ms": syncMs[syncMs.count / 2], "chain_median_ms": chainMs[chainMs.count / 2],
                "sync_ms": syncMs, "chain_ms": chainMs, "outputs_equal": syncOut == chainOut,
                "sync_out": syncOut.map(Int.init), "chain_out": chainOut.map(Int.init)]
    }
}
