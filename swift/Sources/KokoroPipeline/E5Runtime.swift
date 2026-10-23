import Darwin
import Foundation
import IOSurface
import Metal

/// Private ANE dispatch through Espresso's `e5rt_*` C API, the layer Core ML
/// itself calls (traced with lldb on macOS 26; README/Notes/ane-folded-generator.md).
///
/// Why: Core ML needs every tensor to be its own IOSurface of exactly the
/// tensor's shape, so tensors that are never live at the same time cannot
/// share memory. e5rt binds any byte range of a buffer (`create_as_alias`),
/// so the hybrid generator can lay all its tensors out in one arena by
/// lifetime. Dispatch speed is the same as Core ML's (calls are I/O-bound).
///
/// Private and version-fragile: not for App Store builds. Every symbol is
/// looked up with dlsym; `E5Library.init` throws if any is missing.
enum E5 {
    typealias P = UnsafeMutableRawPointer
    typealias F1 = @convention(c) (P?) -> Int64
    typealias F2 = @convention(c) (P?, P?) -> Int64
    typealias F3 = @convention(c) (P?, P?, P?) -> Int64
    typealias F4 = @convention(c) (P?, P?, P?, P?) -> Int64

    struct Failure: Error, CustomStringConvertible {
        let step: String, code: Int64, message: String
        var description: String { "e5rt \(step) failed (\(code)): \(message)" }
    }

    static let loaded: Bool = dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW) != nil

    static func sym<T>(_ name: String, _: T.Type) throws -> T {
        _ = loaded
        guard let p = dlsym(UnsafeMutableRawPointer(bitPattern: -2), name) else {   // RTLD_DEFAULT
            throw Failure(step: name, code: -1, message: "missing symbol")
        }
        return unsafeBitCast(p, to: T.self)
    }

    static func check(_ step: String, _ code: Int64) throws {
        guard code != 0 else { return }
        typealias E = @convention(c) () -> UnsafePointer<CChar>?
        let message = (try? sym("e5rt_get_last_error_message", E.self)).flatMap { $0() }.map { String(cString: $0) } ?? ""
        throw Failure(step: step, code: code, message: message)
    }

    /// A C string that lives for the process (e5rt keeps some name pointers).
    static func cstr(_ s: String) -> P { P(strdup(s)!) }

    /// Calls `e5rt_<kind>_release(&handle)`, the convention Core ML uses.
    static func release(_ kind: String, _ handle: inout P?) {
        guard handle != nil, let f = try? sym("e5rt_\(kind)_release", F1.self) else { return }
        withUnsafeMutablePointer(to: &handle) { _ = f(P($0)) }
        handle = nil
    }
}

/// One compiled multifunction program; one operation per function.
final class E5Library {
    private var ops: [String: E5Operation] = [:]
    private var lib: E5.P?

    /// - Parameters:
    ///   - mil: `model.mil` inside a compiled `.mlmodelc`.
    ///   - functions: entry points to compile.
    init(mil: URL, functions: [String]) throws {
        var compiler: E5.P?, opts: E5.P?, lib: E5.P?
        try E5.check("compiler_create", try E5.sym("e5rt_e5_compiler_create", E5.F1.self)(&compiler))
        try E5.check("options_create", try E5.sym("e5rt_e5_compiler_options_create", E5.F1.self)(&opts))
        // Experiment knobs: KOKORO_E5_MASK (device mask; Core ML passes 5 for
        // CPU+ANE, 4 is ANE only) and KOKORO_E5_ANE_OPTIONS (Core ML passes
        // "EnableLowEffortCPAllocation=true").
        let env = ProcessInfo.processInfo.environment
        let mask = UInt(env["KOKORO_E5_MASK"] ?? "") ?? 4
        try E5.check("set_mask", try E5.sym("e5rt_e5_compiler_options_set_compute_device_types_mask", E5.F2.self)(opts, E5.P(bitPattern: mask)))
        if let aneOptions = env["KOKORO_E5_ANE_OPTIONS"] {
            try E5.check("ane_options", try E5.sym("e5rt_e5_compiler_options_set_custom_ane_compiler_options", E5.F2.self)(opts, E5.cstr(aneOptions)))
        }
        var entries: [E5.P?] = functions.map { E5.cstr($0) }
        try E5.check("set_entry_points", try E5.sym("e5rt_e5_compiler_options_set_mil_entry_points", E5.F3.self)(
            opts, &entries, E5.P(bitPattern: UInt(functions.count))))
        defer { E5.release("e5_compiler_options", &opts); E5.release("e5_compiler", &compiler) }
        try E5.check("compile", try E5.sym("e5rt_e5_compiler_compile", E5.F4.self)(compiler, E5.cstr(mil.path), opts, &lib))
        self.lib = lib
        for name in functions { ops[name] = try E5Operation(library: lib, function: name) }
    }

    deinit {
        ops = [:]   // operations first, then the library they came from
        E5.release("program_library", &lib)
    }

    func operation(_ name: String) -> E5Operation? { ops[name] }
}

/// One function with its own execution stream. Not thread-safe: one caller at a time.
final class E5Operation {
    private var op: E5.P?
    private var stream: E5.P?
    private var ports: [String: E5.P?] = [:]
    private let retainIn, retainOut: E5.F3
    private let bind: E5.F2
    private let prepare, execute, reset: E5.F1
    private let encode: E5.F2
    private let setQoS: E5.F2
    private var prewire: E5.F1?
    /// Stream QoS (Core ML: 0x21). KOKORO_E5_QOS overrides (experiment).
    private static let qos = UInt(ProcessInfo.processInfo.environment["KOKORO_E5_QOS"] ?? "") ?? 0x21

    init(library: E5.P?, function: String) throws {
        var fn: E5.P?, opOpts: E5.P?, op: E5.P?, stream: E5.P?
        try E5.check("retain_function \(function)", try E5.sym("e5rt_program_library_retain_program_function", E5.F3.self)(library, E5.cstr(function), &fn))
        try E5.check("op_options", try E5.sym("e5rt_precompiled_compute_op_create_options_create_with_program_function", E5.F2.self)(&opOpts, fn))
        defer { E5.release("precompiled_compute_op_create_options", &opOpts); E5.release("program_function", &fn) }
        try E5.check("create_op", try E5.sym("e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options", E5.F2.self)(&op, opOpts))
        try E5.check("stream_create", try E5.sym("e5rt_execution_stream_create", E5.F1.self)(&stream))
        // Experiment knobs (README/Notes/ane-folded-generator.md, "Private e5rt arena"):
        // KOKORO_E5_SKIP_FENCES=1 skips the per-call I/O fences (ordering comes
        // from the host waiting on every Metal command buffer); the config goes
        // in by pointer (by value traps in GetSkipIOFences).
        let env = ProcessInfo.processInfo.environment
        if env["KOKORO_E5_SKIP_FENCES"] == "1" {
            var cfg: E5.P?
            try E5.check("stream_config", try E5.sym("e5rt_execution_stream_config_options_create", E5.F1.self)(&cfg))
            try E5.check("skip_fences", try E5.sym("e5rt_execution_stream_config_options_set_skip_io_fences", E5.F2.self)(cfg, E5.P(bitPattern: 1)))
            try withUnsafeMutablePointer(to: &cfg) { p in
                try E5.check("set_config", try E5.sym("e5rt_execution_stream_set_config_options", E5.F2.self)(stream, E5.P(p)))
            }
        }
        if let prio = env["KOKORO_E5_ANE_PRIORITY"].flatMap(UInt.init) {
            try E5.check("ane_priority", try E5.sym("e5rt_execution_stream_set_ane_execution_priority", E5.F2.self)(stream, E5.P(bitPattern: prio)))
        }
        prewire = env["KOKORO_E5_PREWIRE"] == "1" ? try E5.sym("e5rt_execution_stream_prewire_in_use_allocations", E5.F1.self) : nil
        self.op = op
        self.stream = stream
        retainIn = try E5.sym("e5rt_execution_stream_operation_retain_input_port", E5.F3.self)
        retainOut = try E5.sym("e5rt_execution_stream_operation_retain_output_port", E5.F3.self)
        bind = try E5.sym("e5rt_io_port_bind_buffer_object", E5.F2.self)
        prepare = try E5.sym("e5rt_execution_stream_operation_prepare_op_for_encode", E5.F1.self)
        encode = try E5.sym("e5rt_execution_stream_encode_operation", E5.F2.self)
        execute = try E5.sym("e5rt_execution_stream_execute_sync", E5.F1.self)
        reset = try E5.sym("e5rt_execution_stream_reset", E5.F1.self)
        setQoS = try E5.sym("e5rt_execution_stream_set_quality_of_service", E5.F2.self)
    }

    deinit {
        for key in Array(ports.keys) { var p = ports[key]!; E5.release("io_port", &p) }
        E5.release("execution_stream_operation", &op)
        E5.release("execution_stream", &stream)
    }

    private func port(_ name: String, output: Bool) throws -> E5.P? {
        let key = (output ? ">" : "<") + name
        if let p = ports[key] { return p }
        var p: E5.P?
        try E5.check("port \(name)", (output ? retainOut : retainIn)(op, E5.cstr(name), &p))
        ports[key] = p
        return p
    }

    /// Binds e5rt buffer objects to the named ports and runs synchronously, in
    /// Core ML's per-call order: reset, bind, prepare, QoS, encode, execute.
    /// The QoS is set on every call as Core ML does (0x21, user-interactive,
    /// `-[MLE5ExecutionStream _setANEExecutionPriorityWithOptions:]`): set once
    /// at creation, the generator ran ~50% slower after idle gaps on an A14.
    func run(inputs: [String: E5.P], output: (String, E5.P)) throws {
        try E5.check("reset", reset(stream))
        for (name, buf) in inputs { try E5.check("bind \(name)", bind(try port(name, output: false), buf)) }
        try E5.check("bind \(output.0)", bind(try port(output.0, output: true), output.1))
        try E5.check("prepare", prepare(op))
        try E5.check("qos", setQoS(stream, E5.P(bitPattern: Self.qos)))
        try E5.check("encode", encode(stream, op))
        try E5.check("execute", execute(stream))
        if let p = prewire {   // once, after the first run has bound real buffers
            prewire = nil
            // Fails on aliased arena buffers on macOS 26 ("Program IOSurfaces map
            // failure (0x12)"); the run itself already succeeded, so keep going.
            if p(stream) != 0 { fputs("e5rt prewire failed; continuing\n", stderr) }
        }
    }
}

/// Arena memory shared by Metal and the ANE: a few IOSurfaces ("slots",
/// split at `slotStarts`), each wrapped as one untracked MTLBuffer (callers
/// order their work on the host) and as an e5rt buffer object whose byte
/// ranges are bound to ANE ports through aliases (`create_as_alias`).
final class E5Arena {
    private struct Slot { let start: Int, size: Int, surface: IOSurfaceRef, metal: MTLBuffer; var parent: E5.P? }
    private var slots: [Slot] = []
    private struct Range: Hashable { let offset: Int, length: Int }
    private var aliases: [Range: E5.P] = [:]
    private let alias: E5.F4
    private let lock = NSLock()

    init(device: MTLDevice, slotStarts: [Int], size: Int) throws {
        alias = try E5.sym("e5rt_buffer_object_create_as_alias", E5.F4.self)
        let fromSurface = try E5.sym("e5rt_buffer_object_create_from_iosurface", E5.F3.self)
        for (i, start) in slotStarts.enumerated() {
            let bytes = (i + 1 < slotStarts.count ? slotStarts[i + 1] : size) - start
            guard let s = IOSurfaceCreate([kIOSurfaceWidth: bytes / 2, kIOSurfaceHeight: 1, kIOSurfaceBytesPerElement: 2,
                                           kIOSurfaceAllocSize: bytes] as CFDictionary) else { throw HybridGeneratorError.noMetal }
            IOSurfaceLock(s, [], nil)
            let base = IOSurfaceGetBaseAddress(s)
            IOSurfaceUnlock(s, [], nil)
            guard let m = device.makeBuffer(bytesNoCopy: base, length: bytes,
                                            options: [.storageModeShared, .hazardTrackingModeUntracked], deallocator: nil) else {
                throw HybridGeneratorError.noMetal
            }
            var parent: E5.P?
            try E5.check("buffer_from_iosurface", fromSurface(&parent, E5.P(Unmanaged.passUnretained(s).toOpaque()), E5.P(bitPattern: 0x103)))
            slots.append(Slot(start: start, size: bytes, surface: s, metal: m, parent: parent))
        }
    }

    deinit {
        for key in Array(aliases.keys) { var b: E5.P? = aliases[key]; E5.release("buffer_object", &b) }
        for i in slots.indices { E5.release("buffer_object", &slots[i].parent) }
    }

    /// The slot holding arena bytes [offset, offset + length).
    private func slot(_ offset: Int, _ length: Int) throws -> Slot {
        guard let s = slots.last(where: { $0.start <= offset }), offset + length <= s.start + s.size else {
            throw HybridGeneratorError.badPlan("arena range \(offset)+\(length) crosses a slot")
        }
        return s
    }

    /// Metal buffer and offset in it for arena bytes [offset, offset + length).
    func metal(offset: Int, length: Int) throws -> (MTLBuffer, Int) {
        let s = try slot(offset, length)
        return (s.metal, offset - s.start)
    }

    /// The e5rt buffer object for arena bytes [offset, offset + length), cached.
    func buffer(offset: Int, length: Int) throws -> E5.P {
        lock.lock(); defer { lock.unlock() }
        let key = Range(offset: offset, length: length)
        if let b = aliases[key] { return b }
        let s = try slot(offset, length)
        var b: E5.P?
        try E5.check("alias", alias(&b, s.parent, E5.P(bitPattern: UInt(offset - s.start)), E5.P(bitPattern: UInt(length))))
        aliases[key] = b!
        return b!
    }
}
