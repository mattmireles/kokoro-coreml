// Per-model memory attribution: loads each package in its own process run,
// predicts once on zeros, and prints phys_footprint deltas.
// Usage: fpcheck <units: ne|gpu|cpu|all> <model.mlpackage|.mlmodelc>
// Build: swiftc -O tools/ane-lab/fpcheck.swift -o fpcheck
import CoreML
import Foundation

func footprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    _ = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return Double(info.phys_footprint) / 1_048_576
}

let args = CommandLine.arguments
let units: MLComputeUnits = ["ne": .cpuAndNeuralEngine, "gpu": .cpuAndGPU, "cpu": .cpuOnly][args[1]] ?? .all
var url = URL(fileURLWithPath: args[2])
let base = footprintMB()
if url.pathExtension == "mlpackage" { url = try MLModel.compileModel(at: url) }
let afterCompile = footprintMB()
let cfg = MLModelConfiguration()
cfg.computeUnits = units
let model = try MLModel(contentsOf: url, configuration: cfg)
let afterLoad = footprintMB()
var feats: [String: MLFeatureValue] = [:]
for (name, d) in model.modelDescription.inputDescriptionsByName {
    let c = d.multiArrayConstraint!
    let arr = try MLMultiArray(shape: c.shape, dataType: c.dataType)
    memset(arr.dataPointer, 0, arr.count * (c.dataType == .float16 ? 2 : 4))
    feats[name] = MLFeatureValue(multiArray: arr)
}
var peak = afterLoad
for _ in 0..<3 {
    try autoreleasepool { _ = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: feats)) }
    peak = max(peak, footprintMB())
}
print(String(format: "%@ %@: base %.0f, +compile %.0f, +load %.0f, +predict %.0f MB (total %.0f)",
             args[1], url.deletingPathExtension().lastPathComponent, base, afterCompile - base, afterLoad - afterCompile, peak - afterLoad, peak))
