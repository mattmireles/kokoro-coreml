/// A Neural Engine load that fails is retried on the GPU.
///
/// The placement measures a stage at its largest sizes and lets the smaller
/// ones follow, but a smaller program can still be one the Neural Engine
/// compiler rejects: on the M1 (macOS 26.6) the duration t64 program fails
/// with "ANECCompile() FAILED" (through a multifunction package the same
/// failure surfaces as "functionName must be nil unless the model type is
/// ML Program") while t128 and above compile. Falling back keeps synthesis
/// working at the GPU's speed for that one size.

import CoreML

extension MLModel {
    /// The units to retry on after a failed load, or nil when the load was
    /// not a Neural Engine one.
    public static func fallbackComputeUnits(after failed: MLComputeUnits) -> MLComputeUnits? {
        failed == .cpuAndNeuralEngine ? .cpuAndGPU : nil
    }

    /// Runs `load` with `configuration`; when that throws for a Neural Engine
    /// configuration, runs it again with the same configuration on the GPU.
    public static func withGPUFallback<T>(_ configuration: MLModelConfiguration, log: ((String) -> Void)? = nil, _ load: (MLModelConfiguration) throws -> T) throws -> T {
        do {
            return try load(configuration)
        } catch {
            guard let units = fallbackComputeUnits(after: configuration.computeUnits) else { throw error }
            log?("Neural Engine load failed (\(error.localizedDescription)); retrying on the GPU")
            let retry = configuration.copy() as! MLModelConfiguration
            retry.computeUnits = units
            return try load(retry)
        }
    }
}
