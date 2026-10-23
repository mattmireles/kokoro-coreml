// SplitDuration.swift: the duration and F0Ntrain models as two Core ML programs each.
//
// Front: the PL-BERT (ALBERT) encoder and the text-encoder CNN, exported in an
// ANE-friendly layout ((1, C, 1, S) tensors, 1x1 convs, per-head attention):
// 952 of 961 ops on the Neural Engine. Back: the LSTMs and projections, fp16 on
// the CPU. Same outputs as the whole duration model (pred_dur, d, t_en, s,
// ref_s_out). M2 Air, warm, split vs the whole model's best placement:
// t64 6.6 vs 11.1 ms, t512 35.2 vs 67.0 ms (scripts/export_duration_ane.py,
// README/Notes/ane-folded-generator.md).

import CoreML
import Foundation

/// One pipeline stage: a whole `MLModel` or a split pair (`SplitDuration`,
/// `SplitF0N`) with the same inputs and outputs.
public protocol StagePredictor {
    func prediction(from input: MLFeatureProvider) throws -> MLFeatureProvider
    /// Whether the stage takes the bucket's valid-frame `mask` input.
    var takesMask: Bool { get }
}

extension MLModel: StagePredictor {
    public var takesMask: Bool { modelDescription.inputDescriptionsByName["mask"] != nil }
}

private func value(_ provider: MLFeatureProvider, _ name: String) throws -> MLFeatureValue {
    guard let v = provider.featureValue(for: name) else {
        throw PipelineError.modelContractMismatch("split stage: missing \(name)")
    }
    return v
}

public final class SplitDuration: StagePredictor {
    let front: MLModel
    let back: MLModel

    public init(front: MLModel, back: MLModel) {
        self.front = front
        self.back = back
    }

    /// Environment switch: `0` keeps the whole duration model.
    public static let envKey = "KOKORO_SPLIT_DURATION"
    public static var allowed: Bool { ProcessInfo.processInfo.environment[envKey] != "0" }

    /// Package names for a padded token length.
    public static func packageNames(tokenLength: Int) -> (front: String, back: String) {
        ("kokoro_duration_ane_front_t\(tokenLength).mlpackage", "kokoro_duration_back_t\(tokenLength).mlpackage")
    }

    /// Front on the Neural Engine, back on the CPU.
    public static let frontComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
    public static let backComputeUnits: MLComputeUnits = .cpuOnly

    public var takesMask: Bool { false }   // uses attention_mask, built with the duration inputs

    public func prediction(from input: MLFeatureProvider) throws -> MLFeatureProvider {
        let mask = try value(input, "attention_mask")
        let t0 = CFAbsoluteTimeGetCurrent()
        let f = try front.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "input_ids": try value(input, "input_ids"), "attention_mask": mask,
        ]))
        let t1 = CFAbsoluteTimeGetCurrent()
        let out = try back.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "d_en": try value(f, "d_en"), "t_cnn": try value(f, "t_cnn"),
            "ref_s": try value(input, "ref_s"), "speed": try value(input, "speed"), "attention_mask": mask,
        ]))
        if ProcessInfo.processInfo.environment["KOKORO_SPLIT_DURATION_LOG"] == "1" {
            print(String(format: "split duration: front %.1f ms, back %.1f ms", (t1 - t0) * 1000,
                         (CFAbsoluteTimeGetCurrent() - t1) * 1000))
        }
        return out
    }
}

/// F0Ntrain in two parts: the shared BiLSTM on the CPU (fp16) and the F0 and N
/// branches (AdaIN res-blocks, exact polyphase upsample) in an ANE layout, 314
/// of 314 ops on the Neural Engine. Same inputs (en, s, mask) and outputs
/// (F0_pred, N_pred, fp16) as the whole model. M2 Air, warm: T1200 17.0 vs
/// 29.3 ms whole on CPU, T120 2.4 vs 4.2 ms (scripts/export_f0ntrain_split.py).
public final class SplitF0N: StagePredictor {
    let front: MLModel
    let back: MLModel

    public init(front: MLModel, back: MLModel) {
        self.front = front
        self.back = back
    }

    /// Environment switch: `0` keeps the whole F0Ntrain model.
    public static let envKey = "KOKORO_SPLIT_F0N"
    public static var allowed: Bool { ProcessInfo.processInfo.environment[envKey] != "0" }

    public static func packageNames(tFrames: Int) -> (front: String, back: String) {
        ("kokoro_f0ntrain_front_t\(tFrames).mlpackage", "kokoro_f0ntrain_ane_back_t\(tFrames).mlpackage")
    }

    public static let frontComputeUnits: MLComputeUnits = .cpuOnly
    public static let backComputeUnits: MLComputeUnits = .cpuAndNeuralEngine

    public var takesMask: Bool { true }

    public func prediction(from input: MLFeatureProvider) throws -> MLFeatureProvider {
        let mask = try value(input, "mask")
        let f = try front.prediction(from: MLDictionaryFeatureProvider(dictionary: ["en": try value(input, "en"), "mask": mask]))
        return try back.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "x": try value(f, "x"), "s": try value(input, "s"), "mask": mask,
        ]))
    }
}
