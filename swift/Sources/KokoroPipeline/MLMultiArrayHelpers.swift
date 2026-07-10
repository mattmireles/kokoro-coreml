/// Helpers for CoreML MLMultiArray construction, matrix operations,
/// and zero-padding to bucket geometry.
///
/// These replace the numpy array operations in the Python pipeline
/// (``build_decoder_har_post_inputs_np`` padding, ``d @ alignment``
/// matrix multiply, etc.) with native Swift + Accelerate.

import CoreML
import Accelerate

// MARK: - Validation

public enum PipelineValidationError: Error, LocalizedError {
    case unsupportedDurationDataType(MLMultiArrayDataType)
    case invalidArrayShape(operation: String, expected: String, actual: [Int])
    case invalidDurationAgreement(inputKey: String, canonical: Double, observed: Double, toleranceFraction: Double)

    public var errorDescription: String? {
        switch self {
        case .unsupportedDurationDataType(let dataType):
            return "Unsupported pred_dur MLMultiArray data type: \(dataType)"
        case .invalidArrayShape(let operation, let expected, let actual):
            return "\(operation) expected \(expected), got shape \(actual)"
        case .invalidDurationAgreement(let inputKey, let canonical, let observed, let toleranceFraction):
            return "Config F duration mismatch for \(inputKey): observed \(String(format: "%.3f", observed))s vs canonical \(String(format: "%.3f", canonical))s exceeds \(Int(toleranceFraction * 100))% tolerance"
        }
    }
}

/// Read Core ML ``pred_dur`` output into positive integer duration frames.
///
/// ``export_duration.py`` returns ``pred_dur`` as an integer tensor. Reading the
/// raw buffer as Float corrupts every value into a tiny subnormal number, which
/// then rounds to zero and collapses the utterance to one frame per token. Use
/// MLMultiArray indexed access instead of raw pointer traversal so Core ML output
/// strides are respected too.
public func readDurationFrames(from array: MLMultiArray, validCount: Int? = nil) throws -> [Int] {
    let count = max(0, min(validCount ?? array.count, array.count))
    var frames = [Int](repeating: 0, count: count)
    let rank = array.shape.count

    for i in 0..<count {
        let index: [NSNumber]
        if rank == 1 {
            index = [NSNumber(value: i)]
        } else if rank == 2 {
            index = [NSNumber(value: 0), NSNumber(value: i)]
        } else {
            throw PipelineValidationError.unsupportedDurationDataType(array.dataType)
        }

        let value = array[index]
        if array.dataType == .int32 {
            frames[i] = max(1, value.intValue)
        } else {
            frames[i] = max(1, Int(round(value.doubleValue)))
        }
    }

    return frames
}

/// Read a float MLMultiArray in logical row-major order.
///
/// Core ML outputs may have non-trivial strides, so callers that need a flat
/// waveform or tensor snapshot must not assume `dataPointer` is linearly
/// addressable in logical index order.
public func floatValues(from array: MLMultiArray, limit: Int? = nil) -> [Float] {
    let shape = array.shape.map { $0.intValue }
    let count = max(0, min(limit ?? array.count, array.count))
    if count == 0 { return [] }

    let strides = array.strides.map { $0.intValue }
    if array.dataType == .float32 && isContiguousRowMajor(shape: shape, strides: strides) {
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    if array.dataType == .float32 {
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        return stridedValues(from: ptr, shape: shape, strides: strides, limit: count) { $0 }
    }
    if array.dataType == .float16 {
        let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
        return stridedValues(from: ptr, shape: shape, strides: strides, limit: count) { Float($0) }
    }

    var values = [Float]()
    values.reserveCapacity(count)
    for offset in 0..<count {
        values.append(array[multiIndex(offset: offset, shape: shape)].floatValue)
    }
    return values
}

private func stridedValues<Element>(
    from ptr: UnsafeMutablePointer<Element>,
    shape: [Int],
    strides: [Int],
    limit: Int,
    convert: (Element) -> Float
) -> [Float] {
    var values = [Float]()
    values.reserveCapacity(limit)

    switch shape.count {
    case 1:
        for i in 0..<min(shape[0], limit) {
            values.append(convert(ptr[i * strides[0]]))
        }
    case 2:
        outer: for i in 0..<shape[0] {
            let iBase = i * strides[0]
            for j in 0..<shape[1] {
                values.append(convert(ptr[iBase + j * strides[1]]))
                if values.count == limit { break outer }
            }
        }
    case 3:
        outer: for i in 0..<shape[0] {
            let iBase = i * strides[0]
            for j in 0..<shape[1] {
                let jBase = iBase + j * strides[1]
                for k in 0..<shape[2] {
                    values.append(convert(ptr[jBase + k * strides[2]]))
                    if values.count == limit { break outer }
                }
            }
        }
    default:
        for offset in 0..<limit {
            let index = multiIndex(offset: offset, shape: shape).map { $0.intValue }
            var physicalOffset = 0
            for i in 0..<min(index.count, strides.count) {
                physicalOffset += index[i] * strides[i]
            }
            values.append(convert(ptr[physicalOffset]))
        }
    }

    return values
}

private func isContiguousRowMajor(shape: [Int], strides: [Int]) -> Bool {
    guard shape.count == strides.count else { return false }
    var expectedStride = 1
    for i in stride(from: shape.count - 1, through: 0, by: -1) {
        if strides[i] != expectedStride && shape[i] > 1 {
            return false
        }
        expectedStride *= max(1, shape[i])
    }
    return true
}

/// Fail fast when Config F produces an audio length that cannot be the same
/// utterance measured by the bakeoff manifest.
public func validateDurationAgreement(
    inputKey: String,
    canonical: Double?,
    observed: Double,
    toleranceFraction: Double = 0.15
) throws {
    guard let canonical, canonical > 0, observed > 0 else { return }
    let delta = abs(observed - canonical) / canonical
    if delta > toleranceFraction {
        throw PipelineValidationError.invalidDurationAgreement(
            inputKey: inputKey,
            canonical: canonical,
            observed: observed,
            toleranceFraction: toleranceFraction
        )
    }
}

private func multiIndex(offset: Int, shape: [Int]) -> [NSNumber] {
    guard !shape.isEmpty else { return [] }
    var remainder = offset
    var result = [Int](repeating: 0, count: shape.count)
    for dimIndex in stride(from: shape.count - 1, through: 0, by: -1) {
        let dim = max(1, shape[dimIndex])
        result[dimIndex] = remainder % dim
        remainder /= dim
    }
    return result.map { NSNumber(value: $0) }
}

// MARK: - MLMultiArray Construction

/// Create a 3D MLMultiArray of shape (1, channels, time) filled with zeros.
///
/// Matches the numpy pattern: ``np.zeros((1, C, T), dtype=np.float32)``
public func makeZeroArray3D(channels: Int, time: Int) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: [1, NSNumber(value: channels), NSNumber(value: time)], dataType: .float32)
    let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
    memset(ptr, 0, channels * time * MemoryLayout<Float>.size)
    return arr
}

/// Create a 2D MLMultiArray of shape (1, dim) filled with zeros.
public func makeZeroArray2D(dim: Int) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: [1, NSNumber(value: dim)], dataType: .float32)
    let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
    memset(ptr, 0, dim * MemoryLayout<Float>.size)
    return arr
}

/// Copy a flat Float array into a pre-allocated MLMultiArray, zero-padding if needed.
///
/// The source data is copied starting from index 0. If the source is shorter
/// than the MLMultiArray, the remainder stays zero. If longer, it's truncated.
public func copyInto(array: MLMultiArray, from source: [Float]) {
    let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
    let count = min(source.count, array.count)
    _ = source.withUnsafeBufferPointer { srcBuf in
        memcpy(ptr, srcBuf.baseAddress!, count * MemoryLayout<Float>.size)
    }
}

// MARK: - Matrix Multiply

/// Align token-domain hidden states to frame-domain features by repeating each
/// token vector according to ``predDur``.
///
/// Source layout is `(1, tokens, channels)` and result layout is
/// `(1, channels, frameCount)`. This is the direct equivalent of
/// `source.transpose(-1, -2) @ one_hot_alignment`, but avoids materializing a
/// huge sparse alignment matrix and avoids dense GEMM over zeros.
public func alignTokenMajorToFrames(
    source: MLMultiArray,
    predDur: [Int],
    channels: Int,
    frameCount: Int
) throws -> MLMultiArray {
    let sourceShape = source.shape.map { $0.intValue }
    guard sourceShape.count >= 3, sourceShape[0] == 1,
          sourceShape[2] == channels else {
        throw PipelineValidationError.invalidArrayShape(
            operation: "alignTokenMajorToFrames",
            expected: "(1, tokens, \(channels))",
            actual: sourceShape
        )
    }
    let tokenCount = min(predDur.count, sourceShape[1])
    let strides = source.strides.map { $0.intValue }
    let canUsePointer = source.dataType == .float32 && strides.count >= 3

    if canUsePointer {
        let srcPtr = source.dataPointer.assumingMemoryBound(to: Float.self)
        return try alignValuesToFrames(
            predDur: predDur,
            channels: channels,
            tokenCount: tokenCount,
            frameCount: frameCount
        ) { token, channel in
            srcPtr[token * strides[1] + channel * strides[2]]
        }
    }

    return try alignValuesToFrames(
        predDur: predDur,
        channels: channels,
        tokenCount: tokenCount,
        frameCount: frameCount
    ) { token, channel in
        source[[0, token, channel] as [NSNumber]].floatValue
    }
}

/// Align token-domain features to frame-domain features by repeating each
/// token value according to ``predDur``.
///
/// Source layout is `(1, channels, tokens)` and result layout is
/// `(1, channels, frameCount)`. This is the direct equivalent of
/// `source @ one_hot_alignment`, without the sparse matrix and dense GEMM.
public func alignChannelMajorToFrames(
    source: MLMultiArray,
    predDur: [Int],
    channels: Int,
    frameCount: Int
) throws -> MLMultiArray {
    let sourceShape = source.shape.map { $0.intValue }
    guard sourceShape.count >= 3, sourceShape[0] == 1,
          sourceShape[1] == channels else {
        throw PipelineValidationError.invalidArrayShape(
            operation: "alignChannelMajorToFrames",
            expected: "(1, \(channels), tokens)",
            actual: sourceShape
        )
    }
    let tokenCount = min(predDur.count, sourceShape[2])
    let strides = source.strides.map { $0.intValue }
    let canUsePointer = source.dataType == .float32 && strides.count >= 3

    if canUsePointer {
        let srcPtr = source.dataPointer.assumingMemoryBound(to: Float.self)
        return try alignValuesToFrames(
            predDur: predDur,
            channels: channels,
            tokenCount: tokenCount,
            frameCount: frameCount
        ) { token, channel in
            srcPtr[channel * strides[1] + token * strides[2]]
        }
    }

    return try alignValuesToFrames(
        predDur: predDur,
        channels: channels,
        tokenCount: tokenCount,
        frameCount: frameCount
    ) { token, channel in
        source[[0, channel, token] as [NSNumber]].floatValue
    }
}

private func alignValuesToFrames(
    predDur: [Int],
    channels: Int,
    tokenCount: Int,
    frameCount: Int,
    valueAt: (Int, Int) -> Float
) throws -> MLMultiArray {
    let result = try makeZeroArray3D(channels: channels, time: frameCount)
    let dstPtr = result.dataPointer.assumingMemoryBound(to: Float.self)
    var frameStart = 0

    for token in 0..<tokenCount {
        let repeatCount = max(0, min(predDur[token], frameCount - frameStart))
        if repeatCount == 0 { continue }
        for channel in 0..<channels {
            let value = valueAt(token, channel)
            let dstBase = channel * frameCount + frameStart
            for frameOffset in 0..<repeatCount {
                dstPtr[dstBase + frameOffset] = value
            }
        }
        frameStart += repeatCount
        if frameStart >= frameCount { break }
    }

    return result
}

// MARK: - Zero-Pad to Bucket Geometry

/// Zero-pad a 3D array (1, C, T_src) to (1, C, T_target).
///
/// Matches: ``asr_pad = np.zeros((1, 512, frame_count)); asr_pad[:,:,:t] = asr[:,:,:t]``
///
/// Uses subscript access (not raw pointer arithmetic) to handle MLMultiArray
/// strides correctly. CoreML model outputs may have non-trivial strides.
public func zeroPad3D(source: MLMultiArray, channels: Int, targetTime: Int) throws -> MLMultiArray {
    let sourceShape = source.shape.map { $0.intValue }
    if sourceShape.count >= 3,
       sourceShape[0] == 1,
       sourceShape[1] == channels,
       sourceShape[2] == targetTime {
        return source
    }
    guard sourceShape.count >= 3, sourceShape[0] == 1, sourceShape[1] == channels else {
        throw PipelineValidationError.invalidArrayShape(
            operation: "zeroPad3D",
            expected: "(1, \(channels), time)",
            actual: sourceShape
        )
    }

    let result = try makeZeroArray3D(channels: channels, time: targetTime)
    let srcTime = sourceShape[2]
    let copyTime = min(srcTime, targetTime)

    // Check if source has standard contiguous strides (batch=C*T, channel=T, time=1)
    let srcStrides = source.strides.map { $0.intValue }
    let isContiguous = srcStrides.count >= 3 && srcStrides[2] == 1 && srcStrides[1] == srcTime

    if isContiguous {
        // Fast path: memcpy per channel
        let srcPtr = source.dataPointer.assumingMemoryBound(to: Float.self)
        let dstPtr = result.dataPointer.assumingMemoryBound(to: Float.self)
        for c in 0..<channels {
            let srcOffset = c * srcTime
            let dstOffset = c * targetTime
            memcpy(dstPtr + dstOffset, srcPtr + srcOffset, copyTime * MemoryLayout<Float>.size)
        }
    } else {
        // Safe path: subscript access for non-contiguous layouts
        let dstPtr = result.dataPointer.assumingMemoryBound(to: Float.self)
        for c in 0..<channels {
            for t in 0..<copyTime {
                dstPtr[c * targetTime + t] = source[[0, c, t] as [NSNumber]].floatValue
            }
        }
    }
    return result
}

/// Zero-pad flat channel-major values to a 3D ``MLMultiArray`` of shape
/// `(1, channels, targetTime)`.
///
/// This is the direct path for Swift-built tensors such as `har`, where the
/// source is already a flat `(channels, sourceTime)` buffer. It avoids creating
/// a temporary source ``MLMultiArray`` only to copy the same channel rows again.
public func zeroPad3D(
    sourceValues: [Float],
    channels: Int,
    sourceTime: Int,
    targetTime: Int
) throws -> MLMultiArray {
    let expectedCount = channels * sourceTime
    guard channels > 0, sourceTime >= 0, targetTime >= 0, sourceValues.count == expectedCount else {
        throw PipelineValidationError.invalidArrayShape(
            operation: "zeroPad3D",
            expected: "(1, \(channels), \(sourceTime)) flat channel-major count \(expectedCount)",
            actual: [sourceValues.count]
        )
    }

    let result = try makeZeroArray3D(channels: channels, time: targetTime)
    let copyTime = max(0, min(sourceTime, targetTime))
    guard copyTime > 0 else { return result }

    let dstPtr = result.dataPointer.assumingMemoryBound(to: Float.self)
    sourceValues.withUnsafeBufferPointer { srcBuf in
        guard let srcBase = srcBuf.baseAddress else { return }
        for c in 0..<channels {
            let srcOffset = c * sourceTime
            let dstOffset = c * targetTime
            memcpy(dstPtr + dstOffset, srcBase + srcOffset, copyTime * MemoryLayout<Float>.size)
        }
    }

    return result
}

/// Zero-pad a 1D array (1, T_src) to (1, T_target).
///
/// Matches: ``f0_pad = np.zeros((1, full_f0_len)); f0_pad[:,:t] = f0[:,:t]``
public func zeroPad1D(source: [Float], targetLength: Int) -> [Float] {
    var result = [Float](repeating: 0, count: targetLength)
    let copyLen = min(source.count, targetLength)
    for i in 0..<copyLen {
        result[i] = source[i]
    }
    return result
}

// MARK: - Shape Introspection

/// Extract input shapes from a CoreML model spec.
///
/// Returns a dictionary mapping input name → shape array.
public func inputShapes(from model: MLModel) -> [String: [Int]] {
    var result: [String: [Int]] = [:]
    let desc = model.modelDescription
    for (name, feature) in desc.inputDescriptionsByName {
        if let constraint = feature.multiArrayConstraint {
            result[name] = constraint.shape.map { $0.intValue }
        }
    }
    return result
}
