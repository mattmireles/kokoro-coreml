import CoreML
import XCTest
@testable import KokoroPipeline

final class MLMultiArrayBindingTests: XCTestCase {

    func testReadDurationFramesFromInt32PredDur() throws {
        let arr = try MLMultiArray(shape: [1, 5], dataType: .int32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Int32.self)
        [13, 1, 2, 0, -4].enumerated().forEach { idx, value in
            ptr[idx] = Int32(value)
        }

        XCTAssertEqual(try readDurationFrames(from: arr, validCount: 3), [13, 1, 2])
        XCTAssertEqual(try readDurationFrames(from: arr), [13, 1, 2, 1, 1])
    }

    func testReadDurationFramesFromFloatPredDur() throws {
        let arr = try MLMultiArray(shape: [1, 4], dataType: .float32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
        [12.6, 0.2, -3.0, 2.4].enumerated().forEach { idx, value in
            ptr[idx] = Float(value)
        }

        XCTAssertEqual(try readDurationFrames(from: arr), [13, 1, 1, 2])
    }

    func testFloatValuesReadsLogicalOrderForStridedArray() throws {
        let storage = UnsafeMutablePointer<Float>.allocate(capacity: 6)
        for idx in 0..<6 {
            storage[idx] = -100.0 - Float(idx)
        }
        storage[0] = 1.0
        storage[2] = 2.0
        storage[4] = 3.0

        let arr = try MLMultiArray(
            dataPointer: UnsafeMutableRawPointer(storage),
            shape: [1, 3],
            dataType: .float32,
            strides: [6, 2],
            deallocator: { pointer in
                pointer.deallocate()
            }
        )

        XCTAssertEqual(floatValues(from: arr), [1.0, 2.0, 3.0])
    }

    func testFloatValuesHonorsLimitForContiguousArray() throws {
        let arr = try MLMultiArray(shape: [1, 4], dataType: .float32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
        [7.0, 8.0, 9.0, 10.0].enumerated().forEach { idx, value in
            ptr[idx] = Float(value)
        }

        XCTAssertEqual(floatValues(from: arr, limit: 0), [])
        XCTAssertEqual(floatValues(from: arr, limit: 2), [7.0, 8.0])
        XCTAssertEqual(floatValues(from: arr, limit: 10), [7.0, 8.0, 9.0, 10.0])
    }

    func testFloatValuesReadsLogicalOrderForStridedFloat16Array() throws {
        let storage = UnsafeMutablePointer<Float16>.allocate(capacity: 6)
        for idx in 0..<6 {
            storage[idx] = Float16(-100.0 - Float(idx))
        }
        storage[0] = Float16(1.25)
        storage[2] = Float16(-0.5)
        storage[4] = Float16(3.5)

        let arr = try MLMultiArray(
            dataPointer: UnsafeMutableRawPointer(storage),
            shape: [1, 3],
            dataType: .float16,
            strides: [6, 2],
            deallocator: { pointer in
                pointer.deallocate()
            }
        )

        XCTAssertEqual(floatValues(from: arr), [1.25, -0.5, 3.5])
        XCTAssertEqual(floatValues(from: arr, limit: 2), [1.25, -0.5])
    }

    func testAlignTokenMajorToFramesRepeatsTokensWithoutAlignmentMatrix() throws {
        let source = try makeFloatArray(
            shape: [1, 3, 2],
            values: [
                10.0, 100.0,
                20.0, 200.0,
                30.0, 300.0,
            ]
        )

        let aligned = try alignTokenMajorToFrames(
            source: source,
            predDur: [1, 2, 1],
            channels: 2,
            frameCount: 4
        )

        XCTAssertEqual(floatValues(from: aligned), [10.0, 20.0, 20.0, 30.0, 100.0, 200.0, 200.0, 300.0])
    }

    func testAlignChannelMajorToFramesRepeatsTokensWithoutAlignmentMatrix() throws {
        let source = try makeFloatArray(
            shape: [1, 2, 3],
            values: [
                1.0, 2.0, 3.0,
                10.0, 20.0, 30.0,
            ]
        )

        let aligned = try alignChannelMajorToFrames(
            source: source,
            predDur: [1, 2, 1],
            channels: 2,
            frameCount: 4
        )

        XCTAssertEqual(floatValues(from: aligned), [1.0, 2.0, 2.0, 3.0, 10.0, 20.0, 20.0, 30.0])
    }

    func testAlignHelpersRejectWrongChannelShape() throws {
        let tokenMajor = try makeFloatArray(shape: [1, 2, 3], values: [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(
            try alignTokenMajorToFrames(source: tokenMajor, predDur: [1, 1], channels: 2, frameCount: 2)
        )

        let channelMajor = try makeFloatArray(shape: [1, 3, 2], values: [1, 2, 3, 4, 5, 6])
        XCTAssertThrowsError(
            try alignChannelMajorToFrames(source: channelMajor, predDur: [1, 1], channels: 2, frameCount: 2)
        )
    }

    func testZeroPad3DReusesExactShapeArray() throws {
        let source = try makeFloatArray(shape: [1, 2, 3], values: [1, 2, 3, 4, 5, 6])
        let padded = try zeroPad3D(source: source, channels: 2, targetTime: 3)

        XCTAssertTrue(source === padded)
        XCTAssertEqual(floatValues(from: padded), [1, 2, 3, 4, 5, 6])
    }

    func testZeroPad3DFromFlatChannelMajorValues() throws {
        let padded = try zeroPad3D(
            sourceValues: [1, 2, 3, 10, 20, 30],
            channels: 2,
            sourceTime: 3,
            targetTime: 5
        )

        XCTAssertEqual(padded.shape.map { $0.intValue }, [1, 2, 5])
        XCTAssertEqual(floatValues(from: padded), [1, 2, 3, 0, 0, 10, 20, 30, 0, 0])
    }

    func testZeroPad3DFromFlatRejectsMismatchedCount() throws {
        XCTAssertThrowsError(
            try zeroPad3D(
                sourceValues: [1, 2, 3, 4, 5],
                channels: 2,
                sourceTime: 3,
                targetTime: 5
            )
        )
    }

    func testValidateDurationAgreementRejectsHalfLengthAudio() throws {
        XCTAssertThrowsError(
            try validateDurationAgreement(inputKey: "15s", canonical: 13.9, observed: 6.4)
        )
    }

    func testValidateDurationAgreementAllowsCloseAudio() throws {
        XCTAssertNoThrow(
            try validateDurationAgreement(inputKey: "15s", canonical: 13.9, observed: 13.7)
        )
    }

    func testPcmJoinerCrossfadesSyntheticDiscontinuity() {
        let first = Array(repeating: Float(1.0), count: 240)
        let second = Array(repeating: Float(-1.0), count: 240)

        let joined = PcmJoiner.join(segments: [first, second], sampleRate: 24_000, crossfadeMs: 5)

        XCTAssertLessThan(joined.count, first.count + second.count)
        XCTAssertGreaterThan(joined[120], -1.0)
        XCTAssertLessThan(joined[120], 1.0)
    }

    func testPcmJoinerDefaultCrossfadeMatchesProductionPolicy() {
        XCTAssertEqual(PcmJoiner.defaultCrossfadeMs, 5.0)
    }

    func testPcmJoinerPreservesConcatenationWhenDisabled() {
        let joined = PcmJoiner.join(
            segments: [[1.0, 2.0], [], [3.0, 4.0]],
            sampleRate: 24_000,
            crossfadeMs: 0
        )

        XCTAssertEqual(joined, [1.0, 2.0, 3.0, 4.0])
    }

    func testTensorDumpWriterWritesManifestAndPayloads() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("kokoro-tensor-dump-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        var writer = try TensorDumpWriter(directory: dir)
        try writer.writeInt32Array(name: "tokens", values: [1, 2, 3], shape: [1, 3])
        try writer.writeFloatArray(name: "waveform", values: [0.0, 0.25, -0.5], shape: [3])
        try writer.writeManifest(metadata: ["producer": "test"])

        let manifestURL = dir.appendingPathComponent("tensor_manifest.json")
        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any]
        let tensors = manifest?["tensors"] as? [[String: Any]]

        XCTAssertEqual(manifest?["schema_version"] as? Int, 1)
        XCTAssertEqual((manifest?["metadata"] as? [String: Any])?["producer"] as? String, "test")
        XCTAssertEqual(tensors?.count, 2)
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("tokens.i32").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("waveform.f32").path))

        let reader = try TensorDumpReader(directory: dir)
        let waveform = try reader.readFloatArray(name: "waveform")
        XCTAssertEqual(waveform.shape, [3])
        XCTAssertEqual(waveform.values, [0.0, 0.25, -0.5])

        let arr = try makeFloatArray(shape: waveform.shape, values: waveform.values)
        XCTAssertEqual(arr.count, 3)
        XCTAssertEqual(arr[[1] as [NSNumber]].floatValue, 0.25)
    }
}
