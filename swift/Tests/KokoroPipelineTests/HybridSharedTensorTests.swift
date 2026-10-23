/// SharedTensor (HybridGenerator.swift): IOSurface-backed fp16 tensors shared
/// by Core ML and Metal. Host reads must go through `hostView`, which maps the
/// same memory without locking the pixel buffer (a locked buffer is rejected as
/// a Core ML output backing on the next prediction).
import CoreML
import Metal
import XCTest
@testable import KokoroPipeline

final class HybridSharedTensorTests: XCTestCase {
    func testHostViewSeesWrittenValuesWithoutPadding() throws {
        guard #available(macOS 15.0, iOS 18.0, *), let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("needs Metal")
        }
        let t = try SharedTensor(device: device, rows: 2, channels: 3, width: 32)
        let values = (0..<(2 * 3 * 32)).map { Float($0 % 17) * 0.25 }
        values.withUnsafeBufferPointer { t.write($0.baseAddress!, count: $0.count) }
        XCTAssertEqual(floatValues(from: t.hostView), values)
        XCTAssertEqual(t.hostView.dataType, .float16)
    }

    func testUnalignedWidthFallsBackToPointerMemory() throws {
        guard #available(macOS 15.0, iOS 18.0, *), let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("needs Metal")
        }
        // 15 fp16 values = 30 bytes per row: an IOSurface would pad rows, so plain memory is used.
        let t = try SharedTensor(device: device, rows: 1, channels: 2, width: 15)
        XCTAssertNil(t.pixelBuffer)
        let values = (0..<30).map { Float($0) * 0.5 }
        values.withUnsafeBufferPointer { t.write($0.baseAddress!, count: $0.count) }
        XCTAssertEqual(floatValues(from: t.hostView), values)
        XCTAssertEqual(floatValues(from: t.array), values)
    }

    /// Short buckets go to the folded generator: the hybrid's fixed per-call
    /// cost loses below 7 s (README/Notes/ane-folded-generator.md).
    func testHybridServesOnlyBucketsFromSevenSeconds() throws {
        guard #available(macOS 15.0, iOS 18.0, *) else { throw XCTSkip("needs macOS 15") }
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let plan = #"{"core":240,"halo":8,"w0":256,"w1":2560,"w2":15520,"extra2":160,"rows":[1,3,10],"functions":[],"adain":[],"q6":2589,"har_pad":3}"#
        try plan.write(to: dir.appendingPathComponent("plan.json"), atomically: true, encoding: .utf8)
        for r in [1, 3, 10] {
            try FileManager.default.createDirectory(at: dir.appendingPathComponent("hybrid_generator_r\(r).mlpackage"),
                                                    withIntermediateDirectories: true)
        }
        XCTAssertNil(HybridGenerator.rows(bucketSec: 3, xPreFrames: 240, directory: dir))
        XCTAssertEqual(HybridGenerator.rows(bucketSec: 7, xPreFrames: 560, directory: dir), 3)
        XCTAssertEqual(HybridGenerator.rows(bucketSec: 30, xPreFrames: 2400, directory: dir), 10)
    }

    /// Without a 3-row package a 7 s bucket uses the next larger one (fewer
    /// packages, fewer generator reloads on mixed-length text).
    func testHybridUsesSmallestPackagedRowCountThatFits() throws {
        guard #available(macOS 15.0, iOS 18.0, *) else { throw XCTSkip("needs macOS 15") }
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let plan = #"{"core":224,"halo":8,"w0":240,"w1":2400,"w2":14880,"extra2":480,"rows":[2,3,6,11],"functions":[],"adain":[],"q6":2496,"har_pad":3}"#
        try plan.write(to: dir.appendingPathComponent("plan.json"), atomically: true, encoding: .utf8)
        for r in [6, 11] {
            try FileManager.default.createDirectory(at: dir.appendingPathComponent("hybrid_generator_r\(r).mlpackage"),
                                                    withIntermediateDirectories: true)
        }
        XCTAssertEqual(HybridGenerator.rows(bucketSec: 7, xPreFrames: 560, directory: dir), 6)
        XCTAssertEqual(HybridGenerator.rows(bucketSec: 15, xPreFrames: 1200, directory: dir), 6)
        XCTAssertEqual(HybridGenerator.rows(bucketSec: 30, xPreFrames: 2400, directory: dir), 11)
        // Rows follow the audio, not the bucket: an 8 s chunk in a 15 s bucket fits r6;
        // with an r4 package it would take r4 (the iOS 10 s chunk case).
        XCTAssertEqual(HybridGenerator.rows(bucketSec: 15, xPreFrames: 640, directory: dir), 6)
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("hybrid_generator_r3.mlpackage"),
                                                withIntermediateDirectories: true)
        XCTAssertEqual(HybridGenerator.rows(bucketSec: 15, xPreFrames: 640, directory: dir), 3)
    }
}
