/// Regression tests for the row-folded (Neural Engine) generator's host side:
/// fold/unfold geometry, AdaIN masks, and the exp/sin/iSTFT tail against
/// PyTorch CustomSTFT.inverse (fixture printed by the snippet in
/// scripts/probe_generator_ane_fold.py's `tail`).
import CoreML
import XCTest
@testable import KokoroPipeline

final class FoldedGeneratorTests: XCTestCase {
    /// Deterministic logits for 2 x_pre frames: (22, 121), same formula as the fixture.
    private func fixtureSpec(frames: Int) -> [Float] {
        (0..<(22 * frames)).map { i in Float(0.5 * sin(Double(i) * 0.37) + 0.1 * cos(Double(i) * 0.011)) }
    }

    func testTailMatchesTorchCustomSTFTInverseAcrossRows() throws {
        let frames = 2 * 60 + 1
        let spec = fixtureSpec(frames: frames)
        // Two rows of one frame each, halo 1: every row seam and both ends are exercised.
        let g = FoldGeometry(rows: 2, core: 1, halo: 1)
        let folded = try foldRows(spec, channels: 22, time: frames, rate: FoldGeometry.stage2Rate,
                                  extra: FoldGeometry.stage2Extra, geometry: g)
        let audio = foldedGeneratorAudio(logits: folded, validXPreFrames: 2, geometry: g)
        XCTAssertEqual(audio.count, Self.expected.count)
        for (i, (a, e)) in zip(audio, Self.expected).enumerated() {
            XCTAssertEqual(a, e, accuracy: 2e-5, "sample \(i)")
        }
    }

    /// The tail runs in parallel frame chunks; chunk seams must add up to the
    /// single-chunk result (the chunked tail cut ~12 ms off a 30 s clip on an A14).
    func testTailChunksMatchSingleChunk() throws {
        let xFrames = 37
        let frames = xFrames * 60 + 1
        let spec = fixtureSpec(frames: frames)
        let g = FoldGeometry(rows: 3, core: 13, halo: 2)
        let folded = try foldRows(spec, channels: 22, time: frames, rate: FoldGeometry.stage2Rate,
                                  extra: FoldGeometry.stage2Extra, geometry: g)
        let one = foldedGeneratorAudio(logits: folded, validXPreFrames: xFrames, geometry: g, chunks: 1)
        for chunks in [3, 4, 7] {
            let many = foldedGeneratorAudio(logits: folded, validXPreFrames: xFrames, geometry: g, chunks: chunks)
            XCTAssertEqual(many.count, one.count)
            let worst = zip(many, one).map { abs($0 - $1) }.max() ?? 0
            XCTAssertLessThan(worst, 1e-5, "chunks \(chunks)")
        }
    }

    func testStatsMaskCountsEveryValidIndexOnce() throws {
        let g = FoldGeometry(rows: 3, core: 4, halo: 2)
        let valid = 11 * FoldGeometry.stage1Rate  // last row partly padding
        let m = try foldMasks(rate: FoldGeometry.stage1Rate, extra: 0, valid: valid, geometry: g)
        let stats = floatValues(from: m.stats)
        let zero = floatValues(from: m.zero)
        XCTAssertEqual(stats.reduce(0, +), Float(valid))
        for (s, z) in zip(stats, zero) { XCTAssertLessThanOrEqual(s, z) }
    }

    /// Neural Engine outputs are fp16 with padded outer strides; readRun must
    /// honour the stride and convert whole runs (the per-element path cost ~1 s
    /// per 30 s clip on an M1).
    func testReadRunConvertsStridedFloat16Rows() throws {
        var storage = [Float16](repeating: 0, count: 16)
        for i in 0..<5 { storage[i] = Float16(Float(i) * 0.5) }      // row 0
        for i in 0..<5 { storage[8 + i] = Float16(Float(i) - 2) }    // row 1 at stride 8
        let array = try storage.withUnsafeMutableBytes { raw in
            try MLMultiArray(dataPointer: raw.baseAddress!, shape: [2, 5], dataType: .float16, strides: [8, 1])
        }
        var out = [Float](repeating: -9, count: 5)
        out.withUnsafeMutableBufferPointer { readRun(array, offset: 8, count: 5, into: $0.baseAddress!) }
        XCTAssertEqual(out, [-2, -1, 0, 1, 2])
        out.withUnsafeMutableBufferPointer { readRun(array, offset: 1, count: 3, into: $0.baseAddress!) }
        XCTAssertEqual(Array(out[0..<3]), [0.5, 1.0, 1.5])
    }

    /// Macs and pre-A17 iPhones must get the polynomial Snake first: their ANE
    /// has no `sin`, and the native package falls off the ANE there.
    func testFoldedCandidatesPreferPolyOffA17() {
        let c = PipelineConstants.foldedGeneratorCandidates(bucketSec: 30)
        XCTAssertEqual(Set(c), ["kokoro_generator_fold_30s.mlpackage", "kokoro_generator_fold_poly_30s.mlpackage"])
        #if os(macOS)
        XCTAssertFalse(PipelineConstants.aneHasNativeSin)
        XCTAssertEqual(c.first, "kokoro_generator_fold_poly_30s.mlpackage")
        #endif
    }

    /// Duration units by size (M1 measurements in the note): small graphs are
    /// fastest on the CPU, large ones on the ANE; never the GPU, which was
    /// slowest at every size. iOS stays on the CPU (ANECCompile failures).
    func testDurationComputeUnitsBySize() {
        #if os(macOS)
        XCTAssertEqual(PipelineConstants.durationComputeUnits(tokenLength: 64), .cpuOnly)
        XCTAssertEqual(PipelineConstants.durationComputeUnits(tokenLength: 128), .cpuAndNeuralEngine)
        XCTAssertEqual(PipelineConstants.durationComputeUnits(tokenLength: 512), .cpuAndNeuralEngine)
        #else
        XCTAssertEqual(PipelineConstants.durationComputeUnits(tokenLength: 512),
                       PipelineConstants.aneHasNativeSin ? .cpuAndNeuralEngine : .cpuOnly)
        #endif
    }

    /// F0Ntrain on the CPU: fastest at every measured size, GPU slowest.
    func testF0ntrainRunsOnCPU() {
        XCTAssertEqual(PipelineConstants.f0ntrainComputeUnits, .cpuOnly)
    }

    func testGeometryWidthsStayUnderNeuralEngineAxisLimit() {
        let g = FoldGeometry(rows: 20, core: 120, halo: 8)
        XCTAssertEqual(g.w2, 8165)
        XCTAssertEqual(g.w2 % 5, 0)
        XCTAssertLessThanOrEqual(g.w2, 16_384)
        XCTAssertEqual(g.capacity, 2400)
    }

    private static let expected: [Float] = [
-0.0004596412, -0.04559993, 0.04691267, -0.03329174, 0.01788039, 0.01595682, -0.03170362, 0.03507027, -0.01048718, 0.01804001,
0.03320903, -0.02077341, 0.02962107, 0.02448761, 0.0153479, 0.03530064, -0.001062179, 0.01571552, 0.05276994, 0.01669436,
0.02259003, 0.01634002, 0.003103856, 0.06398498, 0.02451951, 0.005854763, 0.02010372, -0.001799278, 0.05953494, 0.03545046,
-0.003106073, 0.008661183, -0.002627596, 0.04798996, 0.04600014, 0.0002173539, -0.009089928, -0.006585095, 0.03424754, 0.05532621,
0.01364039, -0.02152454, -0.01541315, 0.01647295, 0.0608433, 0.03133141, -0.02027682, -0.02393023, -0.007747598, 0.0559506,
0.04425857, -0.00320702, -0.02431301, -0.03484866, 0.03606854, 0.0429169, 0.0225378, -0.01295408, -0.05548675, 0.006748928,
0.02450849, 0.04218649, 0.00439165, -0.06260641, -0.01749294, -0.002525473, 0.04370238, 0.01698671, -0.05923655, -0.02521553,
-0.02398546, 0.0275305, 0.02064194, -0.05501576, -0.01715995, -0.03074016, 0.003995149, 0.02133534, -0.05445508, -0.002302163,
-0.02228448, -0.01695604, 0.02614648, -0.05190182, 0.01073931, -0.002577163, -0.03011508, 0.03439219, -0.03811084, 0.01704668,
0.0214464, -0.03226635, 0.03834224, -0.009720197, 0.01607174, 0.03907236, -0.02150897, 0.03123697, 0.02602985, 0.01254198,
0.04095628, -0.001581559, 0.01548019, 0.05476557, 0.01349961, 0.02765167, 0.01625804, 0.001364795, 0.0659086, 0.02153306,
0.01029121, 0.02050223, -0.004415877, 0.06085743, 0.03309253, 0.0009801257, 0.009560563, -0.005540337, 0.0483242, 0.04432131,
0.00434746, -0.007524179, -0.009358436, 0.03336718, 0.05397303, 0.01808725, -0.01896587, -0.01763636, 0.01430623, 0.05916836,
0.03603992, -0.01642611, -0.02508248, -0.01101883, 0.05331365, 0.04880484, 0.001887478, -0.02386478, -0.03867152, 0.03226849,
0.04674308, 0.02826886, -0.01071459, -0.05907053, 0.002271558, 0.02733641, 0.04757136, 0.008060616, -0.06534597, -0.02173213,
-0.0004461892, 0.04789969, 0.02141097, -0.06102674, -0.02848166, -0.02203168, 0.03021856, 0.02527538, -0.05611858, -0.01931987,
-0.02826995, 0.005301567, 0.02591009, -0.05511433, -0.003791216, -0.01886869, -0.01674364, 0.03045876, -0.05211215, 0.009215404,
0.001904607, -0.03068415, 0.03803152, -0.03767045, 0.01484414, 0.02674659, -0.03322824, 0.04067465, -0.008450839, 0.01288947,
0.04463229, -0.02239507, 0.0317351, 0.02805172, 0.00856651, 0.04616366, -0.002003449, 0.0141234, 0.05712765, 0.009295131,
0.03215686, 0.01644777, -0.001316516, 0.06801275, 0.01770823, 0.01414582, 0.02125108, -0.007677794, 0.06217136, 0.02998487,
0.004533157, 0.01083183, -0.00876588, 0.04852415, 0.04186955, 0.008009591, -0.005577455, -0.01212511, 0.03231838, 0.05174232,
0.02208053, -0.01602993, -0.0195623, 0.01203131, 0.0564928, 0.04022207, -0.01226221, -0.02569377, -0.01424503, 0.04961254,
0.05268182, 0.00711792, -0.02272518, -0.04221883, 0.02746787, 0.04975479, 0.03384247, -0.007783253, -0.06212873, -0.00300864,
0.02928799, 0.05246361, 0.01225671, -0.06737909, -0.02652973, 0.0008251891, 0.05134372, 0.026087, -0.06205556, -0.03215174,
-0.02071685, 0.03204093, 0.02986198, -0.05650828, -0.02190337, -0.02625232, 0.005765674, 0.03016431, -0.05513785, -0.005875843,
-0.01577817, -0.01726404, 0.03419046, -0.05173168, 0.006861723, 0.006090976, -0.03181297, 0.04082699, -0.03663941, 0.01161812,
0.03168133, -0.03451177, 0.04193293, -0.006586121, 0.008614648, 0.04968511, -0.02332451, 0.03104673, 0.03061381, 0.003584331,
0.05070867, -0.002220879, 0.01165265, 0.05988138, 0.004267329, 0.03589979, 0.01698929, -0.004866462, 0.0703036, 0.01322667,
0.01723176, 0.02239132, -0.01147235, 0.06349297, 0.02628352, 0.007381598, 0.01247649, -0.01218262, 0.04864163, 0.03877309,
0.01103434, -0.003289103, -0.01477802, 0.03120251, 0.04874988, 0.02544157, -0.01280565, -0.02111673, 0.009801827, 0.05294504,
0.04369037, -0.00793311, -0.02574214, -0.01722836, 0.04501121, 0.05570813, 0.01227978, -0.02094346, -0.04526969, 0.02187443,
0.05179922, 0.03902037, -0.004287587, -0.06444833, -0.008855895, 0.03025557, 0.05663009, 0.016787, -0.06852757, -0.03165543,
0.001224158, 0.05384488, 0.03078571, -0.0621884, -0.03603156, -0.02008433, 0.03287419, 0.0341669, -0.05608315, -0.02476132,
-0.02473902, 0.005334213, 0.03387732, -0.05443848, -0.00843613, -0.01309661, -0.01850666, 0.03714579, -0.05067666, 0.003798839,
0.009855423, -0.03343698, 0.04262013, -0.0349409, 0.007518816, 0.03608329, -0.03601421, 0.04201538, -0.004071107, 0.003443271,
0.05403554, -0.02418199, 0.02914553, 0.03373544, -0.002167508, 0.05438825, -0.002136377, 0.008120479, 0.06301226, -0.001329819,
0.03868949, 0.01794017, -0.009172171, 0.07275475, 0.008329836, 0.01937949, 0.02393475, -0.01565955, 0.06481481, 0.02220001,
0.009371836, 0.01446499, -0.01565707, 0.04871233, 0.03521626, 0.0132709, -0.0007284356, -0.01721491, 0.03010822, 0.04517184,
0.02801371, -0.009408098, -0.02224645, 0.00775692, 0.048718, 0.04628725, -0.003604807, -0.02524055, -0.01979266, 0.03973899,
0.05774295, 0.0171649, -0.01861154, -0.0476369, 0.01575461, 0.05277426, 0.04358042, -0.0003972435, -0.06586293, -0.01498905,
0.03018663, 0.05987443, 0.02142607, -0.06866913, -0.03684812, 0.0007374436, 0.05526622, 0.03526035, -0.06134921, -0.03990519,
-0.0201361, 0.03265421, 0.0379532, -0.05479708, -0.02772221, -0.02375217, 0.004010825, 0.03683934, -0.05298039, -0.01132049,
-0.01088768, -0.02041296, 0.03915145, -0.04891305, 0.0001921048, 0.0130864, -0.03545754, 0.04328644, -0.03255023, 0.002752673,
0.03979939, -0.03761707, 0.04086472, -0.0009066984, -0.00236732, 0.05750586, -0.02485673, 0.02605544, 0.03737935, -0.008393709,
0.05702258, -0.00167504, 0.003627907, 0.06645428, -0.007192411, 0.0403637, 0.01932411, -0.01408165, 0.07529798, 0.003301016,
0.02045114, 0.02585556, -0.02007503, 0.06609777, 0.01798409, 0.01038262, 0.01673198, -0.01904784, 0.04874642, 0.03142321,
0.01460093, 0.002004474, -0.01934121, 0.02909907, 0.04122867, 0.02967595, -0.005975014, -0.02291979, 0.006005374, 0.04405261,
0.04789755, 0.0005470049, -0.02423294, -0.02180271, 0.03407125, 0.05869679, 0.02157397, -0.01585527, -0.04918731, 0.009411942,
0.05263513, 0.04733158, 0.003690404, -0.06627247, -0.02110427, 0.0290858, 0.06205174, 0.02593434, -0.06775424, -0.04183797,
-0.0005986318, 0.05553452, 0.03926713, -0.05953408, -0.04355453, -0.02083681, 0.03138223, 0.04100203, -0.05267102, -0.03060866,
-0.02328813, 0.001856735, 0.03887061, -0.05079019, -0.01436021, -0.009196181, -0.02288077, 0.0400736, -0.04646846, -0.003762985,
0.01568933, -0.03775071, 0.04274959, -0.02950676, -0.002434269, 0.04269668, -0.03919741, 0.03847843, 0.002839621, -0.008518403,
0.05994524, -0.02525488, 0.02185649, 0.04144391, -0.01476347, 0.05846757, -0.0007952265, -0.001674213, 0.07008574, -0.01299075,
0.04080289, 0.02112325, -0.01940664, 0.07782154, -0.001558671, 0.02035489, 0.02808671, -0.02453493, 0.06726938, 0.01390077,
0.01033845, 0.01917635, -0.02221031, 0.0487247, 0.02763631, 0.01495157, 0.004782184, -0.02107281, 0.02820631, 0.03716388,
0.03035457, -0.002659926, -0.02312515, 0.004614723, 0.03921537, 0.0484582, 0.004346991, -0.02278781, -0.02317739, 0.02830564,
0.05853751, 0.025329, -0.01282187, -0.04985584, 0.003161657, 0.05139653, 0.05012631, 0.007767639, -0.06565494, -0.0269005,
0.02701233, 0.0630783, 0.03007883, -0.0658146, -0.04636956, -0.002704386, 0.05464546, 0.04258643, -0.0568165, -0.04677891,
-0.02212085, 0.02912427, 0.04313284, -0.04979648, -0.03325411, -0.02332314, -0.001014879, 0.03983886, -0.04796009, -0.01738506,
-0.008050885, -0.02577365, 0.03983273, -0.04343551, -0.00785747, 0.01758797, -0.04025698, 0.01825819, -0.04926451, 0.02157762
    ]
}
