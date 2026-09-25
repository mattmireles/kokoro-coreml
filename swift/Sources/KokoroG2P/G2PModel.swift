// Ported from FluidAudio (Apache-2.0), github.com/FluidInference/FluidAudio v0.17.4
// (TTS/G2P/G2PModel.swift).

import CoreML
import Foundation

/// BART encoder/decoder Core ML G2P for English words missing from the lexicon.
/// Greedy decode, CPU only. Not thread-safe: owned by `KokoroEnglishFrontend`.
final class G2PModel {
    private let graphemeToId: [Character: Int]
    private let idToPhoneme: [Int: String]
    private let bosTokenId: Int
    private let eosTokenId: Int
    private let unkTokenId: Int
    private let encoder: MLModel
    private let decoder: MLModel

    /// Loads `g2p_vocab.json`, `G2PEncoder.mlmodelc`, `G2PDecoder.mlmodelc` from `directory`.
    init(directory: URL) throws {
        let vocabURL = directory.appendingPathComponent("g2p_vocab.json")
        guard let vocab = try JSONSerialization.jsonObject(with: Data(contentsOf: vocabURL)) as? [String: Any],
            let g2id = vocab["grapheme_to_id"] as? [String: Int],
            let id2ph = vocab["id_to_phoneme"] as? [String: String]
        else {
            throw KokoroG2PError("g2p_vocab.json: invalid JSON structure")
        }

        var gMap: [Character: Int] = [:]
        for (key, val) in g2id where key.count == 1 {
            gMap[key.first!] = val
        }
        graphemeToId = gMap

        var pMap: [Int: String] = [:]
        for (key, val) in id2ph {
            if let intKey = Int(key) { pMap[intKey] = val }
        }
        idToPhoneme = pMap

        bosTokenId = vocab["bos_token_id"] as? Int ?? 1
        eosTokenId = vocab["eos_token_id"] as? Int ?? 2
        unkTokenId = vocab["unk_token_id"] as? Int ?? 3

        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        encoder = try MLModel(
            contentsOf: directory.appendingPathComponent("G2PEncoder.mlmodelc"), configuration: config)
        decoder = try MLModel(
            contentsOf: directory.appendingPathComponent("G2PDecoder.mlmodelc"), configuration: config)
    }

    /// Word → phoneme tokens, or `nil` if the model produced nothing.
    func phonemize(word: String) throws -> [String]? {
        // Encode: [BOS] + grapheme IDs + [EOS]
        var inputIds: [Int32] = [Int32(bosTokenId)]
        for ch in word {
            inputIds.append(Int32(graphemeToId[ch] ?? unkTokenId))
        }
        inputIds.append(Int32(eosTokenId))

        let encLen = inputIds.count
        let encoderInput = try MLMultiArray(shape: [1, NSNumber(value: encLen)], dataType: .int32)
        for i in 0..<encLen {
            encoderInput[[0, i] as [NSNumber]] = NSNumber(value: inputIds[i])
        }

        let encoderProvider = try MLDictionaryFeatureProvider(
            dictionary: ["input_ids": MLFeatureValue(multiArray: encoderInput)])
        guard let encoderOutput = try? encoder.prediction(from: encoderProvider),
            let encoderHidden = encoderOutput.featureValue(for: "encoder_hidden_states")?.multiArrayValue
        else {
            throw KokoroG2PError("G2P encoder prediction failed")
        }

        // Greedy decode loop
        let maxSteps = 64
        var decoderIds: [Int32] = [Int32(bosTokenId)]

        for _ in 0..<maxSteps {
            let decLen = decoderIds.count

            let decInput = try MLMultiArray(shape: [1, NSNumber(value: decLen)], dataType: .int32)
            for i in 0..<decLen {
                decInput[[0, i] as [NSNumber]] = NSNumber(value: decoderIds[i])
            }

            // position_ids (BART offset = 2)
            let posIds = try MLMultiArray(shape: [1, NSNumber(value: decLen)], dataType: .int32)
            for i in 0..<decLen {
                posIds[[0, i] as [NSNumber]] = NSNumber(value: Int32(i + 2))
            }

            // causal_mask: upper triangular with -1e4
            let mask = try MLMultiArray(
                shape: [1, NSNumber(value: decLen), NSNumber(value: decLen)], dataType: .float32)
            for i in 0..<decLen {
                for j in 0..<decLen {
                    mask[[0, i, j] as [NSNumber]] = NSNumber(value: Float(j > i ? -1e4 : 0))
                }
            }

            let decoderProvider = try MLDictionaryFeatureProvider(
                dictionary: [
                    "decoder_input_ids": MLFeatureValue(multiArray: decInput),
                    "encoder_hidden_states": MLFeatureValue(multiArray: encoderHidden),
                    "position_ids": MLFeatureValue(multiArray: posIds),
                    "causal_mask": MLFeatureValue(multiArray: mask),
                ])
            guard let decoderOutput = try? decoder.prediction(from: decoderProvider),
                let logits = decoderOutput.featureValue(for: "logits")?.multiArrayValue
            else {
                throw KokoroG2PError("G2P decoder prediction failed")
            }

            // Argmax of last position's logits
            let vocabSize = logits.shape.last!.intValue
            let lastPos = decLen - 1
            var bestId = 0
            var bestVal = -Float.infinity
            for v in 0..<vocabSize {
                let val = logits[[0, lastPos, v] as [NSNumber]].floatValue
                if val > bestVal {
                    bestVal = val
                    bestId = v
                }
            }

            if bestId == eosTokenId { break }
            decoderIds.append(Int32(bestId))
        }

        // Token IDs → phonemes, skipping special tokens
        let specialTokens: Set<Int> = [0, bosTokenId, eosTokenId, unkTokenId]
        let phonemes = decoderIds.compactMap { id -> String? in
            specialTokens.contains(Int(id)) ? nil : idToPhoneme[Int(id)]
        }
        return phonemes.isEmpty ? nil : phonemes
    }
}
