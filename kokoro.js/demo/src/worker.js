import { KokoroTTS } from "kokoro-js";
import { detectWebGPU } from "./utils.js";

// Device detection
const device = (await detectWebGPU()) ? "webgpu" : "wasm";
self.postMessage({ status: "device", device });

// Load the model
const model_id = "onnx-community/Kokoro-82M-v1.0-ONNX";
const tts = await KokoroTTS.from_pretrained(model_id, {
  // Using fp16 on WebGPU is a good balance of speed and quality.
  // For devices without WebGPU, we fall back to wasm with q8 quantization.
  dtype: device === "webgpu" ? "fp16" : "q8",
  device,
}).catch((e) => {
  self.postMessage({ status: "error", error: e.message });
  throw e;
});
self.postMessage({ status: "ready", voices: tts.voices, device });

// Listen for messages from the main thread
self.addEventListener("message", async (e) => {
  const { text, voice } = e.data;

  try {
    // Generate speech
    const stream = tts.stream(text, { voice });

    // Post the first chunk back to the main thread
    let first = true;
    for await (const { audio } of stream) {
      const blob = audio.toBlob();
      self.postMessage({
        status: "chunk",
        audio: blob,
        text,
        first,
      });
      first = false;
    }
  } catch (e) {
    self.postMessage({ status: "error", error: e.message });
    return;
  }

  // Send the audio file back to the main thread
  self.postMessage({ status: "complete", text });
});
