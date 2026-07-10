# **A Hacker's Field Guide to Kokoro TTS on Apple Neural Engine: macOS 26.4 and CoreML 9.0**

April 7, 2026

## **The Reality of On-Device Speech Synthesis in 2026**

Deploying generative machine learning models to edge hardware requires abandoning the comforts of cloud-based compute clusters and descending into the rigid, undocumented, and aggressively constrained environment of proprietary mobile silicon. Kokoro-82M is currently a masterpiece of architectural efficiency in the text-to-speech (TTS) domain, representing an 82-million parameter model that yields speech quality historically reserved for massive, multi-gigabyte models.1 Trained for a mere $1,000 on permissive datasets, Kokoro utilizes a decoder-only architecture derived from StyleTTS 2, replacing heavy diffusion components with an ISTFTNet vocoder.1 In cloud environments, the market rate for serving Kokoro over API is roughly $0.06 per hour of audio output or under $1 per million characters.1

However, paying API costs and enduring network latency is unacceptable for responsive, privacy-preserving local macOS or iOS applications. The objective is to force this model to run natively on Apple Silicon. This requires harnessing the Apple Neural Engine (ANE).

Naively passing the raw PyTorch graph of Kokoro through the coremltools compiler guarantees failure. The standard model contains sequence-dependent dimensional expansions, long short-term memory (LSTM) layers, and complex scaling operations that trigger catastrophic panics within Apple's Espresso (E5RT) runtime environment.5 As of macOS 26.4 and the coremltools 9.0 compiler stack, bridging the gap between PyTorch and the ANE requires surgically dismantling the network, rewriting critical subgraphs, utilizing strict memory alignment, and constructing a native Swift orchestration layer.5

This exhaustive field guide documents the precise failure modes, the non-obvious design patterns, and the heavily optimized runtime architecture required to successfully achieve 17x real-time inference speeds for Kokoro-82M on the Apple Neural Engine.6

## **Deconstructing the ANE Black Box**

The Apple Neural Engine is a fixed-function hardware accelerator embedded in Apple Silicon, evolving significantly from the A11 through the modern M4 and A17 Pro architectures.5 Unlike general-purpose graphical processing units (GPUs) which offer highly flexible instruction set architectures, the ANE is a rigidly designed tensor coprocessor explicitly built for dense, low-precision convolutional and matrix multiplication operations.5 Because Apple does not publish the hardware ISA, the CoreML compiler acts as a black box.5 Understanding the compiler's implicit rules is the only way to prevent it from punting execution back to the CPU.

### **Hardware Constraints and Tensor Mechanics**

To execute code, the ANE imposes absolute constraints on tensor geometry and precision. The silicon natively operates in 16-bit floating-point (FLOAT16 or FP16).5 Supplying 32-bit floating-point tensors, or including layers that demand FP32 accumulation, will force the CoreML compiler to insert a CPU-bound casting operation, instantly destroying the pipeline's latency as the tensor travels across the unified memory bus.5

Furthermore, the ANE's memory controllers are physically wired to optimize 4-dimensional image processing arrays. Tensors must strictly conform to a (Batch, Channels, Height, Width) layout.5 For audio data sequences, this mandates a strict (B, C, 1, S) format, where S represents the sequence length.5 Attempting to feed the ANE a 1D tensor or a 3D tensor frequently triggers an automatic, silent CPU fallback.5

The ANE features an extremely limited L2 cache, measuring approximately 32 MB.5 Data that overflows this cache must be spilled to DRAM. To interface with the cache effectively, the last dimension of any tensor must be contiguous in memory and perfectly aligned to a 64-byte boundary.5 Failing to ensure that the largest sequence dimension sits at the end of the tensor layout incurs a massive memory padding penalty, effectively doubling the execution time of otherwise simple convolutional blocks.5

### **The E5RT Static Execution Paradigm**

CoreML operates on modern Apple operating systems (macOS 12+ / iOS 15+ and later, extending into macOS 26.4) via the E5RT, also known as the Espresso runtime.6 E5RT achieves its low latency by pre-allocating absolute memory buffers and compiling a static execution plan long before the user initiates a prediction.6

This static paradigm is entirely hostile to dynamic shapes. If a CoreML graph contains any operation where the output tensor's size is dependent upon the *values* contained within an intermediate tensor, the E5RT execution plan shatters.6 Data-dependent control flows, boolean masking (tensor\[mask\]), and dynamic broadcasting loops trigger the Invalid blob shape panic, forcing the application to crash.6

## **The Kokoro Autopsy: Why Out-of-the-Box Compilation Fails**

The Kokoro-82M architecture consists of a text encoder, a prosody duration predictor derived from StyleTTS 2, and an ISTFTNet vocoder.1 Evaluating the unified PyTorch graph against the constraints of the ANE reveals a minefield of hostile operations.5

### **The G2P and Text Encoding Bottleneck**

Before Kokoro can generate audio, plain text must be converted into phonemes. The standard Python repository relies on the misaki grapheme-to-phoneme (G2P) library, which under the hood invokes the external espeak-ng binary.1 Apple strictly prohibits iOS and sandboxed macOS applications from spawning external executable binaries.5 Consequently, espeak-ng must be entirely stripped from the pipeline.5 Developers must integrate a secondary, CoreML-converted neural G2P model (such as LiteG2P) to translate text characters into phoneme integer IDs dynamically before hitting the Kokoro acoustic pipeline.5

Once phonemes are generated, they enter the text encoder and duration predictor. This segment is saturated with Transformer attention mechanisms and LSTMs.6 The ANE physically lacks the hardware loop counters and accumulator states required to execute recurrent neural networks.5 Consequently, any LSTM operation explicitly forces CPU binding.6 Furthermore, the text encoder utilizes Adaptive Layer Normalization (AdaLayerNorm) to condition the text representations with a 256-dimensional reference voice embedding.6 AdaLayerNorm requires complex variance calculations and instance-level statistics that historically map poorly to the ANE, often creating dynamic shape warnings.5

### **The Data-Dependent Alignment Catastrophe**

The most toxic subgraph within the unified Kokoro architecture is the duration upsampling layer.6 The duration model predicts exactly how many acoustic frames each phoneme should occupy. A matrix multiplication mathematically expands the text representation tensor (t\_en) to match the predicted acoustic time domain.6

Because the target sequence length of the vocoder input is dictated by the *predicted duration values* generated live during inference, the tensor shape expands dynamically at runtime.6 PyTorch handles tensor.repeat\_interleave() effortlessly. CoreML E5RT views this as an illegal violation of its pre-allocated memory graph, resulting in an immediate crash tracked as non\_zero\_0\_classic\_cpu \- \[?, 3\].6

## **The Two-Stage Architectural Pivot**

To survive the E5RT runtime and unlock ANE acceleration, the monolithic Kokoro graph must be surgically cleaved into two distinct models, mediated by a highly optimized native Swift layer.6 This "Two-Stage Pipeline" architecture, successfully implemented in production environments like the TalkToMe macOS application, decouples the unpredictable control flow from the dense, fixed-function mathematical compute.6

### **Stage 1: The Duration Predictor (CPU/GPU Bound)**

The first stage isolates the text encoder, the LSTM layers, and the duration prediction heads.6 Acknowledging that the ANE cannot process LSTMs, this model is deliberately targeted for the CPU and GPU via the CoreML configuration flag cpuAndGPU.6

To prevent memory fragmentation during rapid execution, dynamic length processing via ct.RangeDim is abandoned.6 Instead, the inputs to the duration model are strictly padded to a fixed geometry.6

| Stage 1: Duration Model Interfaces | Tensor Geometry | Data Type | Purpose |
| :---- | :---- | :---- | :---- |
| **Input:** input\_ids | \`\` | int32 / int8 | Tokenized phoneme sequence. |
| **Input:** attention\_mask | \`\` | int32 / int8 | Binary mask for padding tokens. |
| **Input:** ref\_s | \`\` | float32 | Baseline voice embedding / style vector. |
| **Input:** speed | \`\` | float32 | Scalar dictating the speed of the output speech. |
| **Output:** pred\_dur | \`\` | float32 | Predicted frame duration per phoneme. |
| **Output:** t\_en | \`\` | float32 | Encoded acoustic text representations. |
| **Output:** d & s & ref\_s\_out | Variable | float32 | Internal continuous feature vectors. |

By freezing the input shape to 128 tokens, the CPU runtime pre-allocates exact buffers, ensuring jitter-free execution.6

### **The Swift Interstitial Layer: Offloading Dynamic Math**

The outputs of the duration model contain the acoustic features, but they are completely misaligned with the time domain.6 The alignment upsampling—the exact operation that crashes CoreML—is performed natively in the host Swift application utilizing Apple's Accelerate framework.6

The Swift logic executes the following sequence:

1. **Alignment Matrix Construction:** Using the pred\_dur output, Swift constructs a sparse alignment matrix named pred\_aln\_trg with dimensions \[tokens, frames\].6  
2. **Native Matrix Multiplication:** Swift computes the aligned acoustic features (asr) by calculating the dot product: asr \= t\_en @ pred\_aln\_trg.6 This effectively stretches the acoustic features across the required temporal length without ever touching a CoreML graph.6  
3. **Acoustic Curve Derivation:** Simple fundamental frequency (F0) and noise (N) prediction curves are mathematically derived.6  
4. **Channel Normalization:** Crucially, a per-channel min-max normalization is applied to the acoustic features prior to vocoder entry.6  
5. **Static Enforcement:** The final arrays are aggressively padded with zeros or cropped to fit the strict, hardcoded dimensions demanded by the second CoreML stage.6

### **Stage 2: The HAR Decoder Models (ANE Bound)**

The second stage encapsulates the vocoder—the neural network responsible for converting the stretched acoustic features into a high-fidelity 24kHz audio waveform.6 The vocoder utilizes an ISTFTNet architecture, which consists heavily of 1-dimensional convolutions (Conv1d) and transposed convolutions (ConvTranspose1d), alongside element-wise activations like LeakyReLU.5 Because it lacks recurrent loops, the vocoder is perfectly suited for ANE acceleration.6 For placement, handoff cost, iSTFT tail boundaries, and MLX comparison, see [Core ML vs MLX vocoder scheduling](Core%20ML-MLX-Scheduling-1D-ConvTranspose-ISTFTNet-vocoders-guide.md).

However, because the ANE requires strictly pre-compiled tensor sizes to prevent the non\_zero\_0\_classic\_cpu failure mode, a "Fixed-Size Bucketing" strategy is employed.6 Rather than exporting a single vocoder capable of any length, the compiler exports several identical vocoders, each hardcoded to output a specific duration of audio.6

The host Swift app predicts the total length of the required audio, selects the smallest bucket capable of holding it, pads the interstitial tensors to fit that exact bucket, and invokes the ANE.6

| Stage 2: HAR Decoder 3-Second Bucket | Tensor Geometry | Target Hardware | Purpose |
| :---- | :---- | :---- | :---- |
| **Input:** asr | \`\` | ANE | Time-aligned acoustic features. |
| **Input:** F0\_pred | \`\` | ANE | Fundamental frequency predictions (2x ASR time). |
| **Input:** N\_pred | \`\` | ANE | Noise envelope predictions. |
| **Input:** ref\_s | \`\` | ANE | Voice embedding (sliced internally to 128 dims). |
| **Output:** waveform | \`\` | Host | 3 seconds of uncompressed 24kHz PCM audio data. |

Standard deployment buckets include a **3s model** optimized for fast time-to-first-byte (TTFB) in conversational interfaces, a **10s model** for balanced sentences, and a **45s model** for long-form paragraph offline synthesis.6 By eradicating all instances of .expand() and dynamic masking inside the vocoder, the CoreML runtime binds seamlessly to the Neural Engine.6

## **Mastering the iSTFTNet Vocoder and HAR Processing**

Achieving a benchmark of 17x real-time generation speed requires more than simply mapping PyTorch to CoreML; it necessitates deep manipulation of the vocoder's mathematical operations.6 The Kokoro architecture derives its speed from the ISTFTNet architecture, which must be adapted for Apple Silicon via "HAR Processing".2

### **The Bottleneck of Traditional Vocoders**

Traditional high-fidelity generative vocoders, such as HiFi-GAN, treat audio synthesis as a black-box problem.10 They rely on deep convolutional networks containing massive temporal upsampling layers (e.g., 12 residual blocks with 2048 channels) to implicitly recover magnitude, reconstruct phase, and perform frequency-to-time conversion all at once.10 Directly calculating a raw waveform from an 80-dimensional mel-spectrogram requires reconstructing high-dimensional original-scale spectrograms (e.g., FFT size of 1024).10 For the ANE, this massive channel expansion immediately saturates the 32 MB L2 cache, dragging memory from DRAM and plummeting performance.5

### **The ISTFTNet Architecture**

ISTFTNet abandons the monolithic upsampling approach.10 It replaces the massive output-side upsampling layers with a classic inverse Short-Time Fourier Transform (iSTFT).10 The neural network is only responsible for executing lightweight 1D convolutions to reduce the frequency dimension to a much smaller intermediate representation (e.g., an FFT size of 16).10 Once the neural network predicts the intermediate magnitude and phase geometries, the deterministic iSTFT algorithm mathematically generates the raw waveform.10

### **Apple Silicon HAR Optimization**

"HAR Processing" (Harmonic-phase separation) is the practical application of the ISTFTNet paradigm explicitly tuned for CoreML and the ANE.6

While the ANE is highly optimized for the 1D convolutions required to predict the intermediate representations, it is fundamentally incapable of executing the complex number arithmetic and Fourier mathematics required by the final iSTFT layer.5 If the torch.istft operation is left inside the CoreML export graph, the runtime compiler will process 95% of the model on the ANE, encounter the Fourier node, execute a costly ANE-to-CPU memory copy of the massive tensor, and compute the iSTFT on the CPU.6

To optimize this, developers must execute a truncation. The PyTorch export script must slice the iSTFT computation off the tail end of the graph entirely.6 The exported CoreML graph simply outputs the raw harmonic magnitude and phase tensors. The host Swift application intercepts these dual outputs and immediately executes the inverse STFT natively utilizing Apple's vDSP (Accelerate) framework, which is hand-tuned in assembly for CPU Fourier transforms.6

Profiling a 23.7-second audio utterance utilizing the 30-second HAR processing bucket reveals the profound efficiency of this split:

* **ANE Compute (CoreML Predict):** 0.25–0.31s (Dominant neural execution phase).6  
* **CPU Preprocessing:** 0.15–0.17s (Harmonic-plus-noise modeling).6  
* **Inverse STFT (Accelerate):** 0.02–0.03s (Deterministic math generation).6  
* **Orchestration/IO Latency:** 0.55–0.60s.6

This division of labor achieves an overall Real-Time Factor (RTF) of approximately 0.057, allowing the ANE to operate at peak efficiency without memory thrashing.6

## **macOS 26.4 and CoreML 9.0 Munitions**

The rollout of coremltools 9.0 and updates mirroring iOS 26/macOS 26.4 introduce several profound architectural enhancements that developers must integrate to resolve legacy conversion problems.7

### **MultiFunction Models and Weight Deduplication**

A critical drawback of the fixed-size bucketing strategy (shipping a 3s, 10s, and 45s model) is storage bloat. Duplicating the identical 82-million parameter weight tensors three times drastically inflates the application footprint, resulting in hundreds of megabytes of redundant storage.5

CoreML 9.0 solves this via the MultiFunctionDescriptor API.16 Developers can now embed multiple computational graphs (functions) inside a single unified .mlpackage.16 By exporting the 3s, 10s, and 45s vocoder architectures as separate functions within the same multi-function package, the CoreML compiler automatically deduplicates the underlying weights.5 The host application maps the single weight structure into memory (approximately 200MB) and simply dictates which sequence length function to execute during the inference call, drastically improving memory efficiency and cold-start loading times.6

### **Stateful Models and the KV Cache Paradigm**

While the standard Kokoro model generates audio in batches, developers adapting the acoustic encoders for continuous streaming synthesis encounter input/output bottlenecks.17 Historically, persisting an attention matrix or LSTM hidden state required outputting the massive tensor back to the host CPU, holding it in Swift, and passing it back into the CoreML graph as an input on the next tick.17

CoreML 9.0 introduces the StateType abstraction for mutable buffers and Key-Value (KV) caching.7 During conversion, engineers define a StateType tensor using ct.StateType tied to PyTorch buffer registrations.17 The E5RT runtime natively reads and writes to this state matrix in-place directly on the ANE or unified memory partition without ever returning control to the host.7 This fundamentally eliminates IO transit time for streaming acoustic context.17

### **Native INT8 Boundaries**

Another major improvement in the 9.0 stack is native support for int8 data types at the input and output boundaries.7 In older toolchains, the tokenized phoneme IDs and attention masks passed from the Swift application to the duration model required float32 or int32 casting, artificially increasing the memory bandwidth required.5 By defining int8 inputs, developers can feed the dense token arrays directly, streamlining the pipeline's memory throughput right at the onset.7 Furthermore, AllowLowPrecisionAccumulationOnGPU optimization flags now permit looser tolerances on non-ANE fallback paths, recovering latency during CPU/GPU execution.7

## **Quantization Warfare: Engineering the Weights**

Even after separating HAR processing and mitigating dynamic shapes, the fixed-function architecture of the ANE relies entirely on high-speed matrix decompression. Because the 32 MB L2 cache is easily saturated by raw FP16 weights, post-training quantization (PTQ) is mandatory.5 coremltools.optimize provides the tooling necessary to compress the vocoder weights.8

### **The W8A16 Baseline and W8A8 Hardware Acceleration**

For Apple hardware preceding the M4/A17 Pro generation, W8A16 (8-bit weights, 16-bit activations) is the optimal quantization target.5

Using linear symmetric quantization (cto.coreml.linear\_quantize\_weights), the vocoder parameters are crushed down to 8-bit integers, effectively halving the model size.6 During inference, the ANE fetches the dense 8-bit blocks into the L2 cache and executes a highly optimized hardware-level decompression, expanding the weights back to FP16 just-in-time for the matrix multiplication against the FP16 activations.8

For the newest Apple hardware running macOS 26.4 (M4 and A17 Pro architectures), the ANE incorporates dedicated logic gates for int8-int8 compute.8 By applying W8A8 quantization (8-bit weights and 8-bit activations), developers bypass the FP16 expansion entirely.5 This int8 compute path yields an immediate 30-50% latency reduction over W8A16, maximizing the theoretical operations per second (TOPS) of the coprocessor.5

### **Validation and Distortion Mitigation**

Quantization is a destructive process. Pushing audio synthesis layers into 8-bit precision frequently introduces numerical float differences, manifesting as high-pitched digital artifacts or background static in the output waveform.12

Developers must run systematic validation against the golden PyTorch output using perceptual scoring algorithms like Perceptual Evaluation of Speech Quality (PESQ) and Mel Cepstral Distortion (MCD).5 If severe degradation occurs, surgical graph interventions are required. Engineers utilize the coremltools.models.utils.bisect\_model function to split the CoreML graph at the point of distortion.7 The problematic 1D convolutional layers are isolated, exempted from the global quantization configuration via cto.coreml.OpLinearQuantizerConfig, and left in FP16 precision, while the rest of the model is aggressively quantized.7

## **Debugging the Abyss: Failure Modes and Edge Cases**

Operating at the edge of the CoreML framework requires deep-systems debugging. When a converted model produces static or crashes, standard Python profilers are entirely useless.5 Recognizing the pathophysiological signatures of ANE compilation failures is critical.

### **The E5RT non\_zero\_0\_classic\_cpu Panic**

The most prolific error encountered when porting dynamic architectures is the Invalid blob shape … non\_zero\_0\_classic\_cpu \- \[?, 3\] crash.6

This occurs when the CoreML Model Intermediate Language (MIL) compiler encounters an operation that relies on runtime data geometry (like tensor\[mask\]). Unable to allocate ANE hardware blocks for an unknown shape, the compiler inserts a CPU fallback node (designated classic\_cpu).6 However, when the tensor is returned to the strict E5RT execution plan downstream, the runtime recognizes that the tensor's shape has mutated beyond its pre-allocated memory map and immediately terminates the application.6

**The Solution:** Excision. The offending PyTorch subgraph must be rewritten or removed.6 Any occurrence of tensor.expand() or tensor.repeat() whose arguments rely on incoming data lengths must be stripped, hardcoded to match the strict geographic buckets, or ported entirely into the Swift host code prior to prediction.6

### **The Banishment of Unpredictable torch Ops**

Certain PyTorch primitives are structurally incompatible with ANE mapping, forcing fallback memory copies that destroy performance:

1. **torch.where and Conditional Logic:** Operations that route tensors conditionally based on boolean masks create implicit dynamic branching.5 They must be replaced with strict arithmetic masking. For example, output \= torch.where(condition, A, B) must be rewritten as output \= A \* condition.float() \+ B \* (1.0 \- condition.float()) prior to tracing.5  
2. **torch.var and Statistical Functions:** Historical compiler versions struggle with native variance computations on the ANE.5 If normalization layers trigger a fallback, engineers must rewrite the graph to compute variance using base arithmetic, for example `var = mean(x * x) - mean(x) * mean(x)`.5
3. **Dilated Convolutions:** Convolutions with a dilation parameter greater than 1 are frequently rejected by the ANE compiler, falling back to the GPU.5 If the vocoder relies heavily on dilated convolutions, they must be decomposed or executed via composite operators.5

### **Swift App Developer Flags for Debugging**

For engineers utilizing a host harness like the TalkToMe macOS application, diagnostic flags are built directly into the runtime to bypass production caching and expose raw audio layers.6

| UserDefaults Developer Flags | Functionality | Purpose |
| :---- | :---- | :---- |
| talktome.coreml.computeUnits | all, cpuAndNeuralEngine, cpuAndGPU, cpuOnly | Force explicit execution environments to verify ANE speed vs CPU accuracy.6 |
| talktome.coreml.preferDecoderOnly | true / false | Bypasses the unified graph and forces the two-stage pipeline to avoid dynamic-shape panics.6 |
| talktome.coreml.dumpWaveforms | true | Writes synthesized .wav outputs directly to the system temp directory for external acoustic analysis (e.g., Audacity).6 |
| talktome.coreml.dumpSpectrograms | true | Extracts raw CSV spectrogram data directly from the ASR features pre-vocoder to hunt for nan/inf numerical instability.6 |
| talktome.dev.usePythonTokenizer | true | Bridges to a local Python environment via dev\_tokenize.py to test exact phoneme translations directly from the source library.6 |

## **Profiling and Verification: The Hardware Truth**

Because the coremltools API will silently mask CPU fallbacks behind a successful prediction call, developers must utilize deep hardware telemetry to confirm that the ANE is actually processing the tensor graph.5

### **Netron Graph Surgery**

Before running a single line of Swift, the compiled .mlpackage must be structurally analyzed using Netron, an open-source visualizer.5 Opening the CoreML package allows developers to trace the MIL operations from input to output.5

Engineers must scan the visual graph for dimensional collapse. If any intermediate tensor between convolutional blocks is collapsed into a 1D or 3D shape, the ANE memory controller will reject the operation.5 Furthermore, analyzing the data types within the visual nodes ensures that FLOAT16 or INT8 structures dominate the graph. If a rogue FLOAT32 node is spotted, it signals a precision casting failure in the Python export script, guaranteeing a costly ANE-to-CPU transit.5

### **Xcode Instruments and Daemon Threading**

To definitively prove neural engine execution, the host application is run through **Xcode Instruments** utilizing the explicit Core ML template.5

During inference, developers must isolate the system daemon threads.5 The presence of active processing loads on the ANERuntimeEngine or the H11ANEServicesThread is the indisputable forensic marker of successful hardware mapping.5 If these daemon threads remain dormant while overall CPU utilization spikes to 100%, the static execution plan has been rejected, and the model is functioning purely via the classic\_cpu fallback.6 If the developer flag cpuAndGPU is active, special care must be taken to ensure the GPU isn't silently masking a failed ANE compilation.9

### **The Command Line Absolute: powermetrics**

For the most aggressive, irrefutable hardware telemetry on macOS 26.4, engineers bypass the CoreML tracing stack entirely and query the silicon's power states directly via the terminal.5

By running sudo powermetrics \--samplers all during an active audio generation call, the terminal streams live milliwatt consumption across the CPU, GPU, and Neural Engine clusters.5 If the two-stage HAR optimization and static bucket strategies are correctly implemented, the terminal will report the ANE Power metric surging to multiple watts precisely during the 0.25-second inference window, while the GPU power sits at idle.6 If the ANE power remains at 0 mW while the duration model completes its sequence, the vocoder has fundamentally failed to map, and the engineer must return to the PyTorch export scripts to locate the dynamic shape violation.5

## **Conclusion**

Deploying Kokoro-82M to the Apple Neural Engine is a masterclass in fighting proprietary compilation layers. The CoreML E5RT environment demands absolute geometrical precision, memory alignment, and static execution topologies.6 By tearing the monolithic PyTorch model apart into a two-stage architecture, developers isolate the chaotic, variable-length text processing on the CPU and feed meticulously constructed, hardcoded acoustic features into ANE-optimized vocoder buckets.6

By offloading the dynamic upsampling and Fourier operations (HAR processing) entirely to the Swift/Accelerate host environment, the heavy convolutional layers of the ISTFTNet architecture are allowed to saturate the ANE free from data-dependent panics.6 Furthermore, leveraging the advanced capabilities of macOS 26.4 and CoreML 9.0—such as MultiFunction weight deduplication, StateType caching, and W8A8 hardware acceleration—resolves the memory footprint constraints historically associated with fixed-size bucket strategies.7 The ultimate result is a privacy-preserving, entirely offline generative pipeline capable of producing cloud-quality audio output locally at 17x faster than real-time.6

#### **Works cited**

1. hexgrad/Kokoro-82M \- Hugging Face, accessed April 7, 2026, [https://huggingface.co/hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)  
2. Kokoro-82M TTS API | Together AI, accessed April 7, 2026, [https://www.together.ai/models/kokoro-82m](https://www.together.ai/models/kokoro-82m)  
3. Kokoro-82M: Compact, Customizable, & Cutting-Edge TTS Model \- Analytics Vidhya, accessed April 7, 2026, [https://www.analyticsvidhya.com/blog/2025/01/kokoro-82m/](https://www.analyticsvidhya.com/blog/2025/01/kokoro-82m/)  
4. hexgrad/Kokoro-82M \- Demo \- DeepInfra, accessed April 7, 2026, [https://deepinfra.com/hexgrad/Kokoro-82M](https://deepinfra.com/hexgrad/Kokoro-82M)  
5. coreml-conversion-guide.md  
6. PyTorch → CoreML conversion pipeline for Kokoro TTS. Unlocks fast on-device text-to-speech on Apple Neural Engine. \- GitHub, accessed April 7, 2026, [https://github.com/mattmireles/kokoro-coreml](https://github.com/mattmireles/kokoro-coreml)  
7. Releases · apple/coremltools \- GitHub, accessed April 7, 2026, [https://github.com/apple/coremltools/releases](https://github.com/apple/coremltools/releases)  
8. Overview — Guide to Core ML Tools \- Apple, accessed April 7, 2026, [https://apple.github.io/coremltools/docs-guides/source/opt-overview.html](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)  
9. coreml-optimizer \- Skill \- Smithery, accessed April 7, 2026, [https://smithery.ai/skills/ckorhonen/coreml-optimizer](https://smithery.ai/skills/ckorhonen/coreml-optimizer)  
10. iSTFTNet, accessed April 7, 2026, [https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/istftnet/](https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/istftnet/)  
11. Kokoro \- GitHub, accessed April 7, 2026, [https://github.com/hexgrad/kokoro](https://github.com/hexgrad/kokoro)  
12. GitHub \- yl4579/StyleTTS2: StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models, accessed April 7, 2026, [https://github.com/yl4579/styletts2](https://github.com/yl4579/styletts2)  
13. \[2203.02395\] iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time Fourier Transform \- arXiv, accessed April 7, 2026, [https://arxiv.org/abs/2203.02395](https://arxiv.org/abs/2203.02395)  
14. \[2203.02395\] iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time Fourier Transform \- ar5iv, accessed April 7, 2026, [https://ar5iv.labs.arxiv.org/html/2203.02395](https://ar5iv.labs.arxiv.org/html/2203.02395)  
15. Release Notes \- Core ML Tools, accessed April 7, 2026, [https://coremltools.readme.io/v6.3/docs/change-logrelease-notes](https://coremltools.readme.io/v6.3/docs/change-logrelease-notes)  
16. On-Device AI Models and Core ML Tools: Insights From WWDC 2024 | HackerNoon, accessed April 7, 2026, [https://hackernoon.com/on-device-ai-models-and-core-ml-tools-insights-from-wwdc-2024](https://hackernoon.com/on-device-ai-models-and-core-ml-tools-insights-from-wwdc-2024)  
17. Stateful Models — Guide to Core ML Tools \- Apple, accessed April 7, 2026, [https://apple.github.io/coremltools/docs-guides/source/stateful-models.html](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)  
18. Convert Models to Neural Networks — Guide to Core ML Tools \- Apple, accessed April 7, 2026, [https://apple.github.io/coremltools/docs-guides/source/convert-to-neural-network.html](https://apple.github.io/coremltools/docs-guides/source/convert-to-neural-network.html)  
19. Activity · mattmireles/kokoro-coreml \- GitHub, accessed April 7, 2026, [https://github.com/mattmireles/kokoro-coreml/activity](https://github.com/mattmireles/kokoro-coreml/activity)
