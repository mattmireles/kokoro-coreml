# **Core ML Compute Unit Scheduling on Apple Silicon: An Advanced Engineering Field Guide**

April 14, 2026

> **Scope:** Core ML **runtime** dispatch across CPU, GPU, and Neural Engine—verification, silent fallback, and partitioning. Pair with the PyTorch MPS guides for training and export; use this guide for on-device scheduling and proving where the graph actually ran.

## Related Documentation

- **[PyTorch MPS and Core ML field guide](pytorch-mps.md)**: Broader Apple Silicon training and conversion context next to Core ML deployment.
- **[Hugging Face Transformers on Apple Silicon](HF-transformers-MPS-guide.md)**: MPS-side Trainer and dtype pitfalls that surface again at export time.
- **[PyTorch LSTM → Core ML (padding, EnumeratedShapes)](CoreML-LSTM-export-guide.md)**: BiLSTM padding, MIL `lstm` limits, `ct.EnumeratedShapes`, and ANE tradeoffs for recurrent exports.
- **[LSTM enumerated shapes and ANE (deep dive)](CoreML-LSTM-Enumerated-Shapes.md)**: Flexible inputs, `RangeDim` vs enumerated lists, verification tooling, and Swift inference caveats.
- **[Kokoro M1 vocoder boundary research brief](../../Notes/Kokoro-M1-vocoder-boundary-research-brief.md)**: Current Kokoro-specific lower-end Mac gap, falsified Core ML placement paths, and the exact research target for beating laishere without repeating failed `.all` or split-boundary probes.
- **[Core ML vs MLX vocoder scheduling (ConvTranspose / iSTFT)](Core%20ML-MLX-Scheduling-1D-ConvTranspose-ISTFTNet-vocoders-guide.md)**: Fixed per-inference handoff cost, ISTFTNet boundary placement, dual-output ANE anchor, and why MLX short-bucket wins do not imply ANE is beatable.
- **[Core ML ANE compiler failure triage](CoreML-ANE-compiler-failure-triage-guide.md)**: Execution-plan failures, compute-unit matrix discipline, and silent fallback separation.
- **[iPhone Core ML device lab runbook](iPhone-CoreML-device-lab-runbook.md)**: Physical-device setup, foreground policy, and evidence capture for warmed iPhone Core ML rows.

The landscape of on-device machine learning execution on Apple Silicon is defined by a delicate interplay between high-level software abstraction and rigid, heterogeneous hardware architecture. Core ML serves as the primary orchestration layer, designed to seamlessly distribute mathematical operations across the Central Processing Unit (CPU), Graphics Processing Unit (GPU), and the highly specialized Apple Neural Engine (ANE).1 However, beneath this polished abstraction lies a complex, opaque scheduling heuristic characterized by stringent hardware constraints, volatile graph partitioning logic, and a pervasive phenomenon known within the engineering community as "silent fallback".2

When deploying computationally intensive machine learning models—such as Large Language Models (LLMs), high-resolution vision transformers, or stateful diffusion architectures—relying solely on the framework's default API behaviors frequently results in suboptimal performance, excessive thermal throttling, or outright execution failure.3 This exhaustive technical report provides an advanced engineering breakdown of Core ML's compute unit scheduling mechanisms. It details precise verification techniques, catalogs undocumented hardware constraints, analyzes unresolved compiler bugs in the modern ecosystem, and provides programmatic workarounds required to achieve deterministic hardware acceleration on Apple Silicon.

## **The Architectural Foundation of Core ML Scheduling**

To engineer robust machine learning pipelines on Apple devices, one must first deconstruct how Core ML interprets, compiles, and dispatches a model. The framework does not execute source models (e.g., PyTorch or TensorFlow files) directly. Instead, models are translated into a unified intermediate representation—historically the NeuralNetwork format, and more recently, the MLProgram format (introduced in Core ML 5 / iOS 15).1 The MLProgram format supports complex execution pipelines, intermediate computations, conditional branching, and strongly typed execution, making it the standard for modern deployments.4

During the loading phase (MLModel.load), the Core ML compiler evaluates the model's Model Intermediate Language (MIL) operations.6 The compiler dynamically partitions the network graph into sections targeting the CPU, GPU, and ANE based on the hardware capabilities of the host device.4 This partitioning relies on private C++ backend engines housed within the hidden Espresso framework: Espresso::BNNSEngine for CPU execution using Basic Neural Network Subroutines, Espresso::MPSEngine for GPU execution via Metal Performance Shaders, and Espresso::ANERuntimeEngine for the Neural Engine.8

The critical engineering challenge arises from the ANE's hardware rigidity. The Neural Engine is an Application-Specific Integrated Circuit (ASIC) heavily optimized for low-power, fixed-size matrix multiplications and convolutions. It is not a general-purpose compute unit.3 If the Core ML compiler encounters an operation, tensor shape, or memory alignment pattern within the MIL graph that the ANE cannot natively process, it must reject that operation.2

To prioritize application stability over strict hardware adherence, Core ML does not throw a fatal error when an ANE rejection occurs. Instead, it seamlessly re-routes the rejected subgraph to the CPU or GPU.3 No exception is surfaced to the developer, and no warning is logged in the standard console output. Consequently, an engineer may deploy a model specifically architected for the ANE, observe a successful execution, but suffer massive latency regressions and battery drain because the vast majority of the graph silently fell back to the CPU.10

## **Decoding the MLComputeUnits API Surface**

Execution targeting is exposed to the developer via the MLComputeUnits enumeration, passed through the MLModelConfiguration object during initialization.12 Understanding the exact contractual meaning of these configurations is the first line of defense against erratic scheduling.

### **Configuration Analysis**

| Configuration Flag | Hardware Target(s) | Engineering Implications and Behavior |
| :---- | :---- | :---- |
| MLComputeUnits.all | CPU, GPU, ANE | The default option. It signals the Core ML scheduler to heuristically partition the graph across all available hardware.12 This does not guarantee ANE usage. It merely allows the compiler to attempt ANE dispatch, falling back to GPU or CPU upon encountering unsupported operations or dynamic shapes.3 |
| MLComputeUnits.cpuAndGPU | CPU, GPU | Explicitly excludes the Apple Neural Engine.12 This configuration forces all execution through the Metal and BNNS backends. It is frequently utilized as a targeted fallback mechanism when ANE compilation fails, introduces unacceptable precision loss, or triggers severe context-switching overhead.15 |
| MLComputeUnits.cpuAndNeuralEngine | CPU, ANE | Explicitly bypasses the GPU.12 Available from iOS 16+ and macOS 13+ (see Apple’s `MLComputeUnits.cpuAndNeuralEngine` documentation). This option prevents the machine learning workload from saturating the graphics processor.17 This is highly critical in mixed-workload environments, such as visionOS applications, where the GPU is fully saturated rendering the spatial environment.18 It is also used to bypass inefficient ANE-to-GPU memory context switching in specific diffusion models.13 |
| MLComputeUnits.cpuOnly | CPU only | Restricts all execution to the CPU using the BNNS backend.9 Historically accessed via the deprecated usesCPUOnly boolean on VNRequest, this is the only compute unit that strictly guarantees 32-bit floating-point (FP32) precision execution without intermediate 16-bit quantization.4 |

### **The Illusion of ALL**

The most persistent vulnerability in iOS and macOS AI engineering is the assumption that setting computeUnits = .all functions as a hardware enforcement mechanism. It acts purely as a request.12 When a developer passes .all, the framework calculates an internal "estimated cost" for each operation.20 If the estimated cost of an operation—including the data transformation and memory movement overhead required to transfer tensors into ANE-compatible buffers—exceeds the cost of keeping the data on the GPU, the scheduler rejects the ANE assignment.3

Furthermore, if the model architecture triggers a compiler bug or hardware limitation, .all permits the framework to silently bypass the specialized silicon. Thus, .all must be treated as a dynamic execution state requiring rigorous external verification.

## **Telemetry and Verification: Exposing the Execution Hardware**

Because the silent fallback mechanism obscures the true execution path, standard high-level logging is insufficient. Engineers must deploy low-level profiling utilities, power metrics analysis, and symbolic debuggers to trace the exact physical hardware executing the model graph.

### **The "Smoking Gun" Pattern: Powermetrics Analysis**

The absolute, unforgeable source of truth for hardware utilization on Apple Silicon is the macOS powermetrics utility.22 When a model is configured with MLComputeUnits.all but is silently rejected by the ANE, the definitive "smoking gun" is a sustained high CPU or GPU power draw paired with an ANE power draw of exactly 0 mW.10

To capture this telemetry, developers run `powermetrics` with elevated privileges. **For ANE specifically**—the direct signal that the Neural Engine is drawing power—prefer the ANE sampler. The same pattern appears in this repo’s `CLAUDE.md`:

```bash
sudo powermetrics -i 1000 --samplers ane | grep "ANE Power"
```

Sampler names and output format vary by macOS version; use `man powermetrics` on the target machine. For longer captures that include CPU-oriented samplers, an example invocation is:

```bash
sudo powermetrics -i 100 -s cpu_power -n 50 -o powermetrics_output.txt
```

Here `-i 100` samples every 100 ms, `-s cpu_power` selects CPU/cluster power metrics, and `-n 50` stops after 50 samples.23

Upon analyzing the output, isolate the aggregated power block. An illustrative pattern for a failed ANE dispatch (silent fallback to CPU/GPU) might look like:

```text
*** Sampled system activity (1000.00ms elapsed) ***

CPU Power: 4850 mW

GPU Power: 3120 mW

ANE Power: 0 mW
```

If ANE power stays near zero during a continuous, heavy inference loop while the app requested `.all`, a silent fallback is very likely.10 This bypasses high-level logs and shows what actually burned power.

### **Instruments and Time Profiler Analysis**

For developers constrained to the Xcode environment, the Instruments suite provides the Time Profiler and the Core ML template, which are essential for identifying the specific backend engine processing the graph.8

By running an application with the Time Profiler, executing the Core ML model hundreds of times in a tight loop, and expanding the call tree under the -[MLNeuralNetworkEngine predictionFromFeatures:] symbol, the specific Espresso engine functions become visible.8

| Target Hardware | Sub-framework Engine | Function Signature (Call Tree Indicator) |
| :---- | :---- | :---- |
| **ANE** | Espresso::ANERuntimeEngine | -[_ANEModel program] or H11ANE::H11ANEServicesThreadStart |
| **GPU** | Espresso::MPSEngine | Espresso::MPSEngine::context::__launch_kernel |
| **CPU** | Espresso::BNNSEngine | Espresso::BNNSEngine::convolution_kernel::__launch |

If the Time Profiler reveals a high frequency of Espresso::elementwise_kernel_cpu::__launch or Espresso::MPSEngine::blob_container::get_mps_image, the graph has fragmented, and execution has fallen back to the CPU or GPU.8 Conversely, if the call tree demonstrates deep stacks within AppleNeuralEngine or ANERuntimeEngine, the hardware is actively engaged.9

### **LLDB Breakpoint Strategies and Thread Signatures**

Programmatic verification can be executed dynamically by attaching the LLDB debugger to the running application and probing private framework symbols. The Apple Neural Engine relies on a dedicated background thread for managing execution services and hardware communication.

While the application is actively processing a prediction, an engineer can pause execution in the debugger. If a thread named H11ANEServicesThread is actively running, it confirms that the Neural Engine is processing at least a portion of the model.8 A crash dump analysis further reveals that this thread originates from mach_msg2_trap and routes through H11ANE::H11ANEServicesThreadStart, securely locking the hardware interface.28

Alternatively, engineers can configure symbolic breakpoints in LLDB to trap the exact moment a hardware dispatch occurs.8

```text
# Set breakpoint to catch the ANE program execution layer
(lldb) breakpoint set -n "-[_ANEModel program]"

# Set breakpoint to catch GPU fallback dispatches
(lldb) breakpoint set -n "Espresso::MPSEngine::context::__launch_kernel"

# Set breakpoint to catch CPU fallback (BNNS) convolutions
(lldb) breakpoint set -n "Espresso::BNNSEngine::convolution_kernel::__launch"
```

If the `-[_ANEModel program]` breakpoint never fires during the inference cycle, the ANE path is not executing and the developer should audit the model for constraint violations.8

### **Xcode Performance Reports and SpecializationStrategy**

In modern environments (Xcode 14 and later), developers can generate a Core ML Performance Report without writing host application code.20 This tool parses the model, compiles it against connected hardware, and provides an "estimated cost" for each operation, explicitly listing the compute unit assigned to every layer.20

If a model architected for the ANE presents a performance report where heavy linear operations (such as MatMul or Conv2d) are labeled with CPU or GPU assignments, it confirms the compiler has rejected those nodes for ANE execution.21

Furthermore, developers utilizing the ONNX Runtime Execution Provider (EP) for Core ML can invoke advanced diagnostics by configuring the SpecializationStrategy. Setting SpecializationStrategy to ProfileComputePlan logs the specific hardware each operator is dispatched to and the estimated execution time, providing deep visibility into the exact node causing the fallback.1

## **Mechanics of ANE Rejection: Why the Scheduler Falls Back**

The primary driver of the silent fallback phenomenon is the violation of the Neural Engine's rigid hardware constraints. Unlike general-purpose CPUs or highly flexible GPUs, the ANE is optimized for deterministic memory access patterns and specific matrix geometries. Operations that require arbitrary memory hopping, dynamic allocations, or unsupported activation functions will immediately trigger offloading.3

### **The Dynamic Shape and Sequence Length Bottleneck**

The ANE fundamentally conflicts with dynamic tensor shapes. In standard Large Language Model (LLM) execution, attention mechanisms require dynamic sequence lengths to process variable user inputs efficiently. However, the ANE architecture requires memory access patterns and tensor geometries to be predetermined and statically baked into the compiled graph structure.3

When a Core ML model specifies dynamic ranges for its input shapes, the ANE compiler cannot statically pre-allocate the required IOSurface memory buffers.3 Consequently, any node processing dynamic sequence dimensions is summarily rejected and offloaded to the CPU or GPU.3

To force ANE utilization for LLMs or transformers, the sequence length must be rigidly fixed during the coremltools conversion process. When deploying via ONNX, the RequireStaticInputShapes configuration option must be explicitly set to 1 (true) to prevent the Execution Provider from allowing dynamic inputs that degrade performance.1 The downside of this hardware enforcement is the loss of dynamic batching, requiring models to process padding tokens for shorter sequences, which burns compute cycles but preserves hardware acceleration.

### **Memory Layout and Dimensional Alignments**

The ANE imposes strict topological rules on how tensors are stored in physical memory. Violating these rules results in the insertion of highly expensive memory copy (memcpy) operations, or total fallback.30

1. **The 4D Channels-First Requirement:** The ANE hardware stack operates optimally on 4D tensors formatted as (Batch, Channels, Height, Width).30 Standard NLP Transformer implementations routinely utilize 3D (Batch, Sequence, Feature) formats. These are highly inefficient for the ANE. A 3D sequence batch must be explicitly expanded to a 4D format (B, C, 1, S), mapping the sequence length to the last axis.30  
2. **64-Byte Buffer Alignment Padding:** A critical, often overlooked constraint is that the final axis of any ANE buffer must be aligned to a 64-byte boundary. If an engineer designs a model that utilizes the last axis as a singleton (e.g., (B, C, S, 1)), the hardware must forcibly pad that array up to 64 bytes in memory. For 16-bit precision (fp16) models, this results in a staggering **32×** memory bloat. For 8-bit quantization (int8), the bloat is **64×**. This silently transforms a compute-bound model into a catastrophic bandwidth-bound bottleneck, eventually causing the scheduler to abandon the ANE entirely.30  
3. **Maximum Tensor and Block Sizes:** The ANE hardware possesses hard limits on data dimensionality. It cannot load tensors with a single dimension exceeding **16,384** elements. Model block sizes cannot exceed **1,024**. Furthermore, for optimal efficiency, vocabulary sizes in natural language processing models must be artificially padded to the nearest multiple of **64**.31  
4. **Rank 5 Limit:** Operations requiring tensor reshaping or transposing beyond a rank of 5 will trigger immediate rejection by the Core ML compiler.2

## **The Orion Catalog: Undocumented Hardware Constraints**

In early 2026, research exploring direct ANE programming—bypassing Core ML entirely via private _ANEClient and _ANECompiler APIs (codenamed "Orion")—successfully reverse-engineered the rigid constraints governing the ANE's Model Intermediate Language (MIL) IR parser.29 This research cataloged 20 distinct restrictions on memory layout, compilation limits, and numerical behaviors. Understanding these undocumented constraints is critical for any engineer seeking to avoid silent fallback.

| Constraint ID | Hardware Restriction / Symptom | Required Engineering Workaround |
| :---- | :---- | :---- |
| **Constraint 1** | The concat MIL operation is violently rejected by the ANE compiler, causing immediate compilation failure or total CPU fallback.34 | Model architectures must be partitioned, or concat operations must be mathematically rewritten as an unrolled series of element-wise addition and masking operations prior to Core ML conversion.34 |
| **Constraint 4** | The minimum IOSurface allocation size is strictly enforced at approximately **24 KB**.33 | Single-token tensors with shapes like (equating to 3,072 bytes in FP16) trigger execution errors. They must be artificially padded to at least (24,576 bytes) to clear the memory threshold.33 |
| **Constraint 8** | BLOBFILE weight offsets are fixed at exactly 64 bytes from the chunk header, not from the beginning of the file.33 | Custom weight parsing, manual quantization injection, or delta compilation strategies require explicit 64-byte padding offsets. Failing to account for this causes silent, catastrophic weight corruption.33 |
| **Constraint 10** | Native Gaussian Error Linear Unit (GELU) activation functions trigger severe performance regressions or are rejected entirely by the compiler.33 | GELU must be surgically replaced within the source model with its scaled tanh approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))`.33 |
| **Constraint 17** | Standard matrix multiplication (matmul) performs disproportionately poorly on the ANE architecture.33 | Engineers must reformulate dense layer operations. A 1x1 Conv2d formulation executes **3×** faster than the equivalent matmul on the ANE's specialized matrix ALUs.33 |
| **Constraint 18** | Multi-input IOSurface evaluation pipelines must possess uniform memory allocation sizes. Failure results in a hard 0x1d error at runtime evaluation.33 | Allocate all input tensors to match the maximum tensor size required by the model, padding smaller inputs to achieve uniformity across the memory surface.33 |
| **Constraint 19** | Multi-input memory surfaces must be ordered alphabetically by identifier. Failure results in silent wrong data evaluation.33 | Name all model inputs in strictly sorted alphabetical order within the Core ML generation script.33 |
| **Constraint 20** | The ANE fundamentally reads flat buffers as a packed **N-D hardware-strided** geometric array. Failure to align data results in silent wrong data inference.33 | Write packed data precisely at the start of the memory buffer to prevent silent data corruption and tensor misalignment during inference.33 |

By meticulously adhering to the Orion Catalog guidelines during the model construction phase in PyTorch or TensorFlow, developers can construct models that the Core ML compiler has no structural justification to reject, forcing it to bind the graph directly to the ANE.

## **Open Bugs and Unresolved Issues (Circa 2026)**

Even when all hardware constraints are respected, engineers must navigate unresolved bugs within the Core ML stack and lower-level execution engines. Recognizing these bugs is essential, as they often masquerade as architectural failures.

### **The MLState 32-Width Alignment Bug**

With the introduction of the stateful prediction APIs (MLState) in iOS 18 and macOS 15, Apple enabled models to efficiently hold multiple functions and manage state without continuous data reloading.20 However, this API introduced a severe hardware alignment bug.36

If an mlprogram uses the Stateful API and the state tensor's spatial width dimension is not a multiple of 32, the ANE compiler entirely rejects the execution.36

**Symptoms:** The model predicts flawlessly on the CPU and GPU. However, when deployed to the ANE via .all or .cpuAndNeuralEngine, the runtime fails silently or throws a fatal error in Xcode: "Unable to compute the prediction using ML Program. It can be an invalid input data or broken/unsupported model.".36

**The Root Cause:** The ANE driver's internal tiling and memory paging engine mishandles stateful buffers that lack a 32-pixel/element boundary alignment. For example, a state shape of (1, 3, 480, 270) triggers an immediate fallback, whereas (1, 3, 480, 256) succeeds perfectly.36

**The Engineering Workaround:** All state tensors must be artificially padded during the Python export phase so that the width dimension is uniformly divisible by 32. The subsequent application code must then crop the output back to the desired dimension.

```python
import coremltools as ct
from coremltools.converters.mil import Builder as mb

# BAD: shape=(1, 3, 480, 270) -> width 270 is not divisible by 32.
# This can trigger immediate ANE execution failure on stateful mlprograms (iOS 18+).

# GOOD: pad the width to the nearest multiple of 32 -> 288.
fixed_state_spec = mb.StateTensorSpec(
    shape=(1, 3, 480, 288),
    dtype=ct.converters.mil.mil_types.fp16,
)

# Define flex_tensor_spec to match your non-state inputs (example placeholder).
# flex_tensor_spec = mb.TensorSpec(shape=(1, 3, 480, 270), dtype=...)

# The downstream Swift application must crop outputs back to 270 if needed.
# @mb.program(input_specs=[flex_tensor_spec, fixed_state_spec], opset_version=ct.target.iOS18)
# def stateful_workaround_prog(flex_tensor, fixed_state):
#     ...
```

### **The Uncatchable MLIR Compiler Crash**

A critical, unresolved architectural bug exists on older Apple Neural Engine silicon (specifically targeting A13 chips reporting as ANE subtype 0x8030 on recent OS builds). Certain tensor operations cause the lower-level MLIR compiler to completely crash the execution thread via a C++ assertion.38

Because the crash originates deep within the MPSGraph system framework during the graph compilation phase, it cannot be caught via standard Swift do/catch blocks. A single incompatible model loaded onto an affected device will persistently and fatally crash the host application upon initialization.38

**The Engineering Workaround:** There is no configuration flag to disable this assertion. Engineers must build a resilience layer that preemptively maps device architectures, identifying unstable models and forcing them onto the CPU to bypass the MPSGraph and MLIR compiler entirely.38

```swift
import CoreML
import Foundation

/// Illustrative loader: try several `MLComputeUnits` paths or force CPU-only for flagged models.
final class ResilientModelLoader {
    private static let unstableModels: Set<String> = []

    private static let safeComputeUnitsOrder: [MLComputeUnits] = [.cpuOnly]

    private static let normalComputeUnitsOrder: [MLComputeUnits] = [
        .cpuAndNeuralEngine, .all, .cpuAndGPU, .cpuOnly,
    ]

    static func loadModel(named modelName: String) async throws -> MLModel {
        let isUnstable = unstableModels.contains(modelName)
        let computePaths = isUnstable ? safeComputeUnitsOrder : normalComputeUnitsOrder

        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
            throw NSError(
                domain: "ModelLoader",
                code: 404,
                userInfo: [NSLocalizedDescriptionKey: "Missing compiled model resource: \(modelName).mlmodelc"],
            )
        }

        for unit in computePaths {
            let config = MLModelConfiguration()
            config.computeUnits = unit
            do {
                let model = try await MLModel.load(contentsOf: modelURL, configuration: config)
                return model
            } catch {
                print("Failed to compile \(modelName) using \(unit), trying next compute unit…")
            }
        }

        throw NSError(
            domain: "ModelLoader",
            code: 500,
            userInfo: [NSLocalizedDescriptionKey: "All compute unit strategies failed for \(modelName)"],
        )
    }
}
```

### **The Rank 5 / NonMaximumSuppression Anomaly**

Models utilizing complex bounding box filtering from operations like NonMaximumSuppression (NMS) frequently generate tensor rank structures that exceed the Core ML Execution Provider threshold.2 This anomaly locks the output shape fields in Xcode, preventing explicit static definitions and forcing dynamic shape evaluation at runtime.36

Because the ANE strictly requires static memory layouts, these dynamic NMS outputs automatically force all trailing execution blocks in the graph to fall back to the CPU.2

**The Engineering Workaround:** Never compile NonMaximumSuppression into the Core ML model. Engineers must truncate the model immediately before the NMS operation, exporting the raw bounding box coordinates as the final output. The NMS logic must then be implemented purely in Swift using standard Accelerate and vDSP operations. This maintains the static nature of the Core ML graph, preserving ANE execution for the entire neural network payload.

## **Advanced Workarounds and Engineering Best Practices**

To extract deterministic performance from Apple Silicon, engineers must stop viewing Core ML as a black-box model converter and begin treating it as an explicit graph-partitioning compiler.

### **Best Practice 1: Manual Graph Partitioning (Model Splitting)**

When deploying a model containing a mix of ANE-compatible subgraphs and custom/unsupported layers, relying on the automatic .all scheduler frequently incurs disastrous context-switching overheads.16 As data moves between the CPU, GPU, and ANE, the cost of transferring memory contexts outpaces the acceleration gained by the hardware.30

A highly effective "black magic" workaround is manually splitting the neural network into multiple discrete .mlpackage files.16

For example, a complex image captioning architecture should be strictly partitioned into two models:

1. **Feature Extractor (e.g., ResNet or ViT block):** Contains purely static, standard convolution layers that the ANE supports flawlessly.  
2. **Caption Generator (e.g., Autoregressive Decoder):** Contains custom layers, dynamic KV caching, and concat operations that immediately crash the ANE compiler.

The engineer loads the first model explicitly with .cpuAndNeuralEngine and the second explicitly with .cpuAndGPU.4 This programmatic memory barrier forces Core ML to obey strict hardware boundaries. By connecting the output of the first model to the input of the second in Swift, the heavy feature extraction runs entirely on the ANE, while the dynamic decoding runs on the GPU. This manual partitioning frequently results in a **10×** overall speedup compared to allowing the scheduler to blindly thrash data between memory buses.16

```swift
import CoreML

func initializePartitionedPipeline() throws {
    // 1. Static feature extractor: CPU + ANE (no GPU).
    let aneConfig = MLModelConfiguration()
    aneConfig.computeUnits = .cpuAndNeuralEngine
    let featureExtractor = try CaptionMobile_features(configuration: aneConfig)

    // 2. Dynamic decoder: CPU + GPU (no ANE).
    let gpuConfig = MLModelConfiguration()
    gpuConfig.computeUnits = .cpuAndGPU
    let decoder = try CaptionMobile_caption(configuration: gpuConfig)

    _ = (featureExtractor, decoder)
}
```

### **Best Practice 2: Programmatic Fallback Detection via Latency Baselining**

Because Core ML provides no native API to query the *active* execution hardware at runtime (the MLModelConfiguration.computeUnits property only reflects the user's *requested* state, not the compiler's actual execution target 8), developers must build programmatic fallback detection using baseline latency comparisons.

If a model is requested to run on .all but experiences a silent fallback to the CPU, its inference latency will significantly diverge from expected hardware acceleration times. By profiling a warmup pass using .cpuAndGPU versus .all, the application can dynamically detect if the ANE is actively participating and abort operations if thermal or battery limits are threatened.

```swift
import CoreML
import Foundation

/// Heuristic: if `.all` is not materially faster than `.cpuAndGPU`, the ANE may not be helping.
func detectANEParticipation(modelURL: URL) async throws -> Bool {
    let gpuConfig = MLModelConfiguration()
    gpuConfig.computeUnits = .cpuAndGPU
    let gpuModel = try await MLModel.load(contentsOf: modelURL, configuration: gpuConfig)
    let gpuTime = try measureInferenceLatency(model: gpuModel)

    let aneConfig = MLModelConfiguration()
    aneConfig.computeUnits = .all
    let aneModel = try await MLModel.load(contentsOf: modelURL, configuration: aneConfig)
    let aneTime = try measureInferenceLatency(model: aneModel)

    if aneTime >= (gpuTime * 0.95) {
        print("Silent fallback suspected: .all not faster than .cpuAndGPU baseline.")
        return false
    }

    print("ANE likely active. Latency delta (GPU − .all): \(gpuTime - aneTime) s")
    return true
}

/// Replace the dummy shape with your model’s real input geometry.
func measureInferenceLatency(model: MLModel) throws -> TimeInterval {
    let dummyInputArray = try MLMultiArray(shape: [1, 224, 224, 3], dataType: .float32)
    let dummyInput = try MLDictionaryFeatureProvider(dictionary: ["image": dummyInputArray])

    _ = try model.prediction(from: dummyInput)

    let start = CFAbsoluteTimeGetCurrent()
    _ = try model.prediction(from: dummyInput)
    return CFAbsoluteTimeGetCurrent() - start
}
```

### **Best Practice 3: Linear to Conv2d Transformation for Transformers**

To conform to the ANE's hardware preference for (B, C, 1, S) geometric layouts and spatial convolution logic (Constraint 17 of the Orion Catalog), developers converting PyTorch LLMs and Vision Transformers to Core ML must mathematically manipulate the source architecture.30

During the coremltools conversion pipeline, engineers should utilize a PyTorch load_state_dict_pre_hook to dynamically unsqueeze the weights of nn.Linear layers, mapping them seamlessly into nn.Conv2d operations.30 This architectural sleight-of-hand maps the dense matrix multiplication workloads onto the ANE's highly parallel spatial convolution ALUs, bypassing the severe memory bottlenecks associated with standard tensor manipulations on Apple Silicon.

### **Best Practice 4: Eliminating Memory Copies via Einsum**

Reshape and transpose operations scattered inside a neural network frequently trigger forced memory copies on the ANE, because Apple Silicon hardware does not natively support unpacked memory layouts for these dimensional transitions.30 Every memory copy injected into the graph destroys inference latency.

Instead of routing data through sequential reshape and transpose nodes during complex multi-head attention blocks, developers must collapse these execution paths using the einsum operator. Specifically, utilizing the bchq,bkhc->bkhq einsum notation maps batched matrix multiplications directly to the physical hardware without requiring intermediary memory allocations.30

*Note:* The translation of einsum by coremltools must be meticulously monitored. Depending on the exact compiler version, pure einsum operations may act as unsupported nodes.2 The conversion script must ensure the einsum resolves to static mathematical equivalents in the final MIL representation.

## **Worst Practices to Eliminate**

The following patterns reliably degrade performance, inflate application size, or cause immediate runtime failures in Core ML environments. They must be eliminated from production pipelines:

1. **Deploying with Dynamic Input Shapes:** Allowing the coremltools converter to export inputs with dynamic ranges (RangeDim) guarantees CPU fallback. The ANE requires exact, static dimension bounds at compile time to map the neural graph to physical memory addresses.1 Ensure RequireStaticInputShapes is enforced.  
2. **Leaving Singleton Axes Intact:** A model generating outputs with dimensions like **(B, C, S, 1)** (singleton last axis) forces the Neural Engine to pad the final axis up to 64 bytes in physical memory. Leaving this singleton axis intact causes catastrophic memory bloat. Removing the axis or rearranging the tensor geometry prevents this.30  
3. **Blind Trust in the Core ML Heuristic:** Assuming MLComputeUnits.all is universally superior is a fatal flaw. In heavy vision pipelines, the ANE-to-GPU context switch can cost more latency than executing the entire graph on the GPU alone.13 Engineers must always benchmark .cpuAndNeuralEngine against .cpuAndGPU on physical target devices.12  
4. **Neglecting Quantization Hardware Alignment:** When employing 8-bit integer (int8) or 16-bit floating point (fp16) weight compression, ensure the operations within the compressed block are cleanly supported by the ANE. Certain quantization pairings trigger fallback nodes that forcibly unpack the weights back to FP32, routing the graph back to the CPU (BNNSEngine) and entirely negating both the size and speed benefits of the compression.31

## **Synthesized Conclusions**

Mastering Core ML Compute Unit scheduling on Apple Silicon requires peeling back the user-friendly abstraction layers to understand the exact mathematical and physical demands of the Apple Neural Engine. The platform severely penalizes ambiguity. When a developer provides a graph containing dynamic dimensions, incompatible layouts, or unsupported operations, the internal Core ML compiler aggressively defaults to the path of highest stability—silent execution on the CPU or GPU.

By combining powermetrics to detect zero-watt ANE utilization, structuring models to strictly adhere to the 64-byte hardware padding constraints, padding stateful tensors to 32-width boundaries to avoid OS bugs, and manually partitioning mixed-architecture graphs via discrete MLComputeUnits, engineers can successfully circumvent the scheduler's limitations. Deterministic, high-performance machine learning execution on Apple Silicon is not achieved by allowing the operating system to guess the optimal execution path, but by meticulously architecting the model geometry to leave the compiler with only one logical choice.

For SSD-tier model-weight experiments, combine this placement guidance with the
[Apple Silicon NVMe and energy measurement guide](apple-silicon-nvme-energy-measurement-guide.md);
that guide covers `F_NOCACHE`, `fs_usage`, physical disk-read proof, and
joules/token capture for storage-backed prefetch claims.

#### **Works cited**

1. Apple - CoreML | onnxruntime, accessed April 14, 2026, [https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)  
2. [Performance] Why is dynamic shape not supported with the CoreML provider, while CoreML 2+ supports it ? · Issue #14212 · microsoft/onnxruntime - GitHub, accessed April 14, 2026, [https://github.com/microsoft/onnxruntime/issues/14212](https://github.com/microsoft/onnxruntime/issues/14212)  
3. [D] Anyone successfully running LLMs fully on Apple Neural Engine (ANE)? - Reddit, accessed April 14, 2026, [https://www.reddit.com/r/MachineLearning/comments/1n1pcj7/d_anyone_successfully_running_llms_fully_on_apple/](https://www.reddit.com/r/MachineLearning/comments/1n1pcj7/d_anyone_successfully_running_llms_fully_on_apple/)  
4. Typed Execution — Guide to Core ML Tools - Apple, accessed April 14, 2026, [https://apple.github.io/coremltools/docs-guides/source/typed-execution.html](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html)  
5. How to Use Core ML in iOS: A Complete Guide with Examples - Zignuts Technolab, accessed April 14, 2026, [https://www.zignuts.com/blog/how-to-use-core-ml-in-ios-guide](https://www.zignuts.com/blog/how-to-use-core-ml-in-ios-guide)  
6. GitHub - mechramc/Orion: Local AI runtime for training & running small LLMs directly on Apple Neural Engine (ANE). No CoreML. No Metal. Offline, on-device fine-tuning & inference on M-series silicon., accessed April 14, 2026, [https://github.com/mechramc/Orion](https://github.com/mechramc/Orion)  
7. Daily Papers - Hugging Face, accessed April 14, 2026, [https://huggingface.co/papers?q=MIL%20IR](https://huggingface.co/papers?q=MIL+IR)  
8. neural-engine/docs/is-model-using-ane.md at master - GitHub, accessed April 14, 2026, [https://github.com/hollance/neural-engine/blob/master/docs/is-model-using-ane.md](https://github.com/hollance/neural-engine/blob/master/docs/is-model-using-ane.md)  
9. Core ML Survival Guide | PDF | Computer Vision | Machine Learning - Scribd, accessed April 14, 2026, [https://www.scribd.com/document/689236733/Core-ML-Survival-Guide](https://www.scribd.com/document/689236733/Core-ML-Survival-Guide)  
10. My iPhone 16 Pro Max produces garbage output when running MLX LLMs - Hacker News, accessed April 14, 2026, [https://news.ycombinator.com/item?id=46849258](https://news.ycombinator.com/item?id=46849258)  
11. 立委NLP频道, accessed April 14, 2026, [https://liweinlp.com/author/liweinlp](https://liweinlp.com/author/liweinlp)  
12. MLComputeUnits | Apple Developer Documentation, accessed April 14, 2026, [https://developer.apple.com/documentation/coreml/mlcomputeunits](https://developer.apple.com/documentation/coreml/mlcomputeunits)  
13. .all vs .cpuAndNeuralEngine? · Issue #122 · apple/ml-stable-diffusion, accessed April 14, 2026, [https://github.com/apple/ml-stable-diffusion/issues/122](https://github.com/apple/ml-stable-diffusion/issues/122)  
14. MLComputeUnits.cpuAndGPU | Apple Developer Documentation, accessed April 14, 2026, [https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandgpu](https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandgpu)  
15. ANE inference fails on M4 + macOS 26.4 beta with CoreML encoder #3702 - GitHub, accessed April 14, 2026, [https://github.com/ggml-org/whisper.cpp/issues/3702](https://github.com/ggml-org/whisper.cpp/issues/3702)  
16. CoreML with Custom Layers have bug on devices with Apple Neural Engine, accessed April 14, 2026, [https://stackoverflow.com/questions/56194696/coreml-with-custom-layers-have-bug-on-devices-with-apple-neural-engine](https://stackoverflow.com/questions/56194696/coreml-with-custom-layers-have-bug-on-devices-with-apple-neural-engine)  
17. Code=9 "Could not create inference context" CoreML iOS - Stack Overflow, accessed April 14, 2026, [https://stackoverflow.com/questions/75863781/code-9-could-not-create-inference-context-coreml-ios](https://stackoverflow.com/questions/75863781/code-9-could-not-create-inference-context-coreml-ios)  
18. Vision Pro CoreML seem to only run on CPU (10x slower) - Reddit, accessed April 14, 2026, [https://www.reddit.com/r/visionosdev/comments/1aue2jw/vision_pro_coreml_seem_to_only_run_on_cpu_10x/](https://www.reddit.com/r/visionosdev/comments/1aue2jw/vision_pro_coreml_seem_to_only_run_on_cpu_10x/)  
19. Misleading benchmarks? #54 - apple/ml-stable-diffusion - GitHub, accessed April 14, 2026, [https://github.com/apple/ml-stable-diffusion/issues/54](https://github.com/apple/ml-stable-diffusion/issues/54)  
20. Core ML - Machine Learning - Apple Developer, accessed April 14, 2026, [https://developer.apple.com/machine-learning/core-ml/](https://developer.apple.com/machine-learning/core-ml/)  
21. Efficient Accelerator-Rich Computers for Future Applications - eScholarship.org, accessed April 14, 2026, [https://escholarship.org/uc/item/68w3z4vq](https://escholarship.org/uc/item/68w3z4vq)  
22. powermetrics — Firefox Source Docs documentation - Mozilla, accessed April 14, 2026, [https://firefox-source-docs.mozilla.org/performance/powermetrics.html](https://firefox-source-docs.mozilla.org/performance/powermetrics.html)  
23. How to See Individual Core CPU Usage on Mac with powermetrics - OS X Daily, accessed April 14, 2026, [https://osxdaily.com/2024/07/05/how-to-see-individual-core-cpu-usage-on-mac-with-powermetrics/](https://osxdaily.com/2024/07/05/how-to-see-individual-core-cpu-usage-on-mac-with-powermetrics/)  
24. apple-m1-power-consumption-powermetrics/powermetrics-tool-help-text.txt at main - GitHub, accessed April 14, 2026, [https://github.com/singhkays/apple-m1-power-consumption-powermetrics/blob/main/powermetrics-tool-help-text.txt](https://github.com/singhkays/apple-m1-power-consumption-powermetrics/blob/main/powermetrics-tool-help-text.txt)  
25. powermetrics(1) osx man page | unix.com, accessed April 14, 2026, [https://www.unix.com/man_page/osx/1/powermetrics/](https://www.unix.com/man_page/osx/1/powermetrics/)  
26. Figuring out if Core ML models use the Apple Neural Engine - Heartbeat - Comet, accessed April 14, 2026, [https://heartbeat.comet.ml/figuring-out-if-core-ml-models-use-the-apple-neural-engine-b5b07cd55f4b](https://heartbeat.comet.ml/figuring-out-if-core-ml-models-use-the-apple-neural-engine-b5b07cd55f4b)  
27. coreml-optimizer | Skills Marketplace - LobeHub, accessed April 14, 2026, [https://lobehub.com/zh-TW/skills/ckorhonen-claude-skills-coreml-optimizer](https://lobehub.com/zh-TW/skills/ckorhonen-claude-skills-coreml-optimizer)  
28. Crash on generate using converted model · Issue #307 - GitHub, accessed April 14, 2026, [https://github.com/godly-devotion/MochiDiffusion/issues/307](https://github.com/godly-devotion/MochiDiffusion/issues/307)  
29. [2603.06728] Orion: Characterizing and Programming Apple's Neural Engine for LLM Training and Inference - arXiv, accessed April 14, 2026, [https://arxiv.org/abs/2603.06728](https://arxiv.org/abs/2603.06728)  
30. Deploying Transformers on the Apple Neural Engine - Apple ..., accessed April 14, 2026, [https://machinelearning.apple.com/research/neural-engine-transformers](https://machinelearning.apple.com/research/neural-engine-transformers)  
31. Optimization Guidelines for the Apple Neural Engine (ANE) · GitHub, accessed April 14, 2026, [https://gist.github.com/antmikinka/715499ae63630575065b22e5cb6ad8dd](https://gist.github.com/antmikinka/715499ae63630575065b22e5cb6ad8dd)  
32. Daily Papers - Hugging Face, accessed April 14, 2026, [https://huggingface.co/papers?q=program%20caching](https://huggingface.co/papers?q=program+caching)  
33. Orion: Characterizing and Programming Apple's Neural Engine for LLM Training and Inference - arXiv, accessed April 14, 2026, [https://arxiv.org/html/2603.06728v1](https://arxiv.org/html/2603.06728v1)  
34. Orion: Characterizing and Programming Apple's Neural Engine for LLM Training and Inference - ResearchGate, accessed April 14, 2026, [https://www.researchgate.net/publication/401719058_Orion_Characterizing_and_Programming_Apple's_Neural_Engine_for_LLM_Training_and_Inference](https://www.researchgate.net/publication/401719058_Orion_Characterizing_and_Programming_Apple's_Neural_Engine_for_LLM_Training_and_Inference)  
35. Core ML updates | Apple Developer Documentation, accessed April 14, 2026, [https://developer.apple.com/documentation/updates/coreml](https://developer.apple.com/documentation/updates/coreml)  
36. ML Compute | Apple Developer Forums, accessed April 14, 2026, [https://developer.apple.com/forums/tags/ml-compute](https://developer.apple.com/forums/tags/ml-compute)  
37. Core ML | Apple Developer Forums, accessed April 14, 2026, [https://developer.apple.com/forums/forums/topics/machine-learning-and-ai/machine-learning-topic-core-ml?page=2](https://developer.apple.com/forums/forums/topics/machine-learning-and-ai/machine-learning-topic-core-ml?page=2)  
38. The Uncatchable CoreML Crash: How a single MLIR compiler failure on the iPhone SE2 cost me a week : r/swift - Reddit, accessed April 14, 2026, [https://www.reddit.com/r/swift/comments/1s777t4/the_uncatchable_coreml_crash_how_a_single_mlir/](https://www.reddit.com/r/swift/comments/1s777t4/the_uncatchable_coreml_crash_how_a_single_mlir/)  
39. On-Device AI Models and Core ML Tools: Insights From WWDC 2024 | HackerNoon, accessed April 14, 2026, [https://hackernoon.com/on-device-ai-models-and-core-ml-tools-insights-from-wwdc-2024](https://hackernoon.com/on-device-ai-models-and-core-ml-tools-insights-from-wwdc-2024)
