# Kokoro-82M to CoreML: Production Conversion Guide

## Quick Reality Check

**What works on ANE:** Only the iSTFTNet vocoder (~50% of compute)
- Supported: Conv1d, ConvTranspose1d, LeakyReLU operations
- Result: 30-50% overall speedup possible with vocoder on ANE

**What doesn't:** Text encoder and prosody predictor  
- Blocked by: LSTM layers (no ANE support, ever)
- Also blocked: AdaLayerNorm, AdainResBlk1d (custom style injection)

**Required strategy:** Three-model pipeline
1. G2P model (CPU) → phonemes
2. Text encoder + prosody (CPU/GPU) → mel-spectrogram  
3. Vocoder (ANE) → audio waveform

## Prerequisites

```bash
# Verify ARM64 Python (critical - MPS won't work under Rosetta)
python -c "import platform; print(platform.platform())"  # Must show 'arm64'

# Tested versions (July 2025)
pip install coremltools==7.0  # or 7.0b5
pip install torch==2.3.0 torchaudio numpy==1.26.4
```

## Step 1: Replace espeak-ng Dependency

iOS apps can't execute external binaries. Choose one:

### Option A: Neural G2P Model (Recommended)
Use LiteG2P or similar lightweight model:
```python
import torch
import coremltools as ct
import numpy as np

# Load your G2P model (example with hypothetical LiteG2P)
g2p_model = load_liteg2p_model()  # Your G2P model loading code
g2p_model.eval()

# Create example input for tracing
example_text = "Hello world"
example_text_ids = text_to_ids(example_text)  # Your tokenization
example_input = torch.tensor([example_text_ids])

# Trace the model
traced_g2p = torch.jit.trace(g2p_model, example_input)

# Convert with dynamic text length support
g2p_coreml = ct.convert(
    traced_g2p,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS16,
    inputs=[ct.TensorType(
        name="text_ids",
        shape=(1, ct.RangeDim(1, 100)),  # Support 1-100 tokens
        dtype=np.int32
    )]
)
g2p_coreml.save("G2P.mlpackage")
```

### Option B: C++ Integration (Complex)
Use babylon.cpp - requires Objective-C++ wrapper and CMake setup. Skip unless you have specific requirements.

## Step 2: Extract and Prepare the Vocoder

### Automatic Split Detection (Optional)
If you're unsure where to split the model:
```python
import coremltools.models.utils as utils

# This utility can help identify where CPU fallbacks occur
first_part, second_part = utils.bisect_model(
    full_model, 
    "first_cpu_op_name"  # Find this from Performance Report
)
```

### Manual Extraction (Recommended for Kokoro)
```python
import torch
from kokoro import KokoroModel  # Adjust import as needed

# Load full model
full_model = KokoroModel.from_pretrained("kokoro-82M")

# Inspect model structure to find vocoder
print(full_model)  # Look for decoder/vocoder module
# Common paths: full_model.decoder, full_model.vocoder, full_model.generator

# Extract ONLY the iSTFTNet vocoder (adjust path based on inspection)
vocoder = full_model.decoder  # or full_model.vocoder, full_model.generator, etc.

# Critical fix for Kokoro speed parameter
class VocoderWrapper(torch.nn.Module):
    def __init__(self, vocoder):
        super().__init__()
        self.vocoder = vocoder
    
    def forward(self, mel_spec, speed=1.0):
        # Fix scalar broadcasting issue
        speed = speed.float().view(-1, 1, 1)
        return self.vocoder(mel_spec, speed)

wrapped_vocoder = VocoderWrapper(vocoder)
wrapped_vocoder.eval()

# Trace with representative input
# Shape: (batch, mel_channels, 1, sequence_length) - ANE prefers this layout
example_mel = torch.randn(1, 80, 1, 256)
traced_vocoder = torch.jit.trace(wrapped_vocoder, (example_mel, torch.tensor(1.0)))
```

## Step 3: Convert with ANE-Optimal Settings

```python
import coremltools as ct
import numpy as np

# Define dynamic input for variable length sequences
mel_input = ct.TensorType(
    name="mel_spectrogram",
    shape=(1, 80, 1, ct.RangeDim(10, 1024)),  # Set reasonable upper bound
    dtype=np.float16
)

speed_input = ct.TensorType(
    name="speed", 
    shape=(1,),
    dtype=np.float16
)

# Convert with mandatory ANE settings
vocoder_model = ct.convert(
    traced_vocoder,
    convert_to="mlprogram",           # Required for ANE
    compute_precision=ct.precision.FLOAT16,  # ANE native precision
    minimum_deployment_target=ct.target.iOS16,
    compute_units=ct.ComputeUnit.ALL,
    inputs=[mel_input, speed_input]
)

# For iOS 18+: Package all components together
if targeting_ios18:
    # Multifunction models can package CPU and ANE parts in one .mlpackage
    # with automatic weight deduplication (e.g., shared embeddings)
    import coremltools as ct
    
    # Convert each component with same shared weights
    text_encoder_ml = ct.convert(text_encoder_traced, ...)
    vocoder_ml = ct.convert(vocoder_traced, ...)
    
    # Package as multifunction - reduces app size if models share weights
    from coremltools.models import CompiledMLModel
    multifunction_model = ct.models.CompiledMLModel([
        ("text_encoder", text_encoder_ml),
        ("vocoder", vocoder_ml)
    ])
    multifunction_model.save("KokoroComplete.mlpackage")

vocoder_model.save("KokoroVocoder.mlpackage")
```

## Step 4: Handle Unsupported Operations

### Priority 1: Rewrite in PyTorch
```python
# Example: Replace torch.var if unsupported
# Before: output = x.var(dim=-1)
# After:
mean = x.mean(dim=-1, keepdim=True)
output = ((x - mean) ** 2).mean(dim=-1)
```

### Priority 2: Composite Operator
```python
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs

@register_torch_op
def silu(context, node):
    """SiLU/Swish activation: x * sigmoid(x)"""
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    
    sigmoid_x = mb.sigmoid(x=x, name=node.name + "_sigmoid")
    output = mb.mul(x=x, y=sigmoid_x, name=node.name)
    context.add(output)

# For AdaLayerNorm (if needed for other StyleTTS2 models)
@register_torch_op(torch_alias=["ada_layer_norm"])
def ada_layer_norm(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, style_vector = inputs[0], inputs[1]
    
    # Manual normalization
    mean = mb.reduce_mean(x=x, axes=[-1], keep_dims=True)
    variance = mb.reduce_mean(
        x=mb.square(x=mb.sub(x=x, y=mean)), 
        axes=[-1], 
        keep_dims=True
    )
    normalized = mb.mul(
        x=mb.sub(x=x, y=mean),
        y=mb.rsqrt(x=mb.add(x=variance, y=1e-5))
    )
    
    # Apply style-dependent scaling/bias (implementation specific)
    # ...
    
    context.add(output)
```

### Priority 3: Custom Layer (Avoid)
Only if absolutely necessary - kills ANE performance due to memory transfers.

## Step 5: Verify ANE Execution

### Level 0: Visual Inspection with Netron
Drag your `.mlpackage` into [netron.app](https://netron.app) to visualize the model graph. Quickly spot:
- Unsupported operations that might cause fallback
- Data flow bottlenecks
- Layer types and connections

### Level 1: Xcode Performance Report
1. Add .mlpackage to Xcode project
2. Select device → Performance tab
3. Check "Estimated Compute Unit" column - should show Neural Engine for conv layers

### Level 2: Instruments Profiling
```bash
# Profile from Xcode: Cmd+I → Core ML template
# Manually add: Neural Engine, GPU instruments
```

Look for:
- ✅ Activity in Neural Engine track during inference
- ❌ Gaps in Neural Engine track = fallback to CPU/GPU
- Thread names: `H11ANEServicesThread` (ANE), `Espresso::MPSEngine` (GPU), `Espresso::BNNSEngine` (CPU)

### Level 3: Definitive Proof (Symbolic Breakpoints)
In LLDB:
```
br set -n "_ANEModel program"  # ANE execution
br set -n "Espresso::BNNSEngine::convolution_kernel::__launch"  # CPU fallback
br set -n "Espresso::MPSEngine::context::__launch_kernel"  # GPU fallback
```

If CPU/GPU breakpoints hit = silent fallback detected.

### Quick Alternative: powermetrics
For rapid ANE verification without debugger:
```bash
sudo powermetrics -i 1000 --samplers ane | grep "ANE Power"
# Non-zero power = ANE is active
```

## Step 6: Optimize Performance

### Memory Layout Critical Rule
**Last axis must be largest dimension** to avoid 64-byte alignment penalty:
- ✅ Use shape: `(B, C, 1, S)` where S = sequence length
- ❌ Never: `(B, S, C)` - causes up to 64x memory bloat

**Why this matters:** ANE requires the last axis to be contiguous in memory and aligned to 64-byte boundaries. If you put a small dimension (like channels) last, ANE pads it to 64 bytes, multiplying your memory usage dramatically.

**ANE L2 Cache Limit: ~32MB** - chunk operations to fit within this for best performance.

### Advanced Optimizations
```python
# 1. Attention head splitting (if applicable)
# Split large attention operations into per-head computations

# 2. Replace Linear with Conv1d for ANE
# Before: self.linear = nn.Linear(512, 256)
# After:  self.conv = nn.Conv1d(512, 256, kernel_size=1)

# 3. Use EnumeratedShapes for common lengths (better ANE performance)
if common_lengths := [128, 256, 512, 1024]:
    mel_input = ct.TensorType(
        shape=ct.EnumeratedShapes(shapes=[(1, 80, 1, l) for l in common_lengths]),
        dtype=np.float16
    )
```

### Weight Compression
```python
import coremltools.optimize as cto

# After conversion, before saving
op_config = cto.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric",
    dtype="int8"
)
config = cto.coreml.OptimizationConfig(global_config=op_config)
compressed_model = cto.coreml.linear_quantize_weights(vocoder_model, config)
```

**Quantization Trade-offs for Kokoro Vocoder:**
| Method | Audio Quality Impact | Size Reduction | Latency (A17/M4) |
|--------|---------------------|----------------|------------------|
| FP16 (baseline) | None | 2× | Baseline |
| W8A16 | <1% THD increase | 4× | ~Same |
| W8A8 | 1-3% THD increase | 4× | 30-50% faster |

For TTS, test with perceptual metrics:
- **PESQ** (Perceptual Evaluation of Speech Quality): Should stay above 4.0
- **MCD** (Mel Cepstral Distortion): Increase of <0.5 dB is imperceptible
- **A/B listening tests**: Most reliable for production

## Step 7: Swift Integration

```swift
import CoreML
import AVFoundation

// Two-stage inference pipeline
class KokoroTTS {
    let g2pModel: MLModel
    let vocoderModel: MLModel
    
    init() throws {
        let g2pConfig = MLModelConfiguration()
        g2pConfig.computeUnits = .cpuAndGPU  // G2P is usually small
        self.g2pModel = try MLModel(contentsOf: Bundle.main.url(forResource: "G2P", withExtension: "mlpackage")!, configuration: g2pConfig)
        
        let vocoderConfig = MLModelConfiguration()
        vocoderConfig.computeUnits = .all  // Allow ANE for vocoder
        self.vocoderModel = try MLModel(contentsOf: Bundle.main.url(forResource: "KokoroVocoder", withExtension: "mlpackage")!, configuration: vocoderConfig)
    }
    
    func synthesize(text: String) throws -> AVAudioPCMBuffer {
        // Stage 1: Text to phonemes
        let textIds = textToIds(text)  // Your tokenization
        let g2pInput = try MLMultiArray(shape: [1, NSNumber(value: textIds.count)], dataType: .int32)
        // Fill g2pInput with textIds...
        
        let g2pOutput = try g2pModel.prediction(from: ["text_ids": g2pInput])
        let phonemes = g2pOutput.featureValue(for: "phonemes")!.multiArrayValue!
        
        // Stage 2: Phonemes to audio (through your mel-spectrogram generation)
        let melSpec = generateMelSpectrogram(from: phonemes)  // Your implementation
        let melInput = try MLMultiArray(shape: [1, 80, 1, NSNumber(value: melSpec.count)], dataType: .float16)
        let speedInput = try MLMultiArray([1.0])
        // Fill melInput...
        
        let vocoderOutput = try vocoderModel.prediction(from: [
            "mel_spectrogram": melInput,
            "speed": speedInput
        ])
        
        let waveform = vocoderOutput.featureValue(for: "audio")!.multiArrayValue!
        return convertToAudioBuffer(waveform)
    }
    
    private func convertToAudioBuffer(_ waveform: MLMultiArray) -> AVAudioPCMBuffer {
        // Implementation depends on your audio format
        // Typically convert Float16 array to PCM samples
    }
}
```

## Debugging Checklist

| Problem | Solution |
|---------|----------|
| Conversion fails on unsupported op | Check Step 4 - rewrite or use composite operator |
| Slow inference | Run Step 5 verification - check for CPU/GPU fallback |
| Memory errors | Verify tensor shape is (B,C,1,S), set reasonable RangeDim upper_bound |
| Numerical differences | Normal for FP16; validate with audio quality metrics, not tensor equality |
| "NotImplementedError" | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for debugging only |
| Speed parameter error | Add `speed.float().view(-1, 1, 1)` in wrapper's forward method |

## Critical Warnings

1. **Never use custom layers** unless absolutely necessary - they break ANE pipeline
2. **Always set RangeDim upper bounds** - unbounded = memory crashes
3. **Verify with Instruments** - Xcode reports show intent, not reality
4. **Test on target hardware** - M1 ≠ M2 ≠ A17 performance characteristics
5. **ANE↔CPU copies are cheaper than ANE↔GPU** - if fallback is unavoidable, force CPU-only for non-ANE parts

## Final Architecture

```
Text → [G2P CoreML Model] → Phonemes → [CPU: Text Encoder + Prosody] → Mel Spectrogram → [ANE: Vocoder] → Audio
```

Only the vocoder runs on ANE, but that's where the compute-heavy work happens.