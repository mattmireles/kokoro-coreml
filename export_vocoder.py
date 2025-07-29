#!/usr/bin/env python3
"""
Kokoro Vocoder Extraction and CoreML Conversion Script

This script extracts the iSTFTNet vocoder (Decoder) from the full Kokoro model
and converts it to a CoreML package optimized for Apple Neural Engine (ANE).

The vocoder is the compute-heavy component that can run efficiently on ANE,
while the text processing components (BERT, LSTM) must remain on CPU.

Architecture Split:
- CPU: Text encoding, prosody prediction, duration alignment
- ANE: Audio synthesis via iSTFTNet vocoder (this script)

Key Technical Details:
- Extracts model.decoder (contains Generator from istftnet.py)
- Uses FP16 precision for ANE optimization
- Handles proper tensor shape layout for ANE memory efficiency
- Creates wrapper for CoreML-compatible I/O format
"""

import torch
import coremltools as ct
import numpy as np
from kokoro import KModel

# ANE-optimized conversion settings
COMPUTE_PRECISION = ct.precision.FLOAT16  # ANE native precision
MINIMUM_DEPLOYMENT_TARGET = ct.target.iOS16  # Latest ANE features
COMPUTE_UNITS = ct.ComputeUnit.ALL  # Allow ANE + GPU + CPU as needed

class VocoderWrapper(torch.nn.Module):
    """
    CoreML-compatible wrapper around the Kokoro decoder (vocoder).
    
    The original decoder expects multiple input tensors with specific shapes.
    This wrapper provides a clean interface that matches CoreML conversion
    requirements and handles proper tensor formatting.
    
    Key Considerations:
    - ANE memory layout: Last dimension should be largest for efficiency
    - Fixed input shapes for better ANE optimization
    - Proper handling of F0 curve and noise parameters
    """
    
    def __init__(self, decoder):
        """
        Initialize vocoder wrapper.
        
        Args:
            decoder: The extracted decoder module from KModel
        """
        super().__init__()
        self.decoder = decoder
        
    def forward(self, asr, f0_curve, n, s):
        """
        Forward pass through the vocoder.
        
        Args:
            asr: Aligned acoustic features, shape (1, 512, T)
            f0_curve: F0/pitch curve, shape (1, T) 
            n: Noise parameters, shape (1, T)
            s: Voice style embedding, shape (1, 128)
            
        Returns:
            audio: Generated waveform, shape (1, 1, audio_length)
        """
        # The decoder expects specific input format: (asr, F0_curve, N, s)
        audio = self.decoder(asr, f0_curve, n, s)
        
        # Ensure output shape is consistent for CoreML
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # Add channel dimension
        
        return audio

class SimpleGeneratorWrapper(torch.nn.Module):
    """
    Simplified wrapper that extracts just the Generator component.
    
    This is a fallback approach that focuses on the core synthesis
    part which should be more ANE-compatible.
    """
    
    def __init__(self, decoder):
        """
        Initialize with just the generator from the decoder.
        
        Args:
            decoder: The decoder module containing the generator
        """
        super().__init__()
        self.generator = decoder.generator
        
    def forward(self, x, s, f0_curve):
        """
        Direct generator forward pass.
        
        Args:
            x: Processed features, shape (1, 512, T) 
            s: Style embedding, shape (1, 128)
            f0_curve: F0 curve, shape (1, T*2) (upsampled)
            
        Returns:
            audio: Generated waveform
        """
        return self.generator(x, s, f0_curve)

def inspect_model_structure(model):
    """
    Inspect the model structure to understand the decoder architecture.
    
    This helps identify the exact input shapes and requirements for
    the decoder component that we'll be extracting.
    """
    print("\n🔍 Model Structure Analysis:")
    print(f"Model type: {type(model).__name__}")
    print("\nMain components:")
    for name, module in model.named_children():
        print(f"  - {name}: {type(module).__name__}")
        
    print(f"\n📊 Decoder details:")
    decoder = model.decoder
    print(f"Decoder type: {type(decoder).__name__}")
    print("Decoder submodules:")
    for name, module in decoder.named_children():
        print(f"  - {name}: {type(module).__name__}")
        
    return decoder

def create_sample_inputs():
    """
    Create realistic sample inputs that match the decoder's expected format.
    
    These inputs are based on the actual data flow from the full model:
    - asr: Aligned acoustic features from text encoder
    - f0_curve: F0/pitch predictions from prosody predictor  
    - noise: Noise parameters for vocoder
    - style: Voice style embedding (first 128 dims of ref_s)
    
    Returns:
        Dictionary of sample inputs for tracing
    """
    # Typical sequence length for a short phrase (about 2-3 seconds)
    # The decoder has stride=2 convolutions, so we need to account for this
    # F0_curve and N will be downsampled by 2x, so asr needs to match the downsampled length
    sequence_length_input = 400  # Original F0 curve length  
    sequence_length_asr = sequence_length_input // 2  # ASR features after F0/N convolution
    
    # Sample inputs matching decoder expectations
    sample_inputs = {
        "asr": torch.randn(1, 512, sequence_length_asr),     # Acoustic features (downsampled)
        "f0_curve": torch.randn(1, sequence_length_input),   # F0/pitch curve (original)
        "n": torch.randn(1, sequence_length_input),          # Noise parameters (original)
        "s": torch.randn(1, 128)                             # Voice style embedding
    }
    
    print("\n📝 Sample Input Shapes:")
    for name, tensor in sample_inputs.items():
        print(f"  - {name}: {tensor.shape}")
        
    return sample_inputs

def extract_and_convert_vocoder(model):
    """
    Extract the decoder and convert it to CoreML format.
    
    This is the main conversion process that:
    1. Extracts the decoder module 
    2. Wraps it for CoreML compatibility
    3. Traces with sample inputs
    4. Converts to CoreML with ANE optimization
    
    Args:
        model: The loaded KModel instance
        
    Returns:
        Path to the saved CoreML package
    """
    print("\n🔧 Extracting decoder module...")
    decoder = model.decoder
    
    # Try full decoder first, with fallback to generator-only
    print("🔄 Attempting full decoder conversion...")
    try:
        # Create wrapper for clean I/O
        wrapper = VocoderWrapper(decoder)
        wrapper.eval()
        conversion_mode = "full_decoder"
        print("✅ Full decoder extracted and wrapped")
    except Exception as e:
        print(f"⚠️ Full decoder extraction failed: {e}")
        print("🔄 Falling back to generator-only conversion...")
        try:
            wrapper = SimpleGeneratorWrapper(decoder)
            wrapper.eval()
            conversion_mode = "generator_only"
            print("✅ Generator-only wrapper created")
        except Exception as e2:
            print(f"❌ Both approaches failed: {e2}")
            raise
    
    # Create sample inputs for tracing
    sample_inputs = create_sample_inputs()
    
    # Convert to tuple for tracing (matches forward signature)
    if conversion_mode == "full_decoder":
        trace_inputs = (
            sample_inputs["asr"],
            sample_inputs["f0_curve"], 
            sample_inputs["n"],
            sample_inputs["s"]
        )
    else:  # generator_only
        # Generator expects (x, s, f0_curve) where f0_curve is upsampled
        trace_inputs = (
            sample_inputs["asr"],          # x (processed features)
            sample_inputs["s"],            # s (style)
            sample_inputs["f0_curve"]      # f0_curve 
        )
    
    print("\n⚡ Tracing model with torch.jit.trace...")
    try:
        # Use torch.jit.trace with warnings suppressed
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress tracing warnings
            traced_vocoder = torch.jit.trace(wrapper, trace_inputs, strict=False)
        print("✅ Model traced successfully")
    except Exception as e:
        print(f"❌ Tracing failed: {e}")
        print("This may indicate incompatible operations for CoreML conversion")
        raise
    
    print("\n🍎 Converting to CoreML...")
    
    # Define CoreML input specifications with proper types and shapes
    sequence_length_input = 400  # F0 curve input length
    sequence_length_asr = sequence_length_input // 2  # ASR features length
    
    if conversion_mode == "full_decoder":
        inputs = [
            ct.TensorType(name="asr", shape=(1, 512, sequence_length_asr), dtype=np.float32),
            ct.TensorType(name="f0_curve", shape=(1, sequence_length_input), dtype=np.float32),
            ct.TensorType(name="n", shape=(1, sequence_length_input), dtype=np.float32), 
            ct.TensorType(name="s", shape=(1, 128), dtype=np.float32)
        ]
    else:  # generator_only
        inputs = [
            ct.TensorType(name="x", shape=(1, 512, sequence_length_asr), dtype=np.float32),
            ct.TensorType(name="s", shape=(1, 128), dtype=np.float32),
            ct.TensorType(name="f0_curve", shape=(1, sequence_length_input), dtype=np.float32)
        ]
    
    # Convert with ANE optimization settings
    try:
        coreml_model = ct.convert(
            traced_vocoder,
            inputs=inputs,
            convert_to="mlprogram",
            compute_precision=COMPUTE_PRECISION,
            minimum_deployment_target=MINIMUM_DEPLOYMENT_TARGET,
            compute_units=COMPUTE_UNITS
        )
        print("✅ CoreML conversion successful with ANE optimization")
    except Exception as e:
        print(f"⚠️ ANE conversion failed: {e}")
        print("🔄 Trying fallback conversion with CPU-only...")
        try:
            # Fallback to CPU-only with FP32 precision
            coreml_model = ct.convert(
                traced_vocoder,
                inputs=inputs,
                convert_to="mlprogram",
                compute_precision=ct.precision.FLOAT32,
                minimum_deployment_target=ct.target.iOS15,
                compute_units=ct.ComputeUnit.CPU_ONLY
            )
            print("✅ CoreML conversion successful with CPU fallback")
        except Exception as e2:
            print(f"❌ Both conversion attempts failed:")
            print(f"  - ANE optimized: {e}")
            print(f"  - CPU fallback: {e2}")
            raise
    
    # Add model metadata
    coreml_model.author = "Kokoro TTS - Vocoder Module"
    if conversion_mode == "full_decoder":
        coreml_model.short_description = "Complete iSTFTNet decoder for high-quality audio synthesis on Apple Neural Engine"
    else:
        coreml_model.short_description = "iSTFTNet generator core for high-quality audio synthesis on Apple Neural Engine"
    coreml_model.version = "1.0.0"
    
    # Save the model
    output_path = "KokoroVocoder.mlpackage"
    coreml_model.save(output_path)
    
    print(f"✅ CoreML model saved to: {output_path}")
    
    # Verify the conversion
    print("\n🧪 Verifying CoreML model...")
    try:
        # Load and test the converted model
        loaded_model = ct.models.MLModel(output_path)
        
        # Create test inputs (convert to proper numpy format)
        test_inputs = {
            "asr": sample_inputs["asr"].numpy().astype(np.float32),
            "f0_curve": sample_inputs["f0_curve"].numpy().astype(np.float32),
            "n": sample_inputs["n"].numpy().astype(np.float32), 
            "s": sample_inputs["s"].numpy().astype(np.float32)
        }
        
        # Run prediction
        result = loaded_model.predict(test_inputs)
        print("✅ CoreML model verification successful")
        
        # Print output info
        for key, value in result.items():
            if hasattr(value, 'shape'):
                print(f"  Output {key}: {value.shape}")
            
    except Exception as e:
        print(f"⚠️  Verification failed: {e}")
        print("Model was saved but may have issues")
    
    return output_path

def main():
    """
    Main execution function for vocoder extraction and conversion.
    """
    print("🚀 Kokoro Vocoder Extraction & CoreML Conversion")
    print("=" * 50)
    
    print("\n📦 Loading full Kokoro model...")
    try:
        # Load the model exactly as the demo app does
        model = KModel().to('cpu').eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Inspect the model structure to understand the decoder
    decoder = inspect_model_structure(model)
    
    # Extract and convert the vocoder
    try:
        output_path = extract_and_convert_vocoder(model)
        
        print(f"\n🎉 Conversion Complete!")
        print(f"📁 CoreML vocoder saved to: {output_path}")
        print("\nNext steps:")
        print("1. Test the vocoder with test_ane_pipeline.py")
        print("2. Verify ANE usage with Instruments or powermetrics") 
        print("3. Compare performance vs CPU-only pipeline")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()