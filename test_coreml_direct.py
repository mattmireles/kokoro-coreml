#!/usr/bin/env python3
"""Test CoreML model directly without full pipeline dependencies"""
import coremltools as ct
import numpy as np

def test_coreml_model():
    print("🧪 Testing CoreML model directly...")

    # Load the CoreML model
    model_path = "coreml/kokoro_duration.mlpackage"
    print(f"Loading model from: {model_path}")

    try:
        model = ct.models.MLModel(model_path)
        print("✅ CoreML model loaded successfully!")

        # Print model info
        print(f"\n📋 Model Info:")
        print(f"- Author: {model.author}")
        print(f"- Description: {model.short_description}")
        print(f"- Version: {model.version}")

        # Print input specifications
        print(f"\n🔤 Input Specs:")
        for input_spec in model.get_spec().description.input:
            print(f"- {input_spec.name}: {input_spec.type}")

        # Print output specifications
        print(f"\n📤 Output Specs:")
        for output_spec in model.get_spec().description.output:
            print(f"- {output_spec.name}: {output_spec.type}")

        # Test with dummy inputs
        print(f"\n🧪 Testing with dummy inputs...")

        # Create dummy inputs that match the model specs
        test_inputs = {
            "input_ids": np.random.randint(0, 100, (1, 128)).astype(np.int32),
            "ref_s": np.random.randn(1, 256).astype(np.float32),
            "speed": np.array([1.0]).astype(np.float32),
            "attention_mask": np.ones((1, 128)).astype(np.int32)
        }

        # Run prediction
        result = model.predict(test_inputs)
        print("✅ Model prediction successful!")

        # Print output shapes
        print(f"\n📊 Output shapes:")
        for key, value in result.items():
            if hasattr(value, 'shape'):
                print(f"- {key}: {value.shape}")
            else:
                print(f"- {key}: {type(value)}")

        print(f"\n🎉 CoreML model test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error testing CoreML model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_coreml_model()