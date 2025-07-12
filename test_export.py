#!/usr/bin/env python3
"""Test basic functionality"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Python path:", sys.path[0])
print("Current directory:", os.getcwd())

try:
    print("\n1. Testing imports...")
    from kokoro.model import KModel
    print("✓ KModel imported")
    
    print("\n2. Checking model files...")
    config_path = "checkpoints/config.json" 
    print(f"Config exists: {os.path.exists(config_path)}")
    
    checkpoint_path = "checkpoints/kokoro-v1_0.pth"
    print(f"Checkpoint exists: {os.path.exists(checkpoint_path)}")
    
    print("\n3. Testing CoreML import...")
    import coremltools as ct
    print(f"✓ CoreML tools version: {ct.__version__}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()