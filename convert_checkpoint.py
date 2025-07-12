#!/usr/bin/env python3
"""Convert safetensors to PyTorch checkpoint"""
import os
import torch
from safetensors.torch import load_file
from collections import OrderedDict

mlx_resources = "/Users/mattmireles/Documents/GitHub/kokoro-mlx-swift/kokoro-ios/mlxtest/mlxtest/Resources"
safetensors_path = os.path.join(mlx_resources, "kokoro-v1_0.safetensors")
checkpoint_path = "checkpoints/kokoro-v1_0.pth"

print(f"Loading safetensors from: {safetensors_path}")
state_dict = load_file(safetensors_path)

print(f"Found {len(state_dict)} parameters")

# Organize by module
organized_dict = OrderedDict((k, OrderedDict()) for k in ['bert', 'bert_encoder', 'predictor', 'text_encoder', 'decoder'])
for key, value in state_dict.items():
    module_name = key.split('.')[0]
    if module_name in organized_dict:
        organized_dict[module_name][key[len(module_name)+1:]] = value

os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
torch.save(organized_dict, checkpoint_path)
print(f"✅ Saved PyTorch checkpoint to {checkpoint_path}")