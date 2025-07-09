import argparse
import os
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from safetensors.torch import load_file
from collections import OrderedDict

from kokoro.model import KModel

# --- CoreML-Friendly Model Components ---
# These are rewritten versions of the modules in kokoro/modules.py
# that avoid operations incompatible with torch.jit.trace.

from kokoro.modules import LayerNorm, AdaLayerNorm, LinearNorm, AdainResBlk1d

class CoreMLFriendlyTextEncoder(nn.Module):
    """Replaces the original TextEncoder to avoid pack_padded_sequence."""
    def __init__(self, original_encoder):
        super().__init__()
        self.embedding = original_encoder.embedding
        self.cnn = original_encoder.cnn
        self.lstm = original_encoder.lstm

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x.transpose(-1, -2)
        x.masked_fill_(m, 0.0)
        return x

class CoreMLFriendlyDurationEncoder(nn.Module):
    """Replaces the original DurationEncoder to avoid pack_padded_sequence."""
    def __init__(self, original_encoder):
        super().__init__()
        self.lstms = original_encoder.lstms
        self.dropout = original_encoder.dropout

    def forward(self, x, style, text_lengths, m):
        masks = m
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                block.flatten_parameters()
                x, _ = block(x)
                x = nn.functional.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
        return x.transpose(-1, -2)

# --- Model Wrappers for Two-Stage Conversion ---

class DurationModel(nn.Module):
    """First-stage model: Predicts durations and extracts intermediate features."""
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel
        self.kmodel.text_encoder = CoreMLFriendlyTextEncoder(kmodel.text_encoder)
        self.kmodel.predictor.text_encoder = CoreMLFriendlyDurationEncoder(kmodel.predictor.text_encoder)
        if hasattr(self.kmodel.bert.embeddings, 'token_type_ids'):
             delattr(self.kmodel.bert.embeddings, 'token_type_ids')

    def forward(self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: torch.FloatTensor, attention_mask: torch.LongTensor):
        k = self.kmodel
        input_lengths = attention_mask.sum(dim=-1).to(torch.long)
        text_mask = attention_mask == 0
        token_type_ids = torch.zeros_like(input_ids)
        
        bert_dur = k.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        d_en = k.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        
        d = k.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = k.predictor.lstm(d)
        duration = k.predictor.duration_proj(x)
        
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long()
        
        t_en = k.text_encoder(input_ids, input_lengths, text_mask)
        return pred_dur, d, t_en, s, ref_s

class SynthesizerModel(nn.Module):
    """Second-stage model: Synthesizes audio from intermediate features and a fixed-size alignment matrix."""
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel
        self.kmodel.text_encoder = CoreMLFriendlyTextEncoder(kmodel.text_encoder)

    def forward(self, d: torch.FloatTensor, t_en: torch.FloatTensor, s: torch.FloatTensor, ref_s: torch.FloatTensor, pred_aln_trg: torch.FloatTensor):
        k = self.kmodel
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = k.predictor.F0Ntrain(en, s)
        asr = t_en @ pred_aln_trg
        audio = k.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze(0)
        return audio

# --- Main Export Logic ---

def prepare_pytorch_models(config_path, checkpoint_path):
    """Ensures PyTorch models are available, converting from safetensors if needed."""
    if not os.path.exists(checkpoint_path):
        print("PyTorch checkpoint not found. Attempting to convert from safetensors...")
        mlx_resources = "kokoro-mlx-swift/kokoro-ios/mlxtest/mlxtest/Resources"
        safetensors_path = os.path.join(mlx_resources, "kokoro-v1_0.safetensors")
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(f"Cannot find {safetensors_path}.")
        
        state_dict = load_file(safetensors_path)
        organized_dict = OrderedDict((k, OrderedDict()) for k in ['bert', 'bert_encoder', 'predictor', 'text_encoder', 'decoder'])
        for key, value in state_dict.items():
            module_name = key.split('.')[0]
            if module_name in organized_dict:
                organized_dict[module_name][key[len(module_name)+1:]] = value
        
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(organized_dict, checkpoint_path)
        print(f"Saved PyTorch checkpoint to {checkpoint_path}")

    return KModel(config=config_path, model=checkpoint_path, disable_complex=True)

def export_models(kmodel, output_dir):
    """Exports the two-stage model to Core ML using a bucketing strategy."""
    
    # --- 1. Export the (dynamic) DurationModel ---
    print("\n--- Exporting Duration Model ---")
    duration_model = DurationModel(kmodel).eval()
    duration_file = os.path.join(output_dir, "kokoro_duration.mlpackage")
    
    trace_length = 256
    input_ids = torch.randint(0, 100, (1, trace_length), dtype=torch.int32)
    ref_s = torch.randn(1, 256, dtype=torch.float32)
    speed = torch.tensor([1.0], dtype=torch.float32)
    attention_mask = torch.ones(1, trace_length, dtype=torch.int32)
    
    with torch.no_grad():
        traced_duration_model = torch.jit.trace(duration_model, (input_ids, ref_s, speed, attention_mask))

    ml_duration_model = ct.convert(
        traced_duration_model,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, 512)), dtype=np.int32),
            ct.TensorType(name="ref_s", shape=(1, 256), dtype=np.float32),
            ct.TensorType(name="speed", shape=(1,), dtype=np.float32),
            ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(1, 512)), dtype=np.int32)
        ],
        outputs=[ct.TensorType(name="pred_dur"), ct.TensorType(name="d"), ct.TensorType(name="t_en"), ct.TensorType(name="s"), ct.TensorType(name="ref_s_out")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15
    )
    ml_duration_model.save(duration_file)
    print(f"✅ Saved Duration Model to: {duration_file}")

    # --- 2. Export multiple (fixed-size) SynthesizerModels ---
    print("\n--- Exporting Synthesizer Models (Bucketing) ---")
    
    with torch.no_grad():
        _, d, t_en, s, ref_s_out = duration_model(input_ids, ref_s, speed, attention_mask)
    
    buckets = {
        "3s": 3 * 24000,
        "5s": 5 * 24000,
        "10s": 10 * 24000,
        "30s": 30 * 24000
    }

    synthesizer_model_base = SynthesizerModel(kmodel).eval()

    for name, frame_count in buckets.items():
        print(f"Exporting synthesizer for bucket: {name} ({frame_count} frames)")
        synthesizer_file = os.path.join(output_dir, f"kokoro_synthesizer_{name}.mlpackage")

        pred_aln_trg = torch.zeros((trace_length, frame_count), dtype=torch.float32)

        with torch.no_grad():
            traced_synthesizer_model = torch.jit.trace(synthesizer_model_base, (d, t_en, s, ref_s_out, pred_aln_trg))

        d_shape = (1, kmodel.bert.config.hidden_size, trace_length)
        t_en_shape = (1, kmodel.bert.config.hidden_size, trace_length)
        s_shape = (1, 128)
        ref_s_shape = (1, 256)
        pred_aln_trg_shape = (trace_length, frame_count)
        
        ml_synthesizer_model = ct.convert(
            traced_synthesizer_model,
            inputs=[
                ct.TensorType(name="d", shape=d_shape),
                ct.TensorType(name="t_en", shape=t_en_shape),
                ct.TensorType(name="s", shape=s_shape),
                ct.TensorType(name="ref_s", shape=ref_s_shape),
                ct.TensorType(name="pred_aln_trg", shape=pred_aln_trg_shape)
            ],
            outputs=[ct.TensorType(name="waveform")],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS15
        )
        ml_synthesizer_model.save(synthesizer_file)
        print(f"✅ Saved Synthesizer Model to: {synthesizer_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export Kokoro Model to CoreML", add_help=True)
    parser.add_argument("--output_dir", "-o", type=str, default="coreml", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config_path = "checkpoints/config.json"
    checkpoint_path = "checkpoints/kokoro-v1_0.pth"
    
    kmodel = prepare_pytorch_models(config_path, checkpoint_path)
    export_models(kmodel, args.output_dir)
    print("\n\n🎉 Export complete. You're ready to ship.")