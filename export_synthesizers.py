#!/usr/bin/env python3
"""
Exports the Synthesizer models using a bucketing strategy.

This is a standalone script that contains all necessary code to avoid
import issues and environment conflicts. It assumes the .pth checkpoint
and config.json are present in the 'checkpoints' directory.
"""
import argparse
import os
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from safetensors.torch import load_file
from collections import OrderedDict
import time

# --- Model Imports ---
# These are brought in from the kokoro package to make the script self-contained.

from kokoro.model import KModel
from kokoro.modules import LayerNorm, AdaLayerNorm, LinearNorm, AdainResBlk1d

# --- CoreML-Friendly Model Components ---

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
    """Second-stage model: Synthesizes audio from intermediate features."""
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel
        self.kmodel.text_encoder = CoreMLFriendlyTextEncoder(kmodel.text_encoder)

    def forward(self, d: torch.FloatTensor, t_en: torch.FloatTensor, s: torch.FloatTensor, ref_s: torch.FloatTensor, pred_aln_trg: torch.FloatTensor):
        print("[DEBUG] SynthesizerModel.forward: starting")
        k = self.kmodel
        print("[DEBUG] SynthesizerModel.forward: calculating 'en'")
        en = d.transpose(-1, -2) @ pred_aln_trg
        print("[DEBUG] SynthesizerModel.forward: calculating 'F0_pred, N_pred'")
        F0_pred, N_pred = k.predictor.F0Ntrain(en, s)
        print("[DEBUG] SynthesizerModel.forward: calculating 'asr'")
        asr = t_en @ pred_aln_trg
        print("[DEBUG] SynthesizerModel.forward: calling k.decoder")
        audio = k.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze(0)
        print("[DEBUG] SynthesizerModel.forward: k.decoder finished")
        return audio

# --- Main Export Logic ---

def prepare_pytorch_models(config_path, checkpoint_path):
    """Loads the KModel from the specified checkpoint."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    return KModel(config=config_path, model=checkpoint_path, disable_complex=True)

def export_synthesizers(output_dir, buckets_str):
    """Exports the synthesizer models for the specified buckets."""
    config_path = "checkpoints/config.json"
    checkpoint_path = "checkpoints/kokoro-v1_0.pth"
    
    print("--- Loading Model ---")
    kmodel = prepare_pytorch_models(config_path, checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Preparing Intermediate Features ---")
    duration_model = DurationModel(kmodel).eval()
    
    trace_length = 256
    input_ids = torch.randint(0, 100, (1, trace_length), dtype=torch.int32)
    ref_s = torch.randn(1, 256, dtype=torch.float32)
    speed = torch.tensor([1.0], dtype=torch.float32)
    attention_mask = torch.ones(1, trace_length, dtype=torch.int32)
    
    with torch.no_grad():
        _, d, t_en, s, ref_s_out = duration_model(input_ids, ref_s, speed, attention_mask)
    
    # Define buckets
    # e.g., "3s,5s,10s"
    bucket_seconds = [int(b.replace('s','')) for b in buckets_str.split(',')]
    buckets = {f"{sec}s": sec * 24000 for sec in bucket_seconds}

    synthesizer_model_base = SynthesizerModel(kmodel).half().eval()

    for name, frame_count in buckets.items():
        print(f"\n--- Exporting Synthesizer for Bucket: {name} ({frame_count} frames) ---")
        synthesizer_file = os.path.join(output_dir, f"kokoro_synthesizer_{name}.mlpackage")

        pred_aln_trg = torch.zeros((trace_length, frame_count), dtype=torch.float16)
        
        print(f"[{time.ctime()}] Tracing model...")
        with torch.no_grad():
            traced_synthesizer = torch.jit.trace(
                synthesizer_model_base, 
                (d.half(), t_en.half(), s.half(), ref_s_out.half(), pred_aln_trg)
            )
        print(f"[{time.ctime()}] Model tracing complete.")
        
        d_shape = (1, kmodel.bert.config.hidden_size, trace_length)
        t_en_shape = (1, kmodel.bert.config.hidden_size, trace_length)
        s_shape = (1, 128)
        ref_s_shape = (1, 256)
        pred_aln_trg_shape = (trace_length, frame_count)
        
        print(f"[{time.ctime()}] Converting to Core ML...")
        ml_synthesizer = ct.convert(
            traced_synthesizer,
            inputs=[
                ct.TensorType(name="d", shape=d_shape, dtype=np.float16),
                ct.TensorType(name="t_en", shape=t_en_shape, dtype=np.float16),
                ct.TensorType(name="s", shape=s_shape, dtype=np.float16),
                ct.TensorType(name="ref_s", shape=ref_s_shape, dtype=np.float16),
                ct.TensorType(name="pred_aln_trg", shape=pred_aln_trg_shape, dtype=np.float16)
            ],
            outputs=[ct.TensorType(name="waveform")],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS15,
            compute_precision=ct.precision.FLOAT16
        )
        print(f"[{time.ctime()}] Core ML conversion complete.")
        
        ml_synthesizer.save(synthesizer_file)
        print(f"✅ Saved Synthesizer Model ({name}) to: {synthesizer_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Kokoro Synthesizer to CoreML with bucketing.")
    parser.add_argument("--output_dir", "-o", type=str, default="coreml", help="Output directory for mlpackage files.")
    parser.add_argument("--buckets", type=str, default="3s", help="Comma-separated list of bucket sizes in seconds (e.g., '3s,5s,10s').")
    args = parser.parse_args()

    try:
        export_synthesizers(args.output_dir, args.buckets)
        print("\n\n🎉 Synthesizer export complete. You're ready to ship.")
    except Exception as e:
        print(f"\n❌ An error occurred during export: {e}")
        import traceback
        traceback.print_exc()