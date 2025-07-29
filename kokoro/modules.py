    # Neural Network Components for Kokoro TTS Architecture
    #
    # This module implements the core neural network building blocks used throughout
    # the Kokoro TTS model. Components are designed for high-quality speech synthesis
    # with careful attention to CoreML/ONNX export compatibility.
    #
    # Architecture Components:
    # - TextEncoder: Bidirectional LSTM-based phoneme sequence encoder
    # - ProsodyPredictor: Duration, F0, and noise prediction network
    # - CustomAlbert: Simplified BERT variant for phoneme context encoding
    # - AdaLayerNorm: Adaptive layer normalization with style conditioning
    # - DurationEncoder: Multi-layer LSTM encoder for prosody modeling
    #
    # Design Philosophy:
    # - Modular components for flexible architecture composition
    # - Style-conditioned layers for voice adaptation
    # - Dropout layers for training robustness (removed during CoreML export)
    # - Pack/unpack operations optimized for variable-length sequences
    #
    # Cross-file dependencies:
    # - Used by: model.py (KModel architecture), export scripts
    # - Imports from: istftnet.py (AdainResBlk1d for style conditioning)
    # - Based on: StyleTTS2 architecture with Kokoro-specific modifications
    #
    # Export Considerations:
    # - Pack_padded_sequence operations require CoreML-friendly replacements
    # - Dropout layers must be converted to Identity for inference
    # - Variable-length sequences handled via masking instead of packing

# Based on StyleTTS2: https://github.com/yl4579/StyleTTS2/blob/main/models.py
from .istftnet import AdainResBlk1d
from torch.nn.utils.parametrizations import weight_norm
from transformers import AlbertModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNorm(nn.Module):
    # Xavier-initialized linear layer with configurable activation gain.
    #
    # This wrapper around nn.Linear provides automatic Xavier uniform weight
    # initialization, which is crucial for stable training in deep TTS architectures.
    # The initialization gain can be adjusted based on the subsequent activation function.
    #
    # Parameters:
    # - in_dim: Input feature dimension
    # - out_dim: Output feature dimension  
    # - bias: Enable bias term (default: True)
    # - w_init_gain: Xavier initialization gain ('linear', 'relu', 'tanh', etc.)
    #
    # Used by:
    # - ProsodyPredictor: Duration projection layers
    # - Various model components requiring properly initialized linear layers
    #
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
    # Standard linear transformation: y = xW^T + b
        return self.linear_layer(x)


class LayerNorm(nn.Module):
    # Custom layer normalization with channel-wise parameters.
    #
    # This implementation provides layer normalization specifically designed
    # for sequence data with channel dimension handling. It differs from
    # standard PyTorch LayerNorm by operating on specific tensor layouts.
    #
    # Tensor Layout:
    # - Input: (batch, channels, sequence_length)
    # - Normalization: Applied across channels dimension
    # - Output: Same shape as input
    #
    # Parameters:
    # - channels: Number of channels to normalize across
    # - eps: Numerical stability epsilon (default: 1e-5)
    #
    # Learnable Parameters:
    # - gamma: Scale parameter, initialized to ones
    # - beta: Shift parameter, initialized to zeros
    #
    # Used by:
    # - TextEncoder: CNN layer normalization in phoneme processing
    # - Various sequence modeling components
    #
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
    # Apply layer normalization with proper tensor dimension handling.
    #
    # Process:
    # 1. Transpose to move channels to last dimension
    # 2. Apply PyTorch layer_norm function
    # 3. Transpose back to original layout
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    # Bidirectional LSTM-based encoder for phoneme sequence processing.
    #
    # This encoder transforms tokenized phoneme sequences into rich contextual
    # representations suitable for prosody prediction and audio synthesis.
    # It combines convolutional preprocessing with bidirectional LSTM encoding.
    #
    # Architecture Pipeline:
    # 1. Embedding: Phoneme tokens -> dense vectors
    # 2. CNN Stack: Multiple 1D convolutions with normalization and dropout
    # 3. LSTM: Bidirectional processing for temporal context
    # 4. Masking: Padding-aware processing throughout
    #
    # Parameters:
    # - channels: Hidden dimension throughout the encoder
    # - kernel_size: Convolution kernel size for local pattern detection
    # - depth: Number of convolutional layers in the stack
    # - n_symbols: Vocabulary size for phoneme embedding
    # - actv: Activation function (default: LeakyReLU(0.2))
    #
    # Key Features:
    # - Weight normalization for stable training
    # - Dropout for regularization (0.2 rate)
    # - Bidirectional LSTM for full sequence context
    # - Pack/unpack operations for efficient variable-length processing
    #
    # Used by:
    # - model.py: KModel.forward_with_tokens() for text feature extraction
    # - export_coreml.py: Requires CoreML-friendly replacement (CoreMLFriendlyTextEncoder)
    #
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
    # Forward pass with masking and variable-length sequence handling.
    #
    # This method processes phoneme sequences through the complete encoder
    # pipeline while properly handling variable-length inputs via masking
    # and pack/unpack operations.
    #
    # Parameters:
    # - x: Phoneme token IDs, shape (batch, max_length)
    # - input_lengths: Actual sequence lengths, shape (batch,)
    # - m: Padding mask, shape (batch, max_length), True for padding
    #
    # Returns:
    # - torch.Tensor: Encoded features, shape (batch, channels, max_length)
    #
    # Processing Steps:
    # 1. Embed phoneme tokens to dense vectors
    # 2. Apply CNN layers with masking between each layer
    # 3. Pack sequences for efficient LSTM processing
    # 4. Bidirectional LSTM encoding
    # 5. Unpack and reshape for final output
    # 6. Zero-pad to match original sequence length
    #
    # CoreML Compatibility Issues:
    # - pack_padded_sequence not supported in CoreML export
    # - Requires replacement with CoreMLFriendlyTextEncoder for export
    #
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)  # [B, T, chn]
        lengths = input_lengths if input_lengths.device == torch.device('cpu') else input_lengths.to('cpu')
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad
        x.masked_fill_(m, 0.0)
        return x


class AdaLayerNorm(nn.Module):
    # Adaptive Layer Normalization with style conditioning.
    #
    # This module implements style-conditioned normalization that adapts
    # the normalization parameters based on input style vectors. This allows
    # the model to adjust its internal representations based on voice characteristics.
    #
    # Mathematical Operation:
    # 1. Standard layer normalization: x_norm = LayerNorm(x)
    # 2. Style-dependent parameters: gamma, beta = Linear(style)
    # 3. Adaptive scaling: output = (1 + gamma) * x_norm + beta
    #
    # Parameters:
    # - style_dim: Dimension of input style vector
    # - channels: Number of channels to normalize
    # - eps: Numerical stability epsilon
    #
    # Style Vector Structure:
    # - Typically 128-dimensional voice embedding
    # - Projected to 2*channels for gamma and beta parameters
    # - Enables voice-specific feature adaptation
    #
    # Used by:
    # - DurationEncoder: Style-conditioned duration prediction
    # - ProsodyPredictor: Voice-adaptive prosody modeling
    #
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels*2)

    def forward(self, x, s):
    # Apply adaptive layer normalization with style conditioning.
    #
    # Parameters:
    # - x: Input features, shape (batch, channels, sequence)
    # - s: Style vector, shape (batch, style_dim)
    #
    # Returns:
    # - torch.Tensor: Style-adapted features, same shape as input
    #
    # Tensor Manipulation:
    # - Multiple transposes to align dimensions for layer_norm
    # - gamma/beta reshaping for proper broadcasting
    # - Final transpose back to original layout
    #
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class ProsodyPredictor(nn.Module):
    # Multi-branch prosody prediction network for speech synthesis.
    #
    # This module predicts three critical prosodic features for high-quality
    # speech synthesis: phoneme durations, fundamental frequency (F0/pitch),
    # and noise characteristics. Each branch is style-conditioned for voice adaptation.
    #
    # Architecture Branches:
    # 1. Duration Branch: Predicts per-phoneme durations for alignment
    # 2. F0 Branch: Predicts pitch contours for natural prosody
    # 3. Noise Branch: Predicts spectral noise characteristics
    #
    # Style Conditioning:
    # - All branches use AdainResBlk1d for voice-specific adaptation
    # - Shared LSTM processes style-concatenated features
    # - Voice embedding influences all prosodic predictions
    #
    # Parameters:
    # - style_dim: Voice embedding dimension (typically 128)
    # - d_hid: Hidden dimension throughout the network
    # - nlayers: Number of layers in duration encoder
    # - max_dur: Maximum duration prediction per phoneme
    # - dropout: Regularization rate for training
    #
    # Network Flow:
    # 1. Text features + style -> DurationEncoder -> duration predictions
    # 2. Encoded features -> shared LSTM -> F0 and noise branches
    # 3. Style-conditioned AdainResBlk1d processing in each branch
    # 4. Final 1x1 convolutions to single-channel outputs
    #
    # Used by:
    # - model.py: KModel.forward_with_tokens() for prosody prediction
    # - Central component in the TTS synthesis pipeline
    #
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()
        self.text_encoder = DurationEncoder(sty_dim=style_dim, d_model=d_hid,nlayers=nlayers, dropout=dropout)
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        m = m.unsqueeze(1)
        lengths = text_lengths if text_lengths.device == torch.device('cpu') else text_lengths.to('cpu')
        x = nn.utils.rnn.pack_padded_sequence(d, lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]], device=x.device)
        x_pad[:, :x.shape[1], :] = x
        x = x_pad
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=False))
        en = (d.transpose(-1, -2) @ alignment)
        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
    # Predict F0 (pitch) and noise characteristics from aligned features.
    #
    # This method processes duration-aligned features through parallel
    # F0 and noise prediction branches, each conditioned on voice style.
    # The shared LSTM provides common feature processing before branching.
    #
    # Parameters:
    # - x: Duration-aligned features, shape (batch, sequence, hidden)
    # - s: Style vector for voice conditioning, shape (batch, style_dim)
    #
    # Returns:
    # - F0: Fundamental frequency predictions, shape (batch, sequence)
    # - N: Noise characteristics, shape (batch, sequence)
    #
    # Processing Pipeline:
    # 1. Shared LSTM: Common feature extraction from aligned inputs
    # 2. F0 Branch: Style-conditioned residual blocks -> pitch prediction
    # 3. Noise Branch: Parallel processing -> spectral noise modeling
    # 4. Final projections: 1x1 convolutions to single-channel outputs
    #
    # Style Conditioning:
    # - Each AdainResBlk1d adapts to voice characteristics
    # - Enables voice-specific prosody and spectral properties
    # - Critical for voice cloning and style transfer
    #
    # Used by:
    # - model.py: KModel.forward_with_tokens() for prosody synthesis
    # - export_synthesizers.py: Manual F0/N prediction in SynthesizerModel
    #
        x, _ = self.shared(x.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)
        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        return F0.squeeze(1), N.squeeze(1)


class DurationEncoder(nn.Module):
    # Multi-layer LSTM encoder with adaptive normalization for duration modeling.
    #
    # This encoder specializes in processing phoneme sequences for duration prediction,
    # using alternating LSTM and adaptive layer normalization layers. The style
    # conditioning allows for voice-specific duration patterns.
    #
    # Architecture Pattern:
    # - LSTM Layer: Bidirectional processing with style concatenation
    # - AdaLayerNorm: Style-conditioned normalization
    # - Repeat pattern for nlayers iterations
    #
    # Style Integration:
    # - Style vector concatenated to input at each LSTM layer
    # - AdaLayerNorm provides style-specific normalization parameters
    # - Enables voice-specific speaking rate and rhythm patterns
    #
    # Parameters:
    # - sty_dim: Style vector dimension
    # - d_model: Hidden dimension throughout the encoder
    # - nlayers: Number of LSTM-Norm layer pairs
    # - dropout: Regularization rate applied after LSTM layers
    #
    # Used by:
    # - ProsodyPredictor: Duration prediction component
    # - export_coreml.py: Requires CoreML-friendly replacement
    #
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim, d_model // 2, num_layers=1, batch_first=True, bidirectional=True))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

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
                lengths = text_lengths if text_lengths.device == torch.device('cpu') else text_lengths.to('cpu')
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad

        return x.transpose(-1, -2)


class CustomAlbert(AlbertModel):
    # Simplified ALBERT wrapper for phoneme context encoding.
    #
    # This class wraps the Hugging Face ALBERT model to provide a clean
    # interface for phoneme sequence encoding. It extracts only the final
    # hidden states, discarding other BERT outputs not needed for TTS.
    #
    # ALBERT Architecture Benefits:
    # - Parameter sharing across layers (memory efficient)
    # - Factorized embeddings (reduced parameter count)
    # - Cross-layer parameter sharing (better generalization)
    #
    # Customization:
    # - Returns only last_hidden_state (not attention weights, etc.)
    # - Simplified interface for TTS-specific usage
    # - Compatible with standard ALBERT checkpoints
    #
    # Input Processing:
    # - Accepts standard BERT inputs (input_ids, attention_mask, token_type_ids)
    # - Processes phoneme sequences as if they were text tokens
    # - Provides rich contextual representations for prosody prediction
    #
    # Used by:
    # - model.py: KModel phoneme encoding with self.bert
    # - Initialized with AlbertConfig from model configuration
    #
    # Based on: StyleTTS2 PLBERT utilities
    # Source: https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py
    #
    def forward(self, *args, **kwargs):
    # Forward pass returning only the final hidden states.
    #
    # Parameters:
    # - Standard ALBERT inputs (input_ids, attention_mask, etc.)
    #
    # Returns:
    # - torch.Tensor: Final hidden states, shape (batch, sequence, hidden)
    #
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state
