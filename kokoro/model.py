# Core Model Implementation for Kokoro TTS
#
# This module contains the main KModel class that wraps the Kokoro text-to-speech
# neural network architecture. The model is designed for efficient inference and
# CoreML conversion, with particular attention to Apple Neural Engine (ANE) optimization.
#
# Key Components:
# - KModel: Main inference wrapper with phoneme-to-audio generation
# - KModelForONNX: Export-friendly wrapper for ONNX/CoreML conversion
# - Output: Structured dataclass for model outputs with audio and duration predictions
#
# Cross-file dependencies:
# - Called by: pipeline.py (KPipeline.infer), export_coreml.py, export_synthesizers.py
# - Imports from: modules.py (neural components), istftnet.py (decoder)
# - Used by: All inference scripts, demo applications, and export pipelines

from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from transformers import AlbertConfig
from typing import Dict, Optional, Union
import json
import torch

class KModel(torch.nn.Module):
    # Primary neural network model for Kokoro text-to-speech synthesis.
    #
    # KModel serves as the central inference engine that converts phoneme sequences
    # into high-quality audio waveforms. It combines multiple neural components:
    # - BERT-based text encoder for phoneme understanding
    # - Prosody predictor for duration and F0/pitch modeling
    # - iSTFT-based decoder for final audio synthesis
    #
    # Architecture Overview:
    # 1. Phonemes -> BERT embeddings -> prosody prediction -> duration alignment
    # 2. Text features + voice style -> F0/pitch and noise predictions
    # 3. Aligned features -> decoder -> final audio waveform
    #
    # Key Design Decisions:
    # - Language-agnostic: operates on phonemes, not raw text
    # - Voice-conditioned: uses reference voice embeddings for style transfer
    # - CoreML-friendly: designed to be traceable with torch.jit.trace
    # - Memory-efficient: single instance can serve multiple pipelines
    #
    # Called by:
    # - pipeline.py: KPipeline.infer() for production inference
    # - export_coreml.py: DurationModel and SynthesizerModel wrappers
    # - export_synthesizers.py: Two-stage export with bucketing
    # - demo/app.py: Interactive Gradio interface
    #
    # Performance Notes:
    # - Supports variable sequence lengths up to context_length (512 tokens)
    # - Optimized for 24-kHz audio output with 600-frame hop length
    # - Uses disable_complex=True for CoreML compatibility
    #

    # Model repository mappings for Hugging Face Hub downloads.
    #
    # These constants define the mapping between repository IDs and their
    # corresponding checkpoint filenames. Used by __init__ to automatically
    # download model weights when not provided locally.
    #
    # Supported Models:
    # - hexgrad/Kokoro-82M: Original English model (82M parameters)
    # - hexgrad/Kokoro-82M-v1.1-zh: Chinese-enhanced version
    #
    # File Format: PyTorch state_dict organized by module name
    # (bert, bert_encoder, predictor, text_encoder, decoder)
    MODEL_NAMES = {
        'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
        'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
    }

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[Dict, str, None] = None,
        model: Optional[str] = None,
        disable_complex: bool = False
    ):
    # Initialize KModel with automatic weight and configuration loading.
    #
    # This constructor handles the complete model setup process:
    # 1. Downloads config.json and model weights from Hugging Face if needed
    # 2. Initializes all neural network components (BERT, predictor, decoder)
    # 3. Loads pre-trained weights with fallback handling for version mismatches
    #
    # Parameters:
    # - repo_id: Hugging Face repository ID (defaults to 'hexgrad/Kokoro-82M')
    # - config: Model configuration (dict, file path, or None for auto-download)
    # - model: Path to PyTorch checkpoint (None for auto-download)
    # - disable_complex: Use CustomSTFT instead of torch.stft for CoreML export
    #
    # Configuration Keys:
    # - vocab: Phoneme-to-ID mapping dictionary
    # - n_token: Vocabulary size for embedding layers
    # - hidden_dim: Hidden dimension for text and prosody encoders
    # - style_dim: Voice embedding dimension (128 baseline + 128 style)
    # - max_dur: Maximum duration prediction per phoneme
    # - istftnet: Decoder architecture parameters
    #
    # Called by:
    # - pipeline.py: KPipeline.__init__ with device placement
    # - export_coreml.py: prepare_pytorch_models() for conversion
    # - demo/app.py: Global model initialization
    #
        super().__init__()
        if repo_id is None:
            repo_id = 'hexgrad/Kokoro-82M'
            print(f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning.")
        self.repo_id = repo_id
        if not isinstance(config, dict):
            if not config:
                logger.debug("No config provided, downloading from HF")
                config = hf_hub_download(repo_id=repo_id, filename='config.json')
            with open(config, 'r', encoding='utf-8') as r:
                config = json.load(r)
                logger.debug(f"Loaded config: {config}")
        self.vocab = config['vocab']
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], max_dur=config['max_dur'], dropout=config['dropout']
        )
        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'], kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'], n_symbols=config['n_token']
        )
        self.decoder = Decoder(
            dim_in=config['hidden_dim'], style_dim=config['style_dim'],
            dim_out=config['n_mels'], disable_complex=disable_complex, **config['istftnet']
        )
        if not model:
            model = hf_hub_download(repo_id=repo_id, filename=KModel.MODEL_NAMES[repo_id])
        for key, state_dict in torch.load(model, map_location='cpu').items():
            assert hasattr(self, key), key
            try:
                getattr(self, key).load_state_dict(state_dict)
            except:
                logger.debug(f"Did not load {key} from state_dict")
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(state_dict, strict=False)

    @property
    def device(self):
    # Returns the device (CPU/CUDA/MPS) where the model is currently located.
    #
    # This property provides a convenient way to check model placement without
    # inspecting individual parameter tensors. Uses the BERT module as the
    # canonical device reference since it's always present.
    #
    # Returns: torch.device object (cuda:0, cpu, or mps)
    #
    # Used by:
    # - pipeline.py: Device placement for input tensors and voice embeddings
    # - export scripts: Ensuring model and data are on the same device
    #
        return self.bert.device

    @dataclass
    class Output:
    # Structured output container for KModel inference results.
    #
    # This dataclass encapsulates the two primary outputs from model inference:
    # audio waveform and predicted phoneme durations. Used when return_output=True
    # in the forward() method to provide detailed inference information.
    #
    # Fields:
    # - audio: Generated waveform tensor, shape (T,) at 24-kHz sample rate
    # - pred_dur: Per-phoneme duration in frames (40-fps), shape (N,) where N=num_phonemes
    #
    # Duration Interpretation:
    # - Each duration value represents frames at 40-fps (600 samples at 24-kHz)
    # - Used by pipeline.py for word-level timestamp alignment
    # - Essential for lip-syncing and precise timing applications
    #
    # Used by:
    # - pipeline.py: KPipeline.generate_from_tokens() for timestamp calculation
    # - demo applications: Detailed output analysis and debugging
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
    # Core inference method operating directly on tokenized input.
    #
    # This is the primary computational pathway that implements the full
    # text-to-speech synthesis pipeline. It processes pre-tokenized phoneme
    # sequences through the complete neural architecture.
    #
    # Architecture Flow:
    # 1. BERT encoding: input_ids -> contextual phoneme embeddings
    # 2. Prosody prediction: embeddings + voice_style -> duration + F0/pitch
    # 3. Duration alignment: Create alignment matrix from predicted durations
    # 4. Feature extraction: Text encoder -> aligned acoustic features
    # 5. Audio synthesis: Decoder -> final waveform via iSTFT
    #
    # Parameters:
    # - input_ids: Phoneme token IDs, shape (1, N) with <bos>/<eos> padding
    # - ref_s: Voice embedding tensor, shape (1, 256) [128 baseline + 128 style]
    # - speed: Speech rate multiplier (0.5=slow, 2.0=fast)
    #
    # Returns:
    # - audio: Waveform tensor, shape (T,) at 24-kHz
    # - pred_dur: Duration predictions, shape (N,) in 40-fps frames
    #
    # Critical Implementation Details:
    # - Uses torch.no_grad() for inference-only mode
    # - Handles variable sequence lengths with proper masking
    # - Alignment matrix built via torch.repeat_interleave for efficiency
    # - F0/pitch and noise predictions run in parallel branches
    #
    # Called by:
    # - forward(): User-facing phoneme string interface
    # - export_coreml.py: Model tracing and conversion
    # - KModelForONNX.forward(): ONNX export wrapper
    #
        input_lengths = torch.full(
            (input_ids.shape[0],), 
            input_ids.shape[-1], 
            device=input_ids.device,
            dtype=torch.long
        )

        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1)).to(self.device)
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=self.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return audio, pred_dur

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False
    ) -> Union['KModel.Output', torch.FloatTensor]:
    # User-friendly inference interface accepting phoneme strings.
    #
    # This method provides the primary public API for text-to-speech synthesis.
    # It handles phoneme tokenization and delegates to forward_with_tokens()
    # for the actual neural computation.
    #
    # Tokenization Process:
    # 1. Map phoneme characters to vocabulary IDs via self.vocab
    # 2. Filter out unknown phonemes (returns None from vocab.get())
    # 3. Add <bos> (ID=0) and <eos> (ID=0) tokens for BERT compatibility
    # 4. Validate sequence length against context_length (512 tokens max)
    #
    # Parameters:
    # - phonemes: String of phoneme characters (e.g., "hˈɛloʊ wˈɜːld")
    # - ref_s: Voice embedding, shape (1, 256) on model.device
    # - speed: Speech rate control (1.0=normal, <1.0=slower, >1.0=faster) 
    # - return_output: If True, returns Output dataclass; if False, just audio
    #
    # Returns:
    # - KModel.Output: Full results with audio + duration (if return_output=True)
    # - torch.FloatTensor: Audio waveform only (if return_output=False)
    #
    # Error Handling:
    # - Asserts sequence length fits within BERT context window
    # - Automatically moves ref_s to model device
    # - Filters unknown phonemes silently (logs debug info)
    #
    # Called by:
    # - pipeline.py: KPipeline.infer() for production inference
    # - demo/app.py: Direct model usage in Gradio interface
    # - Test scripts: Basic functionality validation
    #
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        assert len(input_ids)+2 <= self.context_length, (len(input_ids)+2, self.context_length)
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)
        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed)
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        logger.debug(f"pred_dur: {pred_dur}")
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio

class KModelForONNX(torch.nn.Module):
    # ONNX/CoreML export wrapper for KModel with simplified interface.
    #
    # This wrapper class provides a clean, tensor-only interface that's more
    # suitable for model export and tracing. It eliminates string processing
    # and focuses purely on the numerical computation pipeline.
    #
    # Design Rationale:
    # - ONNX/CoreML exporters work better with pure tensor operations
    # - Avoids Python string processing that can't be traced
    # - Provides explicit input/output signatures for conversion tools
    # - Maintains compatibility with the full KModel functionality
    #
    # Usage Pattern:
    # 1. Wrap existing KModel: wrapper = KModelForONNX(trained_model)
    # 2. Trace with torch.jit.trace using representative inputs
    # 3. Convert traced model to ONNX or CoreML format
    #
    # Called by:
    # - Export scripts when simple tensor-in/tensor-out interface is needed
    # - Model serving scenarios where tokenization happens externally
    #
    def __init__(self, kmodel: KModel):
    # Initialize ONNX wrapper around existing KModel.
    #
    # Parameters:
    # - kmodel: Pre-trained KModel instance with loaded weights
    #
    # The wrapped model retains all its original functionality but
    # exposes only the tensor-based forward_with_tokens interface.
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
    # Pure tensor-based inference suitable for ONNX/CoreML export.
    #
    # This method directly delegates to the underlying KModel's
    # forward_with_tokens method, providing the same functionality
    # with a cleaner interface for export tools.
    #
    # Parameters:
    # - input_ids: Tokenized phonemes, shape (1, N)
    # - ref_s: Voice embedding, shape (1, 256) 
    # - speed: Speech rate multiplier
    #
    # Returns:
    # - waveform: Audio output, shape (T,)
    # - duration: Per-phoneme durations, shape (N,)
    #
    # Used for:
    # - torch.jit.trace() input for model conversion
    # - ONNX export workflows
    # - CoreML conversion pipelines
        waveform, duration = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return waveform, duration
