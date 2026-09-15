"""Side-effect-free mask-aware wrappers shared by duration export entry points."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from coreml_export_lstm import MaskedBidirectionalLSTM


def _is_masked_lstm(module: nn.Module) -> bool:
    return isinstance(module, MaskedBidirectionalLSTM)


class CoreMLFriendlyTextEncoder(nn.Module):
    """Replace packed text encoding with a static-shape mask-aware LSTM."""

    def __init__(self, original_encoder):
        super().__init__()
        self.embedding = original_encoder.embedding
        self.cnn = original_encoder.cnn
        self.lstm = (
            original_encoder.lstm
            if _is_masked_lstm(original_encoder.lstm)
            else MaskedBidirectionalLSTM(original_encoder.lstm)
        )

    def forward(self, x, input_lengths, padding_mask):
        del input_lengths
        valid_mask = (~padding_mask).to(dtype=torch.long)
        x = self.embedding(x).transpose(1, 2)
        channel_mask = padding_mask.unsqueeze(1)
        x.masked_fill_(channel_mask, 0.0)
        for convolution in self.cnn:
            x = convolution(x)
            x.masked_fill_(channel_mask, 0.0)
        x = self.lstm(x.transpose(1, 2), valid_mask).transpose(-1, -2)
        x.masked_fill_(channel_mask, 0.0)
        return x


class CoreMLFriendlyDurationEncoder(nn.Module):
    """Replace packed duration encoding with static-shape mask-aware LSTMs."""

    def __init__(self, original_encoder):
        super().__init__()
        self.lstms = nn.ModuleList(
            block
            if _is_masked_lstm(block)
            else MaskedBidirectionalLSTM(block)
            if isinstance(block, nn.LSTM)
            else block
            for block in original_encoder.lstms
        )
        self.dropout = original_encoder.dropout

    def forward(self, x, style, text_lengths, padding_mask):
        del text_lengths
        valid_mask = (~padding_mask).to(dtype=torch.long)
        x = x.permute(2, 0, 1)
        # Core ML validates expand as a tile whose inferred reps can reach zero.
        # The static export batch is already carried by style, so repeat only
        # the sequence axis explicitly.
        expanded_style = style.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = torch.cat([x, expanded_style], axis=-1)
        x.masked_fill_(padding_mask.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1).transpose(-1, -2)
        for block in self.lstms:
            if type(block).__name__ == "AdaLayerNorm":
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, expanded_style.permute(1, 2, 0)], axis=1)
                x.masked_fill_(padding_mask.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = block(x.transpose(-1, -2), valid_mask)
                x = nn.functional.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
        return x.transpose(-1, -2)


class DurationModel(nn.Module):
    """Predict durations and emit the features consumed by synthesizer export."""

    def __init__(self, kmodel: Any):
        super().__init__()
        self.kmodel = kmodel
        self.kmodel.text_encoder = CoreMLFriendlyTextEncoder(kmodel.text_encoder)
        self.kmodel.predictor.text_encoder = CoreMLFriendlyDurationEncoder(kmodel.predictor.text_encoder)
        self.duration_lstm = (
            kmodel.predictor.lstm
            if _is_masked_lstm(kmodel.predictor.lstm)
            else MaskedBidirectionalLSTM(kmodel.predictor.lstm)
        )
        if hasattr(self.kmodel.bert.embeddings, "token_type_ids"):
            delattr(self.kmodel.bert.embeddings, "token_type_ids")

    def forward(self, input_ids, ref_s, speed, attention_mask):
        kmodel = self.kmodel
        input_lengths = attention_mask.sum(dim=-1).to(torch.long)
        text_mask = attention_mask == 0
        bert_dur = kmodel.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(input_ids),
        )
        d_en = kmodel.bert_encoder(bert_dur).transpose(-1, -2)
        style = ref_s[:, 128:]
        duration_features = kmodel.predictor.text_encoder(d_en, style, input_lengths, text_mask)
        duration_hidden = self.duration_lstm(duration_features, attention_mask)
        duration = kmodel.predictor.duration_proj(duration_hidden)
        pred_dur = torch.round(torch.sigmoid(duration).sum(axis=-1) / speed).clamp(min=1).long()
        text_features = kmodel.text_encoder(input_ids, input_lengths, text_mask)
        return pred_dur, duration_features, text_features, style, ref_s + torch.zeros_like(ref_s)
