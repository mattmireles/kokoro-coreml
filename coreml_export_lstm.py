"""Side-effect-free LSTM primitive shared by the Core ML export entry points."""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedBidirectionalLSTM(nn.Module):
    """Reproduce packed-sequence semantics on right-padded static input.

    The forward direction may read the full padded tensor because padding
    follows every valid prefix. The backward direction must start at the last
    valid element, so it runs a forward LSTM over a gather-reversed prefix.
    Keeping both directions as stock LSTMs lets Core ML lower them to two
    ``lstm`` operations instead of unrolling one cell per timestep.
    """

    def __init__(self, original_lstm: nn.LSTM):
        super().__init__()
        if original_lstm.num_layers != 1 or not original_lstm.bidirectional or not original_lstm.batch_first:
            raise ValueError("MaskedBidirectionalLSTM expects one-layer batch-first bidirectional LSTM")
        self.hidden_size = original_lstm.hidden_size
        self.fwd = nn.LSTM(original_lstm.input_size, self.hidden_size, num_layers=1, batch_first=True)
        self.bwd = nn.LSTM(original_lstm.input_size, self.hidden_size, num_layers=1, batch_first=True)
        with torch.no_grad():
            self.fwd.weight_ih_l0.copy_(original_lstm.weight_ih_l0)
            self.fwd.weight_hh_l0.copy_(original_lstm.weight_hh_l0)
            self.fwd.bias_ih_l0.copy_(original_lstm.bias_ih_l0)
            self.fwd.bias_hh_l0.copy_(original_lstm.bias_hh_l0)
            self.bwd.weight_ih_l0.copy_(original_lstm.weight_ih_l0_reverse)
            self.bwd.weight_hh_l0.copy_(original_lstm.weight_hh_l0_reverse)
            self.bwd.bias_ih_l0.copy_(original_lstm.bias_ih_l0_reverse)
            self.bwd.bias_hh_l0.copy_(original_lstm.bias_hh_l0_reverse)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        # Keep index arithmetic in int64. fp16 stops representing every integer
        # at 2048, close to the largest f0ntrain bucket.
        valid = attention_mask.to(dtype=torch.long).sum(dim=1, keepdim=True)
        h_fwd, _ = self.fwd(x)

        arange = torch.arange(steps, device=x.device, dtype=torch.long).unsqueeze(0).expand(batch, steps)
        reverse_index = (valid - 1 - arange).clamp(min=0, max=steps - 1)
        expanded_index = reverse_index.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x_reversed = torch.gather(x, 1, expanded_index)
        h_reversed, _ = self.bwd(x_reversed)
        output_index = reverse_index.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        h_bwd = torch.gather(h_reversed, 1, output_index)

        mask = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
        return torch.cat([h_fwd * mask, h_bwd * mask], dim=-1)
