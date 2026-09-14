"""``MaskedBidirectionalLSTM`` must reproduce ``pack_padded_sequence`` semantics.

Kokoro's duration path runs its bidirectional LSTMs over packed sequences. The
export wrapper receives right-padded buckets instead, so the reference here is
the same ``nn.LSTM`` run over ``pack_padded_sequence`` and padded back: valid
positions must match to fp32 rounding and padded positions must be zero.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from export_synth.wrappers import MaskedBidirectionalLSTM


def _packed_reference(lstm: nn.LSTM, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    packed = pack_padded_sequence(x, mask.sum(dim=1).cpu(), batch_first=True, enforce_sorted=False)
    out, _ = lstm(packed)
    out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
    return out


@pytest.mark.parametrize(
    "T,valid_lengths",
    [
        (32, [32]),          # no padding
        (32, [1]),           # almost all padding
        (64, [44]),          # bakeoff 3s prefix in the t64 bucket
        (128, [105, 64]),    # two rows with different valid lengths
        (512, [476, 128]),
        (1200, [601]),       # f0ntrain-sized frame axis
    ],
)
def test_matches_packed_sequence_semantics(T: int, valid_lengths: list[int]) -> None:
    torch.manual_seed(0)
    lstm = nn.LSTM(64, 32, num_layers=1, batch_first=True, bidirectional=True).eval()
    x = torch.randn(len(valid_lengths), T, 64)
    mask = torch.zeros(len(valid_lengths), T, dtype=torch.long)
    for row, valid in enumerate(valid_lengths):
        mask[row, :valid] = 1

    with torch.no_grad():
        got = MaskedBidirectionalLSTM(lstm).eval()(x, mask)
        want = _packed_reference(lstm, x, mask)

    torch.testing.assert_close(got, want, atol=1e-5, rtol=0)
    for row, valid in enumerate(valid_lengths):
        assert torch.count_nonzero(got[row, valid:]) == 0


def test_rejects_unsupported_lstm_shapes() -> None:
    with pytest.raises(ValueError, match="one-layer"):
        MaskedBidirectionalLSTM(nn.LSTM(8, 4, num_layers=2, batch_first=True, bidirectional=True))
    with pytest.raises(ValueError, match="one-layer"):
        MaskedBidirectionalLSTM(nn.LSTM(8, 4, num_layers=1, batch_first=True, bidirectional=False))
