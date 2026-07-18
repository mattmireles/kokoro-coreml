import torch
import numpy as np
import pytest
from kokoro.custom_stft import CustomSTFT
from kokoro.istftnet import TorchSTFT


@pytest.fixture
def sample_audio():
    # Generate a sample audio signal (sine wave)
    sample_rate = 16000
    duration = 1.0  # seconds
    t = torch.linspace(0, duration, int(sample_rate * duration))
    frequency = 440.0  # Hz
    signal = torch.sin(2 * np.pi * frequency * t)
    return signal.unsqueeze(0)  # Add batch dimension


def test_stft_reconstruction(sample_audio):
    """Round-trip must be near-transparent.

    Before 2026-07-14 the inverse used uniform 1/n_fft scaling (no one-sided
    doubling of interior bins, no overlap-add normalization), which passed a
    10 dB threshold while reconstructing 1.2-10.8 kHz at half amplitude. The
    threshold is deliberately strict so that regression cannot return.
    """
    custom_stft = CustomSTFT(filter_length=800, hop_length=200, win_length=800)
    out = custom_stft(sample_audio).squeeze()
    inp = sample_audio.squeeze()
    noise = inp - out
    rms = lambda x: torch.sqrt(torch.mean(x**2))
    snr_db = 20 * torch.log10((rms(inp) + 1e-12) / (rms(noise) + 1e-12))
    assert snr_db > 35.0, f"round-trip SNR too low: {snr_db.item():.2f} dB"


def test_inverse_matches_torch_istft_on_inconsistent_spectrogram():
    """The vocoder feeds the inverse a network-generated (non-consistent)
    magnitude/phase pair, so parity must hold for arbitrary spectrograms, not
    just round trips. Interior samples must match torch.istft exactly; edges
    differ only by the constant-vs-exact overlap-add envelope."""
    torch.manual_seed(1)
    n_fft, hop = 20, 5  # Kokoro generator iSTFT geometry
    frames = 200
    magnitude = torch.rand(1, n_fft // 2 + 1, frames) + 0.1
    phase = (torch.rand(1, n_fft // 2 + 1, frames) * 2 - 1) * np.pi

    custom_stft = CustomSTFT(filter_length=n_fft, hop_length=hop, win_length=n_fft)
    custom_out = custom_stft.inverse(magnitude, phase).reshape(-1)

    window = torch.hann_window(n_fft, periodic=True)
    torch_out = torch.istft(
        magnitude * torch.exp(phase * 1j), n_fft, hop, n_fft, window=window
    ).reshape(-1)

    n = min(custom_out.shape[-1], torch_out.shape[-1])
    interior = slice(n_fft, n - n_fft)
    assert torch.allclose(custom_out[interior], torch_out[interior], atol=1e-5)


def test_magnitude_phase_consistency(sample_audio):
    custom_stft = CustomSTFT(filter_length=800, hop_length=200, win_length=800)
    torch_stft = TorchSTFT(filter_length=800, hop_length=200, win_length=800)

    # Get magnitude and phase from both implementations
    custom_mag, custom_phase = custom_stft.transform(sample_audio)
    torch_mag, torch_phase = torch_stft.transform(sample_audio)

    # Compare magnitudes ignoring the boundary frames
    custom_mag_center = custom_mag[..., 2:-2]
    torch_mag_center = torch_mag[..., 2:-2]
    assert torch.allclose(custom_mag_center, torch_mag_center, rtol=1e-2, atol=1e-2)


def test_batch_processing():
    # Create a batch of signals
    batch_size = 4
    sample_rate = 16000
    duration = 0.1  # shorter duration for faster testing
    t = torch.linspace(0, duration, int(sample_rate * duration))
    frequency = 440.0
    signals = torch.sin(2 * np.pi * frequency * t).unsqueeze(0).repeat(batch_size, 1)

    custom_stft = CustomSTFT(filter_length=800, hop_length=200, win_length=800)

    # Process batch
    output = custom_stft(signals)

    # Check output shape
    assert output.shape[0] == batch_size
    assert len(output.shape) == 3  # (batch, 1, time)


def test_different_window_sizes():
    signal = torch.randn(1, 16000)  # 1 second of random noise

    # Test with different window sizes
    for filter_length in [512, 1024, 2048]:
        custom_stft = CustomSTFT(
            filter_length=filter_length,
            hop_length=filter_length // 4,
            win_length=filter_length,
        )

        # Forward and backward transform
        output = custom_stft(signal)

        # Check that output length is reasonable
        assert output.shape[-1] >= signal.shape[-1]
