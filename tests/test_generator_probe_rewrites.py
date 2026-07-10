import torch
import torch.nn as nn

from export_synth.wrappers import ZeroInsertConvTranspose1d, rewrite_generator_ups_conv_transpose


def test_zero_insert_conv_transpose_rewrite_matches_conv_transpose1d():
    torch.manual_seed(0)
    original = nn.ConvTranspose1d(
        in_channels=3,
        out_channels=4,
        kernel_size=20,
        stride=10,
        padding=5,
    )
    rewritten = ZeroInsertConvTranspose1d(original)
    x = torch.randn(2, 3, 7)

    expected = original(x)
    actual = rewritten(x)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_generator_ups_rewrite_is_idempotent():
    generator = nn.Module()
    generator.ups = nn.ModuleList(
        [
            nn.ConvTranspose1d(3, 4, kernel_size=20, stride=10, padding=5),
            nn.ConvTranspose1d(4, 5, kernel_size=12, stride=6, padding=3),
        ]
    )

    assert rewrite_generator_ups_conv_transpose(generator) == 2
    assert rewrite_generator_ups_conv_transpose(generator) == 0
    assert all(isinstance(layer, ZeroInsertConvTranspose1d) for layer in generator.ups)
