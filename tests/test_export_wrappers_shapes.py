"""Shape contracts for ``export_synth.wrappers`` DurationModel / SynthesizerModel."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def kmodel():
    """Fresh KModel per test (wrappers mutate submodules)."""
    from kokoro import KModel

    try:
        return KModel(disable_complex=True)
    except Exception as exc:
        pytest.skip(f"KModel load failed: {exc}")


def test_duration_model_returns_five_tensors_expected_shapes(kmodel):
    from export_synth.wrappers import DurationModel

    dm = DurationModel(kmodel)
    b, t = 1, 128
    input_ids = torch.zeros((b, t), dtype=torch.long)
    attention_mask = torch.ones((b, t), dtype=torch.long)
    ref_s = torch.randn(b, 256)
    speed = torch.ones(1)
    pred_dur, d, t_en, s, ref_s_out = dm(input_ids, ref_s, speed, attention_mask)

    assert pred_dur.shape == (b, t)
    assert d.ndim == 3 and d.shape[0] == b
    assert t_en.ndim == 3 and t_en.shape[0] == b
    assert s.shape == (b, 128)
    assert ref_s_out.shape == ref_s.shape


def test_duration_model_reuses_already_masked_predictor_lstm(kmodel):
    from export_synth.wrappers import DurationModel, MaskedBidirectionalLSTM

    kmodel.predictor.lstm = MaskedBidirectionalLSTM(kmodel.predictor.lstm)

    dm = DurationModel(kmodel)

    assert dm.duration_lstm is kmodel.predictor.lstm


def test_duration_model_padded_input_matches_exact_valid_prefix(kmodel):
    from export_synth.wrappers import DurationModel

    dm = DurationModel(kmodel)
    dm.eval()
    # Bakeoff 3s token prefix. It is long enough to expose right-padding drift
    # through the shared bidirectional duration LSTM.
    valid_ids = torch.tensor(
        [
            0, 81, 83, 16, 53, 65, 156, 102, 53, 16, 44, 123,
            156, 39, 56, 16, 48, 156, 69, 53, 61, 16, 82, 156,
            138, 55, 58, 61, 16, 156, 31, 64, 83, 123, 16, 81,
            83, 16, 46, 156, 76, 92, 4, 0,
        ],
        dtype=torch.long,
    ).unsqueeze(0)
    padded_ids = torch.zeros((1, 64), dtype=torch.long)
    padded_ids[:, : valid_ids.shape[1]] = valid_ids
    exact_mask = torch.ones_like(valid_ids)
    padded_mask = torch.zeros_like(padded_ids)
    padded_mask[:, : valid_ids.shape[1]] = 1
    torch.manual_seed(1234)
    ref_s = torch.randn(1, 256)
    speed = torch.ones(1)

    with torch.no_grad():
        exact_pred, *_ = dm(valid_ids, ref_s, speed, exact_mask)
        padded_pred, *_ = dm(padded_ids, ref_s, speed, padded_mask)

    assert torch.equal(exact_pred, padded_pred[:, : valid_ids.shape[1]])


def test_synthesizer_model_forward_runs_and_returns_1d_audio(kmodel):
    from kokoro import KModel
    from export_synth.wrappers import DurationModel, SynthesizerModel

    dm = DurationModel(kmodel)
    b, t = 1, 128
    input_ids = torch.zeros((b, t), dtype=torch.long)
    attention_mask = torch.ones((b, t), dtype=torch.long)
    ref_s = torch.randn(b, 256)
    speed = torch.ones(1)
    pred_dur, d, t_en, s, ref_s_out = dm(input_ids, ref_s, speed, attention_mask)

    ttok = d.shape[-1]
    F = 72
    pred_aln_trg = torch.full((ttok, F), 1.0 / F, dtype=torch.float32)

    k2 = KModel(disable_complex=True)
    sm = SynthesizerModel(k2)
    audio = sm(d, t_en, s, ref_s_out, pred_aln_trg)
    if audio.ndim == 2:
        assert audio.shape[0] == 1
        audio = audio.squeeze(0)
    assert audio.ndim == 1
    assert audio.numel() > 0


# ---------------------------------------------------------------------------
# GeneratorFromHar._align_mask_to
# ---------------------------------------------------------------------------
#
# The mask enters on the x_pre axis (T frames) and is re-aligned inside each
# upsample stage: to the noise_convs output and to x. Those axes are integer
# multiples of the mask's axis plus at most one frame (reflection pad, N/hop + 1
# har frames). The pairs are the real ones for the 3 s and 30 s packages.

_ALIGN_PAIRS = [
    (240, 2400), (240, 4800), (2400, 14401), (2400, 28801),  # 3 s package
    (2400, 24000), (2400, 48000), (24000, 144001), (24000, 288001),  # 30 s package
]


def _prefix_mask(cur_t, valid):
    m = torch.zeros(1, 1, cur_t)
    m[:, :, :valid] = 1.0
    return m


@pytest.mark.parametrize("cur_t,target_t", _ALIGN_PAIRS)
def test_align_mask_to_never_invents_padding_in_an_all_ones_mask(cur_t, target_t):
    """An all-ones mask must survive alignment with no zeros introduced.

    All-ones means "every frame is valid". Alignment changes resolution, never
    validity, so a zero appearing here (the +1 frame is the usual suspect) gates
    out audio the caller said was real.
    """
    from export_synth.wrappers import GeneratorFromHar

    out = GeneratorFromHar._align_mask_to(torch.ones(1, 1, cur_t), target_t)
    assert out.shape[-1] == target_t
    assert int((out == 0).sum()) == 0


@pytest.mark.parametrize("cur_t,target_t", _ALIGN_PAIRS)
def test_align_mask_to_puts_the_boundary_at_valid_times_the_factor(cur_t, target_t):
    """A partial mask stays binary, stays a prefix, and ends at ``valid * factor``.

    Over-covering makes the model normalise real frames by padding; under-covering
    gates out real frames. Values must stay 0/1: anything in between means the
    mask was interpolated, which would scale statistics instead of selecting frames.
    """
    from export_synth.wrappers import GeneratorFromHar

    valid = cur_t // 2 + 1
    out = GeneratorFromHar._align_mask_to(_prefix_mask(cur_t, valid), target_t)
    assert out.shape[-1] == target_t
    assert set(out.unique().tolist()) <= {0.0, 1.0}
    ones = int((out == 1).sum())
    assert ones == valid * (target_t // cur_t)
    assert int(out[0, 0, :ones].sum()) == ones, "valid region is not a contiguous prefix"


def test_align_mask_to_traces_with_a_shape_derived_target():
    """The generator passes ``x.shape[-1]`` as the target, which is a tensor under tracing.

    The index must still be built from Python ints so it enters the trace as a
    constant rather than as int32 graph arithmetic.
    """
    from export_synth.wrappers import GeneratorFromHar

    class Align(torch.nn.Module):
        def forward(self, m, x):
            return GeneratorFromHar._align_mask_to(m, x.shape[-1])

    m, x = _prefix_mask(2400, 1204), torch.zeros(1, 1, 144001)
    traced = torch.jit.trace(Align().eval(), (m, x))
    assert torch.equal(traced(m, x), GeneratorFromHar._align_mask_to(m, 144001))
    assert "aten::floor_divide" not in str(traced.graph)
    assert "aten::arange" not in str(traced.graph)


def test_align_mask_to_passes_none_through():
    """``None`` means "no masking"; alignment must not manufacture a mask."""
    from export_synth.wrappers import GeneratorFromHar

    assert GeneratorFromHar._align_mask_to(None, 1234) is None


def test_align_mask_to_rejects_downsampling():
    """numpy floor-divides by zero silently, so a shrinking axis must fail loudly."""
    from export_synth.wrappers import GeneratorFromHar

    with pytest.raises(ValueError):
        GeneratorFromHar._align_mask_to(torch.ones(1, 1, 100), 50)


@pytest.mark.parametrize("cur_t,target_t", [(24000, 144001), (24000, 288001)])
def test_align_mask_to_is_exact_after_core_ml_conversion(cur_t, target_t):
    """Regression: the converted fp16 graph must reproduce PyTorch's mask exactly.

    Core ML has no int64. When the index was left to the graph as
    ``j * cur_t // target_t``, coremltools evaluated it in int32, and at the
    30 s package's second stage (24000 x 144001 > 2^31) the wrapped products
    marked ~18k padded frames valid and cost 3 dB of mid band at 50% fill.
    The index now enters the trace as a constant; this pins that.
    """
    ct = pytest.importorskip("coremltools")
    import numpy as np

    from export_synth.wrappers import GeneratorFromHar

    class Align(torch.nn.Module):
        def forward(self, m):
            return GeneratorFromHar._align_mask_to(m, target_t)

    m = _prefix_mask(cur_t, cur_t // 2 + 1)
    with torch.no_grad():
        expected = Align().eval()(m).numpy()
    traced = torch.jit.trace(Align().eval(), (m,))
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="m", shape=(1, 1, cur_t), dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    got = np.asarray(next(iter(model.predict({"m": m.numpy()}).values())), dtype=np.float32)
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)
