"""Cross-attention injection and downsamplers (speechLM_utils/continuous_fusion_utils.py, downsamplers.py)."""
import pytest
import torch

from speechLM_utils.continuous_fusion_utils import CrossAttention
from speechLM_utils.downsamplers import Conv1DAdapter, ReshapeAdapter, get_downsampler

HIDDEN, AUDIO, HEADS = 32, 16, 4


def make_cross_attention(**kwargs):
    torch.manual_seed(0)
    return CrossAttention(hidden_dim=HIDDEN, audio_dim=AUDIO, num_heads=HEADS, **kwargs).eval()


def open_gate(attn, value=1.0):
    with torch.no_grad():
        attn.gate.fill_(value)
    return attn


# ----------------------------------------------------------------------------- gated cross-attention
def test_zero_initialized_gate_leaves_hidden_states_unchanged():
    """Paper §2.4: at the first update the injected audio must not disturb the pre-trained LLM."""
    attn = make_cross_attention(gated=True)
    hidden = torch.randn(2, 5, HIDDEN)
    out = attn(hidden, torch.randn(2, 7, AUDIO))
    assert torch.equal(out, hidden)


def test_open_gate_adds_audio_context():
    attn = open_gate(make_cross_attention(gated=True))
    hidden = torch.randn(2, 5, HIDDEN)
    out = attn(hidden, torch.randn(2, 7, AUDIO))
    assert out.shape == hidden.shape
    assert not torch.allclose(out, hidden)


def test_padded_audio_frames_get_no_attention():
    attn = make_cross_attention()
    attn.store_attention = True
    attn.attention_map = []
    mask = torch.tensor([[1., 1., 1., 1., 0., 0.]])
    attn(torch.randn(1, 4, HIDDEN), torch.randn(1, 6, AUDIO), inj_audio_mask=mask)

    weights = attn.attention_map[0]  # (batch, heads, text, audio)
    assert weights.shape == (1, HEADS, 4, 6)
    assert torch.all(weights[..., 4:] < 1e-6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, HEADS, 4), atol=1e-5)


def test_audio_is_repeated_for_each_beam():
    """During beam search the text batch is batch * beams while the audio batch is not."""
    attn = open_gate(make_cross_attention(gated=True))
    hidden, audio, beams = torch.randn(2, 5, HIDDEN), torch.randn(2, 7, AUDIO), 3

    with_beams = attn(hidden.repeat_interleave(beams, dim=0), audio)
    attn.clear_cache()
    reference = attn(hidden, audio)

    assert with_beams.shape == (2 * beams, 5, HIDDEN)
    assert torch.allclose(with_beams[::beams], reference, atol=1e-5, rtol=1e-5)


def test_decoding_step_reuses_cached_audio_keys():
    """After the prompt pass, single-token decoding steps reuse the cached audio keys/values."""
    attn = open_gate(make_cross_attention(gated=True))
    audio = torch.randn(1, 7, AUDIO)
    attn(torch.randn(1, 5, HIDDEN), audio)  # prompt pass fills the cache

    step = torch.randn(1, 1, HIDDEN)
    cached = attn(step, torch.zeros_like(audio))  # audio argument ignored: keys come from the cache
    attn.clear_cache()
    fresh = attn(step, audio)

    assert torch.allclose(cached, fresh, atol=1e-6)


# ----------------------------------------------------------------------------- downsamplers
def test_reshape_adapter_stacks_consecutive_frames():
    x = torch.randn(2, 1500, AUDIO)
    y = ReshapeAdapter(5, torch.float32)(x)
    assert y.shape == (2, 300, 5 * AUDIO)
    assert torch.equal(y[0, 0], x[0, :5].reshape(-1))


def test_reshape_adapter_rejects_indivisible_lengths():
    with pytest.raises(ValueError):
        ReshapeAdapter(7, torch.float32)(torch.randn(1, 1500, AUDIO))


def test_conv1d_adapter_downsamples_whisper_frames_by_three():
    """Paper §2.1: 1500 Whisper frames (20 ms) -> 500 frames (60 ms) for cross-attention."""
    y = Conv1DAdapter(AUDIO, 3)(torch.randn(2, 1500, AUDIO))
    assert y.shape == (2, 500, AUDIO)


@pytest.mark.parametrize("method, expected_dim", [
    ("reshape", 5 * AUDIO), ("conv1d", AUDIO), ("avg_pool", AUDIO), ("linear", AUDIO),
])
def test_get_downsampler_reports_its_output_dim(method, expected_dim):
    module, dim = get_downsampler(method, 5, AUDIO, torch.device("cpu"), torch.float32)
    assert dim == expected_dim
    assert module(torch.randn(1, 1500, AUDIO)).shape == (1, 300, expected_dim)
