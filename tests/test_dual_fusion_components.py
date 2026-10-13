"""Model components that need no checkpoint: layer-wise fusion, text perturbation, mask
downsampling and the EOS-safe repetition penalty (speechLM_utils/dual_fusion_model.py)."""
from types import SimpleNamespace

import pytest
import torch
from transformers import RepetitionPenaltyLogitsProcessor

from speechLM_utils.dual_fusion_model import (DualFusionModel, LayerWiseAttention,
                                              SafeRepetitionPenaltyLogitsProcessor, apply_text_dropout)

DIM = 16
TOKENIZER = SimpleNamespace(pad_token_id=0, eos_token_id=1)


# ----------------------------------------------------------------------------- layer-wise fusion (§2.3)
def test_layer_wise_attention_mixes_layers_per_utterance():
    torch.manual_seed(0)
    module = LayerWiseAttention(num_layers=4, hidden_dim=DIM, bottleneck_dim=8)
    states = torch.randn(4, 3, 10, DIM)  # (encoder layers, batch, time, dim)

    fused, weights = module(states, return_weights=True)

    assert fused.shape == (3, 10, DIM)
    assert weights.shape == (4, 3, 1, 1)
    assert torch.allclose(weights.sum(dim=0), torch.ones(3, 1, 1), atol=1e-6)
    assert torch.allclose(fused, (states * weights).sum(dim=0))
    assert not torch.allclose(weights[:, 0], weights[:, 1])  # each utterance gets its own mixture


def test_layer_wise_attention_of_identical_layers_returns_that_layer():
    torch.manual_seed(0)
    module = LayerWiseAttention(num_layers=4, hidden_dim=DIM, bottleneck_dim=8)
    layer = torch.randn(2, 10, DIM)
    fused = module(layer.expand(4, -1, -1, -1))
    assert torch.allclose(fused, layer, atol=1e-5)


# ----------------------------------------------------------------------------- text perturbation (§2.5)
def test_text_dropout_with_zero_probability_changes_nothing():
    labels = torch.tensor([[5, 6, 7, 1, 0]])
    masks = torch.tensor([[1, 1, 1, 1, 0]])
    assert torch.equal(apply_text_dropout(labels, masks, TOKENIZER, dropout_prob=0.0), labels)


def test_text_dropout_masks_only_real_target_tokens():
    labels = torch.tensor([[5, 6, 7, 1, 0, 0]])
    masks = torch.tensor([[1, 1, 1, 1, 0, 0]])

    noisy = apply_text_dropout(labels, masks, TOKENIZER, dropout_prob=1.0)

    assert torch.equal(noisy, torch.tensor([[0, 0, 0, 1, 0, 0]]))  # EOS and padding untouched
    assert torch.equal(labels, torch.tensor([[5, 6, 7, 1, 0, 0]]))  # input not modified in place


def test_text_dropout_rate_matches_the_probability():
    torch.manual_seed(0)
    labels = torch.randint(2, 100, (64, 50))
    noisy = apply_text_dropout(labels, torch.ones_like(labels), TOKENIZER, dropout_prob=0.1)
    rate = (noisy == TOKENIZER.pad_token_id).float().mean().item()
    assert 0.08 < rate < 0.12


# ----------------------------------------------------------------------------- masks
def test_downsampled_mask_keeps_frames_that_contain_any_speech():
    mask = torch.zeros(1, 1500)
    mask[0, :601] = 1  # 601 valid Whisper frames

    # calculate_mask does not use the model's state, so no model needs to be built.
    new_mask = DualFusionModel.calculate_mask(None, mask, torch.zeros(1, 500, 4))

    assert new_mask.shape == (1, 500)
    assert new_mask[0, :201].eq(1).all() and new_mask[0, 201:].eq(0).all()


# ----------------------------------------------------------------------------- decoding
def test_safe_repetition_penalty_never_penalizes_eos():
    vocab, eos, word = 50, 1, 7
    input_ids = torch.tensor([[word, eos, word]])
    scores = torch.full((1, vocab), 10.0)

    standard = RepetitionPenaltyLogitsProcessor(penalty=1.2)(input_ids, scores.clone())
    safe = SafeRepetitionPenaltyLogitsProcessor(penalty=1.2, skip_token_ids=[eos])(input_ids, scores.clone())

    assert standard[0, eos] < 10.0  # the stock processor makes it harder to stop
    assert safe[0, eos] == 10.0  # ours leaves EOS alone
    assert safe[0, word] == pytest.approx(10.0 / 1.2)  # repeated words are still penalized
    assert safe[0, 2] == 10.0  # unseen tokens are untouched
