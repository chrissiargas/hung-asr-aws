"""Evaluation harness (evaluations/fair_eval.py): normalization, scoring, manifests, decoding
presets and the paired bootstrap."""
import json

import numpy as np
import pytest
import torch

from evaluations import fair_eval as fe


# ----------------------------------------------------------------------------- normalization
def test_normalizer_strips_speaker_tags_annotations_and_punctuation():
    assert fe.normalize_nwer("Beszélő 1: Jó napot, [zaj] kívánok!") == "jó napot kívánok"


def test_speaker_tag_is_only_stripped_at_the_start():
    assert fe.normalize_nwer("Azt mondta a speaker: igen") == "azt mondta a speaker igen"


def test_numbers_are_verbalized_in_hungarian_unless_disabled():
    assert fe.normalize_nwer("3 alma") == "három alma"
    assert fe.normalize_nwer("3 alma", verbalize=False) == "3 alma"


def test_punctuation_aware_normalizer_keeps_punctuation_as_tokens():
    assert fe.normalize_wer_punct("Jó napot, kívánok!") == "jó napot , kívánok !"


# ----------------------------------------------------------------------------- scoring
def test_empty_hypothesis_counts_every_reference_word_as_deleted():
    assert fe._utterance_counts(["a b c"], [""]).tolist() == [[0, 3, 0, 3]]


def test_utterance_counts_split_substitutions_deletions_insertions():
    # columns: substitutions, deletions, insertions, reference words
    assert fe._utterance_counts(["a b c"], ["a x c d"]).tolist() == [[1, 0, 1, 3]]


def test_corpus_wer_is_micro_averaged_and_subsets_respect_duration():
    records = [
        {"reference_raw": "egy kettő három", "hypothesis_raw": "egy kettő három", "duration": 2.0},
        {"reference_raw": "négy öt", "hypothesis_raw": "négy hat", "duration": 40.0},
    ]
    metrics = fe.score_records(records)

    assert metrics["all"]["n_wer"] == pytest.approx(1 / 5)  # 1 error over 5 reference words
    assert metrics["le_30s"]["n_utts"] == 1 and metrics["le_30s"]["n_wer"] == 0.0


# ----------------------------------------------------------------------------- frozen manifests
def test_manifest_cannot_change_after_it_is_built(tmp_path):
    path = str(tmp_path / "hungarian_fleurs_test.jsonl")
    rows = [{"id": "fleurs:0", "reference_raw": "jó napot"}]
    fe._write_jsonl(path, rows)
    fe._write_json(f"{path}.meta.json", {"sha256": fe._sha256(path)})

    assert fe.load_manifest(path)[0] == rows

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "fleurs:1", "reference_raw": "új sor"}) + "\n")
    with pytest.raises(RuntimeError):
        fe.load_manifest(path)


# ----------------------------------------------------------------------------- decoding presets
def test_penalized_preset_replaces_builtin_penalties_with_generated_only_ones():
    kwargs, make_processors, description = fe.resolve_decoding("penalized", "dual_fusion", special_ids={0, 1})

    assert kwargs["repetition_penalty"] == 1.0 and kwargs["no_repeat_ngram_size"] == 0
    assert [type(p) for p in make_processors()] == [fe.GeneratedTextRepetitionPenalty,
                                                    fe.GeneratedTextNoRepeatNGram]
    assert description["repetition_penalty"] == 1.15 and description["no_repeat_ngram_size"] == 4


def test_whisper_only_preset_is_rejected_for_the_dual_fusion_model():
    with pytest.raises(ValueError):
        fe.resolve_decoding("whisper_standard", "dual_fusion", special_ids=set())


def test_generated_text_penalty_ignores_prompt_and_special_tokens():
    processor = fe.GeneratedTextRepetitionPenalty(penalty=2.0, special_ids={1})
    scores = torch.full((1, 10), 4.0)

    first = processor(torch.tensor([[5, 6]]), scores.clone())  # first call only records the prompt
    assert torch.equal(first, scores)

    out = processor(torch.tensor([[5, 6, 7, 1]]), scores.clone())  # prompt + "7" + EOS
    assert out[0, 7] == 2.0  # generated token penalized
    assert out[0, 5] == 4.0 and out[0, 6] == 4.0  # prompt tokens untouched
    assert out[0, 1] == 4.0  # special token never penalized


def test_no_repeat_ngram_only_looks_at_generated_tokens():
    processor = fe.GeneratedTextNoRepeatNGram(ngram_size=2)
    processor(torch.tensor([[3, 4]]), torch.zeros(1, 10))  # prompt

    # The bigram (3, 4) occurs only in the prompt: nothing is banned.
    assert torch.isfinite(processor(torch.tensor([[3, 4, 3]]), torch.zeros(1, 10))).all()

    # Generated so far: 3 4 3 -> continuing with 4 would repeat the generated bigram (3, 4).
    out = processor(torch.tensor([[3, 4, 3, 4, 3]]), torch.zeros(1, 10))
    assert out[0, 4] == -float("inf")
    assert torch.isfinite(out[0, [0, 1, 2, 3, 5]]).all()


# ----------------------------------------------------------------------------- significance
def test_paired_bootstrap_of_identical_systems_finds_no_difference():
    errors = np.array([1, 0, 2, 1, 0, 3])
    n_ref = np.array([5, 4, 6, 5, 3, 7])
    delta, low, high, p_value = fe.paired_bootstrap(errors, errors, n_ref, n_boot=200, seed=0)
    assert (delta, low, high, p_value) == (0.0, 0.0, 0.0, 1.0)


def test_paired_bootstrap_detects_a_clearly_better_system():
    rng = np.random.default_rng(1)
    n_ref = rng.integers(5, 15, size=300)
    errors_a, errors_b = (n_ref * 0.1).round(), (n_ref * 0.3).round()

    delta, low, high, p_value = fe.paired_bootstrap(errors_a, errors_b, n_ref, n_boot=500, seed=0)

    assert delta < 0 and high < 0 and p_value < 0.01
