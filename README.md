# Dual-Fusion Speech LLM for Medium-Resource ASR

Code for **"Unifying Speech and Language through Continuous Dual-Fusion for Medium-Resource
Languages: The Cases of Hungarian and Greek"** (C. Siargkas et al., under review at IEEE ICASSP 2027).
This repository holds the Hungarian configuration (Whisper-large-v3 + `elte-nlp/Racka-4B`); the Greek
model uses the same code with `Llama-KriKri-8B-Instruct`.

A frozen Whisper encoder is bridged into a language-specialized LLM through two paths at once:

- **Prompt path:** the last encoder state is downsampled ×5 (reshape) and projected by an MLP into
  the LLM's embedding space as prefix tokens.
- **Cross-attention path:** all encoder layers are mixed per utterance by a learned layer-wise
  attention, downsampled ×3 (Conv1D), and injected into several decoder layers through
  zero-initialized, tanh-gated cross-attention.

Training also masks part of the teacher-forced text history, so the decoder cannot ignore the audio.

| Hungarian test set (WER %) | Dual-Fusion | Whisper-large-v3 |
|---|---|---|
| Common Voice* | 5.9 | 13.3 |
| FLEURS | 8.0 | 13.0 |
| Speech-MASSIVE | 13.9 | 20.1 |
| VoxPopuli | 7.6 | 14.1 |
| YODAS† | 11.3 | 7.1 |

\* Some Common Voice test sentences also appear in training (detected by `fair_eval.py build-manifest`), so read this row as an upper bound.
† YODAS references are Whisper-large-v3 pseudo-labels.

## Where each part of the paper lives

| Paper | Code |
|---|---|
| §2.1 Prompt path (reshape ×5 + MLP adapter) | `DualFusionModel.get_input_embeddings`, `downsamplers.ReshapeAdapter` |
| §2.1 Cross-attention path (Conv1D ×3) | `DualFusionModel.inject`, `downsamplers.Conv1DAdapter` |
| §2.2 Multiple injection layers | `continuous_fusion_utils.InjectionLayer`, `injection_layers` in the config |
| §2.3 Layer-wise fusion of encoder states | `dual_fusion_model.LayerWiseAttention` |
| §2.4 Zero-initialized tanh gate | `continuous_fusion_utils.CrossAttention` (`gate`) |
| §2.5 Text perturbation | `dual_fusion_model.apply_text_dropout` |
| §3.2 Two-stage curriculum (bridge, then QLoRA) | `train.two_stage_train`, `model.load_weights` |
| §3.3 Temperature-based corpus interleaving | `preprocessing/prepare.interleave` |
| §4 Evaluation protocol and significance tests | `evaluations/fair_eval.py` |
| §4.4 Attention entropy and diagonality | `evaluations/utils.calculate_attention_metrics`, `evaluations/vizualizations.py` |

Every component is a config switch (`include_adapter`, `layer_wise_fusion`, `gated_cross_attention`,
`text_perturbation`, `downsample_L`, ...), so each row of the ablation table is a config change, not a
code change.

## Repository layout

```
config/            YAML experiment configs and the parser (my_config.yaml, my_config_expN.yaml)
constructors/      one loader per corpus: download, resample to 16 kHz, write a JSON manifest
preprocessing/     text normalization, data-quality filters, splitting and interleaving
speechLM_utils/    the model, training loop, collator, checkpointing (see its README)
evaluations/       frozen-manifest evaluation harness, baselines, result tables and plots
tb_utils/          TensorBoard helpers
tests/             CPU-only unit tests, run by GitHub Actions on every push
```

## Setup

```bash
conda env create -f environment.yml   # environment "nemo2", Python 3.10, fully pinned
conda activate nemo2
```

Credentials are read from the environment and never stored in the code:

| Variable | Used for |
|---|---|
| `HF_TOKEN` | gated Hugging Face models/datasets (or run `huggingface-cli login`) |
| `WANDB_API_KEY` | experiment tracking (or run `wandb login`) |
| `MDC_API_KEY` | downloading Common Voice from the Mozilla Data Collective |
| `ASSEMBLYAI_API_KEY` | only for the optional AssemblyAI baseline |

Cache, dataset, checkpoint and result paths are set in `data_args` / `main_args` of the config.

## Training

```bash
torchrun --nproc_per_node=4 speechLM_utils/train.py --exp 1
```

`--exp N` selects `config/my_config_expN.yaml` (`0` = `my_config.yaml`). With `two_stage: True` the run
first trains only the bridge with the LLM frozen, then freezes the bridge and adapts the 4-bit LLM with LoRA.

Each run writes its full configuration to `config.json` next to its checkpoints, and resuming a run
reads that file rather than the current YAML. Losses, WER/CER and sample predictions (with audio)
are logged to Weights & Biases and TensorBoard.

## Evaluation

```bash
bash evaluations/run_fair_eval.sh fleurs voxpopuli
python evaluations/collect_results.py runs/hungarian_*_test --bootstrap
```

`evaluations/fair_eval.py` makes system comparisons reproducible and controlled:

- **Frozen test sets.** Each test set is written once as a manifest, and its SHA-256 is checked
  before every run. Utterances whose reference text also occurs in training are flagged.
- **One protocol for every system.** Every system is scored on the same audio, with the same
  30 s input policy and the same normalizer (NFKC, tag and bracket removal, number verbalization,
  Whisper's `BasicTextNormalizer`).
- **Named decoding presets.** `plain` is the controlled comparison, `legacy` is the setting
  behind the paper numbers, and `whisper_standard` is Whisper's own decoding.
- **Restart-safe multi-GPU decoding.** Inference is sharded across GPUs, and the merge step checks
  that every utterance appears exactly once.
- **Self-describing outputs.** Every run writes `predictions.jsonl`, `metrics.json` and a
  `run_config.json` with library versions and a configuration hash.
- **Significance testing.** `compare` reports paired-bootstrap 95% confidence intervals and p-values.

## Tests

```bash
pytest          # inside the project environment (pip install pytest if it is missing)
```

The tests run on CPU without model downloads. They cover the zero-gate identity of the
cross-attention, padding masks, beam expansion, layer-wise fusion weights, the downsamplers,
text perturbation, the repetition-penalty and n-gram processors, and the evaluation normalizer,
manifest check and bootstrap. GitHub Actions installs the CPU-only dependencies in
`tests/requirements.txt` and runs the tests, plus a lint check for undefined names, on every push.

## Data

Public corpora: Common Voice, FLEURS, Speech-MASSIVE, VoxPopuli and YODAS (via Granary). The paper's
two internal corpora (~1,500 h of telephony and broadcast speech) are licensed and not distributed.
The model trained on public data only still outperforms Whisper on the four comparable test sets
(paper, Table 3).
