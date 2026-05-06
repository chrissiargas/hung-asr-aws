# Configuration Guide

This directory contains the configuration files for the Greek ASR system. The primary configuration file is `my_config.yaml`.

## Configuration Sections

The configuration is divided into four main sections:

### 1. `data_args`
Configuration related to data paths and basic audio properties.

*   `tmpdir`: Temporary directory for intermediate files.
*   `hf_cache`: Path to the Hugging Face cache directory.
*   `dataset_path`: Local directory for caching processed datasets.
*   `language`: Target language (default: 'greek').
*   `sampling_rate`: Audio sampling rate (standardized to 16000Hz).
*   `other_to_train`: Boolean flag indicating if other data should be included in training.

### 2. `main_args`
High-level training and workspace settings.

*   `datasets`: List of datasets to include in the training/evaluation run (e.g., `common_voice`, `fleurs`, `css10`, `hparl`).
*   `split_type`: Strategy for dataset splitting (default: 'default').
*   `results_path`: Path for saving evaluation results.
*   `checkpoint_path`: Root directory for saving model checkpoints.

### 3. `dual_fuse_args`
Core configuration for the **Dual Fusion Model** (Whisper + KriKri LLM).

#### Data Configurations
*   `interleave_temperature`: Sampling temperature for mixing multiple datasets.
*   `micro_data`: If `True`, trains on a small subset for debugging.
*   `micro_size`: Number of samples to use when `micro_data` is enabled.
*   `randomize`: Whether to shuffle the data.
*   `norm_mono`: Whether to normalize audio to mono.

#### Regularization & Augmentation
*   `blank_training`: Whether to train on blank (audio-only) samples.
*   `audio_dropout`: Dropout rate for acoustic features.
*   `text_perturbation`: Whether to apply perturbations to the text labels.
*   `text_dropout`: Dropout rate for text tokens (text masking).
*   `spec_augment`: Enables SpecAugment (time/frequency masking) on the audio.

#### Input Injection (Prompt Adapter)
*   `include_adapter`: If `True`, injects audio features into the LLM prompt.
*   `hidden_dim`: Dimensionality of the adapter's projection layers.
*   `static_projector`: If `True`, fixes the weights of the input adapter.
*   `downsample_K`: Factor by which to downsample audio for the prompt adapter.
*   `input_downsample`: Downsampling method (e.g., `reshape`, `avg_pool`, `conv1d`, `cif`).

#### Cross-Attention Injection
*   `static_injection_layers`: If `True`, the cross-attention modules are frozen.
*   `downsample_L`: Factor by which to downsample audio for the cross-attention modules.
*   `injection_downsample`: Downsampling method for cross-attention features.
*   `injection_layers`: List of LLM layer indices where cross-attention is applied (e.g., `[8, 16, 24]`).
*   `pyramid_layers`: If `True`, applies a pyramidal downsampling across layers.
*   `downsamplers`: Strategy for downsamplers (`common`, `injection_common`, or `different`).
*   `causal_fusion`: If `True`, applies a causal mask to the cross-attention.
*   `gated_cross_attention`: Adds a learnable tanh gate to the cross-attention output.
*   `positional_info`: If `True`, adds sinusoidal positional embeddings to audio features.

#### LoRA Configuration
*   `linguistic_lora`: Enables LoRA on the Language Model (LLM).
*   `acoustic_lora`: Enables LoRA on the Whisper Acoustic Encoder.
*   `lora_params`: List of modules to target with LoRA (e.g., `["q_proj", "v_proj"]`).
*   `proj_lr`: Learning rate for the projectors/adapters.
*   `lora_lr`: Learning rate for the LoRA layers.
*   `two_stage`: Enables the two-stage training strategy (Stage 1: Alignment, Stage 2: Refinement).

#### Auxiliary Tasks & Losses
*   `predict_duration`: Enables duration prediction from audio features.
*   `ctc`: Enables CTC (Connectionist Temporal Classification) loss.
*   `audio_forecasting`: Enables future acoustic embedding prediction.
*   `ctc_weight`, `duration_weight`, `audio_weight`: Loss scaling factors for auxiliary tasks.

#### Training & Prompt
*   `early_stopping_patience`: Patience for early stopping based on validation metrics.
*   `first_stage_epochs` / `second_stage_epochs`: Epoch counts for the two-stage strategy.
*   `prompt_instruction`: The instruction string injected into the LLM prompt.
*   `training_args`: Standard Hugging Face `TrainingArguments` (batch size, LR, epochs, etc.).

### 4. `eval_args`
Configuration for the evaluation phase.

*   `training_args`: Hugging Face `TrainingArguments` specific to evaluation (e.g., `per_device_eval_batch_size`, `generation_config`).

## Experiment Files
Files like `my_config_exp1.yaml`, `my_config_exp2.yaml`, etc., are variations used for different experiments, overriding the defaults in `my_config.yaml`. The `Parser` class in `parser.py` handles loading these based on an experiment ID.
