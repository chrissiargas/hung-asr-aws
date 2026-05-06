# Greek ASR Methodology: Dual Fusion (Whisper + KriKri LLM)

This project implements a Greek Automatic Speech Recognition (ASR) system by fusing a pre-trained acoustic model (**Whisper**) with a Greek-optimized Large Language Model (**KriKri**, based on Llama). The methodology centers on a "Dual Fusion" approach, where acoustic information is injected into the LLM at both the input (prompt) level and at intermediate layers via cross-attention.

## 1. Methodology Overview

The core architecture, implemented in \`DualFusionModel\` (\`speechLM_utils/dual_fusion_model.py\`), leverages the linguistic power of an LLM to refine and transcribe acoustic features.

### Key Components:
- **Acoustic Encoder:** OpenAI's Whisper-large-v3 encoder extracts acoustic embeddings from Greek speech.
- **Language Model:** \`Llama-Krikri-8B\` (Base or Instruct) acts as the decoder/transcriber.
- **Input Injection (Prompt Level):** Acoustic features are downsampled and projected into the LLM's embedding space. They are then concatenated with text prompt embeddings (e.g., "Μετάγραψε το...") as if they were special "audio tokens".
- **Cross-Attention Injection (Deep Level):** Intermediate LLM layers (e.g., layers 10, 20, 30) are wrapped with \`InjectionLayer\` modules. These modules perform cross-attention between the LLM's hidden states and the acoustic embeddings, allowing the model to attend to fine-grained acoustic details during the generation process.
- **LoRA (Low-Rank Adaptation):** To keep training efficient and preserve the LLM's knowledge, LoRA is applied to the linguistic model and optionally to the acoustic encoder.
- **Multi-task Learning:** The model can be trained with auxiliary losses like **CTC** (Connectionist Temporal Classification), **Duration Prediction**, and **Audio Forecasting** to better align acoustic and linguistic representations.

### Training Phases:
The system supports a two-stage training strategy (\`speechLM_utils/train.py\`):
1.  **Stage 1 (Adapter Alignment):** Only the adapters and cross-attention modules are trained to align acoustic features with the LLM's space. The LLM remains frozen.
2.  **Stage 2 (LoRA Refinement):** LoRA is enabled on the LLM, allowing it to adapt its linguistic knowledge specifically for the ASR task.

---

## 2. Configuration Guide (\`config/my_config.yaml\`)

The system is highly configurable. Below is an explanation of the primary arguments in \`my_config.yaml\`.

### \`data_args\`
- \`hf_cache\`: Path to the Hugging Face cache.
- \`dataset_path\`: Local directory for caching datasets.
- \`language\`: Target language (default: 'greek').
- \`sampling_rate\`: Audio sampling rate (standardized to 16000Hz).

### \`main_args\`
- \`datasets\`: List of datasets to include (e.g., \`common_voice\`, \`fleurs\`, \`hparl\`).
- \`checkpoint_path\`: Base directory for saving model checkpoints.

### \`dual_fuse_args\` (The Core Model Config)

#### Data & Regularization
- \`interleave_temperature\`: Controls the sampling balance when mixing multiple datasets.
- \`micro_data\` / \`micro_size\`: If enabled, trains on a small subset for debugging.
- \`audio_dropout\`: Probability of zeroing out audio features during training.
- \`text_dropout\`: Probability of masking text tokens (text perturbation).
- \`spec_augment\`: Enables SpecAugment (time/frequency masking) on the audio.

#### Input Injection (Adapter)
- \`include_adapter\`: If \`True\`, injects audio features at the prompt level.
- \`downsample_K\`: Factor by which to downsample audio for the input adapter.
- \`input_downsample\`: Method for downsampling (e.g., \`reshape\`, \`avg_pool\`, \`conv1d\`, \`cif\`).
- \`hidden_dim\`: Dimensionality of the adapter's projection layers.
- \`static_projector\`: If \`True\`, the input adapter weights are frozen.

#### Cross-Attention Injection
- \`injection_layers\`: A list of LLM layer indices where cross-attention is applied (e.g., \`[10, 20, 30]\`).
- \`downsample_L\`: Factor by which to downsample audio for the cross-attention modules.
- \`injection_downsample\`: Method for downsampling features for cross-attention.
- \`downsamplers\`: Strategy for downsamplers (e.g., \`injection_common\` uses the same downsampler for all injection layers).
- \`causal_fusion\`: If \`True\`, applies a causal mask to cross-attention so the LLM only looks "backwards" in the audio relative to the current token.
- \`gated_cross_attention\`: Adds a learnable tanh gate to the cross-attention output.
- \`positional_info\`: If \`True\`, adds sinusoidal positional embeddings to the audio features.

#### LoRA Configuration
- \`linguistic_lora\`: Enables LoRA on the Language Model.
- \`acoustic_lora\`: Enables LoRA on the Whisper Encoder.
- \`lora_params\`: List of modules to target with LoRA (e.g., \`["q_proj", "v_proj"]\`).
- \`proj_lr\` / \`lora_lr\`: Learning rates for the adapters and LoRA layers respectively.
- \`two_stage\`: Enables the two-stage training strategy (Alignment then Refinement).

#### Auxiliary Tasks
- \`ctc\`: Enables CTC loss on the audio projections.
- \`predict_duration\`: Enables a head to predict the duration of the audio clip.
- \`audio_forecasting\`: Enables a task to predict future acoustic embeddings from LLM states.
- \`ctc_weight\` / \`duration_weight\` / \`audio_weight\`: Loss scaling factors for auxiliary tasks.

#### Prompt & Training
- \`prompt_instruction\`: The system instruction injected into the prompt.
- \`training_args\`: Standard Hugging Face \`TrainingArguments\` (batch size, learning rate, epochs, etc.).

---

## 3. Script Descriptions (\`speechLM_utils/\`)

- \`model.py\`: Factory script to instantiate the model and load weights.
- \`dual_fusion_model.py\`: Defines the main architecture, combining the encoder, decoder, adapters, and cross-attention.
- \`continuous_fusion_utils.py\`: Contains the logic for \`CrossAttention\` and the \`InjectionLayer\` wrapper.
- \`downsamplers.py\`: Implements various strategies to reduce the sequence length of acoustic features.
- \`train.py\`: The entry point for starting a training run, managing stages, and orchestrating the \`MultiLossTrainer\`.
- \`trainers.py\`: Custom \`MultiLossTrainer\` that aggregates and logs multiple loss components.
- \`data_collator.py\`: Prepares batches of audio and text, including padding and auxiliary label generation.
- \`checkpoint.py\`: Logic for checkpoint discovery and resuming training.
- \`metrics.py\`: Computes WER/CER and handles logging of audio samples to Weights & Biases.
