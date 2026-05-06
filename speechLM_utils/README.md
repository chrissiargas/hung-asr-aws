# Core Model & Training Utilities (`speechLM_utils`)

This directory is the engine room of the project. It contains the implementation of the **Dual Fusion** architecture, the custom training loops, and the logic for injecting acoustic features into Large Language Models.

## 1. Model Architecture

### `dual_fusion_model.py`
The heart of the project. It defines the `DualFusionModel`, which fuses a **Whisper** acoustic encoder with a **KriKri (Llama-based)** LLM.
- **Input Injection:** Uses a projection adapter to treat downsampled audio features as "audio tokens" in the LLM's prompt.
- **Deep Injection:** Wraps intermediate LLM layers with cross-attention to allow the model to attend to acoustic details during generation.
- **Auxiliary Heads:** Implements predictors for CTC loss, duration prediction, and future audio forecasting.

### `continuous_fusion_utils.py`
Contains the building blocks for deep injection:
- **`CrossAttention`**: A specialized module that allows LLM hidden states to query acoustic embeddings. Supports **Causal Fusion**, ensuring the model only looks at relevant audio segments.
- **`InjectionLayer`**: A wrapper that intercepts standard LLM layers to perform the fusion step.
- **`SinusoidalPositionalEmbedding`**: Adds temporal context to acoustic features.

### `downsamplers.py`
Provides various methods to reduce the 1500-token sequence of Whisper into a manageable size for the LLM:
- `ReshapeAdapter`, `Conv1DAdapter`, `AvgPoolAdapter`, `LinearAdapter`.
- **`CIFireAdapter`**: An implementation of **Continuous Integrate-and-Fire (CIF)** for dynamic, content-aware downsampling.

## 2. Training Orchestration

### `train.py`
The main entry point for training runs.
- **Two-Stage Strategy:** Supports an "Alignment" stage (training only adapters/cross-attention) followed by a "Refinement" stage (enabling LoRA on the LLM).
- **Restart/Resume:** Handles resuming from checkpoints and managing WandB/TensorBoard logging.

### `trainers.py`
Defines the **`MultiLossTrainer`**, a subclass of Hugging Face's `Seq2SeqTrainer`.
- **Loss Aggregation:** Combines the primary Language Modeling loss with auxiliary losses (CTC, Duration, Audio Forecasting).
- **Custom Logging:** Provides detailed tracking of each loss component in the training logs.

## 3. Data & Checkpointing

### `data_collator.py`
A complex collator that prepares multi-modal batches:
- Processes raw audio into Whisper features.
- Tokenizes Greek text and prepares LLM labels.
- Generates targets for auxiliary tasks (CTC character sequences, duration bins).

### `checkpoint.py` & `model.py`
- **`checkpoint.py`**: Logic for automatic checkpoint discovery, loading `config.json`, and managing experiment directories.
- **`model.py`**: Factory functions to instantiate the complex `DualFusionModel` and load weights from multi-shard `safetensors`.

## 4. Support Utilities

- **`metrics.py`**: Orchestrates validation-time evaluation, calculating WER/CER and logging audio-transcript samples to WandB.
- **`utils.py`**: Handles GPU initialization, experiment tagging, and the bridge between CLI arguments and internal configuration.
- **`environment.py`**: Sets up the local environment, including Hugging Face cache and temp directories.
- **`training_info.py`**: Contains default configuration templates for different model variants.

---

## The Dual Fusion Mechanism

The architecture follows a dual-path injection strategy:
1. **Prompt Path:** Audio is downsampled by factor $ and projected into the LLM's embedding space.
2. **Cross-Attention Path:** Audio is downsampled by factor $ and injected into specific LLM layers (e.g., 8, 16, 24) via cross-attention.

This allows the model to have both a high-level "understanding" of the audio as part of the instructions and a low-level "listening" capability during the decoding process.
