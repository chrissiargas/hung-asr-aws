# Greek-ASR: Dual-Fusion Architecture Deep-Dive

This project implements a state-of-the-art Greek ASR system by fusing **Whisper-v3 (Acoustic)** and **Llama-Krikri-8B (Linguistic)** models. It specifically targets the "orthographic fidelity" gap found in standalone models.

## 🛠️ Core Architectural Components

### 1. Dual-Fusion Model (`speechLM_utils/dual_fusion_model.py`)
The orchestrator of the multimodal pipeline. It manages:
*   **Prompt-Level Fusion:** Prepends projected Whisper embeddings to the LLM's text prompt (SLAM-style).
*   **Injection-Level Fusion:** Uses `InjectionLayer` wrappers to insert Cross-Attention at deep LLM layers (e.g., 10, 20, 30).
*   **Multi-Task Objectives:** Simultaneously optimizes for Language Modeling (LM), Character CTC, Audio Forecasting (MSE), and Duration Prediction.

### 2. Alignment & Masking (`speechLM_utils/continuous_fusion_utils.py`)
Implements the heavy-lifting logic for multimodal interaction:
*   **`CrossAttention`:** The bridge between modalities. It features a **Proportional Alignment Mask** that ensures the LLM's causal attention correctly maps text tokens to the corresponding downsampled acoustic frames.
*   **Gated Fusion:** Employs a learnable $\tanh(g)$ gate to dynamically control the flow of acoustic information, preventing noise from degrading purely linguistic transitions.

### 3. Acoustic Compression (`speechLM_utils/downsamplers.py`)
Provides multiple strategies to reduce the Whisper frame rate (standard 1500 frames/30s) to LLM-compatible lengths:
*   **Static:** `Reshape`, `AvgPool`, `Conv1D`.
*   **Dynamic (CIF):** The `CIFireAdapter` uses a learnable convolution to accumulate "acoustic energy" and fire semantic tokens, preserving temporal granularity while drastically reducing sequence length.

### 4. Training Orchestration (`speechLM_utils/train.py`)
Handles the **Two-Stage Curriculum**:
*   **Stage 1:** Freezes the LLM/Whisper core and trains only the `Adapter` and `CrossAttention` modules to align the modalities.
*   **Stage 2:** Enables **LoRA** (`q_proj`, `v_proj`, etc.) on the LLM to refine linguistic knowledge under acoustic conditioning.
*   **Optimizer Grouping:** Assigns different learning rates to bridge adapters vs. LoRA weights.

## 📊 Evaluation & Metrics

### 1. Inference Engine (`evaluations/speechLM.py`)
The primary benchmarking tool. It uses the `Seq2SeqTrainer` in prediction mode to generate transcripts across Greek test sets (Fleurs, Common Voice, etc.) and logs results to **WandB Tables**.

### 2. Aggregation & Viz (`evaluations/utils.py`)
A comprehensive tool for post-training analysis:
*   **Metrics:** Calculates **WER**, **n_WER** (normalized), **Orthographic Gap (OG)**, and **Relative Error Reduction (RER)**.
*   **Visualizations:** Generates Performance Heatmaps, Error Delta charts (comparing to Whisper baseline), and Length Correlation scatter plots (to detect hallucinations).

## ⚙️ Configuration (`config/my_config.yaml`)
The single source of truth for the system:
*   **`dual_fuse_args`:** Controls architecture toggles (pyramid layers, gated attention, downsampling factor $K$ vs $L$).
*   **`training_args`:** Native Titan RTX optimizations (fp16, gradient accumulation, reentrant-free checkpointing).

## 🗃️ Data Pipeline (`speechLM_utils/data_collator.py`)
The `DataCollator` is custom-built for multimodal batches:
*   Processes raw audio through Whisper's feature extractor.
*   Maps text to a custom character-level vocabulary (`GREEK_CHARS`) for CTC.
*   Synthesizes duration tokens (`<|0.50|>`) for the duration modeling task.

---

## 🔍 Technical Analysis: Potential Bottlenecks & Bugs

1.  **Masking Stability:** In `continuous_fusion_utils.py`, the `argmin` on binary masks can return `0` for fully-unmasked sequences, leading to division-by-zero in the `slope` calculation.
2.  **State Dict Filtering:** `model.py` currently filters for `['projector', 'adapter', 'ctc_head']` when loading Stage 1 weights. It must be updated to include `'downsampler'` to avoid losing learned Conv1D/Linear compression weights.
3.  **Audio Dropout Type:** `apply_audio_dropout` incorrectly uses `tokenizer.eos_token` (string) where `tokenizer.eos_token_id` (int) is required for tensor conversion.
