# Evaluation Framework

This directory contains the scripts and utilities for evaluating the Greek ASR models. It supports evaluating the core **Dual Fusion** model as well as various baseline models (Whisper, MMS, Canary, Parakeet).

## Core Evaluation Scripts

### 1. `speechLM.py`
The primary entry point for evaluating Dual Fusion models.
*   **Model Loading:** Dynamically loads the model and tokenizer based on the provided checkpoint. Handles weights in `safetensors` or `pytorch_model.bin` formats.
*   **Inference:** Uses Hugging Face's `Seq2SeqTrainer` and `DataCollator` to perform batch inference on test datasets.
*   **WandB Integration:** Logs predictions and global metrics (WER/CER) to Weights & Biases for experiment tracking.
*   **Multi-GPU Support:** Can be launched with `torchrun` or standard CLI arguments to utilize multiple GPUs.

### 2. `metrics.py`
Defines the scoring logic for all evaluation tasks.
*   **Normalization:** Automatically applies text normalization (removing punctuation, lowercase, etc.) via `preprocessing.normalize` to calculate "Normalized" metrics (`n_wer`, `n_cer`).
*   **Standard Metrics:** Calculates Word Error Rate (WER) and Character Error Rate (CER) using the `jiwer` library.
*   **Stability Metrics:** Tracks the length ratio between predictions and references to detect hallucinations or truncations.
*   **Micro-Averaging:** Supports calculating global metrics across entire datasets.

### 3. `utils.py`
Utilities for post-evaluation analysis and visualization.
*   **Aggregation:** `aggregate_results()` collects `results.csv` files from multiple datasets into a single summary table.
*   **Advanced Analysis:**
    *   **Catastrophic Failure Rate:** Percentage of samples with WER > 50%.
    *   **Length Stability:** Standard deviation of prediction-to-reference length ratios.
    *   **Orthographic Gap (OG):** Difference between raw and normalized WER, measuring the model's sensitivity to punctuation and casing.
*   **Visualization:**
    *   Heatmaps for performance across datasets.
    *   Scatter plots for length correlation (ideal for spotting hallucinations).
    *   Distribution histograms for WER per sample.

## Baseline Evaluation Scripts

These scripts provide standardized evaluation pipelines for comparison against existing models:
*   `whisper.py`: Evaluates OpenAI Whisper (Base or Fine-tuned).
*   `mms.py`: Evaluates Meta's Massively Multilingual Speech model.
*   `canary.py` / `parakeet.py`: Evaluates NVIDIA NeMo-based ASR models.

## Usage Example

To evaluate a Dual Fusion model across multiple Greek datasets:

```bash
python evaluations/speechLM.py --gpus 0,1 --model_name "my_experiment" --datetime "May06_12-00" --datasets common_voice fleurs hparl
```

After evaluation, use `utils.py` to generate an aggregated report:

```bash
python evaluations/utils.py
```
