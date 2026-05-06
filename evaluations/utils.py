import os

import safetensors.torch
import torch
from os.path import dirname, abspath
import sys

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from config.parser import Parser
from evaluations.speechLM import get_results_path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import argparse
from bert_score import score

def generate_semantic_drift_plot(info: Dict):
    conf = Parser()
    conf.get_args()

    for dataset_name in DATASETS:
        local_info = info.copy()
        local_info['test_dataset'] = dataset_name
        results_folder = get_results_path(conf, local_info)
        predictions_path = os.path.join(results_folder, 'predictions.csv')

        if os.path.exists(predictions_path):
            df = pd.read_csv(predictions_path)

        df['reference'] = df['reference'].fillna('')
        df['prediction'] = df['prediction'].fillna('')

        refs = df['reference'].tolist()
        preds = df['prediction'].tolist()

        print("Calculating BERTScore on GPU (using multilingual model for Greek)...")
        P, R, F1 = score(
            preds,
            refs,
            lang="el",
            model_type="bert-base-multilingual-cased",
            batch_size=32,
            verbose=False
        )

        df['semantic_f1'] = F1.tolist()

        plot_df = df[df['wer'] <= 10.0].copy()

        plt.figure(figsize=(11, 8))

        sns.scatterplot(
            data=plot_df,
            x='wer',
            y='semantic_f1',
            hue='is_looping',
            palette={True: '#e74c3c', False: '#3498db'},  # Red for loops, Blue for healthy text
            alpha=0.7,
            edgecolor='w',
            s=70
        )

        # Draw quadrant guidelines
        plt.axhline(y=0.80, color='#2ecc71', linestyle='--', alpha=0.8, label='Acceptable Semantics')
        plt.axvline(x=0.20, color='#f39c12', linestyle='--', alpha=0.8, label='Acceptable WER')

        # Annotate the specific regions to explain the "Lazy Decoder" problem
        plt.text(0.02, 0.95, 'Top-Left\nPerfect', color='green', fontsize=10, alpha=0.8)
        plt.text(1.2, 0.95, 'Top-Right\nParaphrasing / Grammar Fixes\n(Lazy Decoder)', color='orange', fontsize=10,
                 alpha=0.8)
        plt.text(1.2, 0.40, 'Bottom-Right\nCatastrophic Hallucination\n(Lost Meaning)', color='red', fontsize=10, alpha=0.8)

        # Formatting
        plt.title(f"Semantic Drift Analysis - {dataset_name}\n(Whisper Acoustics vs KriKri Semantics)", fontsize=14, pad=15)
        plt.xlabel("Word Error Rate (WER) ➔ Lower is Better", fontsize=12)
        plt.ylabel("Semantic F1 (BERTScore) ➔ Higher is Better", fontsize=12)

        # Customize Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        # Filter out the seaborn legend artifacts
        valid_handles = [h for h, l in zip(handles, labels) if
                         l in ['False', 'True', 'Acceptable Semantics', 'Acceptable WER']]
        valid_labels = [l for l in labels if l in ['False', 'True', 'Acceptable Semantics', 'Acceptable WER']]
        plt.legend(valid_handles, valid_labels, title="Caught in Generation Loop", loc="lower left")

        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        # 5. Save the plot
        save_dir = get_plots_dir(conf, info, 'semantic_drift')
        save_path = os.path.join(save_dir, f"{dataset_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved successfully to: {save_path}\n")

def get_plots_dir(conf, info, viz):
    base_res_dir = get_results_path(conf, info, data_folder=False)
    plots_dir = os.path.join(base_res_dir, 'plots', viz)
    os.makedirs(plots_dir, exist_ok=True)

    return plots_dir


def aggregate_results(info: Dict):
    conf = Parser()
    conf.get_args()

    print("Aggregating results...")

    all_results = []
    for dataset_name in DATASETS:
        info['test_dataset'] = dataset_name
        results_folder = get_results_path(conf, info)
        info['res_folder'] = results_folder

        results_path = os.path.join(results_folder, 'results.csv')
        predictions_path = os.path.join(results_folder, 'predictions.csv')

        if os.path.exists(results_path):
            try:
                df = pd.read_csv(results_path, index_col=0)
                row_data = df.iloc[0].to_dict()
                row_data['dataset'] = dataset_name

                # Calculate Orthographic Gap (OG)
                row_data['og'] = row_data['wer'] - row_data['n_wer']

                # Calculate Catastrophic Rate (CR)
                pred_df = pd.read_csv(predictions_path)
                row_data['catastrophic_rate'] = (pred_df['wer'] > 0.5).mean()

                all_results.append(row_data)
            except Exception as e:
                print(f"Error reading {results_path}: {e}")
        else:
            print(f"Warning: Could not find results for {dataset} at {results_path}")

    if not all_results:
        return pd.DataFrame()

    aggregated_df = pd.DataFrame(all_results)

    # Convert to percentages for readability
    perc_cols = ['wer', 'n_wer', 'cer', 'n_cer', 'og', 'catastrophic_rate']
    for col in perc_cols:
        if col in aggregated_df.columns:
            aggregated_df[col] = (aggregated_df[col] * 100).round(1)

    cols = ['dataset'] + [col for col in aggregated_df.columns if col != 'dataset']
    aggregated_df = aggregated_df[cols]

    results_folder = get_results_path(conf, info, data_folder=False)
    results_path = os.path.join(results_folder, "aggregated_results.csv")
    aggregated_df.to_csv(results_path, index=False)

    return aggregated_df


def compare_models(df_model1, df_model2, name_model1="Model_1", name_model2="Model_2"):
    if df_model1.empty or df_model2.empty:
        return pd.DataFrame()

    comparison_df = pd.merge(
        df_model1,
        df_model2,
        on='dataset',
        suffixes=(f'_{name_model1}', f'_{name_model2}')
    )

    metrics = ['wer', 'n_wer', 'cer', 'n_cer']

    for metric in metrics:
        col1 = f'{metric}_{name_model1}'
        col2 = f'{metric}_{name_model2}'

        if col1 in comparison_df.columns and col2 in comparison_df.columns:
            diff_col = f'{metric}_diff'
            comparison_df[diff_col] = (comparison_df[col2] - comparison_df[col1]).round(2)

            # Relative Error Reduction (RER)
            rer_col = f'{metric}_rer'
            comparison_df[rer_col] = (
                        (comparison_df[col1] - comparison_df[col2]) / (comparison_df[col1] + 1e-9) * 100).round(1)

    return comparison_df


# --- Visualization Functions ---
def plot_sid_stacked_bar(info: Dict):
    conf = Parser()
    conf.get_args()

    sid_data = []

    for dataset_name in DATASETS:
        local_info = info.copy()
        local_info['test_dataset'] = dataset_name
        results_folder = get_results_path(conf, local_info)
        predictions_path = os.path.join(results_folder, 'predictions.csv')

        if os.path.exists(predictions_path):
            df = pd.read_csv(predictions_path)
            total_subs = df['substitutions'].sum()
            total_ins = df['insertions'].sum()
            total_dels = df['deletions'].sum()
            total_errors = total_subs + total_ins + total_dels

            if total_errors > 0:
                sid_data.append({
                    'Dataset': dataset_name,
                    'Substitutions': (total_subs / total_errors) * 100,
                    'Insertions': (total_ins / total_errors) * 100,
                    'Deletions': (total_dels / total_errors) * 100
                })

    if not sid_data:
        print("No S-I-D data found to plot.")
        return

    sid_df = pd.DataFrame(sid_data).set_index('Dataset')

    # Plotting
    ax = sid_df.plot(kind='bar', stacked=True, figsize=(10, 6),
                     color=['#ffb347', '#ff6961', '#aec6cf'], edgecolor='black')

    plt.title("Proportion of Error Types per Dataset (S-I-D)", fontsize=14, pad=15)
    plt.ylabel("Percentage of Total Errors (%)", fontsize=12)
    plt.xlabel("Dataset", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Error Type", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add percentage labels inside the bars
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if height > 5:  # Only label if the chunk is big enough
            ax.text(x + width / 2, y + height / 2, f'{height:.1f}%',
                    horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_wer_vs_duration(info: Dict):
    conf = Parser()
    conf.get_args()

    local_info = info.copy()
    for dataset_name in DATASETS:
        local_info['test_dataset'] = dataset_name
        results_folder = get_results_path(conf, local_info)
        predictions_path = os.path.join(results_folder, 'predictions.csv')

        if not os.path.exists(predictions_path):
            print(f"No predictions found for {dataset_name} at {predictions_path}")
            return

        df = pd.read_csv(predictions_path)

        if 'duration' not in df.columns:
            print(f"Warning: 'duration' column not found in predictions for {dataset_name}. Update metrics.py.")
            return

        # Filter out extreme outliers (WER > 1.5) to keep the trendline accurate
        plot_df = df[df['wer'] <= 1.5]

        plt.figure(figsize=(10, 6))

        # Use seaborn's regplot to automatically calculate and plot the trendline
        sns.regplot(data=plot_df, x='duration', y='wer',
                    scatter_kws={'alpha': 0.5, 'color': '#1f77b4'},
                    line_kws={'color': 'red', 'linewidth': 2})

        # Add a vertical line at 30 seconds (Whisper's typical maximum context)
        plt.axvline(x=30.0, color='black', linestyle='--', alpha=0.7, label='Whisper 30s Limit')

        plt.title(f"WER vs. Audio Duration - {dataset_name}", fontsize=14)
        plt.xlabel("Audio Duration (Seconds)", fontsize=12)
        plt.ylabel("Word Error Rate (WER)", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()
        plt.close()


def plot_length_correlation(info: Dict):
    conf = Parser()
    conf.get_args()

    local_info = info.copy()

    for dataset_name in DATASETS:
        local_info['test_dataset'] = dataset_name

        results_folder = get_results_path(conf, local_info)
        predictions_path = os.path.join(results_folder, 'predictions.csv')

        if not os.path.exists(predictions_path):
            print(f"No predictions found for {dataset_name} at {predictions_path}")
            return

        df = pd.read_csv(predictions_path)

        # FIX: Handle pandas NaN loading for empty strings
        df['reference'] = df['reference'].fillna('')
        df['prediction'] = df['prediction'].fillna('')

        df['ref_len'] = df['reference'].str.len()
        df['pred_len'] = df['prediction'].str.len()

        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df, x='ref_len', y='pred_len', alpha=0.5)

        # FIX: Safe max calculation in case df is entirely empty
        max_val = max(df['ref_len'].max(), df['pred_len'].max()) if not df.empty else 100

        plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal (y=x)')
        plt.xlabel("Reference Length (chars)")
        plt.ylabel("Prediction Length (chars)")
        plt.title(f"Length Correlation - {dataset_name}")
        plt.legend()
        plt.tight_layout()

        save_dir = get_plots_dir(conf, info, 'length_correlation')
        save_path = os.path.join(save_dir, f"{dataset_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_wer_distribution(info: Dict):
    conf = Parser()
    conf.get_args()

    local_info = info.copy()

    for dataset_name in DATASETS:
        local_info['test_dataset'] = dataset_name

        results_folder = get_results_path(conf, local_info)
        predictions_path = os.path.join(results_folder, 'predictions.csv')

        if not os.path.exists(predictions_path):
            print(f"No predictions found for {dataset_name} at {predictions_path}")
            return

        df = pd.read_csv(predictions_path)

        # Prevent crash if WER column is somehow missing or empty
        if 'wer' not in df.columns or df.empty:
            return

        plt.figure(figsize=(10, 6))
        sns.histplot(df['wer'], bins=20, kde=True, color='skyblue')
        plt.axvline(df['wer'].mean(), color='red', linestyle='--', label=f'Mean: {df["wer"].mean():.2f}')
        plt.xlabel("Word Error Rate (WER)")
        plt.title(f"WER Distribution - {dataset_name}")
        plt.legend()
        plt.tight_layout()

        save_dir = get_plots_dir(conf, info, 'wer_distribution')
        save_path = os.path.join(save_dir, f"{dataset_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='dual_fusion')
    parser.add_argument('--datasets', nargs='+', type=str,
                        default=['common_voice', 'fleurs', 'hparl', 'tedx', 'logotypographia'], help='datasets')
    parser.add_argument('--train_datasets', nargs='+', type=str,
                        default=['common_voice', 'fleurs', 'hparl', 'tedx', 'logotypographia'],
                        help='datasets of the trained models')
    parser.add_argument('--checkpoint_folder', type=str, default='dual_fusion_checkpoints')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--speech_encoder_id', type=str, default='openai/whisper-large-v3')
    parser.add_argument('--language_model_id', type=str, default='ilsp/Llama-Krikri-8B-Instruct')
    parser.add_argument('--machine', type=str, default='kronos')
    parser.add_argument('--datetime', type=str, default='')
    parser.add_argument('--turn', type=str, default=None)

    args, unknown = parser.parse_known_args()
    DATASETS = args.datasets

    base_info = {
        'model_type': args.model_type,
        'res_folder': None,
        'train_dataset': args.train_datasets,
        's_': False,
        'checkpoint_folder': args.checkpoint_folder,
        'model_name': args.model_name,
        'speech_encoder_id': args.speech_encoder_id,
        'language_model_id': args.language_model_id,
        'bit4': True,
        'machine': args.machine,
        'datetime': args.datetime,
        'turn': args.turn
    }

    if base_info['model_name'] is None:
        base_info['model_name'] = (base_info['speech_encoder_id'].split('/')[1] + '_' +
                                   base_info['language_model_id'].split('/')[1])

    df1 = plot_length_correlation(base_info)