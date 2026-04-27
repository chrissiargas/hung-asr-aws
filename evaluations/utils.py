from config.parser import Parser
from evaluations.speechLM import get_results_path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_advanced_metrics(predictions_df):
    """Calculates stability and failure rates from per-instance predictions."""
    if predictions_df.empty:
        return {}

    # Catastrophic failure: WER > 50%
    catastrophic_rate = (predictions_df['wer'] > 0.5).mean()

    # Length stability: std of the ratio
    length_stability = predictions_df['ratio'].std()

    return {
        'catastrophic_rate': catastrophic_rate,
        'length_stability': length_stability
    }

def aggregate_results():
    conf = Parser()
    conf.get_args()

    if INFO['model_name'] is None:
        INFO['model_name'] = (INFO['speech_encoder_id'].split('/')[1] + '_' +
                              INFO['language_model_id'].split('/')[1])

    print("Aggregating results...")

    all_results = []
    for dataset in DATASETS:
        INFO['test_dataset'] = dataset
        results_folder = get_results_path(conf, INFO)
        results_path = os.path.join(results_folder, 'results.csv')
        predictions_path = os.path.join(results_folder, 'predictions.csv')

        if os.path.exists(results_path):
            try:
                df = pd.read_csv(results_path, index_col=0)
                row_data = df.iloc[0].to_dict()
                row_data['dataset'] = dataset

                # Calculate Orthographic Gap (OG)
                row_data['og'] = row_data['wer'] - row_data['n_wer']

                # Add advanced metrics from predictions.csv
                if os.path.exists(predictions_path):
                    pred_df = pd.read_csv(predictions_path)
                    adv_metrics = calculate_advanced_metrics(pred_df)
                    row_data.update(adv_metrics)

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

    results_folder = get_results_path(conf, INFO, data_folder=False)
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
            comparison_df[rer_col] = ((comparison_df[col1] - comparison_df[col2]) / (comparison_df[col1] + 1e-9) * 100).round(1)

    return comparison_df

# --- Visualization Functions ---

def plot_performance_heatmap(aggregated_df, metric='n_wer', title="Model Performance Heatmap"):
    """Plots a heatmap of performance across datasets."""
    plt.figure(figsize=(10, 6))
    pivot_df = aggregated_df.set_index('dataset')[[metric]]
    sns.heatmap(pivot_df.T, annot=True, cmap="YlGnBu", fmt=".1f", cbar_kws={'label': metric + ' (%)'})
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_error_delta(comparison_df, metric='n_wer', name1="Baseline", name2="Dual-Fusion"):
    """Bar chart showing WER difference between models."""
    plt.figure(figsize=(12, 6))
    diff_col = f'{metric}_diff'
    sns.barplot(x='dataset', y=diff_col, data=comparison_df, palette="RdYlGn_r")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel(f"$\Delta$ {metric} (%)")
    plt.title(f"Error Difference: {name2} vs {name1} (Negative is better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_length_correlation(info, dataset_name):
    """Scatter plot of reference vs prediction length to detect hallucinations."""
    conf = Parser()
    conf.get_args()
    INFO['test_dataset'] = dataset_name
    results_folder = get_results_path(conf, INFO)
    predictions_path = os.path.join(results_folder, 'predictions.csv')

    if not os.path.exists(predictions_path):
        print(f"No predictions found for {dataset_name}")
        return

    df = pd.read_csv(predictions_path)
    df['ref_len'] = df['reference'].str.len()
    df['pred_len'] = df['prediction'].str.len()

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x='ref_len', y='pred_len', alpha=0.5)
    max_val = max(df['ref_len'].max(), df['pred_len'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal (y=x)')
    plt.xlabel("Reference Length (chars)")
    plt.ylabel("Prediction Length (chars)")
    plt.title(f"Length Correlation - {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_wer_distribution(info, dataset_name):
    """Histogram of WER per sample to identify edge cases."""
    conf = Parser()
    conf.get_args()
    INFO['test_dataset'] = dataset_name
    results_folder = get_results_path(conf, INFO)
    predictions_path = os.path.join(results_folder, 'predictions.csv')

    if not os.path.exists(predictions_path):
        return

    df = pd.read_csv(predictions_path)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['wer'], bins=20, kde=True, color='skyblue')
    plt.axvline(df['wer'].mean(), color='red', linestyle='--', label=f'Mean: {df["wer"].mean():.2f}')
    plt.xlabel("Word Error Rate (WER)")
    plt.title(f"WER Distribution - {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

DATASETS = ['common_voice', 'fleurs', 'hparl', 'tedx', 'logotypographia']
INFO = {
        'model_type': 'dual_fusion',
        'res_folder': None,
        'train_dataset': ['common_voice', 'fleurs', 'hparl', 'tedx', 'logotypographia'],
        's_': False,
        'checkpoint_folder': "dual_fusion_checkpoints",
        'model_name': None,
        'speech_encoder_id': 'openai/whisper-large-v3',
        'language_model_id': 'ilsp/Llama-Krikri-8B-Instruct',
        'bit4': True,
        'machine': 'kronos',
        'datetime': 'Apr03_16-04',
        'turn': '32500'
    }

if __name__ == "__main__":
    df1 = aggregate_results()
    if not df1.empty:
        print(df1)
        # Example plotting
        # plot_performance_heatmap(df1)