from preprocessing.normalize import normalize
import jiwer
import pandas as pd
from typing import Tuple

def get_metrics(predictions, references, indices, with_total_metrics: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    norm_predictions = [normalize(t, with_signs=False) for t in predictions]
    norm_references = [normalize(t, with_signs=False) for t in references]

    predictions = [normalize(t, with_signs=True) for t in predictions]
    references = [normalize(t, with_signs=True) for t in references]

    wers = [jiwer.wer(reference, prediction) for reference, prediction in zip(references, predictions)]
    norm_wers = [jiwer.wer(reference, prediction) for reference, prediction in zip(norm_references, norm_predictions)]
    cers = [jiwer.cer(reference, prediction) for reference, prediction in zip(references, predictions)]
    norm_cers = [jiwer.cer(reference, prediction) for reference, prediction in zip(norm_references, norm_predictions)]
    ratio = [len(prediction) / len(reference) if len(reference) > 0 else 0 for reference, prediction in
             zip(references, predictions)]
    norm_ratio = [len(prediction) / len(reference) if len(reference) > 0 else 0 for reference, prediction in
                  zip(norm_references, norm_predictions)]

    results_per_instance = {'index': indices,
                            'reference': references,
                            'prediction': predictions,
                            'wer': wers, 'n_wer': norm_wers,
                            'cer': cers, 'n_cer': norm_cers,
                            'ratio': ratio, 'n_ratio': norm_ratio}

    results_per_instance = pd.DataFrame(results_per_instance)

    total_results = None
    if with_total_metrics:
        total_wer = jiwer.wer(references, predictions)
        total_n_wer = jiwer.wer(norm_references, norm_predictions)
        total_cer = jiwer.cer(references, predictions)
        total_n_cer = jiwer.cer(norm_references, norm_predictions)

        print("-" * 50)
        print(f"Global Metrics (Micro-Average over {len(predictions)} samples):")
        print(f"Total WER  : {total_wer:.4f}")
        print(f"Total N_WER: {total_n_wer:.4f}")
        print(f"Total CER  : {total_cer:.4f}")
        print(f"Total N_CER: {total_n_cer:.4f}")
        print("-" * 50)

        total_results = {
            'wer': [total_wer],
            'n_wer': [total_n_wer],
            'cer': [total_cer],
            'n_cer': [total_n_cer]
        }

        total_results = pd.DataFrame(total_results)
    
    return results_per_instance, total_results