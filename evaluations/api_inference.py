import os
import pandas as pd
import numpy as np
import scipy.io.wavfile as wavf
import assemblyai as aai
from datasets import Dataset
from tqdm import tqdm

from config.parser import Parser
from preprocessing.split import splitter
from preprocessing.prepare import get_data, concatenate
from evaluations.metrics import get_metrics

def get_results_path(conf, name: str, dataset: str, split: str):
    results_path = os.path.join(os.path.expanduser('~'),
                                conf.results_path,
                                name,
                                dataset,
                                split)

    os.makedirs(results_path, exist_ok=True)

    return results_path

def get_bad_folder_path(conf, dataset: str, split: str):
    bad_folder_path = os.path.join(os.path.expanduser('~'),
                                   conf.dataset_path,
                                   conf.language,
                                   'bad_folder')

    return bad_folder_path

def evaluate_assemblyai(args, dataset: str, split: str = 'test'):
    # 1. Initialize AssemblyAI Client
    # You get a free API key when signing up on their site
    aai.settings.api_key = "3d958aca54974245aba2a3cbd2878428"
    transcriber = aai.Transcriber()

    # Configure for Hungarian ASR
    config = aai.TranscriptionConfig(language_code="hu")

    # 2. Load your dataset split using your framework
    conf = Parser()
    conf.get_args()

    args['res_folder'] = get_results_path(conf, 'assemblyai', dataset, split)
    res_folder = args['res_folder']

    bad_folder = get_bad_folder_path(conf, dataset, split)

    if not isinstance(dataset, list):
        dataset_names = [dataset]
    else:
        dataset_names = dataset

    split_manager = splitter()
    data = split_manager.split(datasets=dataset_names)
    evaluation_data = get_data(data[split], bad_folder, normalized=False, has_duration=True, filters=FILTERS,
                               split=split)
    evaluation_data = concatenate(evaluation_data)

    predictions = []
    references = []
    indices = []
    durations = []

    print(f"Sending {dataset} samples to AssemblyAI...")

    # Create a temporary directory to save raw audio files for uploading
    os.makedirs("temp_api_audio", exist_ok=True)

    for item in tqdm(evaluation_data):
        audio_data = item["audio"]
        sr = audio_data["sampling_rate"]
        arr = np.array(audio_data["array"], dtype=np.float32)

        temp_wav_path = f"temp_api_audio/{item['index']}.wav"
        wavf.write(temp_wav_path, sr, arr)

        try:
            transcript = transcriber.transcribe(temp_wav_path, config=config)

            if transcript.status == aai.TranscriptStatus.error:
                predictions.append("")  # Handled as empty if API failed
            else:
                predictions.append(transcript.text.strip())
        except Exception as e:
            predictions.append("")

        references.append(item["reference"].strip())
        indices.append(item["index"])
        durations.append(item["duration"])

        # Clean up local temp file
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

    # 3. Calculate metrics using your native metrics.py script
    res_samples, total_results = get_metrics(predictions, references, indices, durations)

    # Save results to your results folder
    samples_path = os.path.join(res_folder, "predictions.csv")
    total_path = os.path.join(res_folder, "results.csv")

    res_samples.to_csv(samples_path)
    total_results.to_csv(total_path)

    print(f"✅ AssemblyAI Hungarian Benchmarks saved to {res_folder}")

FILTERS = ['duration', 'length']
import argparse

if __name__ == "__main__":
    all_datasets = ['common_voice',
                    'fleurs',
                    'speech_massive',
                    'voxpopuli',
                    'yodas',
                    'dataocean_asr_657',
                    'dataocean_asr_659',
                    'datatang_asr_1',
                    'datatang_asr_2']

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', type=str, default=all_datasets, help='datasets')
    args, unknown = parser.parse_known_args()

    args_dict = {'res_folder': None}
    DATASETS = args.datasets

    for dataset in DATASETS:
        evaluate_assemblyai(args_dict, dataset, split="test")