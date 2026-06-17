import os
from os.path import dirname, abspath
import sys
import logging

logging.disable(logging.CRITICAL)
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from speechLM_utils.environment import set_environment

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
set_environment()

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from preprocessing.prepare import get_data

from config.parser import Parser
from preprocessing.split import splitter
from speechLM_utils.utils import init_gpu, resume_wandb, init
from typing import Optional
import os
from preprocessing.prepare import concatenate
import numpy as np
from tqdm import tqdm
from preprocessing.parallel import parallelize_process, concat_dataframes
from evaluations.metrics import get_metrics

def gpu_evaluate(data, gpu_id, args):
    data = Dataset.from_dict(data)
    
    torch_dtype = torch.float16
    batch_size = 4

    model = AutoModelForSpeechSeq2Seq.from_pretrained(args['model_name'], torch_dtype=torch_dtype).to(f"cuda:{gpu_id}")
    processor = AutoProcessor.from_pretrained(args['model_name'])

    gen_kwargs = {
        "language": "hu", 
        "task": "transcribe", 
        "return_timestamps": False,
        "num_beams": 5,
        "repetition_penalty": 1.15,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 4
    }

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=f"cuda:{gpu_id}",
        chunk_length_s=30.0,
        batch_size=batch_size,
        generate_kwargs=gen_kwargs
    )

    predictions = []

    def yield_data():
        for i, item in enumerate(data):
            audio_data = item["audio"]

            if isinstance(audio_data["array"], list):
                audio_data["array"] = np.array(audio_data["array"], dtype=np.float32)
            yield audio_data

    for out in tqdm(pipe(yield_data(),
                        batch_size=batch_size,
                        generate_kwargs=gen_kwargs),
                        position=gpu_id,
                        total=len(data),
                        desc=f"GPU {gpu_id}"):

        predictions.append(out['text'].strip())

    references = [t.strip() for t in data["reference"]]

    indices = data['index']
    durations = data['duration']
    res_samples, _ = get_metrics(predictions, references, indices, durations, verbose=False)

    samples_path = os.path.join(args['res_folder'], f"predictions_{gpu_id}.csv")
    res_samples.to_csv(samples_path)

    return samples_path

def get_results_path(conf, args, dataset: str, split: str):
    results_path = os.path.join(os.path.expanduser('~'),
                                conf.results_path,
                                args['model_type'],
                                args['model_name'],
                                dataset,
                                split)

    os.makedirs(results_path, exist_ok=True)

    return results_path

def get_bad_folder_path(conf, dataset: str, split: str):
    bad_folder_path = os.path.join(os.path.expanduser('~'),
                                   conf.dataset_path,
                                   conf.language,
                                   'bad_folder',
                                   dataset,
                                   split)

    return bad_folder_path

def get_total_metrics(base_filename: str):
    merged_df = pd.read_csv(f"{base_filename}.csv")

    predictions = merged_df['prediction'].fillna("").astype(str).tolist()
    references = merged_df['reference'].fillna("").astype(str).tolist()
    indices = merged_df['index'].tolist()
    durations = merged_df['duration'].tolist()

    res_samples, total_results = get_metrics(predictions, references, indices, durations)

    res_samples.to_csv(f"{base_filename}.csv", index=False)
    total_results.to_csv(f"{base_filename}_total_metrics.csv", index=False)

def evaluate(args, dataset: str, split: str = 'test'):
    conf = Parser()
    conf.get_args()

    args['res_folder'] = get_results_path(conf, args, dataset, split)
    bad_folder = get_bad_folder_path(conf, dataset, split)

    split_manager = splitter()
    data = split_manager.split(datasets=[dataset], test_split=args['test_split'])
    evaluation_data = get_data(data[split], bad_folder, has_duration=True)
    evaluation_data = concatenate(evaluation_data)

    print(f"Evaluating on {dataset} ({len(evaluation_data)} samples)...")

    torch.multiprocessing.set_start_method('spawn', force=True)
    parallelize_process(evaluation_data, gpu_evaluate, gpus=[0,1,2,3], info=args)

    print(f"Merging GPU prediction files for {dataset}...")
    base_filename = os.path.join(args['res_folder'], "predictions")

    concat_dataframes(base_filename, remove=True)
    print(f"✅ Final merged predictions saved to: {base_filename}.csv")

    get_total_metrics(base_filename)
    print(f"✅ Unified metrics saved to: {base_filename}_total_metrics.csv\n")

import argparse
if __name__ == "__main__":
    print('Starting...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='GPUs to be used')
    parser.add_argument('--datasets', nargs='+', type=str, default=['common_voice', 'fleurs', 'massive', 'voxpopuli', 'yodas'], help='datasets')
    parser.add_argument('--splits', nargs='+', type=str, default=['train', 'validation', 'test'], help='split sets')
    parser.add_argument('--model_type', type=str, default='whisper')
    parser.add_argument('--model_name', type=str, default='Trendency/whisper-large-v3-hu')
    parser.add_argument('--test_split', type=float, default=0, help='test split percentage')

    args, unknown = parser.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    args_dict = vars(args)

    for dataset in args.datasets:
        for split in args.splits:
            # local_rank, device = init_gpu()
            evaluate(args_dict, dataset, split)

