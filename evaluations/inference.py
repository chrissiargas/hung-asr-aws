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

import torch
from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)
from transformers import pipeline
from preprocessing.prepare import get_data

from config.parser import Parser
from preprocessing.irregularities import bad_folder
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

    gen_kwargs = {"language": "hungarian", "task": "transcribe", "return_timestamps": False}

    pipe = pipeline(
        "automatic-speech-recognition",
        model=args['model_name'],
        device=gpu_id,
        torch_dtype=torch.float16,
        chunk_length_s=30.0,
        batch_size=8,
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
                         batch_size=8,
                         generate_kwargs=gen_kwargs),
                    position=gpu_id,
                    total=len(data),
                    desc=f"GPU {gpu_id}"):

        predictions.append(out['text'].strip())

    references = [t.strip() for t in data["reference"]]

    indices = data['index']
    durations = data['duration']
    res_samples, _ = get_metrics(predictions, references, indices, durations)

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

def get_bad_folder_path(conf, dataset: str, set: str):
    bad_folder_path = os.path.join(os.path.expanduser('~'),
                                   conf.dataset_path,
                                   conf.language,
                                   'bad_folder',
                                   dataset,
                                   set)

    return bad_folder_path

def evaluate(args, dataset: str, set: str = 'test'):
    conf = Parser()
    conf.get_args()

    args['res_folder'] = get_results_path(conf, args, dataset, set)
    bad_folder = get_bad_folder_path(conf, dataset, set)

    split_manager = splitter()
    data = split_manager.split(datasets=[dataset], split_k=0)
    evaluation_data = get_data(data[set], bad_folder, has_duration=True)
    evaluation_data = concatenate(evaluation_data)

    print(f"Evaluating on {dataset} ({len(evaluation_data)} samples)...")

    torch.multiprocessing.set_start_method('spawn', force=True)
    parallelize_process(evaluation_data, gpu_evaluate, gpus=[0,1,2,3], info=args)

    print(f"Merging GPU prediction files for {dataset}...")
    base_filename = os.path.join(args['res_folder'], "predictions")

    concat_dataframes(base_filename, remove=True)

    print(f"✅ Final merged predictions saved to: {base_filename}.csv")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='GPUs to be used')
    parser.add_argument('--datasets', nargs='+', type=str, default=['common_voice', 'fleurs', 'massive', 'voxpopuli', 'yodas'], help='datasets')
    parser.add_argument('--splits', nargs='+', type=str, default=['train', 'validation', 'test'], help='split sets')
    parser.add_argument('--model_type', type=str, default='whisper')
    parser.add_argument('--model_name', type=str, default='Trendency/whisper-large-v3-hu')

    args, unknown = parser.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    args_dict = vars(args)

    for dataset in args.datasets:
        for split in args.splits:
            local_rank, device = init_gpu()
            evaluate(args, dataset, set=split)

