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
from training.data import get_data

from config.parser import Parser
from preprocessing.irregularities import bad_folder
from preprocessing.split import splitter
from speechLM_utils.utils import init_gpu, resume_wandb, init
from typing import Optional
import os
from preprocessing.prepare import concatenate
import numpy as np
from tqdm import tqdm
from preprocessing.parallel import parallelize_process
from evaluations.metrics import get_metrics

def gpu_evaluate(data, gpu_id, args):
    data = Dataset.from_dict(data)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=args['model_name'],
        device=gpu_id,
        torch_dtype=torch.float16,
        chunk_length_s=30.0
    )

    predictions = []

    def yield_data():
        for i, item in enumerate(data):
            audio_data = item["audio"]

            if isinstance(audio_data["array"], list):
                audio_data["array"] = np.array(audio_data["array"], dtype=np.float32)
            yield audio_data

    for out in tqdm(pipe(yield_data(), batch_size=8, generate_kwargs=args['generate_kwargs']), position=gpu_id,
                    total=len(data), desc=f"GPU {gpu_id}"):
        predictions.append(out['text'].strip())

    references = [t.strip() for t in data["reference"]]

    res_samples, _ = get_metrics(predictions, references)

    samples_path = os.path.join(args['res_folder'], f"predictions_{gpu_id}.csv")
    res_samples.to_csv(samples_path)

    return samples_path

def get_results_path(conf, args, dataset: str):
    results_path = os.path.join(os.path.expanduser('~'),
                                conf.results_path,
                                args['model_type'],
                                args['model_name'],
                                dataset)

    os.makedirs(results_path, exist_ok=True)

    return results_path

def evaluate(args, dataset: str, set: str = 'test', device: Optional[str] = None):
    conf = Parser()
    conf.get_args()

    args['res_folder'] = get_results_path(args, dataset)

    split_manager = splitter()
    data = split_manager.split(dataset)
    test_data = get_data(data[set], bad_folder)
    test_data = concatenate(test_data)

    print(f"Evaluating on {dataset} ({len(test_data)} samples)...")

    torch.multiprocessing.set_start_method('spawn', force=True)
    parallelize_process(test_data, gpu_evaluate, gpus=[0,1,2,3], info=args)



import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='GPUs to be used')
    parser.add_argument('--datasets', nargs='+', type=str, default=['common_voice', 'fleurs', 'dataocean_asr_657', 'dataocean_asr_659', 'massive', 'voxpopuli', 'yodas'], help='datasets')
    parser.add_argument('--model_type', type=str, default='whisper')
    parser.add_argument('--model_name', type=str, default='sarpba/whisper-hu-large-v3-turbo-finetuned')

    args, unknown = parser.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    for dataset in args.datasets:
        local_rank, device = init_gpu()
        evaluate(args, dataset, set='test', device=device)

