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

from tqdm import tqdm

import torch
from config.parser import Parser
from evaluations.metrics import get_metrics
from preprocessing.split import splitter
from preprocessing.prepare import get_data
from preprocessing.parallel import parallelize_process

from transformers import (
    Wav2Vec2ForCTC,
    AutoProcessor, pipeline
)
from datasets import Dataset
import numpy as np

def pipe_test_(data, gpu_id, info):
    data = Dataset.from_dict(data)

    model = Wav2Vec2ForCTC.from_pretrained(info['model'])
    processor = AutoProcessor.from_pretrained(info['model'])

    processor.tokenizer.set_target_lang('ell')
    model.load_adapter('ell')

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=gpu_id,
        torch_dtype=torch.float16
    )

    predictions = []

    def yield_data():
        for i, item in enumerate(data):
            audio_data = item["audio"]
            if isinstance(audio_data["array"], list):
                audio_data["array"] = np.array(audio_data["array"], dtype=np.float32)
            yield audio_data

    for out in tqdm(pipe(yield_data(), batch_size=8), position=gpu_id, total=len(data), desc=f"GPU {gpu_id}"):
        predictions.append(out['text'])

    references = [t for t in data["reference"]]

    res_samples, _ = get_metrics(predictions, references)

    samples_path = os.path.join(info['res_folder'], info['dataset'],  f"predictions_{gpu_id}.csv")
    res_samples.to_csv(samples_path)

    return samples_path

def pipe_test(dataset_names, info):
    conf = Parser()
    conf.get_args()

    split = splitter()
    data = split.split(datasets=dataset_names)

    test_sets = get_data(data['test'], process=True)
    test = test_sets[dataset_names[0]]

    print(f"Evaluating on {dataset_names} ({len(test)} samples)...")

    torch.multiprocessing.set_start_method('spawn', force=True)
    parallelize_process(test, pipe_test_, gpus=[0,1,2,3], info=info)