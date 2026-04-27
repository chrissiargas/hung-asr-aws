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
from evaluations.metrics import get_metrics
import nemo.collections.asr as nemo_asr
from config.parser import Parser
from preprocessing.split import splitter
from preprocessing.prepare import get_data
from preprocessing.parallel import parallelize_process

def pipe_test_(data, gpu_id, info):
    device = torch.device(f"cuda:{gpu_id}")

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=info['model'], map_location='cpu')
    model = model.to(device)
    model.eval()

    if hasattr(model, 'decoding') and hasattr(model.decoding, 'decoding'):
        comp = getattr(model.decoding.decoding, 'decoding_computer', None)
        if comp and hasattr(comp, 'disable_cuda_graphs'):
            comp.disable_cuda_graphs()

    filepaths = [entry['audio_filepath'] for entry in data]
    references = [entry['text'] for entry in data]

    output = model.transcribe(
        audio=filepaths,
        batch_size=4,
        verbose=True,
        return_hypotheses=True
    )

    transcripts = [out.text for out in output]

    res_samples, _ = get_metrics(transcripts, references)

    samples_path = os.path.join(info['res_folder'], info['dataset'],  f"predictions_{gpu_id}.csv")
    res_samples.to_csv(samples_path)

    return samples_path

def pipe_test(dataset_names, info):
    conf = Parser()
    conf.get_args()

    split = splitter()
    data = split.split(datasets=dataset_names)
    test_sets = get_data(data['test'], process=False)
    test = test_sets[dataset_names[0]]

    print(f"Evaluating on {dataset_names} ({len(test)} samples)...")

    torch.multiprocessing.set_start_method('spawn', force=True)
    parallelize_process(test, pipe_test_, gpus=[0,1,2,3], info=info)