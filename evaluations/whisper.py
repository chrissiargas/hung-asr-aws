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
import numpy as np
import os
import librosa
from tqdm import tqdm
import pandas as pd
from transformers import pipeline
from config.parser import Parser
from evaluations.metrics import get_metrics
from preprocessing.split import splitter
from preprocessing.prepare import get_data, concatenate
from preprocessing.parallel import parallelize_process
import jiwer
from peft import PeftModel
from preprocessing.normalize import normalize


def get_prediction(x, device, model, processor, info, forced_decoder_ids = None):
    path = x['audio_filepath']

    try:
        audio, sr = librosa.load(path, sr=16000, duration=30.0) #TODO handle longer audio files
        text = x['text'].strip()

        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

        with torch.no_grad():
            if forced_decoder_ids:
                predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, **info['generate_kwargs'])
            else:
                predicted_ids = model.generate(input_features, **info['generate_kwargs'])

            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

            norm_transcription = normalize(transcription, with_signs=False)
            norm_text = normalize(text, with_signs=False)

            transcription = normalize(transcription, with_signs=True)
            text = normalize(text, with_signs=True)

            wer = jiwer.wer(text, transcription)
            n_wer = jiwer.wer(norm_text, norm_transcription)
            cer = jiwer.cer(text, transcription)
            n_cer = jiwer.cer(norm_text, norm_transcription)
            ratio = len(transcription) / len(text)
            norm_ratio = len(norm_transcription) / len(norm_text)

        return wer, n_wer, cer, n_cer, ratio, norm_ratio, transcription, text

    except Exception as e:
        print(f"Error processing {path}: {e}")

def check_predictions_(data, gpu_id, info):
    device = torch.device(f"cuda:{gpu_id}")

    if info['greek_finetuned']:
        base_model = WhisperForConditionalGeneration.from_pretrained(info['base_model']).to(device)
        model = PeftModel.from_pretrained(base_model, info['model']).to(device)
        processor = WhisperProcessor.from_pretrained(info['model'])

    else:
        processor = WhisperProcessor.from_pretrained(info['model'], language="greek", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(info['model']).to(device)

    if info['use_forced_decoder']:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="greek", task="transcribe")
    else:
        forced_decoder_ids = None

    wers = []
    n_wers = []
    cers = []
    n_cers = []
    transcripts = []
    references = []
    ratios = []
    norm_ratios = []

    progress_bar = tqdm(
        data,
        desc=f"GPU {gpu_id}",
        position=gpu_id,
        leave=True,
        ncols=100,  # Fix width to ensure neat alignment
        colour='green'  # Optional: makes it look nicer
    )

    for entry in progress_bar:
        wer, n_wer, cer, n_cer, ratio, norm_ratio, transcript, reference = get_prediction(entry, device, model, processor, info, forced_decoder_ids)

        wers.append(wer)
        n_wers.append(n_wer)
        cers.append(cer)
        n_cers.append(n_cer)
        transcripts.append(transcript)
        references.append(reference)
        ratios.append(ratio)
        norm_ratios.append(norm_ratio)


    results = {'prediction': transcripts, 'reference': references,
               'wer': wers, 'n_wer': n_wers,
               'cer': cers, 'n_cer': n_cers,
               'ratio': ratios, 'n_ratio': norm_ratios}
    results = pd.DataFrame(results)

    path = os.path.join(info['res_folder'], info['dataset'], f"predictions_{gpu_id}.csv")
    results.to_csv(path)

    return path

def entry_test(dataset_names, info):
    conf = Parser()
    conf.get_args()

    split = splitter()
    data = split.split(datasets=dataset_names)
    test_sets = get_data(data['test'], process=False)
    test = test_sets[dataset_names[0]]

    torch.multiprocessing.set_start_method('spawn', force=True)
    parallelize_process(test, check_predictions_, gpus=[0,1,2,3], info=info)

def pipe_test_(data, gpu_id, info):
    data = Dataset.from_dict(data)

    pipe = pipeline(
        "automatic-speech-recognition",
        model = info['model'],
        device = gpu_id,
        torch_dtype = torch.float16,
        chunk_length_s=30.0
    )

    if data is None:
        return

    predictions = []

    def yield_data():
        for i, item in enumerate(data):
            audio_data = item["audio"]

            if isinstance(audio_data["array"], list):
                audio_data["array"] = np.array(audio_data["array"], dtype=np.float32)
            yield audio_data

    for out in tqdm(pipe(yield_data(), batch_size=8, generate_kwargs=info['generate_kwargs']), position=gpu_id, total=len(data), desc=f"GPU {gpu_id}"):
        predictions.append(out['text'].strip())

    references = [t.strip() for t in data["reference"]]

    indices = data['index']
    durations = data['duration']

    res_samples, _ = get_metrics(predictions, references, indices, durations)

    samples_path = os.path.join(info['res_folder'], info['dataset'],  f"predictions_{gpu_id}.csv")
    res_samples.to_csv(samples_path)

    return samples_path

def pipe_test(dataset_names, info):
    conf = Parser()
    conf.get_args()

    split_manager = splitter(splitting=False, validation=False)
    manifests = split_manager.split(datasets=dataset_names)

    data = split.split(datasets=dataset_names)
    test_sets = get_data(data['test'])
    test = concatenate(test_sets)

    print(f"Evaluating on {dataset_names} ({len(test)} samples)...")

    torch.multiprocessing.set_start_method('spawn', force=True)
    parallelize_process(test, pipe_test_, gpus=[0,1,2,3], info=info)







