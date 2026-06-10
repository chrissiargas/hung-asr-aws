import json
import torch
import numpy as np
from pathspec import patterns
from tqdm import tqdm
from cleanlab import Datalab
from transformers import WhisperModel, WhisperFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration
import librosa
import os
from config.parser import Parser
import pandas as pd
from pathlib import Path
import torchaudio
import re
from preprocessing.normalize import normalize
from preprocessing.parallel import parallelize_process
from typing import Dict, List

SIL_MODEL = "snakers4/silero-vad"
LANG_MODEL = "openai/whisper-tiny"
EMBED_MODEL = "openai/whisper-tiny"

OUTPUT_REPORT = "cleanlab_report"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def check_duration(data, info, bad_folder):
    min_thres = 1
    max_thres = 60

    durations = np.array([entry['duration'] for entry in data])

    if max_thres:
        bad_indices = np.where((durations < min_thres) | (durations > max_thres))[0]
    else:
        bad_indices = np.where(durations < min_thres)[0]

    bad_files = pd.DataFrame(bad_indices, columns=['bad_index'])

    bad_files['filepath'] = bad_files.bad_index.map(lambda x: data[x]['audio_filepath'])
    bad_files['text'] = bad_files.bad_index.map(lambda x: data[x]['text'])
    bad_files['duration'] = bad_files.bad_index.map(lambda x: durations[x])

    dataset = info['dataset']
    split = info['split']

    bad_files.to_csv(os.path.join(bad_folder, f"bad_by_duration_{dataset}_{split}.csv"))

def check_length(data, info, bad_folder):
    min_thres = 2
    max_thres = None

    text_lens = np.array([len(normalize(entry['text'])) for entry in data])

    if max_thres:
        bad_indices = np.where((text_lens < min_thres) | (text_lens > max_thres))[0]
    else:
        bad_indices = np.where(text_lens < min_thres)[0]

    bad_files = pd.DataFrame(bad_indices, columns=['bad_index'])

    bad_files['filepath'] = bad_files.bad_index.map(lambda x: data[x]['audio_filepath'])
    bad_files['text'] = bad_files.bad_index.map(lambda x: data[x]['text'])
    bad_files['normalized'] = bad_files.bad_index.map(lambda x: normalize(data[x]['text']))
    bad_files['length'] = bad_files.bad_index.map(lambda x: text_lens[x])

    dataset = info['dataset']
    split = info['split']

    bad_files.to_csv(os.path.join(bad_folder, f"bad_by_length_{dataset}_{split}.csv"))

def check_ratio(data, info, bad_folder):
    max_thres = 30
    min_thres = 2
    durations = [entry['duration'] for entry in data]

    text_lens = [len(normalize(entry['text'])) for entry in data]
    ratios = np.array(text_lens) / np.array(durations)

    bad_indices = np.where((ratios < min_thres) | (ratios > max_thres))[0]
    bad_files = pd.DataFrame(bad_indices, columns=['bad_index'])

    bad_files['filepath'] = bad_files.bad_index.map(lambda x: data[x]['audio_filepath'])
    bad_files['text'] = bad_files.bad_index.map(lambda x: data[x]['text'])
    bad_files['normalized'] = bad_files.bad_index.map(lambda x: normalize(data[x]['text']))
    bad_files['length'] = bad_files.bad_index.map(lambda x: text_lens[x])
    bad_files['duration'] = bad_files.bad_index.map(lambda x: durations[x])
    bad_files['ratio'] = bad_files.bad_index.map(lambda x: ratios[x])

    dataset = info['dataset']
    split = info['split']

    bad_files.to_csv(os.path.join(bad_folder, f"bad_by_ratio_{dataset}_{split}.csv"))

def get_silence(x, device, model, get_model_timestamps):
    path = x['audio_filepath']

    try:
        audio, sr = torchaudio.load(path)
        audio_dev = audio.to(device)

        speech_timestamps = get_model_timestamps(audio_dev, model, sampling_rate=16000)

        total_speech_samples = sum([t['end'] - t['start'] for t in speech_timestamps])
        total_samples = audio.shape[1]
        speech_ratio = total_speech_samples / total_samples

        return speech_ratio

    except Exception as e:
        print(f"Error processing {path}: {e}")

def check_silence_(data, gpu_id, info):
    device = torch.device(f"cuda:{gpu_id}")

    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )

    vad_model.to(device)
    (get_speech_timestamps, _, _, _, _) = utils

    speech = []
    ratios = []

    progress_bar = tqdm(
        data,
        desc=f"GPU {gpu_id}",
        position=gpu_id,
        leave=True,
        ncols=100,  # Fix width to ensure neat alignment
        colour='green'  # Optional: makes it look nicer
    )

    for entry in progress_bar:
        speech.append(get_silence(entry, device, vad_model, get_speech_timestamps))
        if info['to_ratio']:
            ratios.append(len(normalize(entry['text'], with_signs=False)) / entry['duration'])

    speech = np.array(speech)
    bad_indices = np.where(speech < info['threshold'])[0]

    if info['to_ratio']:
        ratios = np.array(ratios)
        bad_indices = np.where(speech / ratios < info['threshold'])[0]

    bad_files = pd.DataFrame(bad_indices, columns=['bad_index'])

    bad_files['filepath'] = bad_files.bad_index.map(lambda x: data[x]['audio_filepath'])
    bad_files['text'] = bad_files.bad_index.map(lambda x: data[x]['text'])
    bad_files['duration'] = bad_files.bad_index.map(lambda x: data[x]['duration'])
    bad_files['speech_ratio'] = bad_files.bad_index.map(lambda x: speech[x])
    bad_files['text_ratio'] = bad_files.bad_index.map(lambda x: ratios[x])

    dataset = info['dataset']
    split = info['split']

    path = os.path.join(BAD_FOLDER, f"bad_by_silence_{dataset}_{split}_{gpu_id}.csv")
    bad_files.to_csv(path)

    return path

def get_issue(x, patterns):
    text = x['text'].strip()

    if not text:
        return 'empty_text'

    if not patterns['hungarian'].search(text):
        return 'without_hungarian_characters'

    elif patterns['acoustic'].search(text):
       return 'acoustic_tag'

    elif patterns['speaker'].search(text):
        return 'speaker_tag'

    clean_text = re.sub(r'[\s\.,;!\?\'"«»\-]', '', text)
    if len(clean_text) > 0:
        hungarian_chars = patterns['hungarian'].findall(clean_text)
        hungarian_ratio = len(hungarian_chars) / len(clean_text)

        if hungarian_ratio < 0.7:
            return "foreign_text"

    invalid_match = patterns['invalid_chars'].search(text)
    if invalid_match:
        bad_char = invalid_match.group(0)
        return f"invalid_character_detected_'{bad_char}'"

    return 'none'

def check_text(data, info, bad_folder):
    progress_bar = tqdm(
        data,
        leave=True,
        ncols=100,
        colour='green'
    )

    patterns = {
        'acoustic': re.compile(r'[\[\(\<\{].*?[\]\)\>\}]'),
        'speaker': re.compile(r'^[a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ_]+\s*\d*:\s'),
        'digits': re.compile(r'\d+'),
        'hungarian': re.compile(r'[a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ]'),
        'invalid_chars': re.compile(r'[^a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ0-9\s\.,;!\?\'"«»„”\-]')
    }

    issues = []
    for i, entry in enumerate(progress_bar):
        issues.append(get_issue(entry, patterns))

    issues = np.array(issues)
    bad_indices = np.where(issues != 'none')[0]
    bad_files = pd.DataFrame(bad_indices, columns=['bad_index'])

    bad_files['filepath'] = bad_files.bad_index.map(lambda x: data[x]['audio_filepath'])
    bad_files['text'] = bad_files.bad_index.map(lambda x: data[x]['text'])
    bad_files['issue'] = bad_files.bad_index.map(lambda x: issues[x])

    dataset = info['dataset']
    split = info['split']

    bad_files.to_csv(os.path.join(bad_folder, f"bad_by_text_{dataset}_{split}.csv"))

if '__main__' == __name__:
    conf = Parser()
    conf.get_args()

    datasets = ['common_voice', 'fleurs', 'massive', 'voxpopuli', 'yodas', 'dataocean_asr_657', 'dataocean_asr_659']
    splits = ['train', 'validation', 'test']

    bad_folder = os.path.join(os.path.expanduser('~'),
                              conf.dataset_path,
                              conf.language,
                              'bad_folder')

    os.makedirs(bad_folder, exist_ok=True)

    for dataset in datasets:
        for split in splits:
            print(f'Filtering data for {dataset}/{split}')

            info = {
                'dataset': dataset,
                'split': split,
                'to_ratio': True,
                'threshold': 0.1
            }

            manifest_folder = os.path.join(os.path.expanduser('~'), conf.dataset_path, conf.language, dataset, 'manifests')

            file = os.path.join(manifest_folder, f'{conf.language}_{split}.json')
            if not os.path.exists(file):
                print(f"No such split {split} for {dataset}")
                continue

            with open(file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]

            print('checking duration anomalies')
            check_duration(data, info, bad_folder)
            print('checking length anomalies')
            check_length(data, info, bad_folder)
            print('checking ratio anomalies')
            check_ratio(data, info, bad_folder)
            print('checking text anomalies')
            check_text(data, info, bad_folder)
            print()

        # torch.multiprocessing.set_start_method('spawn', force=True)
        # parallelize_process(data, check_silence_, gpus=[0,1,2,3], info=info)
        # concat_dataframes(os.path.join(BAD_FOLDER, f"bad_by_{task}_{dataset}_{split}"))
        #


