import json
import torch
import numpy as np
from pathspec import patterns
from tqdm import tqdm
import librosa
import os
from config.parser import Parser
import pandas as pd
from pathlib import Path
import torchaudio
import re
from preprocessing.normalize import normalize, normalize_symbols
from preprocessing.parallel import parallelize_process
from typing import Dict, List

SIL_MODEL = "snakers4/silero-vad"
LANG_MODEL = "openai/whisper-tiny"
EMBED_MODEL = "openai/whisper-tiny"

OUTPUT_REPORT = "cleanlab_report"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def check_duration(data, info, bad_folder, min_thres = 1, max_thres = 30):

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

    bad_files.to_csv(os.path.join(bad_folder, dataset, split, f"bad_by_duration.csv"))

def check_length(data, info, bad_folder, min_thres = 6, max_thres = None):

    text_lens = np.array([len(normalize(entry['text'], with_signs=False)) for entry in data])

    if max_thres:
        bad_indices = np.where((text_lens < min_thres) | (text_lens > max_thres))[0]
    else:
        bad_indices = np.where(text_lens < min_thres)[0]

    bad_files = pd.DataFrame(bad_indices, columns=['bad_index'])

    bad_files['filepath'] = bad_files.bad_index.map(lambda x: data[x]['audio_filepath'])
    bad_files['text'] = bad_files.bad_index.map(lambda x: data[x]['text'])
    bad_files['normalized'] = bad_files.bad_index.map(lambda x: normalize(data[x]['text'], with_signs=False))
    bad_files['length'] = bad_files.bad_index.map(lambda x: text_lens[x])

    dataset = info['dataset']
    split = info['split']

    bad_files.to_csv(os.path.join(bad_folder, dataset, split, f"bad_by_length.csv"))

def check_ratio(data, info, bad_folder, min_thres = 2, max_thres = 30):
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

    bad_files.to_csv(os.path.join(bad_folder, dataset, split, f"bad_by_ratio.csv"))

def get_issue(x, patterns, foreign_max_threshold=0.4):
    text = x['text']
    
    if text is None:
        return 'empty_text'

    foreign_blocks = patterns['foreign_blocks'].findall(text)
    if foreign_blocks:
        total_raw_length = len(text.strip())
        total_foreign_length = sum(len(block.strip()) for block in foreign_blocks)

        foreign_ratio = total_foreign_length / total_raw_length
        if foreign_ratio > foreign_max_threshold:
            return f'too_much_foreign_text_{foreign_ratio:.2f}'

    cleaned_text = normalize(text, with_signs=True)

    if not cleaned_text.strip():
        return 'empty_after_normalization'

    invalid_match = patterns['invalid_chars'].search(cleaned_text)
    if invalid_match:
        bad_char = invalid_match.group(0)
        return f"invalid character detected: '{bad_char}'"

    if not patterns['letters'].search(cleaned_text):
        return 'no_alphabetical_content'

    return 'none'

def check_text(data, info, bad_folder, foreign_max_threshold=0.4):
    progress_bar = tqdm(
        data,
        leave=True,
        ncols=100,
        colour='green'
    )

    patterns = {
        'letters': re.compile(r'[a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ]'),
        'invalid_chars': re.compile(r'[^a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ0-9\s\.,:;!\?\'"«»„”\-%\+\*\/=€$#]'),
        'foreign_blocks': re.compile(r'<lang:[^>]+>(.*?)</lang:[^>]+>', re.IGNORECASE)
    }

    issues = []
    for i, entry in enumerate(progress_bar):
        issues.append(get_issue(entry, patterns, foreign_max_threshold))

    issues = np.array(issues)
    bad_indices = np.where(issues != 'none')[0]
    bad_files = pd.DataFrame(bad_indices, columns=['bad_index'])

    bad_files['filepath'] = bad_files.bad_index.map(lambda x: data[x]['audio_filepath'])
    bad_files['text'] = bad_files.bad_index.map(lambda x: data[x]['text'])
    bad_files['issue'] = bad_files.bad_index.map(lambda x: issues[x])

    dataset = info['dataset']
    split = info['split']

    bad_files.to_csv(os.path.join(bad_folder, dataset, split, f"bad_by_text.csv"))

def run_check(datasets, splits):
    conf = Parser()
    conf.get_args()
    print('Loaded config file')

    bad_folder = os.path.join(os.path.expanduser('~'),
                              conf.dataset_path,
                              conf.language,
                              'bad_folder')

    os.makedirs(bad_folder, exist_ok=True)

    DEFAULT_THRESHOLDS = {
        'duration': {'min_thres': 1, 'max_thres': 30},
        'length': {'min_thres': 2, 'max_thres': None},
        'ratio': {'min_thres': 2, 'max_thres': 30},
        'text': {'foreign_max_threshold': 0.5}
    }

    # 2. Define Custom Thresholds for dataocean_asr_659
    # (Modify these values to your specific requirements)
    DATAOCEAN_659_THRESHOLDS = {
        'duration': {'min_thres': 2, 'max_thres': 30},
        'length': {'min_thres': 8, 'max_thres': None},
        'ratio': {'min_thres': 1, 'max_thres': 25},
        'text': {'foreign_max_threshold': 0.4}
    }

    # Map dataset names to their configurations
    DATASET_CONFIGS = {
        'dataocean_asr_659': DATAOCEAN_659_THRESHOLDS
    }

    for dataset in datasets:
        dataset_thresholds = DATASET_CONFIGS.get(dataset, DEFAULT_THRESHOLDS)

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

            to_folder = Path(os.path.join(bad_folder, dataset, split))
            to_folder.mkdir(parents=True, exist_ok=True)

            print('checking duration anomalies')
            check_duration(data, info, bad_folder, **dataset_thresholds['duration'])
            print('checking length anomalies', )
            check_length(data, info, bad_folder, **dataset_thresholds['length'])
            print('checking ratio anomalies')
            check_ratio(data, info, bad_folder, **dataset_thresholds['ratio'])
            print('checking text anomalies')
            check_text(data, info, bad_folder, **dataset_thresholds['text'])
            print()

import argparse
if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', type=str, default=['dataocean_asr_657', 'dataocean_asr_659'], help='datasets')
    parser.add_argument('--splits', nargs='+', type=str, default=['train', 'validation', 'test'], help='datasets')
    args, unknown = parser.parse_known_args()

    args_dict = vars(args)
    datasets = args_dict['datasets']
    splits = args_dict['splits']

    run_check(datasets, splits)


