from config.parser import Parser
import os
import json
from pathlib import Path
from typing import Dict, Optional, List
import random

class splitter:
    def __init__(self, splitting: bool = True, validation: bool = True, exp: int = 0):
        self.conf = Parser()
        self.conf.get_args(exp)

        self.splitting = splitting
        self.validation = validation

        self.seed = 42

    def perform_speaker_split(self, manifest_folder: str, manifest_file: str, dataset: str, split_k: float):
        print(f"[{dataset}] No pre-existing splits found. Performing Speaker-Disjoint Split on the fly...")

        data = []
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))

        speaker_groups = {}
        speaker_durations = {}
        total_duration = 0

        for entry in data:
            speaker_id = entry.get('subject')
            duration = entry.get('duration')

            if speaker_id not in speaker_groups:
                speaker_groups[speaker_id] = []
                speaker_durations[speaker_id] = 0

            speaker_groups[speaker_id].append(entry)
            speaker_durations[speaker_id] += duration
            total_duration += duration

        random.seed(self.seed)
        speakers = list(speaker_groups.keys())
        random.shuffle(speakers)

        splits = {'train': [], 'validation': [], 'test': []}
        current_durations = {'train': 0.0, 'validation': 0.0, 'test': 0.0}

        train_duration = split_k * total_duration
        val_duration = (1 - split_k / 2) * total_duration

        for speaker in speakers:
            if current_durations['train'] < train_duration:
                splits['train'].append(speaker)
                current_durations['train'] += speaker_durations[speaker]
            elif current_durations['validation'] < val_duration:
                splits['validation'].append(speaker)
                current_durations['validation'] += speaker_durations[speaker]
            else:
                splits['test'].append(speaker)
                current_durations['test'] += speaker_durations[speaker]

        generated_paths = {}
        for split, entries in splits.items():
            output_path = os.path.join(manifest_folder, f'{self.conf.language}_{split}.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            generated_paths[split] = output_path

        print(f"[{dataset}] Successfully generated splits.")
        return generated_paths


    def merge_manifests(self, manifest_files: Dict, split: str):
        output_path = os.path.join(self.conf.dataset_path, split + '.json')

        if Path(output_path).exists():
            os.remove(output_path)

        split_files = manifest_files[split]
        for dataset, file in split_files.items():
            if not os.path.exists(file):
                print(f"Warning: File not found - {file}")
                continue

            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    with open(output_path, 'a', encoding='utf-8') as g:
                        entry = json.loads(line)
                        entry['dataset'] = dataset
                        entry['split'] = split
                        g.write(json.dumps(entry) + '\n')

    def split(self, validation: bool = True, merging: bool = False, datasets: Optional[List] = None, split_k: float = 0.8):
        manifests = {
            'train': {},
            'validation': {},
            'test': {}
        }

        datasets = datasets if datasets else self.conf.datasets

        if self.conf.split_type == 'default':
            for dataset in datasets:
                manifest_folder = os.path.join(os.path.expanduser('~'), self.conf.dataset_path, self.conf.language, dataset, 'manifests')
                splits = len(os.listdir(manifest_folder))

                if splits == 1:
                    if split_k > 0:
                        manifest_file = os.path.join(manifest_folder, f'{self.conf.language}_train.json')
                        if os.path.exists(manifest_file):
                            self.perform_speaker_split(manifest_folder, manifest_file, dataset, split_k)

                            for split in ['train', 'validation', 'test']:
                                manifest_file = os.path.join(manifest_folder, f'{self.conf.language}_{split}.json')
                                if os.path.exists(manifest_file):
                                    manifests[split][dataset] = manifest_file

                    else:
                        manifest_file = os.path.join(manifest_folder, f'{self.conf.language}_train.json')
                        manifests['train'][dataset] = manifest_file

                elif splits == 3:
                    for split in ['train', 'validation', 'test']:
                        manifest_file = os.path.join(manifest_folder, f'{self.conf.language}_{split}.json')
                        if os.path.exists(manifest_file):
                            manifests[split][dataset] = manifest_file
                elif splits == 2:
                    raise TypeError("There are only 2 splits: ", os.listdir(manifest_folder))

        if not validation:
            manifests['train'].update(manifests['validation'])
            del manifests['validation']

        if merging:
            self.merge_manifests(manifests, 'train')
            if validation:
                self.merge_manifests(manifests, 'validation')
            self.merge_manifests(manifests, 'test')

        return manifests

    def split_for_train(self):
        if self.splitting:
            self.split(validation=True)

    def split_for_inference(self):
        if self.splitting:
            self.split(validation=True)

import pandas as pd
pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    B = splitter(splitting=True, validation=True)
    B.split_for_train()

    df = pd.read_json(os.path.join(B.conf.dataset_path, 'train.json'), lines=True)
    common_voice = df[df['dataset'] == 'common_voice']
    print(common_voice)




