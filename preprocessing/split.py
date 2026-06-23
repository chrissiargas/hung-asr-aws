from config.parser import Parser
import os
import json
from pathlib import Path
from typing import Dict, Optional, List
import random
import numpy as np

class splitter:
    def __init__(self, splitting: bool = True, validation: bool = True, exp: int = 0):
        self.conf = Parser()
        self.conf.get_args(exp)

        self.splitting = splitting
        self.validation = validation

        self.seed = 42

    def perform_speaker_split(self, manifest_folder: str, manifest_file: str, dataset: str, test_split: float = 0):
        print(f"[{dataset}] No pre-existing splits found. Performing Speaker-Disjoint Split on the fly...")

        data = pd.read_json(manifest_file, lines=True)
        speaker_stats = data.groupby('subject', as_index=False)['duration'].sum()

        speaker_stats['cum_duration'] = speaker_stats['duration'].cumsum()
        total_duration = speaker_stats['duration'].sum()

        train_threshold = (1. - 2. * test_split) * total_duration
        val_threshold = train_threshold + (test_split * total_duration)

        conditions = [
            speaker_stats['cum_duration'] <= train_threshold,
            (speaker_stats['cum_duration'] > train_threshold) & (speaker_stats['cum_duration'] <= val_threshold)
        ]
        choices = ['train', 'validation']
        speaker_stats['split'] = np.select(conditions, choices, default='test')

        splits_dict = speaker_stats.groupby('split')['subject'].apply(list).to_dict()

        print(f"Train Subjects: {splits_dict.get('train', [])}\n"
              f"Validation Subjects: {splits_dict.get('validation', [])}\n"
              f"Test Subjects: {splits_dict.get('test', [])}\n")

        data = data.merge(speaker_stats[['subject', 'split']], on='subject', how='left')

        generated_paths = {}
        for split_name in ['train', 'validation', 'test']:
            split_data = data[data['split'] == split_name].drop(columns=['split'])
            print(f"{split_name} samples: {len(split_data)}")

            output_path = os.path.join(manifest_folder, f'{self.conf.language}_{split_name}.json')
            split_data.to_json(output_path, orient='records', lines=True, force_ascii=False)
            generated_paths[split_name] = output_path

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

    def split(self, validation: bool = True, merging: bool = False, datasets: Optional[List] = None, test_split: float = 0):
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

                if ('dataocean' in dataset) or ('datatang' in dataset):
                    print(f'Checking if {dataset} needs splitting...')
                    
                    if test_split > 0:
                        manifest_file = os.path.join(manifest_folder, f'{self.conf.language}.json')
                        if os.path.exists(manifest_file):
                            self.perform_speaker_split(manifest_folder, manifest_file, dataset, test_split)

                            for split in ['train', 'validation', 'test']:
                                manifest_file = os.path.join(manifest_folder, f'{self.conf.language}_{split}.json')
                                if os.path.exists(manifest_file):
                                    manifests[split][dataset] = manifest_file
                        
                        else:
                            print(f'{manifest_file} Not Found!')

                    else:
                        if splits > 3:
                            for split in ['train', 'validation', 'test']:
                                manifest_file = os.path.join(manifest_folder, f'{self.conf.language}_{split}.json')
                                if os.path.exists(manifest_file):
                                    manifests[split][dataset] = manifest_file
                        else:
                            manifest_file = os.path.join(manifest_folder, f'{self.conf.language}.json')
                            manifests['train'][dataset] = manifest_file

                else:
                    for split in ['train', 'validation', 'test']:
                        manifest_file = os.path.join(manifest_folder, f'{self.conf.language}_{split}.json')
                        if os.path.exists(manifest_file):
                            manifests[split][dataset] = manifest_file

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




