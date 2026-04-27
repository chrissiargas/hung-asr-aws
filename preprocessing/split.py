from config.parser import Parser
import os
import json
from pathlib import Path
from typing import Dict, Optional, List


class splitter:
    def __init__(self, splitting: bool = True, validation: bool = True, exp: int = 0):
        self.conf = Parser()
        self.conf.get_args(exp)
        self.splitting = splitting
        self.validation = validation

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

    def split(self, validation: bool = True, merging: bool = False, datasets: Optional[List] = None):
        manifests = {
            'train': {},
            'validation': {},
            'test': {}
        }

        datasets = datasets if datasets else self.conf.datasets

        if self.conf.split_type == 'default':
            for dataset in datasets:
                for split in ['train', 'validation', 'test']:
                    manifest_folder = os.path.join(os.path.expanduser('~'), self.conf.dataset_path, self.conf.language, dataset, 'manifests')
                    file = os.path.join(manifest_folder, f'greek_{split}.json')
                    if not os.path.exists(file):
                        print(f"Warning: File not found - {file}")
                        continue

                    manifests[split][dataset] = file

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
    fleurs = df[df['dataset'] == 'common_voice']
    print(fleurs)




