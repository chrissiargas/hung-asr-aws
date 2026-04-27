import json
import os
from typing import Dict
from tqdm import tqdm
from config.parser import Parser
from datasets import load_dataset, Dataset
from pathlib import Path
import soundfile as sf
import shutil
import librosa
from datacollective import DataCollective
import pandas as pd
import subprocess

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
api_key = "a5a7fb5e382a78f691c85763a922b5410f9dda129943d731c79e48b9cafb1ce8"
dataset_id = {'greek':'cmn2cx91x01dno10754vxfu3b', 'hungarian': 'cmj8u3p8900bhnxxb50f37mkm'}

class common_voice:
    def __init__(self, download: bool = False):
        self.conf = Parser()
        self.conf.get_args()

        self.client = DataCollective(api_key=api_key,
                                    download_path=os.path.join(
                                    os.path.expanduser('~'),
                                    self.conf.dataset_path,
                                    self.conf.language,
                                    'common_voice'))

        if download:
            self.client.load_dataset(dataset_id[self.conf.language])


        if self.conf.language == 'greek':
            self.language = 'el'

        elif self.conf.language == 'hungarian':
            self.language = 'hu'

        self.load_path = os.path.join(os.path.expanduser('~'),
                                      self.conf.dataset_path,
                                      self.conf.language,
                                      'common_voice',
                                      'cv-corpus-25.0-2026-03-09',
                                      self.language)

        self.target_path = os.path.join(
            os.path.expanduser('~'),
            self.conf.dataset_path,
            self.conf.language,
            'common_voice'
        )

    def load_common_subset(self, split: str):
        manifest_path = os.path.join(self.load_path, f"{split}.tsv")
        clips_path = os.path.join(self.load_path, "clips")

        dataset = pd.read_csv(manifest_path, sep="\t")
        dataset["audio_path"] = dataset["path"].apply(lambda x: os.path.join(clips_path, x))

        dataset = Dataset.from_pandas(dataset)

        if split == 'dev':
            split = 'validation'

        initial_split = split
        remove_files = False if split == 'other' and self.conf.other_to_train else True
        split = 'train' if split == 'other' and self.conf.other_to_train else split

        audio_dir = Path(os.path.join(self.target_path, 'data', f'{self.conf.language}_{split}_clips'))
        manifest_dir = Path(os.path.join(self.target_path, 'manifests', f"{self.conf.language}_{split}.json"))

        if remove_files:
            if audio_dir.exists():
                shutil.rmtree(audio_dir)

            if manifest_dir.exists():
                os.remove(manifest_dir)

        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.parent.mkdir(parents=True, exist_ok=True)

        if initial_split == 'other' and self.conf.other_to_train:
            with open(manifest_dir, 'r', encoding='utf-8') as f:
                train_manifests = [json.loads(line) for line in f]
            offset = len(train_manifests)
        else:
            offset = 0

        manifest_entries = [self.extract_samples(sample, audio_dir, manifest_dir, offset + idx) for
            idx, sample in enumerate(tqdm(dataset))]

        return manifest_entries

    def extract_samples(self, sample, audio_dir: str, manifest_dir: str, idx: int):
        audio_data, sampling_rate = sf.read(sample['audio_path'])
        audio_data = librosa.resample(y=audio_data, orig_sr=sampling_rate, target_sr=self.conf.sampling_rate)

        audio_filepath = os.path.join(audio_dir, f'{idx:06d}.wav')

        sf.write(str(audio_filepath), audio_data, self.conf.sampling_rate)
        duration = len(audio_data) / self.conf.sampling_rate

        manifest_entry = {
            'audio_filepath': str(audio_filepath),
            'duration': duration,
            'text': sample['sentence']
        }

        with open(manifest_dir, 'a', encoding='utf-8') as f:
            f.write(json.dumps(manifest_entry) + '\n')

        return manifest_entry

if __name__ == '__main__':
    extractor = common_voice(download=False)
    _ = extractor.load_common_subset('train')
    _ = extractor.load_common_subset('dev')
    _ = extractor.load_common_subset('test')
    _ = extractor.load_common_subset('other')

