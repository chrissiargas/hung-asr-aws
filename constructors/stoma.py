import json
import os
from typing import Dict
from tqdm import tqdm
from config.parser import Parser
from datasets import load_dataset
from pathlib import Path
import soundfile as sf
import shutil
import librosa
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class stoma:
    def __init__(self):
        self.conf = Parser()
        self.conf.get_args()

        self.load_path = 'aangelakis/STOMA'
        self.target_path = os.path.join(
            os.path.expanduser('~'),
            self.conf.dataset_path,
            self.conf.language,
            'stoma'
        )

        self.dataset = load_dataset(
            self.load_path,
            split='train'
        )

        self.get_loso_splits()

    def get_loso_splits(self):
        speaker_indices = {}
        for idx, sample in enumerate(tqdm(self.dataset, desc="Analyzing speakers for LOSO split")):
            spk = sample['speaker_id']

            if spk not in speaker_indices:
                speaker_indices[spk] = []

            speaker_indices[spk].append(idx)

        sorted_speakers = sorted(list(speaker_indices.keys()))

        self.test_speaker = sorted_speakers[0]
        self.val_speaker = sorted_speakers[1]
        self.train_speakers = sorted_speakers[2:]

        self.splits = {
            'train': [idx for spk in self.train_speakers for idx in speaker_indices[spk]],
            'validation': speaker_indices[self.val_speaker],
            'test': speaker_indices[self.test_speaker]
        }

    def load_stoma_subset(self, split: str):

        if split == 'full':
            split = 'train'
            dataset = self.dataset

        if split == 'train':
            dataset = self.dataset.select(self.splits['train'])

        if split == 'validation':
            dataset = self.dataset.select(self.splits['validation'])

        if split == 'test':
            dataset = self.dataset.select(self.splits['test'])

        audio_dir = Path(os.path.join(self.target_path, 'data', f'{self.conf.language}_{split}_clips'))
        manifest_dir = Path(os.path.join(self.target_path, 'manifests', f"{self.conf.language}_{split}.json"))

        if audio_dir.exists():
            shutil.rmtree(audio_dir)

        if manifest_dir.exists():
            os.remove(manifest_dir)

        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.parent.mkdir(parents=True, exist_ok=True)  # Ensure manifest folder exists too

        manifest_entries = [self.extract_samples(sample, audio_dir, manifest_dir, idx) for
            idx, sample in enumerate(tqdm(dataset))]

        return manifest_entries

    def extract_samples(self, sample, audio_dir: str, manifest_dir: str, idx: int):
        audio_data = sample['audio']['array']
        sampling_rate = sample['audio']['sampling_rate']

        audio_data = librosa.resample(y=audio_data, orig_sr=sampling_rate, target_sr=self.conf.sampling_rate)

        audio_filepath = os.path.join(audio_dir, f'{idx:06d}.wav')

        sf.write(str(audio_filepath), audio_data, self.conf.sampling_rate)
        duration = len(audio_data) / self.conf.sampling_rate

        manifest_entry = {
            'audio_filepath': str(audio_filepath),
            'duration': duration,
            'text': sample['text']
        }

        with open(manifest_dir, 'a', encoding='utf-8') as f:
            f.write(json.dumps(manifest_entry) + '\n')

        return manifest_entry


if __name__ == '__main__':
    extractor = stoma()
    _ = extractor.load_stoma_subset('train')
    _ = extractor.load_stoma_subset('validation')
    _ = extractor.load_stoma_subset('test')