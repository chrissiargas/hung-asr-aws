import json
import os
from typing import Dict
from tqdm import tqdm
from config.parser import Parser
from datasets import load_dataset
from pathlib import Path
import soundfile as sf
import librosa
import shutil
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class hparl:
    def __init__(self):
        self.conf = Parser()
        self.conf.get_args()

        self.load_path = 'Elormiden/Hellenic-greek-parliamentary-speech'
        self.target_path = os.path.join(
            os.path.expanduser('~'),
            self.conf.dataset_path,
            self.conf.language,
            'hparl'
        )

        if self.conf.language != 'greek':
            print('Error: This dataset only supports Greek language.')
            exit(1)

    def load_hparl_subset(self, split: str):
        dataset = load_dataset(
            self.load_path,
            split=split,
            token=False
        )

        audio_dir = Path(os.path.join(self.target_path, 'data', f'{self.conf.language}_{split}_clips'))
        manifest_dir = Path(os.path.join(self.target_path, 'manifests', f"{self.conf.language}_{split}.json"))

        if audio_dir.exists():
            shutil.rmtree(audio_dir)

        if manifest_dir.exists():
            os.remove(manifest_dir)

        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.parent.mkdir(parents=True, exist_ok=True)

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
            'text': sample['transcription']
        }

        with open(manifest_dir, 'a', encoding='utf-8') as f:
            f.write(json.dumps(manifest_entry) + '\n')

        return manifest_entry


if __name__ == '__main__':
    extractor = hparl()
    _ = extractor.load_hparl_subset('train')
    _ = extractor.load_hparl_subset('validation')
    _ = extractor.load_hparl_subset('test')