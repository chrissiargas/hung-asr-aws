import json
import os
from tqdm import tqdm
from config.parser import Parser
from datasets import load_dataset
from pathlib import Path
import soundfile as sf
import kagglehub
import pandas as pd
import librosa
import shutil
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)

class css10:
    def __init__(self, download: bool = False):
        self.conf = Parser()
        self.conf.get_args()

        if self.conf.language == 'greek':
            self.language = 'el'

        elif self.conf.language == 'hungarian':
            self.language = 'hu'

        if download:
            path = kagglehub.dataset_download(f"bryanpark/{self.conf.language}-single-speaker-speech-dataset")
            print("Path to dataset files:", path)

        self.load_path = os.path.join(
            os.path.expanduser('~'),
            '.cache',
            f'kagglehub/datasets/bryanpark/{self.conf.language}-single-speaker-speech-dataset/versions/1/{self.language}'
        )

        self.target_path = os.path.join(
            os.path.expanduser('~'),
            self.conf.dataset_path,
            self.conf.language,
            'css10'
        )

    def load_css10_subset(self, split: str):
        transcript_path = os.path.join(self.load_path, 'transcript.txt')

        dataset = pd.read_csv(transcript_path, sep='|', header=None,
                         names=['wav_filename', 'original_text', 'normalized_text', 'duration'])

        audio_dir = Path(os.path.join(self.target_path, 'data', f'{self.conf.language}_{split}_clips'))
        manifest_dir = Path(os.path.join(self.target_path, 'manifests', f"{self.conf.language}_{split}.json"))

        if audio_dir.exists():
            shutil.rmtree(audio_dir)

        if manifest_dir.exists():
            os.remove(manifest_dir)

        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.parent.mkdir(parents=True, exist_ok=True)

        manifest_entries = [self.extract_samples(sample, audio_dir, manifest_dir, idx) for
            idx, sample in enumerate(tqdm(dataset.itertuples()))]

        return manifest_entries

    def extract_samples(self, sample, audio_dir: str, manifest_dir: str, idx: int):
        original_audio_path = os.path.join(self.load_path, sample.wav_filename)
        audio_data, sampling_rate = sf.read(original_audio_path)

        audio_data = librosa.resample(y=audio_data, orig_sr=sampling_rate, target_sr=self.conf.sampling_rate)

        audio_filepath = os.path.join(audio_dir, f'{idx:06d}.wav')

        sf.write(str(audio_filepath), audio_data, self.conf.sampling_rate)
        duration = len(audio_data) / self.conf.sampling_rate

        manifest_entry = {
            'audio_filepath': str(audio_filepath),
            'duration': duration,
            'text': sample.original_text
        }

        with open(manifest_dir, 'a', encoding='utf-8') as f:
            f.write(json.dumps(manifest_entry) + '\n')

        return manifest_entry


if __name__ == '__main__':
    extractor = css10(download=True)
    _ = extractor.load_css10_subset('train')