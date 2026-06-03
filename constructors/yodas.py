import json
import os
from tqdm import tqdm
from config.parser import Parser
from datasets import load_dataset
from pathlib import Path
import soundfile as sf
import warnings
import re

warnings.simplefilter(action='ignore', category=FutureWarning)
import librosa
import shutil

class yodas:
    def __init__(self):
        self.conf = Parser()
        self.conf.get_args()

        self.load_path = 'espnet/yodas-granary'
        self.target_path = os.path.join(
            os.path.expanduser('~'),
            self.conf.dataset_path,
            self.conf.language,
            'yodas'
        )

        if self.conf.language.lower() == 'greek':
            self.subset = 'Greek'
        elif self.conf.language.lower() == 'hungarian':
            self.subset = 'Hungarian'

        self.dataset = load_dataset(
            self.load_path,
            self.subset,
            split='asr_only',
            trust_remote_code=True,
            verification_mode="no_checks"
        )

        split_train = 0.8
        split_val = 0.1
        split_test = 1 - split_train - split_val

        train_samples = int(split_train * len(self.dataset))
        val_samples = int(split_val * len(self.dataset))
        test_samples = int(split_test * len(self.dataset))

        self.splits = {
            'train': list(range(0, train_samples)),
            'validation': list(range(train_samples, train_samples + val_samples)),
            'test': list(range(train_samples + val_samples, train_samples + val_samples + test_samples))
        }

    def load_yodas_subset(self, split: str = 'train'):

        if split == 'full':
            split = 'train'
            dataset = self.dataset

        else:
            dataset = self.dataset.select(self.splits[split])

        audio_dir = Path(os.path.join(self.target_path, 'data', f'{self.conf.language}_{split}_clips'))
        manifest_dir = Path(os.path.join(self.target_path, 'manifests', f"{self.conf.language}_{split}.json"))

        if audio_dir.exists():
            shutil.rmtree(audio_dir)
        if manifest_dir.exists():
            os.remove(manifest_dir)

        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.parent.mkdir(parents=True, exist_ok=True)

        idx = 0
        for sample in tqdm(dataset):
            text = sample['text'].strip()

            if self.is_synthetic_heuristic(text):
                continue

            self.extract_samples(sample, audio_dir, manifest_dir, idx)
            idx += 1

        return

    def extract_samples(self, sample, audio_dir: str, manifest_dir: str, idx: int):
        if self.is_synthetic_heuristic(sample.get('text')):
            return {}

        audio_data = sample['audio']['array']
        sampling_rate = sample['audio']['sampling_rate']

        if sampling_rate != self.conf.sampling_rate:
            audio_data = librosa.resample(y=audio_data, orig_sr=sampling_rate, target_sr=self.conf.sampling_rate)

        audio_filepath = os.path.join(audio_dir, f'{idx:06d}.wav')

        sf.write(str(audio_filepath), audio_data, self.conf.sampling_rate)
        duration = len(audio_data) / self.conf.sampling_rate

        text = sample['text']

        manifest_entry = {
            'audio_filepath': str(audio_filepath),
            'duration': duration,
            'text': text
        }

        with open(manifest_dir, 'a', encoding='utf-8') as f:
            f.write(json.dumps(manifest_entry) + '\n')

        return manifest_entry

    def is_synthetic_heuristic(self, text: str) -> bool:
        if not re.search(r'[.,!?;:]', text):
            return True

        if text == text.lower():
            return True

        forbidden_tags = ['[zene]', '[taps]', '[nevetés]', '[music]', '[applause]']
        if any(tag in text.lower() for tag in forbidden_tags):
            return True

        return False

if __name__ == '__main__':
    extractor = yodas()

    # Run the custom splits
    extractor.load_yodas_subset('train')
    extractor.load_yodas_subset('validation')
    extractor.load_yodas_subset('test')


