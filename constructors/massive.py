import json
import os

from debugpy._vendored.pydevd._pydevd_bundle._debug_adapter import __main__pydevd_gen_debug_adapter_protocol
from tqdm import tqdm
from config.parser import Parser
from datasets import load_dataset
from pathlib import Path
import soundfile as sf

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import librosa
import shutil

class speech_massive:
    def __init__(self):
        self.conf = Parser()
        self.conf.get_args()

        self.target_path = os.path.join(
            os.path.expanduser('~'),
            self.conf.dataset_path,
            self.conf.language,
            'speech_massive'
        )

        if self.conf.language == 'greek':
            self.language = 'el-GR'
        elif self.conf.language == 'hungarian':
            self.language = 'hu-HU'


    def load_massive_subset(self, split: str, how: str):
        if split == 'test':
            load_paths = ['FBK-MT/Speech-MASSIVE-test']
            set_splits = [['test']]
        if split == 'train':
            if how == 'all':
                load_paths = ['FBK-MT/Speech-MASSIVE', 'FBK-MT/Speech-MASSIVE-test']
                set_splits = [['train_115', 'validation'], ['test']]
            elif how == 'train_val':
                load_paths = ['FBK-MT/Speech-MASSIVE']
                set_splits = [['train_115', 'validation']]
            elif how == 'train_test':
                load_paths = ['FBK-MT/Speech-MASSIVE', 'FBK-MT/Speech-MASSIVE-test']
                set_splits = [['train_115'], ['test']]
        if split == 'validation':
            load_paths = ['FBK-MT/Speech-MASSIVE']
            set_splits = [['validation']]

        audio_dir = Path(os.path.join(self.target_path, 'data', f'{self.conf.language}_{split}_clips'))
        manifest_dir = Path(os.path.join(self.target_path, 'manifests', f"{self.conf.language}_{split}.json"))

        if audio_dir.exists():
            shutil.rmtree(audio_dir)

        if manifest_dir.exists():
            os.remove(manifest_dir)

        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.parent.mkdir(parents=True, exist_ok=True)

        idx_offset = 0
        for load_path, orig_splits in zip(load_paths, set_splits):
            for orig_split in orig_splits:
                dataset = load_dataset(
                    load_path,
                    self.language,
                    split=orig_split,
                    trust_remote_code=True
                )

                manifest_entries = [self.extract_samples(sample, audio_dir, manifest_dir, idx_offset + idx)
                                    for idx, sample in enumerate(tqdm(dataset))]
                idx_offset += len(dataset)

                return manifest_entries

    def extract_samples(self, sample, audio_dir: str, manifest_dir: str, idx: int):
        audio_data = sample['audio']['array']
        sampling_rate = sample['audio']['sampling_rate']

        audio_data = librosa.resample(y=audio_data, orig_sr=sampling_rate, target_sr=self.conf.sampling_rate)
        audio_filepath = os.path.join(audio_dir, f'{idx:06d}.wav')

        sf.write(str(audio_filepath), audio_data, self.conf.sampling_rate)
        duration = len(audio_data) / self.conf.sampling_rate

        text = sample.get('utt', sample.get('text', ''))
        manifest_entry = {
            'audio_filepath': str(audio_filepath),
            'duration': duration,
            'text': text
        }

        with open(manifest_dir, 'a', encoding='utf-8') as f:
            f.write(json.dumps(manifest_entry) + '\n')

        return manifest_entry


if __name__ == '__main__':
    extractor = speech_massive()
    _ = extractor.load_massive_subset('train', how='all')

