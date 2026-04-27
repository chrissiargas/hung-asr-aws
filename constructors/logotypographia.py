import json
import os

import numpy as np
from tqdm import tqdm
from triton.knobs import language

from config.parser import Parser
from datasets import load_dataset
from pathlib import Path
import soundfile as sf
import shutil
import librosa
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Logotypographia:
    def __init__(self):
        self.conf = Parser()
        self.conf.get_args()

        self.load_path = 'ddamianos/logotypografia'

        self.target_path = os.path.join(
            os.path.expanduser('~'),
            self.conf.dataset_path,
            self.conf.language,
            'logotypographia'
        )

    def load_subset(self, split: str):
        dataset = load_dataset(
            self.load_path,
            split=split
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

    def extract_samples(self, sample, audio_dir: Path, manifest_dir: Path, idx: int):
        audio_data = np.array(sample['audio'])
        sampling_rate = 16000

        # Resample based on your parser config
        audio_data = librosa.resample(y=audio_data, orig_sr=sampling_rate, target_sr=self.conf.sampling_rate)

        audio_filepath = audio_dir / f'{idx:06d}.wav'

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
    extractor = Logotypographia()

    # Logotypographia typically uses standard 'train' and 'test' splits
    _ = extractor.load_subset('train')
    _ = extractor.load_subset('test')