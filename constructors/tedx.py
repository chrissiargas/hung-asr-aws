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
import torchaudio.functional as F
import yaml
import torchaudio

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class tedx:
    def __init__(self):
        self.conf = Parser()
        self.conf.get_args()

        if self.conf.language == 'greek':
            self.language = 'el'

        self.load_path = os.path.join(os.path.expanduser('~'),
                                      self.conf.dataset_path,
                                      self.conf.language,
                                      'tedx',
                                      'el-el',
                                      'data')

        self.target_path = os.path.join(
            os.path.expanduser('~'),
            self.conf.dataset_path,
            self.conf.language,
            'tedx'
        )

    def extract_sample(self, seg_info: Dict, text: str, seg_audios: Dict, audio_dir: Path, manifest_dir: Path, idx: int):
        wav_filename = seg_info['wav']
        full_audio, sr = seg_audios[wav_filename]

        start_frame = int(seg_info['offset'] * sr)
        end_frame = start_frame + int(seg_info['duration'] * sr)

        sliced_audio = full_audio[:, start_frame:end_frame]

        clip_filepath = audio_dir / f'{idx:06d}.wav'
        torchaudio.save(str(clip_filepath), sliced_audio, self.conf.sampling_rate)

        manifest_entry = {
            'audio_filepath': str(clip_filepath),
            'duration': seg_info['duration'],
            'text': text
        }

        with open(manifest_dir, 'a', encoding='utf-8') as f:
            f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')

    def load_tedx_subset(self, split: str):

        data_path = os.path.join(self.load_path, split)
        text_path = os.path.join(data_path, f"txt")
        clips_path = os.path.join(data_path, "wav")

        yaml_path = os.path.join(text_path, f'{split}.yaml')
        text_path = os.path.join(text_path, f'{split}.{self.language}')

        with open(yaml_path, 'r', encoding='utf-8') as yf:
            segmentation_map = yaml.safe_load(yf)

        with open(text_path, 'r', encoding='utf-8') as tf:
            transcripts = tf.read().splitlines()

        audio_dir = Path(os.path.join(self.target_path, 'data', f'{self.conf.language}_{split}_clips'))
        manifest_dir = Path(os.path.join(self.target_path, 'manifests', f"{self.conf.language}_{split}.json"))

        if audio_dir.exists():
            shutil.rmtree(audio_dir)

        if manifest_dir.exists():
            os.remove(manifest_dir)

        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.parent.mkdir(parents=True, exist_ok=True)

        seg_audios = {}

        for seg_info in tqdm(segmentation_map, desc=f"Loading audio for {split} split"):
            wav_filename = seg_info['wav']

            if wav_filename.endswith('.wav'):
                wav_filename_ = wav_filename.removesuffix('.wav')
                wav_filename_ += '.flac'
            else:
                wav_filename_ = wav_filename

            full_audio_path = os.path.join(clips_path, wav_filename_)
            if wav_filename not in seg_audios:
                audio, sr = torchaudio.load(full_audio_path)
                if sr != self.conf.sampling_rate:
                    audio = F.resample(audio, orig_freq=sr, new_freq=self.conf.sampling_rate)
                seg_audios[wav_filename] = (audio, self.conf.sampling_rate)

        manifest_entries = [self.extract_sample(seg_info, text, seg_audios, audio_dir, manifest_dir, idx) for
                            idx, (seg_info, text) in enumerate(tqdm(zip(segmentation_map, transcripts),
                                                                    desc=f"Processing segments for {split} split", total=len(segmentation_map)))]

        return manifest_entries

if __name__ == '__main__':
    extractor = tedx()
    _ = extractor.load_tedx_subset('train')
    _ = extractor.load_tedx_subset('valid')
    _ = extractor.load_tedx_subset('test')

