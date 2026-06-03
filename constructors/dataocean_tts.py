import json
import os
import glob
from tqdm import tqdm
from config.parser import Parser
from pathlib import Path
import soundfile as sf
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import librosa
import shutil
import pandas as pd

class dataocean_tts:
    def __init__(self):
        self.conf = Parser()
        self.conf.get_args()

        self.load_paths = os.path.join(os.path.expanduser('~'),
                                      self.conf.dataset_path,
                                      self.conf.language,
                                      'dataocean_tts')

        self.target_path = os.path.join(
            os.path.expanduser('~'),
            self.conf.dataset_path,
            self.conf.language,
            'dataocean_tts'
        )

    def load_transcripts(self, batch_folder: str) -> dict:
        text_map = {}

        with open(batch_folder, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                parts = line.split('\t')
                if len(parts) >= 2:
                    file_id = parts[0].strip()
                    transcript = parts[1].strip()
                    text_map[file_id] = transcript

        return text_map

    def load_dataocean_subset(self, split: str):
        audio_dir = Path(os.path.join(self.target_path, 'data', f'{self.conf.language}_{split}_clips'))
        manifest_dir = Path(os.path.join(self.target_path, 'manifests', f"{self.conf.language}_{split}.json"))

        if audio_dir.exists():
            shutil.rmtree(audio_dir)

        if manifest_dir.exists():
            os.remove(manifest_dir)

        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.parent.mkdir(parents=True, exist_ok=True)

        with open(manifest_dir, 'a', encoding='utf-8') as manifest_f:
            for load_path in os.listdir(self.load_paths):

                manifest_load_dir = os.path.join(load_path, "corpus_text.txt")
                clips_load_dir = os.path.join(load_path, "wavs")
                text_mapping = self.load_transcripts(manifest_load_dir)

                total_processed = 0
                for wav_path in tqdm(os.listdir(clips_load_dir), desc="Converting & Formatting Audio"):
                    file_id = os.path.splitext(os.path.basename(wav_path))[0]

                    transcript = text_mapping.get(file_id)

                    audio_data, _ = librosa.load(wav_path, sr=self.conf.sampling_rate)

                    target_filename = f'{total_processed:06d}.wav'
                    target_filepath = os.path.join(audio_dir, target_filename)
                    sf.write(str(target_filepath), audio_data, self.conf.sampling_rate)
                    duration = len(audio_data) / self.conf.sampling_rate

                    manifest_entry = {
                        'audio_filepath': str(target_filepath),
                        'duration': duration,
                        'text': transcript
                    }

                    manifest_f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
                    total_processed += 1

