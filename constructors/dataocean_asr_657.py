import json
import os
import glob

from jmespath.ast import subexpression
from tqdm import tqdm
from config.parser import Parser
from pathlib import Path
import soundfile as sf
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import librosa
import shutil

class dataocean_asr_657:
    def __init__(self):
        self.conf = Parser()
        self.conf.get_args()

        self.load_dir = os.path.join(os.path.expanduser('~'),
                                      self.conf.dataset_path,
                                      self.conf.language,
                                      'dataocean_asr_657',
                                      'DATA')

        self.target_path = os.path.join(
            os.path.expanduser('~'),
            self.conf.dataset_path,
            self.conf.language,
            'dataocean_asr_657'
        )

    def load_transcripts(self, manifest_folder: str) -> dict:
        text_map = {}

        with open(manifest_folder, 'r', encoding='utf-8') as f:
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

        total_processed = 0
        with open(manifest_dir, 'a', encoding='utf-8') as manifest_f:
            for batch_folder in os.listdir(self.load_dir):
                channel = batch_folder[-1]

                if 'CHANNEL' in batch_folder:
                    batch_path = os.path.join(self.load_dir, batch_folder)
                    manifests_path = os.path.join(batch_path, 'SCRIPT')
                    clips_path = os.path.join(batch_path, 'WAVE')

                    for subject_clip_folder in os.listdir(clips_path):
                        subject_id = subject_clip_folder.replace('SPEAKER', '')

                        manifest_load_dir = os.path.join(manifests_path, f"{channel}{subject_id}0.TXT")
                        clips_load_dir = os.path.join(clips_path, subject_clip_folder, f'SESSION0')
                        text_mapping = self.load_transcripts(manifest_load_dir)

                        for wav_file in tqdm(os.listdir(clips_load_dir), desc="Converting & Formatting Audio"):
                            file_id = os.path.splitext(os.path.basename(wav_file))[0]
                            transcript = text_mapping.get(file_id)
                            wav_path = os.path.join(clips_load_dir, wav_file)

                            audio_data, _ = librosa.load(wav_path, sr=self.conf.sampling_rate)

                            target_filename = f'{total_processed:06d}.wav'
                            target_filepath = os.path.join(audio_dir, target_filename)
                            sf.write(str(target_filepath), audio_data, self.conf.sampling_rate)
                            duration = len(audio_data) / self.conf.sampling_rate

                            manifest_entry = {
                                'audio_filepath': str(target_filepath),
                                'audio_source': str(wav_path),
                                'subject': subject_id,
                                'channel': channel,
                                'duration': duration,
                                'text': transcript,
                            }

                            manifest_f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
                            total_processed += 1

if __name__ == '__main__':
    extractor = dataocean_asr_657()
    extractor.load_dataocean_subset('train')