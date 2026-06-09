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
import textgrid

class dataocean_asr_659:
    def __init__(self):
        self.conf = Parser()
        self.conf.get_args()

        self.load_dir = os.path.join(os.path.expanduser('~'),
                                    'asr-data-segr',
                                    '5th_lang',
                                    'DataOcean',
                                    'King-ASR-659',
                                    'DATA')

        self.target_path = os.path.join(
            os.path.expanduser('~'),
            self.conf.dataset_path,
            self.conf.language,
            'dataocean_asr_659'
        )

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
            manifests_path = os.path.join(self.load_dir, 'SCRIPT')
            clips_path = os.path.join(self.load_dir, 'WAVE')

            for batch_clip_folder in os.listdir(clips_path):
                batch_clip_path = os.path.join(clips_path, batch_clip_folder)
                if os.path.isfile(batch_clip_path):
                   continue

                for wav_file in os.listdir(batch_clip_path):

                    wav_path = os.path.join(batch_clip_path, wav_file)
                    audio_data, _ = librosa.load(wav_path, sr=self.conf.sampling_rate)

                    wav_file_name = os.path.splitext(os.path.basename(wav_file))[0]
                    session_id = wav_file_name[-2:]
                    subject_id = wav_file_name.replace('HU-HU_U', '').replace(f'_{session_id}', '')

                    manifest_file = os.path.join(manifests_path, f"HU-HU_U{subject_id}_{session_id}.TextGrid")

                    tg = textgrid.TextGrid.fromFile(manifest_file)
                    tier = tg.tiers[0]

                    print('\nSession:', session_id, 'SUBJECT:', subject_id, 'CLIPS:', len(tier))
                    for interval in tqdm(tier):
                        raw_text = interval.mark.strip()

                        if not raw_text or raw_text in ['<sil>', 'sp', 'sil', '[no-speech]']:
                            continue

                        current_start = interval.minTime
                        current_end = interval.maxTime

                        merged_text = " ".join(raw_text)

                        start_sample = int(current_start * self.conf.sampling_rate)
                        end_sample = int(current_end * self.conf.sampling_rate)
                        audio_slice = audio_data[start_sample:end_sample]
                        duration = len(audio_slice) / self.conf.sampling_rate

                        target_filename = f'{total_processed:06d}.wav'
                        target_filepath = os.path.join(audio_dir, target_filename)
                        sf.write(str(target_filepath), audio_slice, self.conf.sampling_rate)

                        manifest_entry = {
                            'audio_filepath': str(target_filepath),
                            'audio_source': str(wav_path),
                            'subject': subject_id,
                            'channel': session_id,
                            'duration': duration,
                            'text': merged_text,
                        }

                        manifest_f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
                        total_processed += 1

if __name__ == '__main__':
    extractor = dataocean_asr_659()
    extractor.load_dataocean_subset('train')