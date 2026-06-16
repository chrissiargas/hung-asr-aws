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
from preprocessing.full import check_full_dataocean_659
import re

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

    def load_dataocean_subset(self, split: str, remove: bool = False):
        audio_dir = Path(os.path.join(self.target_path, 'data', f'{self.conf.language}_{split}_clips'))
        manifest_dir = Path(os.path.join(self.target_path, 'manifests', f"{self.conf.language}_{split}.json"))
        source_file = Path(os.path.join(self.target_path, 'all_wav_files.txt'))

        if remove:
            if audio_dir.exists():
                print(f"Found existing directory. Deleting old files from network storage (please wait)...")
                shutil.rmtree(audio_dir)
                print("Deletion complete!")

            if manifest_dir.exists():
                os.remove(manifest_dir)

        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.parent.mkdir(parents=True, exist_ok=True)

        total_processed = 0
        if not remove:
            incomplete = check_full_dataocean_659(manifest_dir, source_file)

            if len(incomplete) == 0:
                print("✅ Dataset is 100% complete! No further processing needed.")
                return

            else:
                print(f"⚠️ Found {len(incomplete)} incomplete (Channel, Speaker) pairs.")
                print("Cleaning partial entries from manifest to prepare for targeted re-loading...")

                kept_lines = []
                pattern = re.compile(r'WAVE[/\\](C\d+)[/\\].*?(U\d+)', re.IGNORECASE)

                max_id = -1
                with open(manifest_dir, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            source = entry.get('audio_source', '')
                            match = pattern.search(source)
                            if match:
                                ch, spk = match.group(1).upper(), match.group(2).upper()
                                if (ch, spk) in incomplete:
                                    continue

                            kept_lines.append(line)

                            filepath = entry.get('audio_filepath', '')
                            filename = os.path.basename(filepath)
                            name_part = os.path.splitext(filename)[0]  # Extracts '001452' from '001452.wav'
                            try:
                                file_id = int(name_part)
                                if file_id > max_id:
                                    max_id = file_id
                            except ValueError:
                                pass

                        except json.JSONDecodeError:
                            continue

                # Overwrite the manifest with only the perfectly completed data
                with open(manifest_dir, 'w', encoding='utf-8') as f:
                    for line in kept_lines:
                        f.write(line)

                print(f"✅ Scrub complete. Manifest now safely retains {len(kept_lines)} completed files.")
                file_mode = 'a'  # Append new data to the clean manifest

                print("\n🚀 Beginning Targeted Audio Processing...")

                total_processed = max_id + 1

        with open(manifest_dir, 'a', encoding='utf-8') as manifest_f:
            manifests_path = os.path.join(self.load_dir, 'SCRIPT')
            clips_path = os.path.join(self.load_dir, 'WAVE')

            for batch_clip_folder in os.listdir(clips_path):
                batch_clip_path = os.path.join(clips_path, batch_clip_folder)
                if os.path.isfile(batch_clip_path):
                    continue

                for wav_file in os.listdir(batch_clip_path):
                    wav_file_name = os.path.splitext(os.path.basename(wav_file))[0]
                    session_id = wav_file_name[-1]
                    subject_id = wav_file_name.replace('HU-HU_U', '').replace(f'_S{session_id}', '')

                    if not remove:
                        if (f'C{session_id}', f'U{subject_id}') not in incomplete:
                            print(f"Skipping {session_id} {subject_id} as it is already complete.")
                            continue

                    wav_path = os.path.join(batch_clip_path, wav_file)
                    audio_data, _ = librosa.load(wav_path, sr=self.conf.sampling_rate)

                    manifest_file = os.path.join(manifests_path, f"HU-HU_U{subject_id}_S{session_id}.TextGrid")

                    tg = textgrid.TextGrid.fromFile(manifest_file)
                    tier = tg.tiers[0]

                    print('\nSession:', session_id, 'SUBJECT:', subject_id, 'CLIPS:', len(tier))
                    for interval in tqdm(tier):
                        raw_text = interval.mark.strip()

                        if not raw_text or raw_text in ['<sil>', 'sp', 'sil', '[no-speech]']:
                            continue

                        current_start = interval.minTime
                        current_end = interval.maxTime

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
                            'text': raw_text,
                        }

                        manifest_f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
                        total_processed += 1

if __name__ == '__main__':
    extractor = dataocean_asr_659()
    extractor.load_dataocean_subset('train')