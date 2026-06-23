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

# Assuming you added check_full_datatang_part2 to your full.py
from preprocessing.full import check_full_datatang_part2

class datatang_asr_part2:
    def __init__(self):
        self.conf = Parser()
        self.conf.get_args()

        # Target the root directory containing the 'category' folder
        self.load_dir = os.path.join(
            os.path.expanduser('~'),
            'asr-data-segr',
            '5th_lang',
            'DataTang',
            'data',
            'category'
        )

        self.target_path = os.path.join(
            os.path.expanduser('~'),
            self.conf.dataset_path,
            self.conf.language,
            'datatang_asr_2'
        )

    def load_datatang_subset(self, split: str, remove: bool = False):
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
            incomplete = check_full_datatang_part2(manifest_dir, source_file)

            if len(incomplete) == 0:
                print("✅ Dataset is 100% complete! No further processing needed.")
                return

            else:
                print(f"⚠️ Found {len(incomplete)} incomplete base audio files.")
                print("Cleaning partial entries from manifest to prepare for targeted re-loading...")

                kept_lines = []
                max_id = -1

                if manifest_dir.exists():
                    with open(manifest_dir, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                entry = json.loads(line)
                                source = entry.get('audio_source', '')

                                if os.path.normpath(source) in incomplete:
                                    continue

                                kept_lines.append(line)

                                filepath = entry.get('audio_filepath', '')
                                filename = os.path.basename(filepath)
                                name_part = os.path.splitext(filename)[0]
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
                print("\n🚀 Beginning Targeted Audio Processing...")

                total_processed = max_id + 1

        with open(manifest_dir, 'a', encoding='utf-8') as manifest_f:
            print("\n🚀 Beginning Part 2 Processing: Nested 1-to-1 utterance mapping...")
            
            # Explicitly target the text files inside their subfolders (e.g., G00001/G00001S0001.txt)
            part2_txts = glob.glob(os.path.join(self.load_dir, 'G*', 'G*.txt'))

            for txt_file in tqdm(part2_txts, desc="Processing Part 2 Files"):
                
                # The .wav file is sitting right next to the .txt file
                wav_path = os.path.splitext(txt_file)[0] + '.wav'

                if not os.path.exists(wav_path):
                    print(f"⚠️ Missing corresponding audio for {os.path.basename(txt_file)}")
                    continue

                if not remove:
                    if os.path.normpath(wav_path) not in incomplete:
                        continue

                # Isolate the speaker ID from the filename (e.g., "G00001S0001" -> "G00001")
                big_utt_id = os.path.basename(txt_file).replace('.txt', '')
                speaker = big_utt_id.split('S')[0]

                # Read text directly using utf-8-sig to clear any potential BOM artifacts
                with open(txt_file, "r", encoding="utf-8-sig") as f:
                    raw_text = f.read().strip()

                if raw_text in ['<sil>', 'sp', 'sil', '[no-speech]', '[N]', ''] or not raw_text:
                    continue

                # Load audio and determine duration
                audio_data, sr = librosa.load(wav_path, sr=self.conf.sampling_rate)
                duration = len(audio_data) / self.conf.sampling_rate

                target_filename = f'{total_processed:06d}.wav'
                target_filepath = os.path.join(audio_dir, target_filename)
                
                # Write the standard sample rate audio file
                sf.write(str(target_filepath), audio_data, self.conf.sampling_rate)

                manifest_entry = {
                    'audio_filepath': str(target_filepath),
                    'audio_source': str(wav_path),
                    'subject': speaker,
                    'duration': duration,
                    'text': raw_text,
                }

                manifest_f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
                total_processed += 1


if __name__ == '__main__':
    extractor = datatang_asr_part2()
    extractor.load_datatang_subset('train', remove=False)