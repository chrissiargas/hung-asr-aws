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
from preprocessing.full import check_full_datatang_part1

class datatang_asr_part1:
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
            'datatang_asr_1'
        )

    def assign_session(self):
        manifest_dir = Path(os.path.join(self.target_path, 'manifests', f"{self.conf.language}.json"))
        source_file = Path(os.path.join(self.target_path, 'all_wav_files.txt'))

        incomplete = check_full_datatang_part1(manifest_dir, source_file)

        if len(incomplete) == 0:
            print("Dataset is 100% complete! Start assigning...")

            temp_manifest = manifest_dir.with_suffix('.tmp')
            with open(manifest_dir, 'r', encoding='utf-8') as f_in, \
                open(temp_manifest, 'w', encoding='utf-8') as f_out:

                for line in f_in:
                    entry = json.loads(line)
                    entry.pop('subject', None)

                    filepath = entry.get('audio_source', '')
                    filename = os.path.basename(filepath)
                    session = os.path.splitext(filename)[0]  # Extracts '001452' from '001452.wav'
                    session_id = session.split('_')[0]
                    entry['session'] = session_id
                    f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

            temp_manifest.replace(manifest_dir)
            print("✅ Session IDs successfully appended to all lines!")

        else:
            print("⚠️ Cannot assign sessions: Dataset is incomplete.")

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
            incomplete = check_full_datatang_part1(manifest_dir, source_file)

            if len(incomplete) == 0:
                print("✅ Dataset is 100% complete! No further processing needed.")
                return

            else:
                print(f"⚠️ Found {len(incomplete)} incomplete (Channel, Speaker) pairs.")
                print("Cleaning partial entries from manifest to prepare for targeted re-loading...")

                kept_lines = []
                max_id = -1

                with open(manifest_dir, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            source = entry.get('audio_source', '').split('::')[0]

                            if os.path.normpath(source) in incomplete:
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
                print("\n🚀 Beginning Targeted Audio Processing...")

                total_processed = max_id + 1

        with open(manifest_dir, 'a', encoding='utf-8') as manifest_f:
            print("\n🚀 Beginning Part 1 Processing: Segmenting long-form audio...")
            part1_txts = glob.glob(os.path.join(self.load_dir, '*.txt'))

            for txt_file in part1_txts:
                big_utt_id = os.path.basename(txt_file).replace('.txt', '')

                if big_utt_id.startswith('G'):
                    continue

                wav_path = os.path.join(self.load_dir, f'{big_utt_id}.wav')

                if not os.path.exists(wav_path):
                    print(f"⚠️ Missing corresponding audio for {txt_file}")
                    continue

                if not remove:
                    if os.path.normpath(wav_path) not in incomplete:
                        print(f"Skipping {big_utt_id} as it is already complete.")
                        continue

                audio_data, sr = librosa.load(wav_path, sr=self.conf.sampling_rate)

                with open(txt_file, "r", encoding="utf-8-sig") as f:
                    lines = f.readlines()

                print(f'\nFILE: {big_utt_id} | CLIPS: {len(lines)}')

                for line in tqdm(lines, desc="Converting & Formatting Audio"):
                    line = line.strip()
                    parts = line.split('\t')

                    if len(parts) < 4:
                        print(f'⚠️ Missing parts for {line}')
                        continue

                    current_start, current_end, speaker, raw_text = parts[0], parts[1], parts[2], parts[3]
                    raw_text = raw_text.strip()

                    if raw_text in ['<sil>', 'sp', 'sil', '[no-speech]', '[N]', ''] or not raw_text:
                        print(f'Silence Instance: {big_utt_id} - {raw_text}')
                        continue
                        
                    try:
                        current_start, current_end = float(current_start), float(current_end)
                    except ValueError:
                        continue

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
                        'subject': speaker,
                        'duration': duration,
                        'text': raw_text,
                    }

                    manifest_f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
                    total_processed += 1



if __name__ == '__main__':
    extractor = datatang_asr_part1()
    extractor.assign_session()
