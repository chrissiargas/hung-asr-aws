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
from datacollective import DataCollective
import pandas as pd
import subprocess
import requests
import tarfile

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
API_KEY = "a5a7fb5e382a78f691c85763a922b5410f9dda129943d731c79e48b9cafb1ce8"
DATASET_ID = {'greek':'cmn2cx91x01dno10754vxfu3b',
              'hungarian': 'cmj8u3p8900bhnxxb50f37mkm',
              'english': 'cmqim2hn800ssnr07gvmpcnwu',
              'german': 'cmqim3xpi00t6nr07k0myqtkr'}

TARGET_DIR = os.path.expanduser(os.path.join("~", "asr-shared", "csiargka", "cache", "datasets", "hungarian", "common_voice"))
LANGUAGE_ID = {'greek': 'el', 'hungarian': 'hu', 'english': 'en', 'german': 'de'}

def download(language: str):
    dataset_id = DATASET_ID[language]
    archive_path = os.path.join(TARGET_DIR, f"common_voice_{language}.tar.gz")
    extract_dir = os.path.join(TARGET_DIR, f"common_voice_{language}")

    print("Fetching presigned download URL...")
    api_url = f"https://mozilladatacollective.com/api/datasets/{dataset_id}/download"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(api_url, headers=headers)
    response.raise_for_status()  # Raises an exception if the request failed

    download_url = response.json().get("downloadUrl")
    print("Successfully retrieved download URL.")

    # 2. Download the file in chunks (Streaming)
    print(f"Downloading archive to {archive_path}...")
    with requests.get(download_url, stream=True) as download_response:
        download_response.raise_for_status()

        # Get total file size from headers if available (for progress bar)
        total_size = int(download_response.headers.get("content-length", 0))

        with open(archive_path, "wb") as f, tqdm(
                desc=archive_path,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
        ) as progress_bar:
            for chunk in download_response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                progress_bar.update(size)

    print("Download complete.")

    # 3. Extract the .tar.gz archive
    print(f"Extracting contents to {extract_dir}...")

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    print("Extraction complete!")

class common_voice:
    def __init__(self, language: str = 'hungarian', do_download: bool = False):
        self.conf = Parser()
        self.conf.get_args()
        self.language = language

        if do_download:
            download(language=self.language)

        extract_dir = os.path.join(TARGET_DIR, f"common_voice_{self.language}")
        self.load_path = os.path.join(extract_dir, LANGUAGE_ID[self.language])

        self.target_path = os.path.join(
            os.path.expanduser('~'),
            self.conf.dataset_path,
            self.conf.language,
            f'common_voice_{self.language}'
        )

    def load_common_subset(self, split: str):
        manifest_path = os.path.join(self.load_path, f"{split}.tsv")
        clips_path = os.path.join(self.load_path, "clips")

        dataset = pd.read_csv(manifest_path, sep="\t", low_memory=False)
        dataset["audio_path"] = dataset["path"].apply(lambda x: os.path.join(clips_path, x))

        dataset = Dataset.from_pandas(dataset)

        if split == 'dev':
            split = 'validation'

        initial_split = split
        remove_files = False if split == 'other' and self.conf.other_to_train else True
        split = 'train' if split == 'other' and self.conf.other_to_train else split

        audio_dir = Path(os.path.join(self.target_path, 'data', f'{self.language}_{split}_clips'))
        manifest_dir = Path(os.path.join(self.target_path, 'manifests', f"{self.language}_{split}.json"))

        if remove_files:
            if audio_dir.exists():
                shutil.rmtree(audio_dir)

            if manifest_dir.exists():
                os.remove(manifest_dir)

        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.parent.mkdir(parents=True, exist_ok=True)

        if initial_split == 'other' and self.conf.other_to_train:
            with open(manifest_dir, 'r', encoding='utf-8') as f:
                train_manifests = [json.loads(line) for line in f]
            offset = len(train_manifests)
        else:
            offset = 0

        manifest_entries = []
        for idx, sample in enumerate(tqdm(dataset)):
            entry = self.extract_samples(sample, audio_dir, manifest_dir, offset + len(manifest_entries))
            if entry is not None:
                manifest_entries.append(entry)

        return manifest_entries

    def extract_samples(self, sample, audio_dir: str, manifest_dir: str, idx: int):
        try:
            audio_data, _ = librosa.load(sample['audio_path'], sr=self.conf.sampling_rate)
        except Exception as e:
            # Safely skip unreadable or corrupt mp3 files
            return None

        audio_filepath = os.path.join(audio_dir, f'{idx:06d}.wav')

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

import argparse
from distutils.util import strtobool

if __name__ == '__main__':
    print('Starting...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default='hungarian')
    parser.add_argument('--do_download', default=False, type=lambda x: bool(strtobool(x)))
    args, unknown = parser.parse_known_args()
    args_dict = vars(args)

    extractor = common_voice(language=args_dict['language'], do_download=args_dict['do_download'])
    _ = extractor.load_common_subset('train')
    _ = extractor.load_common_subset('dev')
    _ = extractor.load_common_subset('test')
    _ = extractor.load_common_subset('other')

