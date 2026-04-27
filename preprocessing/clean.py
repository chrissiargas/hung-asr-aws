import json
import os
import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm
from config.parser import Parser

class DataLabeler:
    def __init__(self):
        self.conf = Parser()
        self.conf.get_args()

        self.language = self.conf.language
        self.bad_folder = os.path.join(Path(__file__).parent.parent, "bad_folder")

    def get_suspect_filepaths(self, dataset: str, split: str):
        suspect_filepaths = set()
        all_csvs = glob.glob(os.path.join(self.bad_folder, "*.csv"))

        for csv_path in all_csvs:
            if dataset in csv_path and split in csv_path:
                try:
                    df = pd.read_csv(csv_path)
                    if 'filepath' in df.columns:
                        paths = df['filepath'].dropna().tolist()
                        suspect_filepaths.update(paths)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")

        return suspect_filepaths

    def label_manifest(self, dataset: str, split: str):
        print(f"Labeling {dataset} - {split} split...")

        manifest_folder = os.path.join(os.path.expanduser('~'),
                                       self.conf.dataset_path,
                                       self.language,
                                       dataset, 'manifests')

        original_manifest_path = os.path.join(manifest_folder, f"{self.language}_{split}.json")

        if not os.path.exists(original_manifest_path):
            print(f"Manifest not found: {original_manifest_path}")
            return

        suspect_filepaths = self.get_suspect_filepaths(dataset, split)

        suspect_count = 0
        clean_count = 0

        with open(original_manifest_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        with open(original_manifest_path, 'w', encoding='utf-8') as f:
            for line in tqdm(lines, desc="Applying flags"):
                entry = json.loads(line)
                is_suspect = entry.get('audio_filepath') in suspect_filepaths
                entry['suspect'] = is_suspect

                if is_suspect:
                    suspect_count += 1
                else:
                    clean_count += 1

                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"Done! Labeled manifest saved to: {original_manifest_path}")
        print(f"Summary -> Clean: {clean_count} | Suspect: {suspect_count}\n")

if __name__ == "__main__":
    labeler = DataLabeler()

    datasets = ['common_voice', 'fleurs', 'hparl', 'tedx']
    splits = ['train', 'validation', 'test']

    for dataset in datasets:
        for split in splits:
            labeler.label_manifest(dataset, split)