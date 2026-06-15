from collections import defaultdict
import json
import os
from pathlib import Path
import pandas as pd

def check_full_dataocean(manifest_path, original_data_dir):
    print("=" * 60)
    print(" 🚀 VECTORIZED DATAOCEAN EXHAUSTION AUDIT ")
    print("=" * 60)

    print(f"Reading manifest: {manifest_path}...")
    df = pd.read_json(manifest_path, lines=True)

    processed_files = set(df['audio_source'].dropna().apply(os.path.normpath))

    print(f"✅ Loaded {len(processed_files)} processed audio files.")

    print(f"Scanning original directory: {original_data_dir}...")
    original_files = set(
        map(lambda p: os.path.normpath(str(p)), Path(original_data_dir).rglob('*.[wW][aA][vV]'))
    )

    print(f"📁 Found {len(original_files)} physical WAV files in the folder.\n")

    missing_in_json = original_files - processed_files
    missing_in_folder = processed_files - original_files

    if not missing_in_json:
        print("✅ SUCCESS: The JSON successfully exhausted EVERY audio file in the folder!")
    else:
        print(f"❌ FAILED: Found {len(missing_in_json)} WAV files in the folder that are NOT in your JSON.")
        print("Top 5 missed files:")
        # Print up to 5 missing files by joining a sliced list (NO LOOPS)
        print("   -> " + "\n   -> ".join(list(missing_in_json)[:5]))

    if missing_in_folder:
        print(
            f"\n⚠️ GHOST FILES: Found {len(missing_in_folder)} files in the JSON that do NOT exist in the folder anymore.")
        print("   -> " + "\n   -> ".join(list(missing_in_folder)[:5]))

    print("=" * 60)

if __name__ == '__main__':
    MANIFEST_PATH = '/home/jovyan/asr-shared/csiargka/cache/datasets/hungarian/dataocean_asr_657/manifests/hungarian_train.json'
    ORIGINAL_DATA_DIR = '/home/jovyan/asr-data-segr/5th_lang/DataOcean/King-ASR-657/DATA'

    check_full_dataocean(MANIFEST_PATH, ORIGINAL_DATA_DIR)