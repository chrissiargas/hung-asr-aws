from collections import defaultdict
import json
import os
from pathlib import Path
from collections import Counter
import pandas as pd
import re

def check_full_dataocean_657(manifest_path, source_path):
    print("=" * 60)
    print(" 🚀 VECTORIZED DATAOCEAN EXHAUSTION AUDIT ")
    print("=" * 60)

    print(f"Reading manifest: {manifest_path}...")
    json_lines = Path(manifest_path).read_text(encoding='utf-8').splitlines()

    parsed_json = map(json.loads, json_lines)
    audio_sources = map(lambda x: x.get('audio_source'), parsed_json)
    valid_sources = filter(None, audio_sources)

    processed_files = set(map(os.path.normpath, valid_sources))

    print(f"✅ Loaded {len(processed_files)} processed audio files.")

    print(f"Reading cache file: {source_path}...")

    cache_lines = Path(source_path).read_text(encoding='utf-8').splitlines()

    stripped_lines = map(str.strip, cache_lines)
    valid_lines = filter(None, stripped_lines)
    original_files = set(map(os.path.normpath, valid_lines))

    print(f"✅ Found {len(original_files)} physical WAV files in the folder.\n")

    missing_in_json = original_files - processed_files
    missing_in_folder = processed_files - original_files

    if not missing_in_json:
        print("✅ SUCCESS: The JSON successfully exhausted EVERY audio file in the folder!")
    else:
        print(f"❌ FAILED: Found {len(missing_in_json)} WAV files in the folder that are NOT in your JSON.")

        pattern = re.compile(r'(CHANNEL\d+)[/\\]WAVE[/\\](SPEAKER\d+)', re.IGNORECASE)

        breakdown = Counter()
        unmatched_paths = 0

        for path in missing_in_json:
            match = pattern.search(path)
            if match:
                channel = match.group(1).upper()
                speaker = match.group(2).upper()
                breakdown[(channel, speaker)] += 1
            else:
                unmatched_paths += 1

        print("-" * 60)

        sorted_breakdown = sorted(breakdown.items(), key=lambda x: (x[0][0], x[0][1]))

        report_data = []
        for (channel, speaker), count in sorted_breakdown:
            print(f"{channel:<15} | {speaker:<15} | {count}")
            report_data.append({"Channel": channel, "Subject": speaker, "Missing_Count": count})

        if unmatched_paths > 0:
            print(
                f"\n⚠️ Note: {unmatched_paths} missing files were in unexpected folder structures and couldn't be categorized.")

        if missing_in_folder:
            print(
                f"\n⚠️ GHOST FILES: Found {len(missing_in_folder)} files in the JSON that do NOT exist in the folder anymore.")
            print("   -> " + "\n   -> ".join(list(missing_in_folder)[:5]))

        print("=" * 60)

        incomplete_pairs = set()
        for row in report_data:
            incomplete_pairs.add((row['Channel'].upper(), row['Subject'].upper()))

        return incomplete_pairs


def check_full_dataocean_659(manifest_path, source_path):
    print("=" * 60)
    print(" 🚀 VECTORIZED DATAOCEAN EXHAUSTION AUDIT ")
    print("=" * 60)

    print(f"Reading manifest: {manifest_path}...")
    json_lines = Path(manifest_path).read_text(encoding='utf-8').splitlines()

    parsed_json = map(json.loads, json_lines)
    audio_sources = map(lambda x: x.get('audio_source'), parsed_json)
    valid_sources = filter(None, audio_sources)

    processed_files = set(map(os.path.normpath, valid_sources))

    print(f"✅ Loaded {len(processed_files)} processed audio files.")

    print(f"Reading cache file: {source_path}...")

    cache_lines = Path(source_path).read_text(encoding='utf-8').splitlines()

    stripped_lines = map(str.strip, cache_lines)
    valid_lines = filter(None, stripped_lines)
    original_files = set(map(os.path.normpath, valid_lines))

    print(f"✅ Found {len(original_files)} physical WAV files in the folder.\n")

    missing_in_json = original_files - processed_files
    missing_in_folder = processed_files - original_files

    if not missing_in_json:
        print("✅ SUCCESS: The JSON successfully exhausted EVERY audio file in the folder!")
    else:
        print(f"❌ FAILED: Found {len(missing_in_json)} WAV files in the folder that are NOT in your JSON.")

        pattern = re.compile(r'WAVE[/\\](C\d+)[/\\].*?(U\d+)', re.IGNORECASE)

        breakdown = Counter()
        unmatched_paths = 0

        for path in missing_in_json:
            match = pattern.search(path)
            if match:
                channel = match.group(1).upper()
                speaker = match.group(2).upper()
                breakdown[(channel, speaker)] += 1
            else:
                unmatched_paths += 1

        print("-" * 60)

        sorted_breakdown = sorted(breakdown.items(), key=lambda x: (x[0][0], x[0][1]))

        report_data = []
        for (channel, speaker), count in sorted_breakdown:
            print(f"{channel:<15} | {speaker:<15} | {count}")
            report_data.append({"Channel": channel, "Subject": speaker, "Missing_Count": count})

        if unmatched_paths > 0:
            print(
                f"\n⚠️ Note: {unmatched_paths} missing files were in unexpected folder structures and couldn't be categorized.")

        if missing_in_folder:
            print(
                f"\n⚠️ GHOST FILES: Found {len(missing_in_folder)} files in the JSON that do NOT exist in the folder anymore.")
            print("   -> " + "\n   -> ".join(list(missing_in_folder)[:5]))

        print("=" * 60)

        incomplete_pairs = set()
        for row in report_data:
            incomplete_pairs.add((row['Channel'].upper(), row['Subject'].upper()))

        return incomplete_pairs


def check_full_datatang_part1(manifest_path, source_path):
    print("=" * 60)
    print(" 🚀 VECTORIZED DATATANG PART 1 EXHAUSTION AUDIT ")
    print("=" * 60)

    print(f"Reading manifest: {manifest_path}...")
    try:
        json_lines = Path(manifest_path).read_text(encoding='utf-8').splitlines()
        parsed_json = map(json.loads, json_lines)

        audio_sources = map(lambda x: x.get('audio_source', '').split('::')[0], parsed_json)
        valid_sources = filter(None, audio_sources)

        processed_files = set(map(os.path.normpath, valid_sources))
        print(f"✅ Loaded {len(processed_files)} processed base audio files from manifest.")

    except FileNotFoundError:
        print("⚠️ Manifest not found. Assuming 0 processed files.")
        processed_files = set()

    print(f"Reading cache file: {source_path}...")
    try:
        cache_lines = Path(source_path).read_text(encoding='utf-8').splitlines()
        stripped_lines = map(str.strip, cache_lines)
        valid_lines = filter(None, stripped_lines)

        original_files = set(map(os.path.normpath, valid_lines))
        print(f"✅ Found {len(original_files)} physical base WAV files in the cache.\n")

    except FileNotFoundError:
        print(f"❌ FAILED: Cache file {source_path} not found! Please run your find command first.")
        return set()

    missing_in_json = original_files - processed_files
    missing_in_folder = processed_files - original_files

    if not missing_in_json and original_files:
        print("✅ SUCCESS: The JSON successfully exhausted EVERY audio file in the folder!")
    else:
        print(f"❌ FAILED: Found {len(missing_in_json)} base WAV files in the cache that are NOT in your JSON.")

        print("-" * 60)
        if missing_in_json:
            print(f"Missing Examples:\n -> " + "\n -> ".join(list(missing_in_json)[:5]))

        if missing_in_folder:
            print(
                f"\n⚠️ GHOST FILES: Found {len(missing_in_folder)} files in the JSON that do NOT exist in the folder anymore.")
            print(" -> " + "\n -> ".join(list(missing_in_folder)[:5]))

        print("=" * 60)
        return missing_in_json

    return set()

if __name__ == '__main__':
    part = '9'
    MANIFEST_PATH = f'/home/jovyan/asr-shared/csiargka/cache/datasets/hungarian/dataocean_asr_65{part}/manifests/hungarian_train.json'
    ORIGINAL_DATA_DIR = f'/home/jovyan/asr-shared/csiargka/cache/datasets/hungarian/dataocean_asr_65{part}/all_wav_files.txt'
    
    if part == '7':
        check_full_dataocean_657(MANIFEST_PATH, ORIGINAL_DATA_DIR)
    elif part == '9':
        check_full_dataocean_659(MANIFEST_PATH, ORIGINAL_DATA_DIR)