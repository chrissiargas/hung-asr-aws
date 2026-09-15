import os
import json
import pandas as pd
import torchaudio

# Base path derived from your my_config.yaml
BASE_DIR = "/home/jovyan/asr-shared/csiargka/cache/datasets"
LANGUAGE = "hungarian"
DATASETS = [
    'dataocean_asr_657',
    'dataocean_asr_659',
    'datatang_asr_1',
    'datatang_asr_2'
]


def analyze_dataset(dataset_name):
    # Adjust filename if your manifest uses a different naming convention
    manifest_path = os.path.join(BASE_DIR, LANGUAGE, dataset_name, 'manifests', f'{LANGUAGE}.json')

    if not os.path.exists(manifest_path):
        print(f"❌ Manifest not found for {dataset_name} at {manifest_path}")
        return

    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)

    # 1. Calculate Total Duration
    if 'duration' in df.columns:
        total_duration_sec = df['duration'].sum()
        total_duration_hours = total_duration_sec / 3600
    else:
        total_duration_hours = 0.0

    # 2. Extract Speaker/Session Info
    if 'subject' in df.columns:
        num_speakers = df['subject'].nunique()
        speaker_type = "Subjects"
    elif 'session' in df.columns:
        num_speakers = df['session'].nunique()
        speaker_type = "Sessions"
    else:
        num_speakers = "Unknown"
        speaker_type = "N/A"

    # 3. Check Sampling Rate & Channels using torchaudio
    sample_rate = "Unknown"
    channels = "Unknown"
    if 'audio_filepath' in df.columns and len(df) > 0:
        # Grab the first valid audio file to check metadata
        sample_audio = df.iloc[0]['audio_filepath']
        if not os.path.isabs(sample_audio):
            sample_audio = os.path.join(BASE_DIR, LANGUAGE, dataset_name, sample_audio)

        try:
            metadata = torchaudio.info(sample_audio)
            sample_rate = metadata.sample_rate
            channels = metadata.num_channels
        except Exception as e:
            sample_rate = f"Error reading audio: {e}"

    # 4. Sample Texts for Qualitative Domain Inference
    text_col = 'text' if 'text' in df.columns else ('reference' if 'reference' in df.columns else None)
    if text_col:
        sample_texts = df[text_col].dropna().sample(min(5, len(df)), random_state=42).tolist()
    else:
        sample_texts = []

    # Print Results
    print(f"==================================================")
    print(f"📊 DATASET: {dataset_name.upper()}")
    print(f"==================================================")
    print(f"Total Duration : {total_duration_hours:.2f} hours")
    print(f"Total Samples  : {len(df):,}")
    print(f"Unique {speaker_type}: {num_speakers}")
    print(f"Sampling Rate  : {sample_rate} Hz")
    print(f"Channels       : {channels} (1=Mono, 2=Stereo)")
    print(f"\n📝 Sample Transcripts (Use these to identify the domain):")
    for txt in sample_texts:
        print(f"  - \"{txt}\"")
    print("\n")


if __name__ == "__main__":
    for ds in DATASETS:
        analyze_dataset(ds)