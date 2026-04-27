import json
import torch
import numpy as np
from pathspec import patterns
from tqdm import tqdm
from cleanlab import Datalab
from transformers import WhisperModel, WhisperFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration
import librosa
import os
from config.parser import Parser
import pandas as pd
from pathlib import Path
import torchaudio
import re
from preprocessing.normalize import normalize
from preprocessing.parallel import parallelize_process
from typing import Dict, List

SIL_MODEL = "snakers4/silero-vad"
LANG_MODEL = "openai/whisper-tiny"
EMBED_MODEL = "openai/whisper-tiny"

OUTPUT_REPORT = "cleanlab_report"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def check_duration(data, info, bad_folder):
    min_thres = 0.5
    max_thres = None

    durations = np.array([entry['duration'] for entry in data])

    if max_thres:
        bad_indices = np.where((durations < min_thres) | (durations > max_thres))[0]
    else:
        bad_indices = np.where(durations < min_thres)[0]

    bad_files = pd.DataFrame(bad_indices, columns=['bad_index'])

    bad_files['filepath'] = bad_files.bad_index.map(lambda x: data[x]['audio_filepath'])
    bad_files['text'] = bad_files.bad_index.map(lambda x: data[x]['text'])
    bad_files['duration'] = bad_files.bad_index.map(lambda x: durations[x])

    dataset = info['dataset']
    split = info['split']

    bad_files.to_csv(os.path.join(bad_folder, f"bad_by_duration_{dataset}_{split}.csv"))

def check_length(data, info, bad_folder):
    min_thres = 2
    max_thres = None

    text_lens = np.array([len(normalize(entry['text'], with_signs=False)) for entry in data])

    if max_thres:
        bad_indices = np.where((text_lens < min_thres) | (text_lens > max_thres))[0]
    else:
        bad_indices = np.where(text_lens < min_thres)[0]

    bad_files = pd.DataFrame(bad_indices, columns=['bad_index'])

    bad_files['filepath'] = bad_files.bad_index.map(lambda x: data[x]['audio_filepath'])
    bad_files['text'] = bad_files.bad_index.map(lambda x: data[x]['text'])
    bad_files['normalized'] = bad_files.bad_index.map(lambda x: normalize(data[x]['text'], with_signs=False))
    bad_files['length'] = bad_files.bad_index.map(lambda x: text_lens[x])

    dataset = info['dataset']
    split = info['split']

    bad_files.to_csv(os.path.join(bad_folder, f"bad_by_length_{dataset}_{split}.csv"))

def check_ratio(data, info, bad_folder):
    max_thres = 30
    min_thres = 2
    durations = [entry['duration'] for entry in data]

    text_lens = [len(normalize(entry['text'], with_signs=False)) for entry in data]
    ratios = np.array(text_lens) / np.array(durations)

    bad_indices = np.where((ratios < min_thres) | (ratios > max_thres))[0]
    bad_files = pd.DataFrame(bad_indices, columns=['bad_index'])

    bad_files['filepath'] = bad_files.bad_index.map(lambda x: data[x]['audio_filepath'])
    bad_files['text'] = bad_files.bad_index.map(lambda x: data[x]['text'])
    bad_files['normalized'] = bad_files.bad_index.map(lambda x: normalize(data[x]['text'], with_signs=False))
    bad_files['length'] = bad_files.bad_index.map(lambda x: text_lens[x])
    bad_files['duration'] = bad_files.bad_index.map(lambda x: durations[x])
    bad_files['ratio'] = bad_files.bad_index.map(lambda x: ratios[x])

    dataset = info['dataset']
    split = info['split']

    bad_files.to_csv(os.path.join(bad_folder, f"bad_by_ratio_{dataset}_{split}.csv"))

def get_silence(x, device, model, get_model_timestamps):
    path = x['audio_filepath']

    try:
        audio, sr = torchaudio.load(path)
        audio_dev = audio.to(device)

        speech_timestamps = get_model_timestamps(audio_dev, model, sampling_rate=16000)

        total_speech_samples = sum([t['end'] - t['start'] for t in speech_timestamps])
        total_samples = audio.shape[1]
        speech_ratio = total_speech_samples / total_samples

        return speech_ratio

    except Exception as e:
        print(f"Error processing {path}: {e}")

def check_silence_(data, gpu_id, info):
    device = torch.device(f"cuda:{gpu_id}")

    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )

    vad_model.to(device)
    (get_speech_timestamps, _, _, _, _) = utils

    speech = []
    ratios = []

    progress_bar = tqdm(
        data,
        desc=f"GPU {gpu_id}",
        position=gpu_id,
        leave=True,
        ncols=100,  # Fix width to ensure neat alignment
        colour='green'  # Optional: makes it look nicer
    )

    for entry in progress_bar:
        speech.append(get_silence(entry, device, vad_model, get_speech_timestamps))
        if info['to_ratio']:
            ratios.append(len(normalize(entry['text'], with_signs=False)) / entry['duration'])

    speech = np.array(speech)
    bad_indices = np.where(speech < info['threshold'])[0]

    if info['to_ratio']:
        ratios = np.array(ratios)
        bad_indices = np.where(speech / ratios < info['threshold'])[0]

    bad_files = pd.DataFrame(bad_indices, columns=['bad_index'])

    bad_files['filepath'] = bad_files.bad_index.map(lambda x: data[x]['audio_filepath'])
    bad_files['text'] = bad_files.bad_index.map(lambda x: data[x]['text'])
    bad_files['duration'] = bad_files.bad_index.map(lambda x: data[x]['duration'])
    bad_files['speech_ratio'] = bad_files.bad_index.map(lambda x: speech[x])
    bad_files['text_ratio'] = bad_files.bad_index.map(lambda x: ratios[x])

    dataset = info['dataset']
    split = info['split']

    path = os.path.join(BAD_FOLDER, f"bad_by_silence_{dataset}_{split}_{gpu_id}.csv")
    bad_files.to_csv(path)

    return path

def get_noise(x, device, x_timestamps = None):
    path = x['audio_filepath']

    try:
        wav, sr = torchaudio.load(path)
        wav = wav.to(device)
        wav = wav.squeeze()

        if x_timestamps is None:
            wav_power = torch.mean(wav ** 2)
            sorted_power = torch.sort(wav ** 2)
            sil_samples = sorted_power[int(0.1 * len(sorted_power))]
            sil_power = torch.mean(sil_samples)

            snr = 10 * torch.log10(wav_power / sil_power + 1e-10)

        else:
            mask = torch.zeros_like(wav, dtype=torch.bool, device=device)
            for t in x_timestamps:
                mask[t['start']:t['end']] = True

            wav_power = torch.mean(wav[mask] ** 2)
            sil_power = torch.mean(wav[~mask] ** 2) if (~mask).any() else 1e-9

            snr = 10 * torch.log10(wav_power / sil_power + 1e-10)

        return snr.item()

    except Exception as e:
        print(f"Error processing {path}: {e}")

def check_noise_(data, id, speech_timestamps = None):
    progress_bar = tqdm(
        data,
        desc=f"GPU {id}",
        position=id,
        leave=True,
        ncols=100,
        colour='green'
    )

    SNRs = []

    for i, entry in enumerate(progress_bar):
        SNRs.append(get_noise(entry, speech_timestamps[i]))

    bad_indices = np.where(SNRs < 5.0)[0]
    bad_files = pd.DataFrame(bad_indices, columns=['bad_index'])

    bad_files['filepath'] = bad_files.bad_index.map(lambda x: data[x]['audio_filepath'])
    bad_files['text'] = bad_files.bad_index.map(lambda x: data[x]['text'])
    bad_files['SNR'] = bad_files.bad_index.map(lambda x: SNRs[x])

    path = os.path.join(BAD_FOLDER, f"bad_by_SNR_{id}.csv")
    bad_files.to_csv(path)

    return path

def check_silence_snr_(data, gpu_id):
    device = torch.device(f"cuda:{gpu_id}")

    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )

    vad_model.to(device)
    (get_speech_timestamps, _, _, _, _) = utils

    speech = []
    SNRs = []

    progress_bar = tqdm(
        data,
        desc=f"GPU {gpu_id}",
        position=gpu_id,
        leave=True,
        ncols=100,  # Fix width to ensure neat alignment
        colour='green'  # Optional: makes it look nicer
    )

    for entry in progress_bar:
        speech_ratio, speech_timestamps = get_silence(entry, device, vad_model, get_speech_timestamps)
        snr = get_noise(entry, device, speech_timestamps)

        speech.append(speech_ratio)
        SNRs.append(snr)

    SNRs = np.array(SNRs)
    snr_indices = np.where(SNRs < 5.0)[0]
    snr_files = pd.DataFrame(snr_indices, columns=['bad_index'])

    snr_files['filepath'] = snr_files.bad_index.map(lambda x: data[x]['audio_filepath'])
    snr_files['text'] = snr_files.bad_index.map(lambda x: data[x]['text'])
    snr_files['SNR'] = snr_files.bad_index.map(lambda x: SNRs[x])

    snr_path = os.path.join(BAD_FOLDER, f"bad_by_SNR_{gpu_id}.csv")
    snr_files.to_csv(snr_path)

    speech = np.array(speech)
    sil_indices = np.where(speech < 0.2)[0]
    sil_files = pd.DataFrame(sil_indices, columns=['bad_index'])

    sil_files['filepath'] = sil_files.bad_index.map(lambda x: data[x]['audio_filepath'])
    sil_files['text'] = sil_files.bad_index.map(lambda x: data[x]['text'])
    sil_files['speech_ratio'] = sil_files.bad_index.map(lambda x: speech[x])

    sil_path = os.path.join(BAD_FOLDER, f"bad_by_silence_{gpu_id}.csv")
    sil_files.to_csv(sil_path)

    del SNRs, speech, snr_files, sil_files

    return sil_path, snr_path

def get_language(x, device, model, processor, greek_token_id):
    path = x['audio_filepath']

    try:
        audio, sr = librosa.load(path, sr=16000, duration=30.0)
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]]).to(device)

        with torch.no_grad():
            outputs = model(input_features, decoder_input_ids=decoder_input_ids)

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        greek_conf = probs[greek_token_id].item()

        top_prob, top_id = torch.max(probs, dim=-1)
        top_token = processor.tokenizer.convert_ids_to_tokens(top_id.item())

        return top_token, top_prob.item(), greek_conf

    except Exception as e:
        print(f"Error processing {path}: {e}")

def check_language_(data, gpu_id):
    device = torch.device(f"cuda:{gpu_id}")

    processor = WhisperProcessor.from_pretrained(SIL_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(SIL_MODEL).to(device)
    greek_token_id = processor.tokenizer.convert_tokens_to_ids("<|el|>")

    languages = []
    scores = []
    confs = []

    progress_bar = tqdm(
        data,
        desc=f"GPU {gpu_id}",
        position=gpu_id,
        leave=True,
        ncols=100,  # Fix width to ensure neat alignment
        colour='green'  # Optional: makes it look nicer
    )

    for entry in progress_bar:
        lang, score, conf = get_language(entry, device, model, processor, greek_token_id)
        languages.append(lang)
        scores.append(score)
        confs.append(conf)

    languages = np.array(languages)
    scores = np.array(scores)
    confs = np.array(confs)

    bad_indices = np.where((languages != '<|el|>') & (scores > 0.75))[0]
    bad_files = pd.DataFrame(bad_indices, columns=['bad_index'])

    bad_files['filepath'] = bad_files.bad_index.map(lambda x: data[x]['audio_filepath'])
    bad_files['text'] = bad_files.bad_index.map(lambda x: data[x]['text'])
    bad_files['language'] = bad_files.bad_index.map(lambda x: languages[x])
    bad_files['score'] = bad_files.bad_index.map(lambda x: scores[x])
    bad_files['greek_confidence'] = bad_files.bad_index.map(lambda x: confs[x])

    path = os.path.join(BAD_FOLDER, f"bad_by_language_{gpu_id}.csv")
    bad_files.to_csv(path)

    return path

def get_issue(x, patterns):
    text = x['text'].strip()

    if not text:
        return 'empty_text'

    elif patterns['acoustic'].search(text):
       return 'acoustic_tag'

    elif patterns['speaker'].search(text):
        return 'speaker_tag'

    clean_text = re.sub(r'[\s\.,;!\?\'"«»\-]', '', text)
    if len(clean_text) > 0:
        greek_chars = patterns['greek'].findall(clean_text)
        greek_ratio = len(greek_chars) / len(clean_text)

        if greek_ratio < 0.7:
            return "foreign_text"

    return 'none'

def check_text(data, info):
    progress_bar = tqdm(
        data,
        leave=True,
        ncols=100,
        colour='green'
    )

    patterns = {
        'acoustic': re.compile(r'[\[\(\<\{].*?[\]\)\>\}]'),
        'speaker': re.compile(r'^[\w\u0370-\u03FF\u1F00-\u1FFF]+\s*\d*:\s'),
        'digits': re.compile(r'\d+'),
        'latin': re.compile(r'[a-zA-Z]'),
        'greek': re.compile(r'[\u0370-\u03FF\u1F00-\u1FFF]')
    }

    issues = []
    for i, entry in enumerate(progress_bar):
        issues.append(get_issue(entry, patterns))

    issues = np.array(issues)
    bad_indices = np.where(issues != 'none')[0]
    bad_files = pd.DataFrame(bad_indices, columns=['bad_index'])

    bad_files['filepath'] = bad_files.bad_index.map(lambda x: data[x]['audio_filepath'])
    bad_files['text'] = bad_files.bad_index.map(lambda x: data[x]['text'])
    bad_files['issue'] = bad_files.bad_index.map(lambda x: issues[x])

    dataset = info['dataset']
    split = info['split']

    bad_files.to_csv(os.path.join(BAD_FOLDER, f"bad_by_text_{dataset}_{split}.csv"))

def get_prediction(x, device, model, processor, forced_decoder_ids = None):
    path = x['audio_filepath']

    try:
        audio, sr = librosa.load(path, sr=16000, duration=30.0)
        text = x['text']

        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        labels = processor(text=text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_features, labels=labels)
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            loss = outputs.loss.item()

        return np.exp(-loss), transcription

    except Exception as e:
        print(f"Error processing {path}: {e}")

def check_predictions_(data, gpu_id, use_forced_decoder: bool = True):
    device = torch.device(f"cuda:{gpu_id}")

    processor = WhisperProcessor.from_pretrained(EMBED_MODEL, language="greek", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(EMBED_MODEL).to(device)

    if use_forced_decoder:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="greek", task="transcribe")
    else:
        forced_decoder_ids = None

    confidences = []
    transcriptions = []

    progress_bar = tqdm(
        data,
        desc=f"GPU {gpu_id}",
        position=gpu_id,
        leave=True,
        ncols=100,  # Fix width to ensure neat alignment
        colour='green'  # Optional: makes it look nicer
    )

    for entry in progress_bar:
        conf, transcription = get_prediction(entry, device, model, processor, forced_decoder_ids)
        confidences.append(conf)
        transcriptions.append(transcription)

    scores = np.array(confidences)
    threshold = np.percentile(scores, 10)  # Look at bottom 10%
    bad_indices = np.where(scores < threshold)[0]
    bad_files = pd.DataFrame(bad_indices, columns=['bad_index'])

    bad_files['filepath'] = bad_files.bad_index.map(lambda x: data[x]['audio_filepath'])
    bad_files['text'] = bad_files.bad_index.map(lambda x: data[x]['text'])
    bad_files['confidence_score'] = bad_files.bad_index.map(lambda x: scores[x])
    bad_files['transcription'] = bad_files.bad_index.map(lambda x: transcriptions[x])

    path = os.path.join(BAD_FOLDER, f"bad_by_prediction_{gpu_id}.csv")
    bad_files.to_csv(path)

    return path

def get_embedding(x, device, model, feature_extractor):
    path = x['audio_filepath']

    try:
        audio, sr = librosa.load(path, sr=16000, duration=30.0)
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            last_hidden_state = model(inputs).last_hidden_state
            embedding = last_hidden_state.mean(dim=1).cpu().numpy()

        return embedding.squeeze()

    except Exception as e:
        print(f"Error processing {path}: {e}")

def check_embeddings_(data, gpu_id):
    device = torch.device(f"cuda:{gpu_id}")

    model = WhisperModel.from_pretrained(EMBED_MODEL).encoder.to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(EMBED_MODEL)

    progress_bar = tqdm(
        data,
        desc=f"GPU {gpu_id}",
        position=gpu_id,
        leave=True,
        ncols=100,  # Fix width to ensure neat alignment
        colour='green'  # Optional: makes it look nicer
    )

    embeddings = []
    filepaths = []

    for entry in progress_bar:
        embeddings.append(get_embedding(entry, gpu_id, model, feature_extractor))

    embeddings = np.array(embeddings)

    lab = Datalab(data={"filepath": filepaths}, label_name=None)
    lab.find_issues(features=embeddings, issue_types={"outlier": {}, "near_duplicate": {"metric": "cosine",
                                                                                        "threshold": 0.3}})

    outliers = lab.get_issues("outlier")
    dupes = lab.get_issues("near_duplicate")


    outlier_indices = outliers[outliers["outlier_score"] < 0.02].index
    duplicate_indices = dupes[dupes["is_near_duplicate_issue"] == True].index

    bad_indices = set(outlier_indices).union(set(duplicate_indices))
    bad_files = pd.DataFrame(bad_indices, columns=['bad_index'])

    with open(manifest_path, 'r') as f:
        all_data = [json.loads(line) for line in f]
        bad_files['filepath'] = bad_files.bad_index.map(lambda x: all_data[x]['audio_filepath'])
        bad_files['text'] = bad_files.bad_index.map(lambda x: all_data[x]['text'])
        bad_files['outlier_score'] = bad_files.bad_index.map(lambda x: outliers['outlier_score'][x])
        bad_files['near_duplicates'] = bad_files.bad_index.map(lambda x: dupes['near_duplicate_sets'][x])

    bad_files = bad_files.sort_values('outlier_score', ascending=True)
    path = os.path.join(BAD_FOLDER, f"bad_by_embedding_{gpu_id}.csv")
    bad_files.to_csv(path)


def check_duplicate_texts(data, info):
    text_tracker = {}

    for dataname, dataset in data.items():
        for idx, entry in enumerate(tqdm(dataset)):
            text = normalize(entry['text'], with_signs=False)
            if text not in text_tracker:
                text_tracker[text] = []
            text_tracker[text].append((dataname, idx))

        bad_indices = []
        for text, indices in text_tracker.items():
            if len(indices) > 1:
                bad_indices.extend(indices)

        bad_files = pd.DataFrame(bad_indices, columns=['bad_index'])

        bad_files['filepath'] = bad_files.bad_index.map(lambda x: data[x]['audio_filepath'])
        bad_files['dataset'] = dataname
        bad_files['text'] = bad_files.bad_index.map(lambda x: data[x]['text'])
        bad_files['duplicates'] = bad_files.bad_index.map(lambda x: text_tracker[data[x]['text']])

    dataset = info['dataset']
    split = info['split']

    bad_files.to_csv(os.path.join(BAD_FOLDER, f"bad_by_duplicate_{dataset}_{split}.csv"))

# import os
# # Disable tokenizer parallelism warning and transformers info
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"

def duplicates():
    conf = Parser()
    conf.get_args()

    datasets = ['common_voice', 'fleurs']
    splits = ['train', 'validation', 'test']

    for split in splits:
        info = {
            'dataset': datasets,
            'split': split
        }

        data = {}
        for dataname, dataset in datasets:
            manifest_folder = os.path.join(conf.dataset_path, dataname, 'manifests')
            file = os.path.join(manifest_folder, f'greek_{split}.json')

            with open(file, 'r', encoding='utf-8') as f:
                data[dataname] = [json.loads(line) for line in f]

        check_duplicate_texts(data, info)

if '__main__' == __name__:
    conf = Parser()
    conf.get_args()

    datasets = ['common_voice', 'fleurs', 'hparl', 'logotypographia', 'tedx', 'stoma']
    splits = ['train', 'validation', 'test']

    bad_folder = os.path.join(os.path.expanduser('~'),
                              conf.dataset_path,
                              conf.language,
                              'bad_folder')

    os.makedirs(bad_folder, exist_ok=True)

    for dataset in datasets:
        for split in splits:
            info = {
                'dataset': dataset,
                'split': split,
                'to_ratio': True,
                'threshold': 0.1
            }

            manifest_folder = os.path.join(os.path.expanduser('~'), conf.dataset_path, conf.language, dataset, 'manifests')

            file = os.path.join(manifest_folder, f'greek_{split}.json')
            with open(file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]

            check_duration(data, info, bad_folder)
            check_length(data, info, bad_folder)
            check_ratio(data, info, bad_folder)

        # torch.multiprocessing.set_start_method('spawn', force=True)
        # parallelize_process(data, check_silence_, gpus=[0,1,2,3], info=info)
        # concat_dataframes(os.path.join(BAD_FOLDER, f"bad_by_{task}_{dataset}_{split}"))
        #


