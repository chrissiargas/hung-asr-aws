import warnings
warnings.filterwarnings("ignore")

import os
import sys
from os.path import dirname

sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))
from speechLM_utils.environment import set_environment

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
set_environment()

import gc
import torch
import os
from typing import List
from config.parser import Parser
from pathlib import Path
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import evaluate
perplexity_metric = evaluate.load("perplexity", module_type="metric")
from tqdm import tqdm

def init_gpu():
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        print(f"Process launched on GPU: {local_rank}")
    else:
        device = "cuda"
        print("Running in standard mode (DataParallel or Single GPU)")

    return local_rank, device


def calculate_perplexity(model, tokenizer, transcripts, device):
    nlls = []
    total_length = 0

    print(f"Evaluating {len(transcripts)} transcripts...")

    with torch.no_grad():
        for text in tqdm(transcripts):
            inputs = tokenizer(text, return_tensors="pt").to(device)
            input_ids = inputs.input_ids

            if input_ids.size(1) == 0:
                continue

            target_ids = input_ids.clone()

            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * input_ids.size(1)
            nlls.append(neg_log_likelihood)
            total_length += input_ids.size(1)

    ppl = torch.exp(torch.stack(nlls).sum() / total_length)
    return ppl.item()

def main(datasets: List[str], models: List[str]):
    local_rank, device = init_gpu()

    conf = Parser()
    conf.get_args()

    base_dir = os.path.join(
        os.path.expanduser('~'),
        conf.dataset_path,
        conf.language
    )

    print("\n--- Downloading & Extracting Datasets ---")
    corpus_texts = {}

    for dataset in datasets:
        dataset_path = os.path.join(base_dir, dataset, 'manifests')
        for split_file in os.listdir(dataset_path):
            split = split_file.split('.')[0].split('_')[-1]

            manifest_path = Path(os.path.join(base_dir, dataset, 'manifests', split_file))

            print(f"Loading {dataset} ({split_file})...")

            with open(manifest_path, 'r', encoding='utf-8') as f:
                corpus_texts[dataset][split] = [json.loads(line).get('text', '') for line in f]

    print(corpus_texts)

    results = {}
    for model_id in models:
        print(f"\n>> Loading Model: {model_id}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                attn_implementation='sdpa',
            )

            model.eval()

            for ds_label, split_texts in corpus_texts.items():
                print(f"Evaluating on {ds_label}...")
                for split, texts in split_texts.items():
                    print(f"  -> Split: {split}")

                    ppl = calculate_perplexity(model, tokenizer, texts, device)
                    results[model_id][ds_label] = ppl
                    print(f" -> Perplexity (PPL): {ppl:.2f}")

            # Quick Generative Sanity Check
            print("\n--- Generative Sanity Check ---")
            sample_prompt = "Kérlek, írd le a beszédfelismerés fontosságát:"
            inputs = tokenizer(sample_prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id
            )
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))

            # Aggressively free memory before loading the next model
            del model
            del tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f" -> Error evaluating {model_id}: {e}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='GPUs to be used')
    parser.add_argument('--datasets', nargs='+', type=str,
                        default=['common_voice', 'fleurs', 'hparl', 'tedx', 'logotypographia'], help='datasets')
    parser.add_argument('--models', nargs='+', type=str, default=['elte-nlp/Racka-4B'])
    args, unknown = parser.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    main(args.datasets, args.models)

