import os

import safetensors.torch
import torch
from os.path import dirname, abspath
import sys

from plotly.graph_objs.layout.slider import currentvalue

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from speechLM_utils.environment import set_environment

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
set_environment()

import transformers
transformers.logging.set_verbosity_error()

from config.parser import Parser
from preprocessing.split import splitter
from preprocessing.prepare import get_data
from preprocessing.prepare import concatenate
from speechLM_utils.data_collator import DataCollator
from speechLM_utils.train import get_checkpoint
from transformers import Seq2SeqTrainingArguments, GenerationConfig, Seq2SeqTrainer
import numpy as np
from metrics import get_metrics
from speechLM_utils.model import get_model
import wandb

from transformers import TrainerCallback
from tqdm.auto import tqdm
from speechLM_utils.utils import init_gpu, resume_wandb, init


class PredictionProgressCallback(TrainerCallback):
    def __init__(self):
        self.prediction_bar = None
        # Strictly rely on the OS environment variable you already trust
        self.is_main_process = int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]

    def on_prediction_step(self, args, state, control, **kwargs):
        if self.is_main_process:
            if self.prediction_bar is None:
                eval_dataloader = kwargs.get("eval_dataloader", None)
                total_batches = len(eval_dataloader) if eval_dataloader else None

                # Force output to stdout so multi-GPU launchers don't swallow it
                self.prediction_bar = tqdm(
                    total=total_batches,
                    desc="Evaluating Batches",
                    leave=True,
                    file=sys.stdout
                )

            self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if self.is_main_process and self.prediction_bar is not None:
            self.prediction_bar.close()
            self.prediction_bar = None

def load_model(model_type, args, info, device='cuda'):
    print(f"Initializing model...")

    model, _ = get_model(model_type, args, info, device)

    checkpoint_path, checkpoint_dir, _, _, _ = get_checkpoint(args, info, info['model_name'], False)
    safetensors_path = os.path.join(checkpoint_dir, 'model.safetensors')

    try:
        state_dict = safetensors.torch.load_file(safetensors_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"[{device}] Weights loaded. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    except Exception as e:
        print(f"[{device}] Error: Could not load weights. {e}")

    model.eval()
    return model

def debug_predictions(predictions, references, generated_ids, label_ids, tokenizer):
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    for pred, ref, gen, label in zip(predictions, references, generated_ids, label_ids):
        gen_clean = [token for token in gen if token != pad_id]
        label_clean = [token for token in label if token != -100 and token != pad_id]

        print(pred)
        print(ref)
        print()

        gen_len = len(gen_clean)
        ref_len = len(label_clean)

        print(gen_clean)
        print(label_clean)
        print()

        print(gen_len)
        print(ref_len)


def get_eval_metrics(data, model_type, conf, info, device = 'cuda'):
    if model_type == 'continuous_fusion':
        args = conf.cont_fuse_args
    elif model_type == 'slam_asr':
        args = conf.slam_args
    elif model_type == 'dual_fusion':
        args = conf.dual_fuse_args

    training_args = conf.eval_args.training_args
    training_args['report_to'] = "none"
    training_args['generation_config'] = GenerationConfig(**training_args['generation_config'])
    training_args['disable_tqdm'] = True
    training_args = Seq2SeqTrainingArguments(**training_args)

    model = load_model(model_type, args, info, device)
    tokenizer = model.language_tokenizer

    collator = DataCollator(processor=model.processor,
                            language_tokenizer=tokenizer,
                            padding='max_length',
                            truncation=True,
                            has_audio_lb_tokens=args['has_decoder'],
                            has_duration_lb=args['predict_duration'],
                            contain_index=False)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        args=training_args,
        callbacks=[PredictionProgressCallback()]
    )

    print(f"Starting inference on {len(data)} samples...")

    output = trainer.predict(data)
    indices = data['index']
    generated_ids = output.predictions
    label_ids = output.label_ids

    data_len = len(indices)
    generated_ids = generated_ids[:data_len]
    label_ids = label_ids[:data_len]

    generated_ids = np.where(generated_ids != -100, generated_ids, tokenizer.pad_token_id)
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

    predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # debug_predictions(predictions, references, generated_ids, label_ids, tokenizer)

    print(f"Computing metrics...")

    res_samples, res_total = get_metrics(predictions, references, indices, with_total_metrics=True)

    samples_path = os.path.join(info['res_folder'], "predictions.csv")
    total_path = os.path.join(info['res_folder'], "results.csv")

    res_samples.to_csv(samples_path)
    res_total.to_csv(total_path)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank in [-1, 0] and wandb.run is not None:
        dataset_name = info['test_dataset']
        current_step = int(info.get('turn', 0))

        wandb.log({
            f"{dataset_name}/predictions": wandb.Table(dataframe=res_samples),
            f"{dataset_name}/results": wandb.Table(dataframe=res_total)
        }, step=current_step)

    return samples_path, total_path

def get_results_path(conf, info, data_folder: bool = True):
    results_path = os.path.join(os.path.expanduser('~'),
                                conf.results_path,
                                info['model_type'],
                                info['machine'],
                                info['model_name'],
                                info['datetime'],
                                info['turn'])

    if data_folder:
        results_path = os.path.join(results_path, info['test_dataset'])

    os.makedirs(results_path, exist_ok=True)

    return results_path

def evaluate(info, set='test', device='cuda', iters = None):
    conf, args, _, bad_folder, _, _, _, _ = init(info['model_type'], info)

    res_folder = get_results_path(conf, info)
    info['res_folder'] = res_folder

    if type(info['test_dataset']) is not list:
        dataset_name = [info['test_dataset']]

    split = splitter()
    data = split.split(datasets=dataset_name)
    data_set = get_data(data[set], bad_folder, iters=iters, normalized=False, has_duration=True, filters=FILTERS, split=set)
    data_set = concatenate(data_set)

    print(f"Evaluating on {dataset_name} ({len(data_set)} samples)...")
    get_eval_metrics(data_set, info['model_type'], conf, info, device)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
DATASETS = ['fleurs', 'common_voice', 'hparl', 'tedx', 'logotypographia']
FILTERS = ['duration', 'length']

BASE_INFO = {
        'model_type': 'dual_fusion',
        'res_folder': None,
        'train_dataset': ['common_voice', 'fleurs', 'hparl', 'tedx', 'logotypographia'],
        's_': False,
        'checkpoint_folder': "dual_fusion_checkpoints",
        'model_name': None,
        'speech_encoder_id': 'openai/whisper-large-v3',
        'language_model_id': 'ilsp/Llama-Krikri-8B-Instruct',
        'bit4': True,
        'machine': 'kronos',
        'datetime': 'Apr03_16-04',
        'turn': '32500'
    }

if __name__ == "__main__":
    local_rank, device = init_gpu()
    resume_wandb(local_rank, BASE_INFO)

    try:
        for dataset in DATASETS:
            info = BASE_INFO.copy()
            info['test_dataset'] = dataset

            evaluate(info,
                     set='test',
                     device=device,
                     iters=None)

        wandb.finish()

    finally:
        if local_rank in [-1, 0] and wandb.run is not None:
            wandb.finish()