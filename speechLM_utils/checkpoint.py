from typing import Dict, List
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from os.path import exists, join, isdir
import os
import socket

def get_max_step(checkpoint_dir):
    max_step = 0
    for filename in os.listdir(checkpoint_dir):
        if isdir(join(checkpoint_dir, filename)) and filename.startswith(
            "checkpoint"
        ):
            max_step = max(max_step, int(filename.replace("checkpoint-", "")))

    return max_step

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, "completed"))
        if is_completed:
            return checkpoint_dir, True

        max_step = get_max_step(checkpoint_dir)
        if max_step == 0:
            return None, is_completed

        checkpoint_dir = join(checkpoint_dir, f"checkpoint-{max_step}")

        return checkpoint_dir, is_completed

    return None, False

def get_last_name(args: Dict, info: Dict, restart: bool):
    lora = 'LoRA_' + str(int(args['linguistic_lora']))
    dataset = info['train_dataset']
    date = '' if info['datetime'] is None else info['datetime']

    if restart:
        date = datetime.now().strftime("%b%d_%H-%M")

    return f'{lora}_{dataset}@{date}', date

def get_checkpoint(args: Dict, info: Dict, model_name: str, restart: bool = False):
    last_name, date = get_last_name(args, info, restart)

    if 'machine' in info and info['machine'] is not None:
        machine = info['machine']
    else:
        machine = socket.gethostname()

    checkpoint_path = os.path.join(os.path.expanduser('~'),
                                   args.checkpoint_path,
                                   info['checkpoint_folder'],
                                   machine,
                                   model_name,
                                   last_name)

    os.makedirs(checkpoint_path, exist_ok=True)
    writer = SummaryWriter(log_dir=checkpoint_path)

    if 'turn' in info and info['turn'] is not None:
        checkpoint_dir = join(checkpoint_path, f"checkpoint-{info['turn']}")
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
    else:
        checkpoint_dir, completed_training = get_last_checkpoint(checkpoint_path)

        if completed_training:
            print("Detected that training was already completed!")

    if checkpoint_dir is not None:
        print(f"Found checkpoint at: {checkpoint_dir}")

    return checkpoint_path, checkpoint_dir, writer, checkpoint_path, date