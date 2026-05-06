import warnings

warnings.filterwarnings("ignore")
import os
import sys
from os.path import dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))

import shutil
from config.parser import Parser
from datetime import datetime
from pprint import pprint

def cleanup_short_runs(base_dir, min_steps, cutoff_date, contains_one_dataset=False, contains_dataset=None, dry_run=True, keep_after: bool = False):
    kept = []
    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} does not exist.")
        return

    for machine in os.listdir(base_dir):
        print('-------------------------------------------------------------')
        print(f'DELETING CHECKPOINT FOLDERS FOR {machine}')

        machine_path = os.path.join(base_dir, machine)
        if not os.path.isdir(machine_path): continue

        for model in os.listdir(machine_path):
            model_path = os.path.join(machine_path, model)
            if not os.path.isdir(model_path): continue

            for run in os.listdir(model_path):
                run_path = os.path.join(model_path, run)
                if not os.path.isdir(run_path): continue

                try:
                    date_string = run.split('@')[-1]
                    parsed_date = datetime.strptime(date_string, "%b%d_%H-%M")
                except ValueError:
                    shutil.rmtree(run_path, ignore_errors=True)
                    print(f"Skipping {run} due to date parsing error.")
                    continue

                current_year = datetime.now().year
                folder_mtime = parsed_date.replace(year=current_year)
                is_older_than_cutoff = folder_mtime < cutoff_date

                if keep_after:
                    if is_older_than_cutoff:
                        continue
                    else:
                        kept.append(run_path)

                else:
                    contains = False

                    if contains_one_dataset:
                        if len(run.split('@')[-2].split(',')) == 1:
                            contains = True

                    if contains_dataset is not None:
                        if contains_dataset in run.split('@')[-2]:
                            contains = True

                    max_step = 0
                    for item in os.listdir(run_path):
                        if os.path.isdir(os.path.join(run_path, item)) and item.startswith("checkpoint-"):
                            try:
                                step = int(item.replace("checkpoint-", ""))
                                max_step = max(max_step, step)
                            except ValueError:
                                pass

                        elif os.path.isdir(os.path.join(run_path, item)) and item.startswith('stage'):
                            run_path = os.path.join(run_path, item)
                            for item in os.listdir(run_path):
                                if os.path.isdir(os.path.join(run_path, item)) and item.startswith("checkpoint-"):
                                    try:
                                        step = int(item.replace("checkpoint-", ""))
                                        max_step = max(max_step, step)
                                    except ValueError:
                                        pass

                            break

                    if contains and is_older_than_cutoff:
                        print(f"[DELETE] Run: {run} | Contains dataset: {contains}")
                        if not dry_run:
                            shutil.rmtree(run_path, ignore_errors=True)

                    elif max_step < min_steps and is_older_than_cutoff:
                        print(f"[DELETE] Run: {run} | Max step: {max_step}")
                        if not dry_run:
                            shutil.rmtree(run_path, ignore_errors=True)

                    else:
                        print(f"[KEEP] Run: {run} (Max step: {max_step})")
                        kept.append(run_path)

    print()
    pprint(kept)

    return kept

if __name__ == "__main__":
    conf = Parser()
    conf.get_args()

    task = 'dual_fusion_checkpoints'
    checkpoint_dir = os.path.join(os.path.expanduser('~'),
                                conf.dual_fuse_args.checkpoint_path,
                                task)
    min_steps = 5000
    cutoff_date = datetime(2026, 6, 7, 5, 0)
    contains_one_dataset = True
    contains_dataset = 'eurospeech'

    kept = cleanup_short_runs(checkpoint_dir, min_steps, cutoff_date, contains_one_dataset, contains_dataset, dry_run=False)