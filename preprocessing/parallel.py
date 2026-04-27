import json
import math
import torch
from concurrent.futures import ProcessPoolExecutor
import random
import pandas as pd
import glob
from pathlib import Path
import os
from typing import Optional, List

BAD_FOLDER = os.path.join(Path(__file__).parent.parent, "bad_folder")

def concat_dataframes(filename, remove: bool = True):
    file_list = glob.glob(f"{filename}_*.csv")

    results = pd.concat([pd.read_csv(f) for f in file_list], ignore_index=True)
    result_path = Path(filename + '.csv')
    if result_path.exists():
        os.remove(result_path)

    if remove:
        for file in file_list:
            os.remove(file)

    results.to_csv(result_path, index=False)

    return results

def parallelize_process(data, func, gpus: Optional[List] = None, info: Optional[dict] = None):
    total_files = len(data)

    if gpus is None:
        n_gpus = torch.cuda.device_count()
        gpus = range(n_gpus)

    else:
        n_gpus = len(gpus)

    chunk_size = math.ceil(total_files / n_gpus)
    file_chunks = [data[chunk_size * i: chunk_size * (i+1)] for i in range(n_gpus - 1)]
    file_chunks.append(data[chunk_size * (n_gpus - 1):])

    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        futures = []
        for g, gpu in enumerate(gpus):
            futures.append(executor.submit(func, file_chunks[g], gpu, info))

    for future in futures:
        future.result()

if __name__ == "__main__":
    concat_dataframes(os.path.join(BAD_FOLDER, "bad_by_silence"))
