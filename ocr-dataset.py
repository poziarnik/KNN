import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from langdetect import detect
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

splits = {
    "ground-truth.split1": {
        "model-1": "transcriptions.model1_split1",
        "model-2": "transcriptions.model2_split1", 
    },
    "ground-truth.split2": {
        "model-1": "transcriptions.model1_split2",
        "model-2": "transcriptions.model2_split2", 
    }
}

def get_text(line: str) -> str:
    return line.split(' ', 1)[1]

def process_line(data: tuple[str, str, str]):
    return get_text(data[0]), get_text(data[1]), get_text(data[2])

df = pd.DataFrame()

for split_name, split in splits.items():
    gt_lines = (Path("./KNN-data") / split_name).read_text().splitlines()
    model1_lines = (Path("./KNN-data") / split["model-1"]).read_text().splitlines()
    model2_lines = (Path("./KNN-data") / split["model-2"]).read_text().splitlines()

    data_generator = zip(gt_lines, model1_lines, model2_lines)

    out = []

    with Pool(processes=cpu_count()) as p, tqdm(total=len(gt_lines), desc="Processing data.") as pbar:
        for result in p.map(process_line, data_generator):
            out.append({'gt': result[0], 'model1': result[1], 'model2': result[2]})
            pbar.update()

    df = pd.concat([df, pd.DataFrame.from_dict(out)])

def process(x):
    try:
        return detect(x) == 'cs'
    except Exception:
        return False


df = df[df['gt'].parallel_apply(process)]

df.to_json("./data/ocr-data.jsonl", orient="records", lines=True)

