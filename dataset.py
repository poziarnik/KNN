from pathlib import Path
import requests # type: ignore
import zipfile
from io import BytesIO
import csv
from util import WordSpoiler
import argparse

# download URL
URL = "https://wiki.korpus.cz/lib/exe/fetch.php/seznamy:syn2015_word_abc_utf8.zip"

# basic parser
parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True)
parser.add_argument("--eval", action="store_true", default=False)

args = parser.parse_args()

BASE_DIR = Path(args.dir).absolute()
BASE_DIR.mkdir(exist_ok=True, parents=True)

# dataset path
DATASET = BASE_DIR / "syn2015_word_abc_utf8.tsv"
OUTPUT_JSONL = BASE_DIR / "word-spoiler" / "samples.jsonl"
OUTPUT = BASE_DIR / "output.csv"

if not DATASET.exists():
    print(f"Downloading dataset from {URL}")
    response = requests.get(URL)
    if response.status_code != 200:
        raise RuntimeError(f"Download failed with status code: {response.status_code}")

    print("Download successful. Extracting dataset...")

    zip_file = zipfile.ZipFile(BytesIO(response.content))
    zip_file.extractall(path=DATASET.parent)
    zip_file.close()
    print("Dataset extracted.")

# load words from the TSV file
with DATASET.open('r', encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    
    # read words longer than 2 characters
    words = [row[1] for row in reader if len(row[1]) > 2]

    spoiler = WordSpoiler(spoil_prob=0.8) # probability of spoiling the word is 0.8
spoiled_words = spoiler.spoil_words_generator(words, num_samples=40000)

if args.eval is True:
    OUTPUT_JSONL.parent.mkdir(exist_ok=True, parents=True)
    # store data in the jsonl format (used in openai evals framework)
    with OUTPUT_JSONL.open('w') as f:
        for correct, spoiled in spoiled_words:
            f.write(f'{{"input": [{{"role": "system", "content": "Úloha: Oprav následující chybně napsané slovo"}}, {{"role": "user", "content": "{spoiled}"}}], "ideal": "{correct}"}}\n')
else:
    with OUTPUT.open('w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(spoiled_words)
            
print(f"Output file: '{f.name}'")
