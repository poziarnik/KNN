import random
import requests
import zipfile
from io import BytesIO
from pathlib import Path
import csv
from word_spoilers import WordSpoiler

URL = "https://wiki.korpus.cz/lib/exe/fetch.php/seznamy:syn2015_word_abc_utf8.zip"

DATASET = Path("./dataset") / "syn2015_word_abc_utf8.tsv"
DATASET.parent.mkdir(exist_ok=True, parents=True)

spoil = WordSpoiler(spoil_prob=0.8)

def generate_corrupted_words(words, num_samples):
    """
    Function to generate randomly corrupted words.

    For each randomly selected word from the original list (words),
    generates a randomly corrupted word (corrupted_word) using a random
    spoiler (spoil).

    Adds a pair of original word (word) and corrupted word (corrupted_word)
    to a new list (corrupted_words).

    Returns this list of pairs.
    """
    corrupted_words = []
    for _ in range(num_samples):
        word = random.choice(words)
        corrupted_word = spoil.spoil(word)
        corrupted_words.append((word, corrupted_word))
    return corrupted_words

# Download the dataset from the URL if it doesn't already exist
if not DATASET.exists():
    print(f"Downloading dataset from {URL}")
    response = requests.get(URL)
    if response.status_code != 200:
        raise RuntimeError(f"Download failed with status code: {response.status_code}")

    print("Download successful. Extracting dataset...")

    zip_file = zipfile.ZipFile(BytesIO(response.content))
    zip_file.extractall(path=DATASET.parent)
    print("Dataset extracted.")


with open(DATASET, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    
    # read words longer than 2 characters
    words = [row[1] for row in reader if len(row[1]) > 2]

num_samples = 40000
corrupted_data = generate_corrupted_words(words, num_samples)

# print corrupted data in jsonl format
for original, corrupted in corrupted_data:
    print(f'{{"input": [{{"role": "system", "content": "Úloha: Oprav následující chybně napsané slovo"}}, {{"role": "user", "content": "{corrupted}"}}], "ideal": "{original}"}}')