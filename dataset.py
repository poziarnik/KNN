from huggingface_hub import hf_hub_download
from datasets import load_dataset, load_from_disk
from pathlib import Path
import random
import string

BASE_DATA_DIR = Path("./data")
BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "BUT-FIT/BUT-LCC"
FILE_NAME = "train_0.jsonl.gz"

FILTERED = BASE_DATA_DIR / "cs-wiki"

if not FILTERED.exists():
    print("Downloading dataset...")
    dataset_path = hf_hub_download(repo_id=REPO_ID, filename=FILE_NAME, repo_type="dataset")
    print(f"Dataset was downloaded '{dataset_path}'")

    print("Loading dataset...")
    dataset = load_dataset('json', data_files=dataset_path, split='train')

    print("Filtering dataset...")
    dataset = dataset.filter(lambda x: x["part"] == "cswiki-20230101")

    print(f"Saving dataset to '{FILTERED.absolute()}'")
    dataset.save_to_disk(FILTERED)


# Function to introduce multiple random errors into text
def spoil_text(text):
    # Number of errors to introduce (can be adjusted)
    num_errors = random.randint(6, 8)  # Randomly choose 1 to 3 errors

    for _ in range(num_errors):
        error_type = random.choice(["typo", "missing", "extra"])

        if error_type == "typo":
            # Introduce a typo by changing one random character
            if len(text) > 1:
                idx = random.randint(0, len(text) - 1)
                text = text[:idx] + random.choice(string.ascii_lowercase) + text[idx+1:]

        elif error_type == "missing":
            # Remove one random character (simulate a missing character)
            if len(text) > 1:
                idx = random.randint(0, len(text) - 1)
                text = text[:idx] + text[idx+1:]

        elif error_type == "extra":
            # Add one random character (simulate an extra character)
            idx = random.randint(0, len(text))
            text = text[:idx] + random.choice(string.ascii_lowercase) + text[idx:]

    return text

dataset = load_from_disk(FILTERED)
df = dataset.to_pandas()
df = df[["text"]]
df["spoiled text"] =  df["text"].apply(spoil_text)
df.to_csv("dataset.csv", index=False, sep=";")
