from huggingface_hub import hf_hub_download
from datasets import load_dataset, load_from_disk
from pathlib import Path
import random
import string
from tqdm import tqdm


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


# Function to introduce errors into text
def introduce_errors(text: str):
    # Simulate introducing errors (e.g., typos, punctuation changes)
    words = text.split(' ')
    modified_words = []
    for word in words:
        if random.random() < 0.3:  # Adjust error introduction rate as needed
            # Introduce a typo by randomly changing a character in the word
            random_char_index = random.randint(0, len(word) - 1)
            modified_word = word[:random_char_index] + random.choice(string.ascii_lowercase) + word[random_char_index + 1:]
            modified_words.append(modified_word)
        else:
            modified_words.append(word)
    return ' '.join(modified_words)

dataset = load_from_disk(FILTERED)
# dataset = dataset.filter(lambda x: len(x["text"]) <= 500)
df = dataset.to_pandas()
df = df[["text"]]

ERR_RATE = 0.5  # 50% of texts will have errors

# Randomly select texts to introduce errors into
texts_to_modify = df.sample(frac=ERR_RATE, random_state=42).index

text_to_not_modify = df.index.difference(texts_to_modify)
df.loc[text_to_not_modify, 'error'] = df.loc[text_to_not_modify, 'text']

# Introduce errors into selected texts using tqdm for progress tracking
tqdm.pandas(desc="Introducing Errors")
df.loc[texts_to_modify, 'error'] = df.loc[texts_to_modify, 'text'].progress_apply(introduce_errors)

print("Saving dataset...")
df.to_csv("dataset.csv", index=False, sep=";")
