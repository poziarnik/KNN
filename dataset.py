from huggingface_hub import hf_hub_download
from datasets import load_dataset, load_from_disk
from pathlib import Path
import random
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')


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
def introduce_errors(text: str) -> str:
    for _ in range(random.randint(2, 5)):
        error_type = random.choice(['insert', 'delete'])
        if error_type == 'insert':
            position = random.randint(0, len(text))
            random_char = random.choice('abcdefghijklmnopqrstuvwxyz0123456789.,!? ')
            text = text[:position] + random_char + text[position:]
        elif error_type == 'delete':
            if len(text) > 0:
                position = random.randint(0, len(text) - 1)
                text = text[:position] + text[position+1:]
    return text

dataset = load_from_disk(FILTERED)
# dataset = dataset.filter(lambda x: len(x["text"]) <= 500)
df = dataset.to_pandas()
df = df[["text"]]


# Apply the tokenization function to the 'text' column
print("Tokenizing dataset, into sentences...")
df['sentences'] = df['text'].apply(lambda x: sent_tokenize(x))

# Explode the 'sentences' column to create a single 'sentence' column
print("Exploding dataset...")
df = df.explode('sentences')

# Rename the 'sentences' column to 'sentence'
print("Renaming dataset...")
df.rename(columns={'sentences': 'sentence'}, inplace=True)

# Drop the 'text' column
print("Dropping unused data...")
df.drop(columns=['text'], inplace=True)

# Reset the index
df.reset_index(drop=True, inplace=True)

for err_rate in tqdm((0.1, 0.2, 0.3, 0.4, 0.5)):
    df_new = df.copy()
    
    # Randomly select texts to introduce errors into
    sentences_to_modify = df_new.sample(frac=err_rate, random_state=42).index

    sentences_to_not_modify = df_new.index.difference(sentences_to_modify)
    df_new.loc[sentences_to_not_modify, 'error'] = df_new.loc[sentences_to_not_modify, 'sentence']

    # Introduce errors into selected texts using tqdm for progress tracking
    tqdm.pandas(desc="Introducing Errors")
    df_new.loc[sentences_to_modify, 'error'] = df_new.loc[sentences_to_modify, 'sentence'].progress_apply(introduce_errors)
    
    print("Saving dataset...")
    df_new.to_csv(f"dataset-{err_rate}.csv", index=False, sep=";")