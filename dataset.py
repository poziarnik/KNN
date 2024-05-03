from huggingface_hub import hf_hub_download
from datasets import load_dataset, load_from_disk, set_caching_enabled
from pathlib import Path
import random
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import swifter
from swifter import set_defaults
set_defaults(
    dask_threshold=1,
    scheduler="processes",
    progress_bar=True,
    progress_bar_desc=None,
    allow_dask_on_strings=False,
    force_parallel=False,
)

set_caching_enabled(False)

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
    dataset.flatten_indices()
    dataset.save_to_disk(FILTERED)


# Function to introduce errors into text
def introduce_errors(original: list[str]) -> tuple[list[str], list[int]]:
    words = original.copy()
    
    # 0 - correct
    # 1 - incorrect
    labels: list[int] = [0 for _ in range(len(words))]

    for _ in range(random.randint(2, 5)):
        error_type = random.choice(['insert', 'delete'])
        word_position = random.randint(0, len(words) - 1)
        char_position = random.randint(0, len(words[word_position]))
        if error_type == 'insert':
            random_char = random.choice('abcdefghijklmnopqrstuvwxyz0123456789.,!? ')
            if random_char == ' ':
                words = words[:word_position] + [words[word_position][:char_position]] + [words[word_position][char_position:]] + words[word_position:]
                labels = labels[:word_position] + [1, 1] + labels[word_position:]
            else:
                words[word_position] = words[word_position][:char_position] + random_char + words[word_position][char_position:]
                labels[word_position] = 1
        elif error_type == 'delete':
            if len(words) > 0:
                if len(words[word_position]) > 1:
                    words[word_position] = words[word_position][:char_position - 1] + words[word_position][char_position:]
                    labels[word_position] = 1

    return words, labels

dataset = load_from_disk(FILTERED, keep_in_memory=False)
# dataset = dataset.filter(lambda x: len(x["text"]) <= 500)

df = dataset.to_pandas()
df = df[["text"]]

# Apply the tokenization function to the 'text' column
print("Tokenizing text, into sentences")
df['sentences'] = df['text'].swifter.apply(lambda x: sent_tokenize(x))

# Drop the 'text' column
print("Dropping unused data...")
df.drop(columns=['text'], inplace=True)

# Explode the 'sentences' column to create a single 'sentence' column
print("Exploding dataset...")
df = df.explode('sentences')

# Reset the index
df.reset_index(drop=True, inplace=True)

print("Tokenizing sentences, into words")
df['sentences'] = df['sentences'].swifter.apply(lambda x: nltk.tokenize.word_tokenize(x))

# Rename the 'sentences' column to 'sentence'
print("Renaming dataset...")
df.rename(columns={'sentences': 'sentence'}, inplace=True)

# Reset the index
df.reset_index(drop=True, inplace=True)

# create labels
print("Creating labels...")
df['error'] = df['sentence']
df['labels'] = df['sentence'].swifter.apply(lambda x: [0 for _ in range(len(x))])

# Reset the index
df.reset_index(drop=True, inplace=True)

for err_rate in tqdm((0.5, )):
    df_new = df.copy()

    # Randomly select texts to introduce errors into
    sentences_to_modify = df_new.sample(frac=err_rate, random_state=42).index

    # Introduce errors into selected texts using tqdm for progress tracking
    print("Introducing errors")
    pd_tuple = df_new.loc[sentences_to_modify, 'sentence'].swifter.apply(introduce_errors)
    for i, result in enumerate(pd_tuple):
        error, labels = result[0], result[1]
        df_new.at[sentences_to_modify[i], 'error'] = error
        df_new.at[sentences_to_modify[i], 'labels'] = labels

    print("Saving dataset...")
    df_new.to_json(BASE_DATA_DIR / f"dataset-{err_rate}.json", index=False, lines=True, orient="records")