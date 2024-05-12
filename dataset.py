import random
import nltk
import string

from huggingface_hub import hf_hub_download
from datasets import load_dataset, load_from_disk, set_caching_enabled
from pathlib import Path
from nltk.tokenize import sent_tokenize
from pandarallel import pandarallel
from tqdm import tqdm

tqdm.pandas()

pandarallel.initialize(progress_bar=True)

set_caching_enabled(False)

nltk.download('punkt')


BASE_DATA_DIR = Path("./data")
BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "BUT-FIT/BUT-LCC"
FILE_NAME = "train_0.jsonl.gz"

FILTERED = BASE_DATA_DIR / "cs-wiki"

NUM_EXAMPLES = 100000

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

    for _ in range(random.randint(3, 6)):
        error_type = random.choice(['insert', 'delete'])
        word_position = random.randint(0, len(words) - 1)
        char_position = random.randint(0, len(words[word_position]))
        if error_type == 'insert':
            random_char = random.choice(string.ascii_letters + string.digits + ' ')
            if random_char == ' ':
                words = words[:word_position] + [words[word_position][:char_position]] + [words[word_position][char_position:]] + words[word_position:]
                labels = labels[:word_position] + [1, 1] + labels[word_position:]
            else:
                words[word_position] = words[word_position][:char_position] + random_char + words[word_position][char_position:]
                labels[word_position] = 1
        elif error_type == 'delete':
            if len(words) > 0:
                if len(words[word_position]) > 1:
                    words[word_position] = words[word_position][:char_position] + words[word_position][char_position + 1:]
                    labels[word_position] = 1

    return words, labels

def tokenize_and_remove_punctuation(text):
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    # Remove punctuation from each sentence
    sentences = [''.join(c for c in s if c not in string.punctuation) for s in sentences]
    return sentences

dataset = load_from_disk(FILTERED, keep_in_memory=False)
# dataset = dataset.select(range(NUM_EXAMPLES))

df = dataset.to_pandas()
df = df[["text"]]

# Apply the tokenization function to the 'text' column
print("Tokenizing text, into sentences")
df['sentences'] = df['text'].parallel_apply(tokenize_and_remove_punctuation)

# Drop the 'text' column
print("Dropping unused data...")
df.drop(columns=['text'], inplace=True)

# Explode the 'sentences' column to create a single 'sentence' column
print("Exploding dataset...")
df = df.explode('sentences')

# Reset the index
df.reset_index(drop=True, inplace=True)

print("Tokenizing sentences, into words")
df['sentences'] = df['sentences'].parallel_apply(lambda x: x.split(' '))

# Rename the 'sentences' column to 'sentence'
print("Renaming dataset...")
df.rename(columns={'sentences': 'sentence'}, inplace=True)

# Reset the index
df.reset_index(drop=True, inplace=True)

# create labels
print("Creating labels...")
df['error'] = df['sentence']
df['labels'] = df['sentence'].progress_apply(lambda x: [0 for _ in range(len(x))])

# Reset the index
df.reset_index(drop=True, inplace=True)

# filter dataset
df = df.sample(n=NUM_EXAMPLES, random_state=42)
df.reset_index(drop=True, inplace=True)

for err_rate in (0.5, 0.8):
    df_new = df.copy()

    # Randomly select texts to introduce errors into
    sentences_to_modify = df_new.sample(frac=err_rate, random_state=42).index

    # Introduce errors into selected texts using tqdm for progress tracking
    print("Introducing errors")
    pd_tuple = df_new.loc[sentences_to_modify, 'sentence'].parallel_apply(introduce_errors).to_list()
    for i, result in enumerate(pd_tuple):
        error, labels = result[0], result[1]
        df_new.at[sentences_to_modify[i], 'error'] = error
        df_new.at[sentences_to_modify[i], 'labels'] = labels

    # Reset the index
    df_new.reset_index(drop=True, inplace=True)

    # Split dataset into train, validate, and test
    train = df_new.sample(frac=0.8)
    validate = (df_new.drop(train.index)).sample(frac=0.5)
    test = df_new.drop(train.index).drop(validate.index)

    # Save datasets
    print("Saving dataset...")

    dir = (BASE_DATA_DIR / f"err-{err_rate}")
    dir.mkdir(parents=True, exist_ok=True)

    train.to_json(dir / "train.jsonl", index=False, lines=True, orient="records")
    validate.to_json(dir / "validate.jsonl", index=False, lines=True, orient="records")
    test.to_json(dir / "test.jsonl", index=False, lines=True, orient="records")