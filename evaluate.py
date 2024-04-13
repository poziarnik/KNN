from openai import OpenAI
from os import environ
from typing import Iterator
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm

NUM_EXAMPLES = 50

# Set your OpenAI API key
client = OpenAI(api_key=environ["OPENAI_API_KEY"])

def data_generator() -> Iterator[tuple[str, str]]:
    c = 0
    for _, row in df.iterrows():
        if c == NUM_EXAMPLES:
            break

        if len(row["text"]) < 5000: # for input length limitation
            yield row["text"], row["error"]
            c += 1

# Function to evaluate text using OpenAI API
def evaluate(data: tuple[str, str]) -> tuple[str, str, str | None]:
    text, error = data

    prompt = (
        f"Jsi poslušný asistent.\n"
        f"Oprav chyby v textu, pokud tam jsou: `{error or text}`"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )

    corrected_text = response.choices[0].message.content

    return text, error, corrected_text


print("Loading dataset...")
df = pd.read_csv("dataset.csv", sep=";")

print("Randomizing dataset...")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Initialize an empty DataFrame to store evaluation results
evaluation_df = pd.DataFrame(columns=['text', 'error', 'correction'])

with Pool(processes=cpu_count()) as p, tqdm(total=NUM_EXAMPLES, desc="Sending API calls to GPT.") as pbar:
    for result in p.imap_unordered(evaluate, data_generator()):
        if result[2] is not None:
            evaluation_df = evaluation_df._append({'text': result[0], 'error': result[1], 'correction': result[2]}, ignore_index=True)
        pbar.update()

print("Saving evaluation data...")
evaluation_df.to_csv("evaluation.csv", sep=";", index=False)
