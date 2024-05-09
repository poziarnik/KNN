from openai import OpenAI
from os import environ
from typing import Iterator
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Set your OpenAI API key
client = OpenAI(api_key=environ["OPENAI_API_KEY"])

def data_generator() -> Iterator[tuple[str, str]]:
    for _, row in df.iterrows():

        if len(row["sentence"]) < 5000: # for input length limitation
            yield row["sentence"], row["error"]

# Function to evaluate text using OpenAI API
def evaluate(data: tuple[str, str]) -> tuple[str, str, str | None]:
    sentence, error = data

    prompt = (
        f"Jsi poslušný asistent.\n"
        f"Oprav následující zkreslený text, který byl získán z OCR: `{error}`\n\n"
        "Instrukce:\n"
        "- Pokud je věta správná a neobsahuje žádné chyby, ponech ji nezměněnou.\n"
        "- Oprav chyby ve formě nesprávných písmen, chybějících nebo přidaných znaků, špatně interpretovaných slov atd.\n"
        "- Zachovej gramatickou správnost a srozumitelnost textu.\n"
        "- Zohledni kontext a význam přilehlých slov pro co nejlepší opravu.\n"
        "Odpověz pouze opraveným textem bez dalších informací nebo vysvětlení."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            timeout=30
        )
    except Exception:
        return sentence, error, None

    corrected_text = response.choices[0].message.content

    return sentence, error, corrected_text

# find dataset
file = Path("data/err-0.5/test.json")

print(f"Loading {file.name}...")
df = pd.read_json(file, orient="records", lines=True)

df[["sentence", "error"]] = df.apply(lambda row: (" ".join(row["sentence"]), " ".join(row["error"])), axis=1, result_type="expand")

# print("Randomizing dataset...")
df = df.sample(frac=0.5).reset_index(drop=True)

# Initialize an empty DataFrame to store evaluation results
evaluation_df = pd.DataFrame(columns=['sentence', 'error', 'correction'])

with Pool(processes=cpu_count()) as p, tqdm(total=len(df), desc="Sending API calls to GPT.") as pbar:
    for result in p.imap_unordered(evaluate, data_generator()):
        if result[2] is not None:
            evaluation_df = evaluation_df._append({'sentence': result[0], 'error': result[1], 'correction': result[2]}, ignore_index=True)
        pbar.update()

print("Saving evaluation data...")
evaluation_df.to_json(f"evaluation-{file.name}", orient="records", lines=True, index=False)
