from openai import OpenAI
from os import environ
from typing import Iterator
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm
from pathlib import Path

NUM_EXAMPLES = 100

# Set your OpenAI API key
client = OpenAI(api_key=environ["OPENAI_API_KEY"])

def data_generator() -> Iterator[tuple[str, str]]:
    c = 0
    for _, row in df.iterrows():
        if c == NUM_EXAMPLES:
            break

        if len(row["sentence"]) < 5000: # for input length limitation
            yield row["sentence"], row["error"]
            c += 1

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
        "Příklad opravy:\n"
        "Původní text: 'Tolto je píklad zkresleného texbu z OCS.'\n"
        "Opravený text: 'Toto je příklad zkresleného textu z OCR.'\n\n"
        "Odpověz pouze opraveným textem bez dalších informací nebo vysvětlení."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )

    corrected_text = response.choices[0].message.content

    return sentence, error, corrected_text

# find datasets
for file in Path(".").rglob("dataset*.csv"):
    if file.name != "dataset-0.5.csv":
        continue

    print(f"Loading {file.name}...")
    df = pd.read_csv(file, sep=";")

    # print("Randomizing dataset...")
    # df = df.sample(frac=1, random_state=52).reset_index(drop=True)

    # Initialize an empty DataFrame to store evaluation results
    evaluation_df = pd.DataFrame(columns=['sentence', 'error', 'correction'])

    with Pool(processes=cpu_count()) as p, tqdm(total=NUM_EXAMPLES, desc="Sending API calls to GPT.") as pbar:
        for result in p.imap_unordered(evaluate, data_generator()):
            if result[2] is not None:
                evaluation_df = evaluation_df._append({'sentence': result[0], 'error': result[1], 'correction': result[2]}, ignore_index=True)
            pbar.update()

    print("Saving evaluation data...")
    evaluation_df.to_csv(f"evaluation-{file.name}", sep=";", index=False)
