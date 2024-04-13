from openai import OpenAI
from openai.types.chat import ChatCompletion
from os import environ
from typing import Iterator
from multiprocessing import Pool, Manager
import pandas as pd
from functools import partial
from tqdm import tqdm
import csv

NUM_EXAMPLES = 10

client = OpenAI(
    api_key=environ["OPENAI_API_KEY"]
)

print("Loading dataset...")
df = pd.read_csv("dataset.csv", sep=";")

print("Randomizing dataset...")
df = df.sample(frac=1).reset_index(drop=True)

def generator(num: int) -> Iterator[tuple[str, str]]:
    c = 0
    for _, row in df.iterrows():
        if len(row["text"]) < 10000: # for input length limitation
            c+=1
            yield row["text"], row["spoiled text"]

        if c == num:
            break

def evaluate(lock, data: tuple[str, str]) -> None:
    response: ChatCompletion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Jsi poslušný asistent\n"
                f"Oprav chyby v textu ak tam jsou: '{data[1]}'"
            }
        ]
    )

    lock.acquire()
    with open("evaluation.csv", "a") as f:
        writer = csv.writer(f, delimiter=";")
        # original (valid);spoiled;corrected by GPT
        writer.writerow((data[0], data[1], response.choices[0].message.content))
    lock.release()

g = generator(num=NUM_EXAMPLES)

with open("evaluation.csv", "w") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(("text","spoiled text","correction"))

with Manager() as m:
    lock = m.Lock()

    with Pool(processes=4) as p:
        func = partial(evaluate, lock)
        for _ in tqdm(p.imap_unordered(func, g), total=NUM_EXAMPLES):
            pass
