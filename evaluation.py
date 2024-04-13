from openai import OpenAI
from openai.types.chat import ChatCompletion
from os import environ
from typing import Iterator
from multiprocessing import Pool, Manager
import pandas as pd
from functools import partial
from tqdm import tqdm

client = OpenAI(
    api_key=environ["OPENAI_API_KEY"]
)

df = pd.read_csv("dataset.csv", sep=";")
df = df.sample(frac=1).reset_index(drop=True)

def generator(num: int) -> Iterator[tuple[str, str]]:
    for index, row in df.iterrows():
        yield row["text"], row["spoiled text"]

        if index == num:
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
        f.write(f'"{data[0]}";{data[1]};{response.choices[0].message.content}\n')
    lock.release()

g = generator(num=10)

# TODO: somehow make it work using built-in csv module (between multiple processes)
with open("evaluation.csv", "w") as f:
    f.write(f"text;spoiled text;correction\n")


with Manager() as m:
    lock = m.Lock()

    with Pool(processes=4) as p:
        func = partial(evaluate, lock)
        for _ in tqdm(p.imap_unordered(func, g), total=10):
            pass
