import pandas as pd
from typing import Iterator
from termcolor import colored

def get_from_events(iterator: Iterator) -> Iterator[str]:
    try:
        while True:
            word = next(iterator)[1]['data']['prompt'][1]['content']
            result = next(iterator)[1]['data']
            color = "red" if result['correct'] == False else "green"

            expected = result['expected']
            picked = result['sampled']

            yield f"{word}\t{expected}\t{colored(picked, color)}"
    except StopIteration:
        StopIteration()

events = "/tmp/evallogs/240331162620JA276Z7I_gpt-3.5-turbo_spell-check.jsonl"

print(f"Word\tExpected\tPicked")

with open(events) as f:
    events_df = pd.read_json(f, lines=True)

iterator = events_df.iterrows()
next(iterator)
next(iterator)

for i in get_from_events(iterator):
    print(i)