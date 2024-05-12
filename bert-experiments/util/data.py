import pandas as pd

def _correct_row(row: pd.Series) -> str | None:
    corrected: str = row["masked"]

    try:
        for fix in row["replace"]:
            corrected = corrected.replace("[MASK]", fix, 1)
    except TypeError:
        return None
    
    return corrected

def masked_correct(df: pd.DataFrame) -> pd.DataFrame:
    df.reset_index(drop=True, inplace=True)
    
    df["corrected"] = None

    index = df[df["masked"].notnull()].index
    _df = df.loc[index]

    _df.to_json("tmp.jsonl", orient="records", lines=True)

    _df["corrected"] = _df.apply(_correct_row, axis=1)
    df.loc[index] = _df
    
    return df