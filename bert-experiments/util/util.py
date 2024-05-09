import numpy as np
import pandas as pd
import math

def split_df(df: pd.DataFrame, n_splits: int) -> list[pd.DataFrame]:
    dfs = np.array_split(df, n_splits)

    for df in dfs:
        df.reset_index(drop=True, inplace=True)

    return dfs

def optimal_split(df: pd.DataFrame) -> list[pd.DataFrame]:
    return split_df(df, math.ceil(len(df) / 1000))