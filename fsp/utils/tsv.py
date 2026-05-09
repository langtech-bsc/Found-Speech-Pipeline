from pathlib import Path

import pandas as pd


def read_tsv_with_optional_header(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
    df = df.replace("", pd.NA).dropna(how="all").fillna("").reset_index(drop=True)

    row_count = len(df)
    if row_count == 2:
        return df.iloc[1:].reset_index(drop=True)
    if row_count == 1:
        return df

    raise ValueError(f"TSV must contain 1 data row, with an optional header row: {path}")

__all__ = ["read_tsv_with_optional_header"]
