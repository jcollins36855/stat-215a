"""Cleaning utilities for STAT 215A Lab 2 (dialect survey data).

`clean_data(raw_df, ...)` copies its input and returns a new frame.
"""

import numpy as np
import pandas as pd

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
    "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}


def question_columns(df):
    """Lexical question columns Q050..Q121 present in the frame."""
    return [c for c in df.columns if c.startswith("Q")]


def clean_data(raw_df, drop_junk_states=False):
    """Return a cleaned copy of the raw lingData frame.

    - Flags rows with non-US STATE codes in a `state_valid` column
      (159 rows: 'XX' placeholders, Canadian provinces, typos like
      '00'/'94'/'C)'). Optionally drops them.
    - Flags rows with missing coordinates in a `geo_valid` column
      (1,020 rows lack lat/long and cannot be mapped).
    - Leaves 0 ("no response") codes in place; `encode_binary`
      excludes them from the analysis matrix.
    """
    df = raw_df.copy()
    st = df["STATE"].fillna("MISSING").astype(str)
    df["state_valid"] = st.isin(US_STATES)
    df["geo_valid"] = df[["lat", "long"]].notna().all(axis=1)
    if drop_junk_states:
        df = df[df["state_valid"]].reset_index(drop=True)
    return df


def encode_binary(df, questions=None):
    """One-hot encode categorical question responses.

    Each (question, answer-code) pair with code != 0 becomes one binary
    column, yielding p = 468 columns on the full data. Code 0 ("no
    response") is excluded from the column set, so it leaves an all-zero
    block for its question; rows are otherwise kept as-is.
    Returns (binary_df, column_labels).
    """
    if questions is None:
        questions = question_columns(df)
    blocks, labels = [], []
    for q in questions:
        codes = sorted(c for c in df[q].dropna().unique() if c != 0)
        for code in codes:
            blocks.append((df[q].values == code).astype(np.int8))
            labels.append(f"{q}={int(code)}")
    binary = pd.DataFrame(
        np.column_stack(blocks), index=df.index, columns=labels,
        dtype=np.int8,
    )
    return binary, labels
