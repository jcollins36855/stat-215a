import numpy as np

def clean_data(raw, codebook):
    clean = raw.copy()

    na92_vars = codebook[
        codebook["Values"].astype(str).str.contains(
            r"92\s+Not applicable",
            case=False,
            na=False,
            regex=True
        )
    ]["Variable"].tolist()

    for col in na92_vars:
        if col in clean.columns:
            clean[col] = clean[col].replace(92, np.nan)

    return clean