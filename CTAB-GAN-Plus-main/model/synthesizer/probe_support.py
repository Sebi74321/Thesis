"""Audit categorical support without modifying the fitted CTAB+ encoder."""

import numpy as np
import pandas as pd


def supported_probe_rows(data_prep, frame):
    """Keep rows encodable by the training vocabulary and report all exclusions.

    Unknown values have no learned discriminator coordinate. Do not invent one,
    refit the encoder, or silently substitute a known category. Row positions
    refer to the supplied source frame, before balanced sampling and splitting.
    """
    missing = [column for column in data_prep.df.columns if column not in frame]
    if missing:
        raise ValueError(f"Discriminator probe is missing columns: {missing}")
    supported = np.ones(len(frame), dtype=bool)
    unknown_categories = {}
    for item in data_prep.label_encoder_list:
        column, encoder = item["column"], item["label_encoder"]
        # Match DataPrep's exact missing-value and string conversion rules.
        values = frame[column].astype(object)
        values = values.mask(values.isna() | values.eq(" "), "empty").astype(str)
        unknown = ~values.isin(encoder.classes_).to_numpy()
        if unknown.any():
            counts = values.iloc[np.flatnonzero(unknown)].value_counts().sort_index()
            unknown_categories[column] = {str(value): int(count) for value, count in counts.items()}
            supported &= ~unknown
    excluded = np.flatnonzero(~supported)
    report = {
        "input_rows": int(len(frame)),
        "supported_rows": int(supported.sum()),
        "excluded_rows": int(len(excluded)),
        "excluded_fraction": float(len(excluded) / len(frame)) if len(frame) else 0.0,
        "excluded_row_positions": excluded.tolist(),
        "unknown_categories": unknown_categories,
    }
    return frame.iloc[np.flatnonzero(supported)].copy(deep=True), report
