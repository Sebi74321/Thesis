# support_coverage.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _quantile_bin_edges(real_series: pd.Series, n_bins: int) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins + 1)
    edges = real_series.quantile(qs).values
    edges = np.unique(edges)
    return edges


def _digitize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Digitize using edges. If edges are degenerate, return all zeros.
    """
    if len(edges) <= 1:
        return np.zeros(len(values), dtype=int)
    # exclude endpoints for digitize cutpoints
    cutpoints = edges[1:-1]
    return np.digitize(values, cutpoints, right=True).astype(int)


@dataclass
class SupportCoverage:
    """
    Support Coverage (SC) for rare multivariate combinations using discretization.

    - n_bins: quantile bins per numeric feature (computed on REAL, applied to both)
    - rare_threshold: combinations with real frequency <= this fraction are "rare"
    - include_label_in_combo: include Class in the combination tuple (recommended)
    """
    label_col: str = "Class"
    n_bins: int = 5
    rare_threshold: float = 0.01
    include_label_in_combo: bool = True
    feature_cols: Optional[List[str]] = None  # if None, all non-label cols

    def evaluate(self, real_df: pd.DataFrame, syn_df: pd.DataFrame) -> Dict[str, float]:
        syn_df = syn_df[real_df.columns]

        cols = self.feature_cols
        if cols is None:
            cols = [c for c in real_df.columns if c != self.label_col]

        # bin each feature based on REAL quantiles
        real_bin_df = pd.DataFrame(index=real_df.index)
        syn_bin_df = pd.DataFrame(index=syn_df.index)

        for col in cols:
            edges = _quantile_bin_edges(real_df[col], self.n_bins)
            real_bin_df[col] = _digitize(real_df[col].values, edges)
            syn_bin_df[col] = _digitize(syn_df[col].values, edges)

        if self.include_label_in_combo:
            real_bin_df[self.label_col] = real_df[self.label_col].values
            syn_bin_df[self.label_col] = syn_df[self.label_col].values

        # tuples represent "support points" in discretized space
        real_combos = list(real_bin_df.itertuples(index=False, name=None))
        syn_combos = list(syn_bin_df.itertuples(index=False, name=None))

        # frequency of each combo in REAL
        real_freq = pd.Series(real_combos).value_counts(normalize=True)

        rare_real = set(real_freq[real_freq <= self.rare_threshold].index)
        syn_support = set(syn_combos)

        if len(rare_real) == 0:
            return {
                "support_coverage": float("nan"),
                "num_rare_combos": 0,
                "rare_threshold": float(self.rare_threshold),
                "n_bins": int(self.n_bins),
                "include_label": float(self.include_label_in_combo),
            }

        covered = rare_real.intersection(syn_support)
        sc = len(covered) / len(rare_real)

        return {
            "support_coverage": float(sc),
            "num_rare_combos": int(len(rare_real)),
            "rare_threshold": float(self.rare_threshold),
            "n_bins": int(self.n_bins),
            "include_label": float(self.include_label_in_combo),
        }

    def evaluate_paths(self, real_path: str, syn_path: str) -> Dict[str, float]:
        real_df = pd.read_csv(real_path)
        syn_df = pd.read_csv(syn_path)
        return self.evaluate(real_df, syn_df)
