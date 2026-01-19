# cdf_tail_metrics.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional


def _ecdf(values: np.ndarray):
    """Return an ECDF callable F(x) = P(X <= x) for a 1D array."""
    values = np.sort(np.asarray(values))
    n = len(values)

    def F(x: float) -> float:
        return float(np.searchsorted(values, x, side="right") / n)

    return F


@dataclass
class CDFTailMetrics:
    """
    Tail-restricted CDF divergence (KS-like) for tabular continuous features.

    - tau: tail threshold computed on REAL data (e.g., 0.9, 0.95, 0.99)
    - grid_size: number of x points to evaluate in [q_tau, max(real)]
    """
    label_col: str = "Class"
    tau: float = 0.9
    grid_size: int = 200
    feature_cols: Optional[List[str]] = None  # if None, uses all non-label cols

    def _tail_divergence_1d(self, real_col: np.ndarray, syn_col: np.ndarray) -> float:
        real_col = np.asarray(real_col)
        syn_col = np.asarray(syn_col)

        if len(real_col) == 0 or len(syn_col) == 0:
            return float("nan")

        q_tau = float(np.quantile(real_col, self.tau))
        max_x = float(np.max(real_col))
        if max_x <= q_tau:
            # degenerate tail (all values identical or very narrow range)
            return 0.0

        F_real = _ecdf(real_col)
        F_syn = _ecdf(syn_col)

        xs = np.linspace(q_tau, max_x, num=self.grid_size)
        diffs = np.abs([F_real(x) - F_syn(x) for x in xs])
        return float(np.max(diffs))

    def evaluate(self, real_df: pd.DataFrame, syn_df: pd.DataFrame) -> Dict[str, float]:
        # align cols
        syn_df = syn_df[real_df.columns]

        cols = self.feature_cols
        if cols is None:
            cols = [c for c in real_df.columns if c != self.label_col]

        divergences = []
        out: Dict[str, float] = {}

        for col in cols:
            d = self._tail_divergence_1d(real_df[col].values, syn_df[col].values)
            out[f"cdf_tail_div_{col}"] = d
            divergences.append(d)

        out["cdf_tail_div_mean"] = float(np.nanmean(divergences)) if divergences else float("nan")
        return out

    def evaluate_paths(self, real_path: str, syn_path: str) -> Dict[str, float]:
        real_df = pd.read_csv(real_path)
        syn_df = pd.read_csv(syn_path)
        return self.evaluate(real_df, syn_df)
