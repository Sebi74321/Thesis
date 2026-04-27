# support_coverage.py
import itertools
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pandas.api.types import is_numeric_dtype, is_bool_dtype


MISSING_TOKEN = "__MISSING__"


def _quantile_bin_edges(real_series: pd.Series, n_bins: int) -> np.ndarray:
    s = pd.to_numeric(real_series, errors="coerce").dropna()
    if len(s) == 0:
        return np.array([])

    qs = np.linspace(0, 1, n_bins + 1)
    return np.unique(np.quantile(s, qs))


def _digitize(values: pd.Series, edges: np.ndarray) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")

    if len(edges) <= 1:
        out = pd.Series(np.zeros(len(s), dtype=object), index=s.index)
        out[s.isna()] = MISSING_TOKEN
        return out

    cutpoints = edges[1:-1]
    binned = pd.Series(
        np.digitize(s.fillna(edges[0]), cutpoints, right=True),
        index=s.index,
        dtype=object,
    )
    binned[s.isna()] = MISSING_TOKEN
    return binned


def _encode_categorical(values: pd.Series) -> pd.Series:
    s = values.astype("object")
    s = s.where(~s.isna(), MISSING_TOKEN)
    return s.astype(str)


def _is_numeric_feature(series: pd.Series) -> bool:
    return is_numeric_dtype(series) and not is_bool_dtype(series)


@dataclass
class SupportCoverage:
    """
    K-way mixed-type support coverage.

    Numeric features are discretized into quantile bins using REAL data.
    Categorical/boolean/object features are compared as categories.

    Instead of evaluating the full joint distribution across all columns,
    this metric evaluates fixed-size feature subsets of size k and averages
    the support coverage across those subsets.

    This makes results more comparable across datasets with different
    numbers of features.
    """

    label_col: str = "Class"
    n_bins: int = 5
    rare_threshold: float = 0.05
    min_rare_count = 3


    # k-way settings
    k: int = 2
    max_subsets: Optional[int] = 100
    random_state: int = 42

    # column settings
    include_label_in_combo: bool = True
    feature_cols: Optional[List[str]] = None
    drop_all_missing_rows: bool = True

    def _prepare_columns(self, real_df: pd.DataFrame, syn_df: pd.DataFrame) -> List[str]:
        if self.feature_cols is None:
            cols = [c for c in real_df.columns if c != self.label_col]
        else:
            cols = list(self.feature_cols)

        missing_real = [c for c in cols if c not in real_df.columns]
        missing_syn = [c for c in cols if c not in syn_df.columns]

        if missing_real:
            raise ValueError(f"Real data is missing required columns: {missing_real}")
        if missing_syn:
            raise ValueError(f"Synthetic data is missing required columns: {missing_syn}")

        if self.include_label_in_combo:
            if self.label_col not in real_df.columns:
                raise ValueError(f"Label column '{self.label_col}' missing in real data.")
            if self.label_col not in syn_df.columns:
                raise ValueError(f"Label column '{self.label_col}' missing in synthetic data.")

        if self.k < 1:
            raise ValueError("k must be >= 1.")
        if self.k > len(cols):
            raise ValueError(f"k={self.k} is larger than the number of feature columns ({len(cols)}).")

        return cols

    def _transform_all_features(
        self,
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        cols: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        real_t = pd.DataFrame(index=real_df.index)
        syn_t = pd.DataFrame(index=syn_df.index)

        for col in cols:
            if _is_numeric_feature(real_df[col]):
                edges = _quantile_bin_edges(real_df[col], self.n_bins)
                real_t[col] = _digitize(real_df[col], edges)
                syn_t[col] = _digitize(syn_df[col], edges)
            else:
                real_t[col] = _encode_categorical(real_df[col])
                syn_t[col] = _encode_categorical(syn_df[col])

        if self.include_label_in_combo:
            real_t[self.label_col] = _encode_categorical(real_df[self.label_col])
            syn_t[self.label_col] = _encode_categorical(syn_df[self.label_col])

        return real_t, syn_t

    def _select_subsets(self, cols: List[str]) -> List[Tuple[str, ...]]:
        all_subsets = list(itertools.combinations(cols, self.k))

        if self.max_subsets is not None and len(all_subsets) > self.max_subsets:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(all_subsets), size=self.max_subsets, replace=False)
            all_subsets = [all_subsets[i] for i in idx]

        return all_subsets

    def _coverage_for_subset(
        self,
        real_t: pd.DataFrame,
        syn_t: pd.DataFrame,
        subset: Tuple[str, ...],
    ) -> Dict[str, float]:
        combo_cols = list(subset)

        if self.include_label_in_combo:
            combo_cols = combo_cols + [self.label_col]

        real_sub = real_t[combo_cols].copy()
        syn_sub = syn_t[combo_cols].copy()

        if self.drop_all_missing_rows:
            real_all_missing = (real_sub == MISSING_TOKEN).all(axis=1)
            syn_all_missing = (syn_sub == MISSING_TOKEN).all(axis=1)
            real_sub = real_sub.loc[~real_all_missing]
            syn_sub = syn_sub.loc[~syn_all_missing]

        real_combos = list(real_sub.itertuples(index=False, name=None))
        syn_combos = list(syn_sub.itertuples(index=False, name=None))

        if len(real_combos) == 0:
            return {
                "support_coverage": np.nan,
                "num_rare_combos": 0,
                "num_covered_rare_combos": 0,
                "num_real_combos": 0,
                "num_syn_combos": len(set(syn_combos)),
            }

        real_freq = pd.Series(real_combos).value_counts(normalize=True)
        real_freq = real_counts / len(real_combos)
        rare_real = set(set(real_counts[(real_counts >= self.min_rare_count) &(real_freq <= self.rare_threshold)].index)
        syn_support = set(syn_combos)

        if len(rare_real) == 0:
            return {
                "support_coverage": np.nan,
                "num_rare_combos": 0,
                "num_covered_rare_combos": 0,
                "num_real_combos": len(set(real_combos)),
                "num_syn_combos": len(syn_support),
            }

        covered = rare_real.intersection(syn_support)
        return {
            "support_coverage": len(covered) / len(rare_real),
            "num_rare_combos": len(rare_real),
            "num_covered_rare_combos": len(covered),
            "num_real_combos": len(set(real_combos)),
            "num_syn_combos": len(syn_support),
        }

    def evaluate(self, real_df: pd.DataFrame, syn_df: pd.DataFrame) -> Dict[str, float]:
        # Align synthetic columns to real where possible
        shared_cols = [c for c in real_df.columns if c in syn_df.columns]
        real_df = real_df[shared_cols].copy()
        syn_df = syn_df[shared_cols].copy()

        cols = self._prepare_columns(real_df, syn_df)
        real_t, syn_t = self._transform_all_features(real_df, syn_df, cols)
        subsets = self._select_subsets(cols)

        subset_scores = []
        rare_counts = []
        covered_counts = []
        real_combo_counts = []
        syn_combo_counts = []

        for subset in subsets:
            res = self._coverage_for_subset(real_t, syn_t, subset)

            if not pd.isna(res["support_coverage"]):
                subset_scores.append(res["support_coverage"])

            rare_counts.append(res["num_rare_combos"])
            covered_counts.append(res["num_covered_rare_combos"])
            real_combo_counts.append(res["num_real_combos"])
            syn_combo_counts.append(res["num_syn_combos"])

        if len(subset_scores) == 0:
            mean_sc = float("nan")
            std_sc = float("nan")
        else:
            mean_sc = float(np.mean(subset_scores))
            std_sc = float(np.std(subset_scores))

        return {
            "support_coverage": mean_sc,
            "support_coverage_std": std_sc,
            "k": int(self.k),
            "num_subsets_evaluated": int(len(subsets)),
            "num_valid_subsets": int(len(subset_scores)),
            "avg_num_rare_combos": float(np.mean(rare_counts)) if rare_counts else float("nan"),
            "avg_num_covered_rare_combos": float(np.mean(covered_counts)) if covered_counts else float("nan"),
            "avg_num_real_combos": float(np.mean(real_combo_counts)) if real_combo_counts else float("nan"),
            "avg_num_syn_combos": float(np.mean(syn_combo_counts)) if syn_combo_counts else float("nan"),
            "rare_threshold": float(self.rare_threshold),
            "min_rare_count" : float(self.min_rare_count),
            "n_bins": int(self.n_bins),
            "include_label": float(self.include_label_in_combo),
            "max_subsets": -1 if self.max_subsets is None else int(self.max_subsets),
        }

    def evaluate_paths(self, real_path: str, syn_path: str) -> Dict[str, float]:
        real_df = pd.read_csv(real_path)
        syn_df = pd.read_csv(syn_path)
        return self.evaluate(real_df, syn_df)