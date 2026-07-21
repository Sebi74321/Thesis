"""Feature signals, underrepresented regions, and row-level weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

MISSING = "__MISSING__"


def normalize_signal(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    maximum = float(clean.max()) if len(clean) else 0.0
    return clean / maximum if maximum > 0 else clean * 0.0


def _categorical(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), MISSING).astype(str)


def _probabilities(series: pd.Series, categories: Sequence[str]) -> np.ndarray:
    frequencies = _categorical(series).value_counts(normalize=True)
    return np.array([float(frequencies.get(category, 0.0)) for category in categories])


def compute_feature_components(
    real_audit: pd.DataFrame,
    synthetic_audit: pd.DataFrame,
    shap_importance: Mapping[str, float] | pd.Series,
    categorical_cols: Iterable[str],
    continuous_cols: Iterable[str],
    rare_threshold: float = 0.05,
    target_col: str | None = None,
) -> pd.DataFrame:
    categorical_cols = list(categorical_cols)
    continuous_cols = list(continuous_cols)
    features = [c for c in real_audit.columns if c in set(categorical_cols + continuous_cols)]
    missing = [c for c in features if c not in synthetic_audit]
    if missing:
        raise ValueError(f"Synthetic audit data is missing columns: {missing}")

    mismatch_raw: Dict[str, float] = {}
    tail_raw: Dict[str, float] = {}
    for col in continuous_cols:
        if col not in features:
            continue
        real = pd.to_numeric(real_audit[col], errors="coerce").dropna().to_numpy()
        syn = pd.to_numeric(synthetic_audit[col], errors="coerce").dropna().to_numpy()
        if len(real) == 0 or len(syn) == 0:
            mismatch_raw[col] = 0.0
            tail_raw[col] = 0.0
            continue
        q25, q75 = np.quantile(real, [0.25, 0.75])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.std(real))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        mismatch_raw[col] = float(wasserstein_distance(real, syn) / scale)
        q05, q95 = np.quantile(real, [0.05, 0.95])
        low_gap = max(0.0, float(np.mean(real <= q05) - np.mean(syn <= q05)))
        high_gap = max(0.0, float(np.mean(real >= q95) - np.mean(syn >= q95)))
        tail_raw[col] = low_gap + high_gap

    for col in categorical_cols:
        if col not in features:
            continue
        real_cat, syn_cat = _categorical(real_audit[col]), _categorical(synthetic_audit[col])
        categories = sorted(set(real_cat.unique()).union(syn_cat.unique()))
        p_real = _probabilities(real_cat, categories)
        p_syn = _probabilities(syn_cat, categories)
        mismatch_raw[col] = float(jensenshannon(p_real, p_syn, base=2.0))
        minority_value = real_cat.value_counts(normalize=True).idxmin() if col == target_col else None
        tail_raw[col] = float(
            sum(
                max(0.0, pr - ps)
                for category, pr, ps in zip(categories, p_real, p_syn)
                if pr <= rare_threshold or category == minority_value
            )
        )

    shap = pd.Series(dict(shap_importance), dtype=float).reindex(features, fill_value=0.0)
    mismatch = pd.Series(mismatch_raw, dtype=float).reindex(features, fill_value=0.0)
    tail = pd.Series(tail_raw, dtype=float).reindex(features, fill_value=0.0)
    return pd.DataFrame(
        {
            "feature": features,
            "shap_raw": shap.values,
            "mismatch_raw": mismatch.values,
            "tail_raw": tail.values,
            "shap": normalize_signal(shap).values,
            "mismatch": normalize_signal(mismatch).values,
            "tail": normalize_signal(tail).values,
        }
    )


def compute_feature_priority(components: pd.DataFrame, variant: str, top_k: int = 5) -> pd.DataFrame:
    required = {"feature", "shap", "mismatch", "tail"}
    if not required.issubset(components.columns):
        raise ValueError(f"components must contain {sorted(required)}")
    variant = variant.upper()
    if variant == "A2":
        combined = components["mismatch"].astype(float)
    elif variant == "A4":
        combined = 0.625 * components["shap"] + 0.375 * components["mismatch"]
    elif variant == "A5":
        combined = 0.5 * components["shap"] + 0.3 * components["mismatch"] + 0.2 * components["tail"]
    else:
        raise ValueError("Feature priorities are defined only for A2, A4, and A5")

    result = components.copy()
    result["combined_raw"] = combined
    order = result.sort_values("combined_raw", ascending=False, kind="mergesort").index[: max(0, top_k)]
    result["selected"] = False
    result.loc[order, "selected"] = True
    selected_sum = float(result.loc[order, "combined_raw"].sum())
    result["priority"] = 0.0
    if selected_sum > 0:
        result.loc[order, "priority"] = result.loc[order, "combined_raw"] / selected_sum
    return result.sort_values(["selected", "priority", "feature"], ascending=[False, False, True]).reset_index(drop=True)


@dataclass(frozen=True)
class RegionDefinition:
    kind: str
    gaps: Dict[str, float]
    edges: tuple[float, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        edges = [None if np.isinf(x) else float(x) for x in self.edges]
        return {"kind": self.kind, "gaps": self.gaps, "edges": edges}


def _continuous_bins(series: pd.Series, edges: Sequence[float]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    values = np.digitize(numeric.fillna(0.0), np.asarray(edges)[1:-1], right=True).astype(str)
    result = pd.Series(values, index=series.index, dtype="object")
    result.loc[numeric.isna()] = MISSING
    return result


def build_region_definitions(
    real_train: pd.DataFrame,
    real_audit: pd.DataFrame,
    synthetic_audit: pd.DataFrame,
    categorical_cols: Iterable[str],
    continuous_cols: Iterable[str],
) -> Dict[str, RegionDefinition]:
    definitions: Dict[str, RegionDefinition] = {}
    for col in continuous_cols:
        train = pd.to_numeric(real_train[col], errors="coerce").dropna()
        if train.empty:
            edges = (-np.inf, np.inf)
        else:
            quantiles = np.unique(np.quantile(train, [0.0, 0.05, 0.25, 0.75, 0.95, 1.0]))
            edges = tuple([-np.inf, *[float(x) for x in quantiles], np.inf])
        real_bins = _continuous_bins(real_audit[col], edges)
        syn_bins = _continuous_bins(synthetic_audit[col], edges)
        keys = sorted(set(real_bins.unique()).union(syn_bins.unique()))
        real_p = real_bins.value_counts(normalize=True)
        syn_p = syn_bins.value_counts(normalize=True)
        raw = {str(k): max(0.0, float(real_p.get(k, 0.0) - syn_p.get(k, 0.0))) for k in keys}
        maximum = max(raw.values(), default=0.0)
        gaps = {k: (v / maximum if maximum > 0 else 0.0) for k, v in raw.items()}
        definitions[col] = RegionDefinition("continuous", gaps, edges)

    for col in categorical_cols:
        real_cat, syn_cat = _categorical(real_audit[col]), _categorical(synthetic_audit[col])
        keys = sorted(set(real_cat.unique()).union(syn_cat.unique()))
        real_p = real_cat.value_counts(normalize=True)
        syn_p = syn_cat.value_counts(normalize=True)
        raw = {str(k): max(0.0, float(real_p.get(k, 0.0) - syn_p.get(k, 0.0))) for k in keys}
        maximum = max(raw.values(), default=0.0)
        gaps = {k: (v / maximum if maximum > 0 else 0.0) for k, v in raw.items()}
        definitions[col] = RegionDefinition("categorical", gaps)
    return definitions


def compute_region_scores(df: pd.DataFrame, definitions: Mapping[str, RegionDefinition]) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for feature, definition in definitions.items():
        if definition.kind == "continuous":
            labels = _continuous_bins(df[feature], definition.edges)
        else:
            labels = _categorical(df[feature])
        result[feature] = labels.map(definition.gaps).fillna(0.0).astype(float)
    return result


def compute_row_weights(
    real_train: pd.DataFrame,
    feature_priority: pd.DataFrame,
    region_definitions: Mapping[str, RegionDefinition],
    alpha: float = 1.0,
    w_max: float = 2.0,
) -> tuple[pd.Series, pd.DataFrame]:
    if alpha < 0 or w_max < 1:
        raise ValueError("alpha must be non-negative and w_max must be at least 1")
    selected = feature_priority.loc[feature_priority["selected"] & (feature_priority["priority"] > 0)]
    features = selected["feature"].tolist()
    scores = compute_region_scores(real_train, {f: region_definitions[f] for f in features})
    contributions = pd.DataFrame(index=real_train.index)
    weighted_sum = np.zeros(len(real_train), dtype=float)
    for row in selected.itertuples(index=False):
        contribution = float(row.priority) * scores[row.feature].to_numpy(dtype=float)
        contributions[row.feature] = contribution
        weighted_sum += contribution
    weights = pd.Series(np.minimum(1.0 + alpha * weighted_sum, w_max), index=real_train.index, name="weight")
    if not np.all(np.isfinite(weights)) or (weights < 1).any() or (weights > w_max).any():
        raise RuntimeError("Computed invalid row weights")
    return weights, contributions


def weight_diagnostics(weights: pd.Series, w_max: float) -> Dict[str, float]:
    desc = weights.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    return {
        **{str(k): float(v) for k, v in desc.items()},
        "fraction_capped": float(np.mean(np.isclose(weights.to_numpy(), w_max))),
        "fraction_above_one": float(np.mean(weights.to_numpy() > 1.0)),
    }
