# rare_event_recall.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler


@dataclass
class RareEventRecall:
    """
    Rare-event recall:
    Train a classifier on SYNTHETIC data (X_syn, y_syn), test on REAL (X_real, y_real),
    and compute recall for the rarest class in REAL.

    Notes:
    - For new-thyroid, features are numeric → scaling is safe.
    - Uses RandomForest by default (robust).
    """
    label_col: str = "Class"
    random_state: int = 42
    n_estimators: int = 200

    def evaluate(self, real_df: pd.DataFrame, syn_df: pd.DataFrame) -> Dict[str, float]:
        syn_df = syn_df[real_df.columns]

        # identify rare class from REAL distribution
        counts = real_df[self.label_col].value_counts()
        rare_class = counts.idxmin()

        X_syn = syn_df.drop(columns=[self.label_col])
        y_syn = syn_df[self.label_col]

        X_real = real_df.drop(columns=[self.label_col])
        y_real = real_df[self.label_col]

        scaler = StandardScaler()
        X_syn_s = scaler.fit_transform(X_syn)
        X_real_s = scaler.transform(X_real)

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        clf.fit(X_syn_s, y_syn)
        y_pred = clf.predict(X_real_s)

        # compute recall for that rare class (as "positive" in a one-vs-rest sense)
        y_true_bin = (y_real == rare_class).astype(int)
        y_pred_bin = (y_pred == rare_class).astype(int)

        rare_recall = recall_score(y_true_bin, y_pred_bin)

        return {
            "rare_class": float(rare_class) if isinstance(rare_class, (int, float, np.number)) else str(rare_class),
            "rare_event_recall": float(rare_recall),
            "rare_class_count_real": int(counts.min()),
            "n_classes_real": int(counts.shape[0]),
        }

    def evaluate_paths(self, real_path: str, syn_path: str) -> Dict[str, float]:
        real_df = pd.read_csv(real_path)
        syn_df = pd.read_csv(syn_path)
        return self.evaluate(real_df, syn_df)
