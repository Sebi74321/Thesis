# rare_event_recall.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


@dataclass
class RareEventRecall:
    """
    Rare-event recall with baseline:

    1) Baseline: train on REAL (split A), test on REAL (split B)
    2) GAN eval: train on SYNTHETIC, test on REAL

    Returns:
    - recall for both settings
    - absolute number of correctly detected rare events
    """

    label_col: str = "Class"
    random_state: int = 42
    n_estimators: int = 200
    test_size: float = 0.3

    def _train_and_eval(self, X_train, y_train, X_test, y_test, rare_class):
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        y_true_bin = (y_test == rare_class).astype(int)
        y_pred_bin = (y_pred == rare_class).astype(int)

        recall = recall_score(y_true_bin, y_pred_bin)

        # absolute counts
        true_rare = int(y_true_bin.sum())
        correctly_detected = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())

        return recall, true_rare, correctly_detected

    def evaluate(self, real_df: pd.DataFrame, syn_df: pd.DataFrame) -> Dict[str, float]:
        syn_df = syn_df[real_df.columns]

        # identify rare class from REAL distribution
        counts = real_df[self.label_col].value_counts()
        rare_class = counts.idxmin()

        X_real = real_df.drop(columns=[self.label_col])
        y_real = real_df[self.label_col]

        # -------------------------
        # 1) BASELINE: REAL → REAL
        # -------------------------
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_real,
            y_real,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_real
        )

        baseline_recall, baseline_total, baseline_correct = self._train_and_eval(
            X_train_r, y_train_r, X_test_r, y_test_r, rare_class
        )

        # -------------------------
        # 2) SYN → REAL
        # -------------------------
        X_syn = syn_df.drop(columns=[self.label_col])
        y_syn = syn_df[self.label_col]

        syn_recall, syn_total, syn_correct = self._train_and_eval(
            X_syn, y_syn, X_real, y_real, rare_class
        )

        return {
            "rare_class": float(rare_class) if isinstance(rare_class, (int, float, np.number)) else str(rare_class),

            # baseline
            "rare_event_recall_real_baseline": float(baseline_recall),
            "rare_event_total_real_baseline": int(baseline_total),
            "rare_event_correct_real_baseline": int(baseline_correct),

            # synthetic
            "rare_event_recall_synthetic": float(syn_recall),
            "rare_event_total_synthetic": int(syn_total),
            "rare_event_correct_synthetic": int(syn_correct),

            # general info
            "rare_class_count_real": int(counts.min()),
            "n_classes_real": int(counts.shape[0]),
        }

    def evaluate_paths(self, real_path: str, syn_path: str) -> Dict[str, float]:
        real_df = pd.read_csv(real_path)
        syn_df = pd.read_csv(syn_path)
        return self.evaluate(real_df, syn_df)