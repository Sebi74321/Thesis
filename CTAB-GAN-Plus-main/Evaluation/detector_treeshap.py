import os
import json
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Optional, List

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

import shap


@dataclass
class DetectorTreeSHAP:
    label_real: int = 1
    label_syn: int = 0
    test_size: float = 0.3
    random_state: int = 42
    n_estimators: int = 300
    max_depth: Optional[int] = None
    output_dir: str = "artifacts/detector_treeshap"

    def _build_dataset(self, real_df: pd.DataFrame, syn_df: pd.DataFrame):
        # align columns
        common_cols = [c for c in real_df.columns if c in syn_df.columns]
        real_df = real_df[common_cols].copy()
        syn_df = syn_df[common_cols].copy()

        real_df["_is_real"] = self.label_real
        syn_df["_is_real"] = self.label_syn

        df = pd.concat([real_df, syn_df], axis=0, ignore_index=True)
        X = df.drop(columns=["_is_real"])
        y = df["_is_real"]

        return X, y

    def _build_preprocessor(self, X: pd.DataFrame):
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols)
        ])

        return preprocessor, numeric_cols, categorical_cols

    def _feature_names(self, preprocessor, numeric_cols, categorical_cols):
        names = []

        names.extend(numeric_cols)

        if categorical_cols:
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
            names.extend(cat_names)

        return names

    def evaluate(self, real_df: pd.DataFrame, syn_df: pd.DataFrame) -> Dict[str, float]:
        os.makedirs(self.output_dir, exist_ok=True)

        X, y = self._build_dataset(real_df, syn_df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        preprocessor, numeric_cols, categorical_cols = self._build_preprocessor(X_train)

        X_train_p = preprocessor.fit_transform(X_train)
        X_test_p = preprocessor.transform(X_test)

        feature_names = self._feature_names(preprocessor, numeric_cols, categorical_cols)

        detector = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced"
        )

        detector.fit(X_train_p, y_train)

        proba = detector.predict_proba(X_test_p)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metrics = {
            "detector_auc": float(roc_auc_score(y_test, proba)),
            "detector_accuracy": float(accuracy_score(y_test, pred)),
            "detector_average_precision": float(average_precision_score(y_test, proba)),
            "n_real": int((y == self.label_real).sum()),
            "n_synthetic": int((y == self.label_syn).sum()),
            "n_features_original": int(X.shape[1]),
            "n_features_encoded": int(X_train_p.shape[1]),
        }

        # TreeSHAP
        explainer = shap.TreeExplainer(detector)
        shap_values = explainer.shap_values(X_test_p)

        # For binary RF, shap_values may be list[class0, class1]
        if isinstance(shap_values, list):
            shap_real = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # shape: (n_samples, n_features, n_classes)
            shap_real = shap_values[:, :, 1]
        else:
            shap_real = shap_values

        mean_abs_shap = np.abs(shap_real).mean(axis=0)

        shap_importance = (
            pd.DataFrame({
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap
            })
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        shap_path = os.path.join(self.output_dir, "detector_shap_importance.csv")
        shap_importance.to_csv(shap_path, index=False)

        metrics_path = os.path.join(self.output_dir, "detector_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def evaluate_paths(self, real_path: str, syn_path: str) -> Dict[str, float]:
        real_df = pd.read_csv(real_path)
        syn_df = pd.read_csv(syn_path)
        return self.evaluate(real_df, syn_df)