import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn import svm,tree
from sklearn.ensemble import RandomForestClassifier
from dython.nominal import compute_associations
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from typing import Dict, Iterable, List, Optional
import warnings

warnings.filterwarnings("ignore")

def supervised_model_training(x_train, y_train, x_test, 
                              y_test, model_name,problem_type):
  
  if model_name == 'lr':
    model  = LogisticRegression(random_state=42,max_iter=500) 
  elif model_name == 'svm':
    model  = svm.SVC(random_state=42,probability=True)
  elif model_name == 'dt':
    model  = tree.DecisionTreeClassifier(random_state=42)
  elif model_name == 'rf':      
    model = RandomForestClassifier(random_state=42)
  elif model_name == "mlp":
    model = MLPClassifier(random_state=42,max_iter=100)
  elif model_name == "l_reg":
    model = LinearRegression()
  elif model_name == "ridge":
    model = Ridge(random_state=42)
  elif model_name == "lasso":
    model = Lasso(random_state=42)
  elif model_name == "B_ridge":
    model = BayesianRidge()
  
  model.fit(x_train, y_train)
  pred = model.predict(x_test)

  if problem_type == "Classification":
    if len(np.unique(y_train))>2:
      predict = model.predict_proba(x_test)        
      acc = metrics.accuracy_score(y_test,pred)*100
      auc = metrics.roc_auc_score(y_test, predict,average="weighted",multi_class="ovr")
      f1_score = metrics.precision_recall_fscore_support(y_test, pred,average="weighted")[2]
      return [acc, auc, f1_score] 

    else:
      predict = model.predict_proba(x_test)[:,1]    
      acc = metrics.accuracy_score(y_test,pred)*100
      auc = metrics.roc_auc_score(y_test, predict)
      f1_score = metrics.precision_recall_fscore_support(y_test,pred)[2].mean()
      return [acc, auc, f1_score] 
  
  else:
    mse = metrics.mean_absolute_percentage_error(y_test,pred)
    evs = metrics.explained_variance_score(y_test, pred)
    r2_score = metrics.r2_score(y_test,pred)
    return [mse, evs, r2_score]





def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float64,
        )

def get_one_hot_feature_names(encoder, input_features):
    """
    Return OneHotEncoder output names across sklearn versions.
    """
    if hasattr(encoder, "get_feature_names_out"):
        return encoder.get_feature_names_out(input_features)

    if hasattr(encoder, "get_feature_names"):
        return encoder.get_feature_names(input_features)

    # Fallback for very old sklearn versions.
    names = []

    for feature, categories in zip(
        input_features,
        encoder.categories_,
    ):
        for category in categories:
            names.append(f"{feature}_{category}")

    return np.asarray(names, dtype=object)


def dense_transform(encoder, data):
    transformed = encoder.transform(data)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return transformed

    
def get_utility_metrics(
    real_path,
    fake_paths,
    scaler="MinMax",
    type={"Classification": ["lr", "dt", "rf", "mlp"]},
    test_ratio=0.20,
    target_col=None,
    cat_cols=None,
):
    real = pd.read_csv(real_path)

    if target_col is None:
        target_col = real.columns[-1]

    feature_cols = [col for col in real.columns if col != target_col]

    if cat_cols is None:
        cat_cols = [
            col
            for col in feature_cols
            if (
                pd.api.types.is_object_dtype(real[col])
                or isinstance(real[col].dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(real[col])
            )
        ]
    else:
        cat_cols = list(cat_cols)

    numeric_cols = [
        col for col in feature_cols if col not in cat_cols
    ]

    X_real = real[feature_cols].copy()
    y_real = real[target_col].copy()

    problem = list(type.keys())[0]
    models = list(type.values())[0]

    stratify = y_real if problem == "Classification" else None

    X_train_real, X_test_real, y_train_real, y_test_real = (
        model_selection.train_test_split(
            X_real,
            y_real,
            test_size=test_ratio,
            stratify=stratify,
            random_state=42,
        )
    )

    for col in cat_cols:
        X_train_real[col] = (
            X_train_real[col].astype("object").fillna("__MISSING__")
        )
        X_test_real[col] = (
            X_test_real[col].astype("object").fillna("__MISSING__")
        )

    encoder = _make_one_hot_encoder()

    if cat_cols:
        encoder.fit(X_train_real[cat_cols])

        encoded_cat_cols = get_one_hot_feature_names(
            encoder,
            cat_cols,
        )

        train_real_cat = pd.DataFrame(
            encoder.transform(X_train_real[cat_cols]),
            columns=encoded_cat_cols,
            index=X_train_real.index,
        )
        test_real_cat = pd.DataFrame(
            encoder.transform(X_test_real[cat_cols]),
            columns=encoded_cat_cols,
            index=X_test_real.index,
        )
    else:
        encoded_cat_cols = []
        train_real_cat = pd.DataFrame(index=X_train_real.index)
        test_real_cat = pd.DataFrame(index=X_test_real.index)

    X_train_real_encoded = pd.concat(
        [X_train_real[numeric_cols], train_real_cat],
        axis=1,
    )

    X_test_real_encoded = pd.concat(
        [X_test_real[numeric_cols], test_real_cat],
        axis=1,
    )

    if scaler == "MinMax":
        fitted_scaler = MinMaxScaler()
    else:
        fitted_scaler = StandardScaler()

    # Fit only on real training data.
    fitted_scaler.fit(X_train_real_encoded)

    X_train_real_scaled = fitted_scaler.transform(
        X_train_real_encoded
    )
    X_test_real_scaled = fitted_scaler.transform(
        X_test_real_encoded
    )

    all_real_results = []

    for model in models:
        result = supervised_model_training(
            X_train_real_scaled,
            y_train_real,
            X_test_real_scaled,
            y_test_real,
            model,
            problem,
        )
        all_real_results.append(result)

    all_fake_results = []

    for fake_path in fake_paths:
        fake = pd.read_csv(fake_path)

        missing_cols = set(real.columns) - set(fake.columns)
        if missing_cols:
            raise ValueError(
                f"{fake_path} is missing columns: {sorted(missing_cols)}"
            )

        fake = fake.loc[:, real.columns]

        X_fake = fake[feature_cols].copy()
        y_fake = fake[target_col].copy()

        stratify_fake = y_fake if problem == "Classification" else None

        X_train_fake, _, y_train_fake, _ = (
            model_selection.train_test_split(
                X_fake,
                y_fake,
                test_size=test_ratio,
                stratify=stratify_fake,
                random_state=42,
            )
        )

        for col in cat_cols:
            X_train_fake[col] = (
                X_train_fake[col]
                .astype("object")
                .fillna("__MISSING__")
            )

        if cat_cols:
            train_fake_cat = pd.DataFrame(
                dense_transform(
                    encoder,
                    X_train_fake[cat_cols],
                ),
                columns=encoded_cat_cols,
                index=X_train_fake.index,
            )
        else:
            train_fake_cat = pd.DataFrame(index=X_train_fake.index)

        X_train_fake_encoded = pd.concat(
            [X_train_fake[numeric_cols], train_fake_cat],
            axis=1,
        )

        # Enforce exactly the same feature order.
        X_train_fake_encoded = X_train_fake_encoded.reindex(
            columns=X_train_real_encoded.columns,
            fill_value=0.0,
        )

        # Important: use the real-training scaler.
        X_train_fake_scaled = fitted_scaler.transform(
            X_train_fake_encoded
        )

        fake_model_results = []

        for model in models:
            result = supervised_model_training(
                X_train_fake_scaled,
                y_train_fake,
                X_test_real_scaled,
                y_test_real,
                model,
                problem,
            )
            fake_model_results.append(result)

        all_fake_results.append(fake_model_results)

    real_results_array = np.asarray(all_real_results, dtype=float)
    fake_results_array = np.asarray(all_fake_results, dtype=float)

    diff_results = (
        real_results_array - fake_results_array.mean(axis=0)
    )

    return {
    "real": real_results_array,
    "synthetic_mean": fake_results_array.mean(axis=0),
    "gap_real_minus_synthetic": (
        real_results_array
        - fake_results_array.mean(axis=0)
        )
    }


def get_association_matrix(df, cat_cols):
    result = compute_associations(
        df,
        nominal_columns=cat_cols,
    )

    if isinstance(result, dict):
        return result["corr"]

    return result


def plot_statistical_distributions(
    real,
    fake,
    cat_cols=None,
    model_name="Synthetic",
    columns=None,
    bins=30,
):
    """
    Plot real and synthetic marginal distributions.

    Continuous columns:
        Overlaid density-normalized histograms.

    Categorical columns:
        Side-by-side probability bars.
    """
    if cat_cols is None:
        cat_cols = []

    if columns is None:
        columns = list(real.columns)

    for column in columns:
        if column not in real.columns:
            print(f"Skipping unknown column: {column}")
            continue

        if column in cat_cols:
            real_pdf = (
                real[column]
                .fillna("__MISSING__")
                .astype(str)
                .value_counts(normalize=True)
            )

            fake_pdf = (
                fake[column]
                .fillna("__MISSING__")
                .astype(str)
                .value_counts(normalize=True)
            )

            categories = real_pdf.index.union(fake_pdf.index)

            real_pdf = real_pdf.reindex(
                categories,
                fill_value=0.0,
            )

            fake_pdf = fake_pdf.reindex(
                categories,
                fill_value=0.0,
            )

            x = np.arange(len(categories))
            width = 0.4

            fig, ax = plt.subplots(figsize=(9, 5))

            ax.bar(
                x - width / 2,
                real_pdf.to_numpy(),
                width,
                label="Real",
            )

            ax.bar(
                x + width / 2,
                fake_pdf.to_numpy(),
                width,
                label=model_name,
            )

            ax.set_title(f"{column}: real vs synthetic")
            ax.set_xlabel(column)
            ax.set_ylabel("Probability")
            ax.set_xticks(x)
            ax.set_xticklabels(
                categories,
                rotation=45,
                ha="right",
            )
            ax.legend()

            fig.tight_layout()
            plt.show()

        else:
            real_values = pd.to_numeric(
                real[column],
                errors="coerce",
            ).dropna()

            fake_values = pd.to_numeric(
                fake[column],
                errors="coerce",
            ).dropna()

            fig, ax = plt.subplots(figsize=(9, 5))

            ax.hist(
                real_values,
                bins=bins,
                density=True,
                alpha=0.5,
                label="Real",
            )

            ax.hist(
                fake_values,
                bins=bins,
                density=True,
                alpha=0.5,
                label=model_name,
            )

            ax.set_title(f"{column}: real vs synthetic")
            ax.set_xlabel(column)
            ax.set_ylabel("Density")
            ax.legend()

            fig.tight_layout()
            plt.show()

def _extract_association_matrix(result):
    """
    Handle different dython versions.

    compute_associations() may return either:
    - a DataFrame, or
    - a dictionary containing the correlation matrix under 'corr'.
    """
    if isinstance(result, dict):
        return result["corr"]

    return result
    
def stat_sim(
    real_path,
    fake_path,
    cat_cols=None,
    model_name=None,
    plot=False,
    plot_columns=None,
    bins=30,
):
    """
    Compare real and synthetic tabular data.

    Continuous columns:
        Min-max-scaled Wasserstein distance.

    Categorical columns:
        Jensen-Shannon distance between category distributions.

    Dependency structure:
        Frobenius norm between real and synthetic association matrices.

    Parameters
    ----------
    real_path : str
        Path to the real CSV.

    fake_path : str
        Path to one synthetic CSV.

    cat_cols : list[str], optional
        Names of categorical columns.

    model_name : str, optional
        Label used in returned tables and plots. If omitted, the synthetic
        filename is used.

    plot : bool, default=False
        Whether to plot real and synthetic distributions.

    plot_columns : list[str], optional
        Subset of columns to plot. If None, all columns are plotted.

    bins : int, default=30
        Number of bins for continuous histograms.

    Returns
    -------
    summary : pd.DataFrame
        One-row summary containing average WD, average JSD and correlation
        distance.

    feature_results : pd.DataFrame
        One row per feature with metric type and distance.

    categorical_distributions : pd.DataFrame
        Long-format category probabilities for real and synthetic data.
    """
    if cat_cols is None:
        cat_cols = []

    real = pd.read_csv(real_path)
    fake = pd.read_csv(fake_path)

    if model_name is None:
        model_name = Path(fake_path).stem

    missing_in_fake = set(real.columns) - set(fake.columns)
    extra_in_fake = set(fake.columns) - set(real.columns)

    if missing_in_fake or extra_in_fake:
        raise ValueError(
            "Real and synthetic datasets must contain the same columns.\n"
            f"Missing in synthetic data: {sorted(missing_in_fake)}\n"
            f"Extra in synthetic data: {sorted(extra_in_fake)}"
        )

    # Enforce identical column order.
    fake = fake.loc[:, real.columns]

    invalid_cat_cols = set(cat_cols) - set(real.columns)

    if invalid_cat_cols:
        raise ValueError(
            f"Unknown categorical columns: {sorted(invalid_cat_cols)}"
        )

    feature_rows = []
    categorical_distribution_rows = []

    for column in real.columns:
        if column in cat_cols:
            real_pdf = (
                real[column]
                .fillna("__MISSING__")
                .astype(str)
                .value_counts(normalize=True)
            )

            fake_pdf = (
                fake[column]
                .fillna("__MISSING__")
                .astype(str)
                .value_counts(normalize=True)
            )

            categories = real_pdf.index.union(fake_pdf.index)

            real_pdf = real_pdf.reindex(
                categories,
                fill_value=0.0,
            )

            fake_pdf = fake_pdf.reindex(
                categories,
                fill_value=0.0,
            )

            jsd = distance.jensenshannon(
                real_pdf.to_numpy(),
                fake_pdf.to_numpy(),
                base=2.0,
            )

            feature_rows.append(
                {
                    "Model": model_name,
                    "Feature": column,
                    "Feature type": "Categorical",
                    "Metric": "JSD",
                    "Distance": jsd,
                }
            )

            for category in categories:
                categorical_distribution_rows.append(
                    {
                        "Model": model_name,
                        "Feature": column,
                        "Category": category,
                        "Dataset": "Real",
                        "Probability": real_pdf.loc[category],
                    }
                )

                categorical_distribution_rows.append(
                    {
                        "Model": model_name,
                        "Feature": column,
                        "Category": category,
                        "Dataset": "Synthetic",
                        "Probability": fake_pdf.loc[category],
                    }
                )

        else:
            real_values = pd.to_numeric(
                real[column],
                errors="coerce",
            )

            fake_values = pd.to_numeric(
                fake[column],
                errors="coerce",
            )

            real_values = real_values.dropna()
            fake_values = fake_values.dropna()

            if real_values.empty or fake_values.empty:
                wd = np.nan
            elif real_values.nunique() <= 1:
                # MinMaxScaler is not informative for a constant column.
                wd = wasserstein_distance(
                    real_values.to_numpy(),
                    fake_values.to_numpy(),
                )
            else:
                scaler = MinMaxScaler()

                scaler.fit(
                    real_values.to_numpy().reshape(-1, 1)
                )

                real_scaled = scaler.transform(
                    real_values.to_numpy().reshape(-1, 1)
                ).ravel()

                fake_scaled = scaler.transform(
                    fake_values.to_numpy().reshape(-1, 1)
                ).ravel()

                wd = wasserstein_distance(
                    real_scaled,
                    fake_scaled,
                )

            feature_rows.append(
                {
                    "Model": model_name,
                    "Feature": column,
                    "Feature type": "Continuous",
                    "Metric": "Wasserstein",
                    "Distance": wd,
                }
            )

    feature_results = pd.DataFrame(feature_rows)

    categorical_distributions = pd.DataFrame(
        categorical_distribution_rows
    )

    real_associations = _extract_association_matrix(
        compute_associations(
            real,
            nominal_columns=cat_cols,
        )
    )

    fake_associations = _extract_association_matrix(
        compute_associations(
            fake,
            nominal_columns=cat_cols,
        )
    )

    fake_associations = fake_associations.reindex(
        index=real_associations.index,
        columns=real_associations.columns,
    )

    correlation_difference = (
        real_associations.to_numpy()
        - fake_associations.to_numpy()
    )

    correlation_distance = np.linalg.norm(
        correlation_difference,
        ord="fro",
    )

    continuous_distances = feature_results.loc[
        feature_results["Feature type"] == "Continuous",
        "Distance",
    ]

    categorical_distances = feature_results.loc[
        feature_results["Feature type"] == "Categorical",
        "Distance",
    ]

    summary = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Average WD": continuous_distances.mean(),
                "Median WD": continuous_distances.median(),
                "Maximum WD": continuous_distances.max(),
                "Worst continuous feature": (
                    feature_results.loc[
                        feature_results["Feature type"]
                        == "Continuous"
                    ]
                    .sort_values("Distance", ascending=False)
                    ["Feature"]
                    .iloc[0]
                    if not continuous_distances.empty
                    else None
                ),
                "Average JSD": categorical_distances.mean(),
                "Maximum JSD": categorical_distances.max(),
                "Worst categorical feature": (
                    feature_results.loc[
                        feature_results["Feature type"]
                        == "Categorical"
                    ]
                    .sort_values("Distance", ascending=False)
                    ["Feature"]
                    .iloc[0]
                    if not categorical_distances.empty
                    else None
                ),
                "Correlation distance": correlation_distance,
            }
        ]
    )

    if plot:
        plot_statistical_distributions(
            real=real,
            fake=fake,
            cat_cols=cat_cols,
            model_name=model_name,
            columns=plot_columns,
            bins=bins,
        )

    return summary, feature_results, categorical_distributions
def privacy_metrics(
    real_path,
    fake_path,
    data_percent=15,
    target_col=None,
    cat_cols=None,
    include_target=True,
):
    real = pd.read_csv(real_path).drop_duplicates(
        keep=False
    )
    fake = pd.read_csv(fake_path).drop_duplicates(
        keep=False
    )

    if target_col is None:
        target_col = real.columns[-1]

    fake = fake.reindex(columns=real.columns)

    if include_target:
        columns_for_distance = list(real.columns)
        encoder_target = None

        # The generic encoder excludes the declared target. To include every
        # field in the privacy calculation, use a temporary non-existent target.
        # It is clearer here to encode predictors and target separately below.
        real_work = real.copy()
        fake_work = fake.copy()

        inferred_cat_cols = (
            list(cat_cols)
            if cat_cols is not None
            else [
                col
                for col in columns_for_distance
                if (
                    pd.api.types.is_object_dtype(real_work[col])
                    or pd.api.types.is_object_dtype(fake_work[col])
                    or isinstance(
                        real_work[col].dtype,
                        pd.CategoricalDtype,
                    )
                    or isinstance(
                        fake_work[col].dtype,
                        pd.CategoricalDtype,
                    )
                    or pd.api.types.is_bool_dtype(real_work[col])
                    or pd.api.types.is_bool_dtype(fake_work[col])
                )
            ]
        )

        numeric_cols = [
            col
            for col in columns_for_distance
            if col not in inferred_cat_cols
        ]

        for col in inferred_cat_cols:
            real_work[col] = (
                real_work[col]
                .astype("object")
                .fillna("__MISSING__")
            )
            fake_work[col] = (
                fake_work[col]
                .astype("object")
                .fillna("__MISSING__")
            )

        encoder = _make_one_hot_encoder()

        if inferred_cat_cols:
            encoder.fit(
                pd.concat(
                    [
                        real_work[inferred_cat_cols],
                        fake_work[inferred_cat_cols],
                    ],
                    ignore_index=True,
                )
            )
            encoded_names = get_one_hot_feature_names(
                encoder,
                inferred_cat_cols,
            )

            real_cat = pd.DataFrame(
                encoder.transform(
                    real_work[inferred_cat_cols]
                ),
                columns=encoded_names,
                index=real_work.index,
            )

            fake_cat = pd.DataFrame(
                encoder.transform(
                    fake_work[inferred_cat_cols]
                ),
                columns=encoded_names,
                index=fake_work.index,
            )
        else:
            real_cat = pd.DataFrame(index=real_work.index)
            fake_cat = pd.DataFrame(index=fake_work.index)

        real_encoded = pd.concat(
            [real_work[numeric_cols], real_cat],
            axis=1,
        )
        fake_encoded = pd.concat(
            [fake_work[numeric_cols], fake_cat],
            axis=1,
        )

    else:
        real_encoded, fake_encoded, _ = (
            one_hot_encode_real_fake(
                real,
                fake,
                target_col=target_col,
                categorical_cols=cat_cols,
                fit_on="combined",
            )
        )

        real_encoded = real_encoded.drop(
            columns=[target_col]
        )
        fake_encoded = fake_encoded.drop(
            columns=[target_col]
        )

    n_real = int(len(real_encoded) * data_percent / 100)
    n_fake = int(len(fake_encoded) * data_percent / 100)

    if n_real < 3 or n_fake < 3:
        raise ValueError(
            "The sampled real and fake subsets must each contain "
            "at least three observations."
        )

    real_refined = real_encoded.sample(
        n=n_real,
        random_state=42,
    ).to_numpy(dtype=float)

    fake_refined = fake_encoded.sample(
        n=n_fake,
        random_state=42,
    ).to_numpy(dtype=float)

    # One shared scaling transformation.
    scaler = StandardScaler()
    scaler.fit(real_refined)

    real_scaled = scaler.transform(real_refined)
    fake_scaled = scaler.transform(fake_refined)

    dist_rf = metrics.pairwise_distances(
        real_scaled,
        fake_scaled,
        metric="minkowski",
        n_jobs=-1,
    )

    dist_rr = metrics.pairwise_distances(
        real_scaled,
        metric="minkowski",
        n_jobs=-1,
    )

    dist_ff = metrics.pairwise_distances(
        fake_scaled,
        metric="minkowski",
        n_jobs=-1,
    )

    # Exclude self-distances without reshaping the matrices.
    np.fill_diagonal(dist_rr, np.inf)
    np.fill_diagonal(dist_ff, np.inf)

    def two_smallest(distances):
        partitioned = np.partition(
            distances,
            kth=1,
            axis=1,
        )
        return partitioned[:, :2]

    smallest_two_rf = two_smallest(dist_rf)
    smallest_two_rr = two_smallest(dist_rr)
    smallest_two_ff = two_smallest(dist_ff)

    epsilon = np.finfo(float).eps

    nn_ratio_rf = (
        smallest_two_rf[:, 0]
        / np.maximum(smallest_two_rf[:, 1], epsilon)
    )
    nn_ratio_rr = (
        smallest_two_rr[:, 0]
        / np.maximum(smallest_two_rr[:, 1], epsilon)
    )
    nn_ratio_ff = (
        smallest_two_ff[:, 0]
        / np.maximum(smallest_two_ff[:, 1], epsilon)
    )

    fifth_perc_rf = np.percentile(
        smallest_two_rf[:, 0],
        5,
    )
    fifth_perc_rr = np.percentile(
        smallest_two_rr[:, 0],
        5,
    )
    fifth_perc_ff = np.percentile(
        smallest_two_ff[:, 0],
        5,
    )

    nn_fifth_perc_rf = np.percentile(
        nn_ratio_rf,
        5,
    )
    nn_fifth_perc_rr = np.percentile(
        nn_ratio_rr,
        5,
    )
    nn_fifth_perc_ff = np.percentile(
        nn_ratio_ff,
        5,
    )

    return np.array(
        [
            fifth_perc_rf,
            fifth_perc_rr,
            fifth_perc_ff,
            nn_fifth_perc_rf,
            nn_fifth_perc_rr,
            nn_fifth_perc_ff,
        ]
    ).reshape(1, 6)  