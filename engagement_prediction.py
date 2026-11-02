"""Predict video engagement with ridge regression and three classifiers.

The preprocessing and model choices retain an existing experiment. Preprocessing
is fitted before splitting and also clips the regression target; this script is
not a leakage-free estimate of out-of-sample performance. See README.md.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from sklearn.model_selection import train_test_split


def preprocess_lecture_dataset(dataset):
    """Apply the retained full-dataset preprocessing sequence. See README limitations."""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    df = dataset.copy()
    # Drop the high-cardinality topic URL rather than expanding it into dummy columns.
    df = df.drop(columns=["most_covered_topic"], errors="ignore")
    df = df.fillna({"subject_domain": "unknown", "has_parts": "unknown"})
    num_cols_full = df.select_dtypes(include=["float64", "int64"]).columns
    for c in num_cols_full:
        df[c] = df[c].fillna(df[c].median())
    # This also clips the target; retained for historical comparability, not recommended.
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for c in numeric_cols:
        Q1 = df[c].quantile(0.25)
        Q3 = df[c].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[c] = df[c].clip(lower=lower, upper=upper)
    skewed_cols = [
        "duration",
        "word_count",
        "silent_period_rate",
        "speaker_speed",
        "title_word_count",
        "freshness",
    ]
    for c in skewed_cols:
        if c in df.columns:
            df[c] = df[c].clip(lower=0)
            df[c] = np.log1p(df[c])
    categorical_cols = ["subject_domain", "lecture_type", "has_parts"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)
    target_col = "median_engagement"
    numeric_cols_after = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if target_col in numeric_cols_after:
        numeric_cols_after.remove(target_col)
    onehot_cols = [
        c
        for c in df.columns
        if c.startswith("subject_domain_")
        or c.startswith("lecture_type_")
        or c.startswith("has_parts_")
    ]
    numeric_cols_to_scale = [c for c in numeric_cols_after if c not in onehot_cols]
    # The scaler is fitted on the full dataset here; see the documented leakage limitation.
    scaler = StandardScaler()
    df[numeric_cols_to_scale] = scaler.fit_transform(df[numeric_cols_to_scale])
    preprocessed_dataset = df
    return preprocessed_dataset


def prepare_final_dataset(preprocessed_dataset):
    """Separate the processed target from the feature matrix."""
    target_col = "median_engagement"
    y = preprocessed_dataset[target_col].values
    X = preprocessed_dataset.drop(columns=[target_col])
    return (preprocessed_dataset, X, y)


def train_ridge_model(X, y, hyperparams):
    """Fit ridge regression using the specified parameters."""
    from sklearn.linear_model import Ridge

    ridge_model = Ridge(**hyperparams)
    ridge_model.fit(X, y)
    return ridge_model


def cont_to_class(y_cont):
    """Label engagement values at or above 0.2 as positive."""
    if y_cont >= 0.2:
        y_class = 1
    else:
        y_class = 0
    return y_class


def train_logistic_model(X, y, hyperparams):
    """Fit the logistic-regression classifier."""
    from sklearn.linear_model import LogisticRegression

    log_reg_model = LogisticRegression(**hyperparams)
    log_reg_model.fit(X, y)
    return log_reg_model


def binary_f1_score(y_actual, y_predicted):
    """Compute positive-class F1, returning zero when precision or recall is undefined."""
    y_actual = np.asarray(y_actual).astype(int)
    y_predicted = np.asarray(y_predicted).astype(int)
    TP = np.sum((y_actual == 1) & (y_predicted == 1))
    FP = np.sum((y_actual == 0) & (y_predicted == 1))
    FN = np.sum((y_actual == 1) & (y_predicted == 0))
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    metric = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    return metric


def binary_balanced_accuracy(y_actual, y_predicted):
    """Average positive- and negative-class recall."""
    y_actual = np.asarray(y_actual).astype(int)
    y_predicted = np.asarray(y_predicted).astype(int)
    TP = np.sum((y_actual == 1) & (y_predicted == 1))
    TN = np.sum((y_actual == 0) & (y_predicted == 0))
    FP = np.sum((y_actual == 0) & (y_predicted == 1))
    FN = np.sum((y_actual == 1) & (y_predicted == 0))
    TPR = TP / (TP + FN) if TP + FN > 0 else 0.0
    TNR = TN / (TN + FP) if TN + FP > 0 else 0.0
    metric = 0.5 * (TPR + TNR)
    return metric


def train_random_forest_model(X, y, hyperparams):
    """Fit the random-forest classifier with the supplied parameters."""
    rf_model = RandomForestClassifier(**hyperparams)
    rf_model.fit(X, y)
    return rf_model


CLASSIFIER_NAMES = ["Logistic Regression", "Random Forest", "Gradient Boosting"]


def save_figure(path):
    plt.gcf().savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def explore_data(data, output):
    """Save the original types of distribution, missingness, and correlation analysis."""
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "missing_count": data.isna().sum(),
            "missing_percent": data.isna().mean().mul(100).round(2),
        }
    ).to_csv(output / "missing_values.csv")
    data.describe().to_csv(output / "numeric_summary.csv")
    numeric = data.select_dtypes(include=["float64", "int64"])
    outliers = {}
    for column in numeric:
        values = data[column]
        q1, q3 = values.quantile([0.25, 0.75])
        lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        outliers[column] = int(((values < lower) | (values > upper)).sum())
        plt.figure(figsize=(5, 3))
        plt.hist(values.dropna(), bins=30, color="#2563eb")
        plt.title(f"Distribution of {column}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        save_figure(output / f"distribution_{column}.png")
    pd.Series(outliers, name="outlier_count").to_csv(output / "outlier_counts.csv")
    for column in data.select_dtypes(exclude=np.number):
        data[column].value_counts().to_csv(output / f"counts_{column}.csv")
    correlation = numeric.corr()
    correlation.to_csv(output / "correlation.csv")
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation)), correlation.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(correlation)), correlation.columns, fontsize=7)
    fig.colorbar(image, ax=ax)
    ax.set_title("Numerical feature correlations")
    save_figure(output / "correlation.png")


def evaluate_classifier(model, X_train, y_train, X_test, y_test, name, output):
    """Use the same labels, decision rule, F1, balanced accuracy, and ROC-AUC."""
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    row = {
        "model": name,
        "train_f1": binary_f1_score(y_train, train_predictions),
        "test_f1": binary_f1_score(y_test, test_predictions),
        "train_balanced_accuracy": binary_balanced_accuracy(y_train, train_predictions),
        "test_balanced_accuracy": binary_balanced_accuracy(y_test, test_predictions),
        "test_roc_auc": roc_auc_score(y_test, probabilities),
    }
    row["f1_gap"] = row["train_f1"] - row["test_f1"]
    row["balanced_accuracy_gap"] = (
        row["train_balanced_accuracy"] - row["test_balanced_accuracy"]
    )
    slug = name.lower().replace(" ", "_")
    display = ConfusionMatrixDisplay.from_predictions(
        y_test, test_predictions, cmap="Blues"
    )
    display.ax_.set_title(f"{name}: test confusion matrix")
    save_figure(output / f"{slug}_confusion.png")
    display = RocCurveDisplay.from_predictions(y_test, probabilities)
    display.ax_.set_title(f"{name}: test ROC curve")
    save_figure(output / f"{slug}_roc.png")
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(
            model.feature_importances_, index=X_train.columns
        ).sort_values()
        importance.rename("importance").to_csv(
            output / f"{slug}_feature_importance.csv"
        )
        importance.plot.barh(figsize=(10, 10), color="#2563eb")
        plt.title(f"{name}: impurity-based feature importance")
        save_figure(output / f"{slug}_feature_importance.png")
    return row


def plot_model_comparison(rows, path, title):
    """Compare recorded scores; the chart does not imply statistical significance."""
    by_name = {row["model"]: row for row in rows}
    names = [name for name in CLASSIFIER_NAMES if name in by_name]
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for offset, (key, label, color) in enumerate(
        [
            ("test_f1", "Test F1", "#2563eb"),
            ("test_balanced_accuracy", "Test balanced accuracy", "#0d9488"),
            ("test_roc_auc", "Test ROC-AUC", "#d97706"),
        ]
    ):
        values = [float(by_name[name][key]) for name in names]
        bars = ax.bar(
            positions + (offset - 1) * 0.24,
            values,
            width=0.24,
            label=label,
            color=color,
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set_xticks(positions, names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=8)
    save_figure(path)


def main():
    parser = argparse.ArgumentParser(
        description="Run video-engagement prediction experiments."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/lectures_dataset.csv"),
        help="Input CSV with the documented 22-column schema.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/latest"),
        help="Destination for new metrics and figures.",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip distribution plots and summary tables.",
    )
    parser.add_argument(
        "--plot-recorded",
        action="store_true",
        help="Regenerate the comparison chart from recorded metrics without training.",
    )
    args = parser.parse_args()
    if args.plot_recorded:
        root = Path(__file__).resolve().parent
        with (root / "results/recorded_metrics.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        (root / "figures").mkdir(exist_ok=True)
        plot_model_comparison(
            rows,
            root / "figures/model_comparison.png",
            "Recorded classification results (preprocessing before split)",
        )
        return
    if not args.data.is_file():
        parser.error(
            "Input data not found. Supply --data with a compatible CSV; see data/README.md."
        )
    data = pd.read_csv(args.data)
    required = {
        "auxiliary_rate",
        "conjugate_rate",
        "normalization_rate",
        "tobe_verb_rate",
        "preposition_rate",
        "pronoun_rate",
        "document_entropy",
        "easiness",
        "fraction_stopword_coverage",
        "fraction_stopword_presence",
        "subject_domain",
        "freshness",
        "title_word_count",
        "word_count",
        "most_covered_topic",
        "topic_coverage",
        "duration",
        "lecture_type",
        "has_parts",
        "speaker_speed",
        "silent_period_rate",
        "median_engagement",
    }
    missing = required - set(data.columns)
    if missing:
        parser.error(f"Missing required columns: {', '.join(sorted(missing))}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_eda:
        explore_data(data, args.output_dir / "eda")

    # Preserve the original ordering. This fits preprocessing on all rows before the split.
    processed = preprocess_lecture_dataset(data)
    _, X, y = prepare_final_dataset(processed)
    np.save(args.output_dir / "regression_labels.npy", y)
    X.to_csv(args.output_dir / "features.csv", index=False)
    # Retain the original CSV/NumPy save-and-reload boundary before fitting models.
    X = pd.read_csv(args.output_dir / "features.csv")
    y = np.load(args.output_dir / "regression_labels.npy", allow_pickle=False)

    # Regression uses the clipped target and is scored on its training data, not a holdout.
    ridge = train_ridge_model(X, y, {"alpha": 1.0, "fit_intercept": True})
    regression = {
        "model": "Ridge",
        "evaluation": "in_sample_clipped_target",
        "r2": float(ridge.score(X, y)),
    }
    with (args.output_dir / "regression_metrics.json").open("w") as handle:
        json.dump(regression, handle, indent=2)

    y_class = np.vectorize(cont_to_class)(y)
    np.save(args.output_dir / "classification_labels.npy", y_class)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    logistic_params = {"penalty": "l2", "C": 1.0, "max_iter": 1000, "solver": "lbfgs"}
    models = {
        "Logistic Regression": train_logistic_model(X_train, y_train, logistic_params),
        "Random Forest": train_random_forest_model(
            X_train,
            y_train,
            {
                "n_estimators": 300,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "bootstrap": True,
                "random_state": 42,
                "n_jobs": -1,
            },
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=42,
        ).fit(X_train, y_train),
    }
    rows = [
        evaluate_classifier(
            model, X_train, y_train, X_test, y_test, name, args.output_dir
        )
        for name, model in models.items()
    ]
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_model_comparison(
        rows,
        args.output_dir / "model_comparison.png",
        "Current run (preprocessing before split)",
    )
    summary = {
        "samples": len(data),
        "features": X.shape[1],
        "positive_fraction": float(y_class.mean()),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "changed_regression_targets": int(
            np.count_nonzero(y != data["median_engagement"].to_numpy())
        ),
        "preprocessing_fitted_before_split": True,
    }
    with (args.output_dir / "run_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(
        pd.DataFrame(rows).to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    print("Ridge R2 (in-sample, clipped target):", f"{regression['r2']:.4f}")
    print(
        "Interpretation: preprocessing uses all rows before splitting; these scores are not leakage-free holdout estimates."
    )
    print("New outputs written to:", args.output_dir)


if __name__ == "__main__":
    main()
