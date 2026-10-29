"""Compare leakage-conscious models for an after-semester-1 prediction cutoff.

This research workflow does not modify datasets or persist/replace application
model artifacts. Preprocessing is fitted inside each cross-validation fold.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "dataset" / "data.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
TARGET_COLUMN = "Target"

PRE_ENROLLMENT_FEATURES = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
]

SEMESTER_1_FEATURES = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
]

CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Gender",
    "Scholarship holder",
    "International",
]


def load_dataset(path=DATASET_PATH):
    """Load and validate the project's semicolon-delimited training data."""
    frame = pd.read_csv(path, sep=";")
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.replace(r"^\s*$", np.nan, regex=True)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Dataset is missing target column {TARGET_COLUMN!r}.")
    if frame[TARGET_COLUMN].isna().any():
        frame = frame.dropna(subset=[TARGET_COLUMN])
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(str).str.strip()
    frame = frame[frame[TARGET_COLUMN] != ""].drop_duplicates().reset_index(drop=True)
    if frame.empty:
        raise ValueError("Dataset contains no rows with a usable target.")
    return frame


def get_after_semester1_features(columns):
    """Return the explicit, cutoff-safe enrollment and first-semester inputs."""
    features = PRE_ENROLLMENT_FEATURES + SEMESTER_1_FEATURES
    missing = [column for column in features if column not in columns]
    if missing:
        raise ValueError(f"Dataset is missing eligible features: {missing}")
    return features


def make_preprocessor(feature_columns):
    categorical = [column for column in CATEGORICAL_FEATURES if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def make_models(class_count):
    """Use a consistent class-balancing strategy for the three candidates."""
    return {
        "Logistic Regression": (
            LogisticRegression(
                solver="newton-cg",
                max_iter=1500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=2,
            ),
            False,
        ),
        "XGBoost": (
            XGBClassifier(
                objective="multi:softprob",
                num_class=class_count,
                eval_metric="mlogloss",
                n_estimators=220,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=2,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=2,
                tree_method="hist",
            ),
            True,
        ),
    }


def _score(y_true, predictions):
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision_macro": precision_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_true, predictions, average="macro", zero_division=0),
    }


def evaluate_model(
    estimator,
    uses_sample_weights,
    preprocessor,
    X_train,
    y_train,
    X_test,
    y_test,
    class_labels,
):
    """Evaluate one candidate using training-only stratified CV and a holdout."""
    cv = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    fold_scores = []
    oof_predictions = np.full(len(y_train), -1, dtype=int)

    for train_indices, validation_indices in cv.split(X_train, y_train):
        pipeline = Pipeline(
            [("preprocessor", clone(preprocessor)), ("model", clone(estimator))]
        )
        fit_params = {}
        if uses_sample_weights:
            fit_params["model__sample_weight"] = compute_sample_weight(
                "balanced", y_train.iloc[train_indices]
            )
        pipeline.fit(
            X_train.iloc[train_indices],
            y_train.iloc[train_indices],
            **fit_params,
        )
        predictions = pipeline.predict(X_train.iloc[validation_indices])
        oof_predictions[validation_indices] = predictions
        fold_scores.append(
            _score(y_train.iloc[validation_indices], predictions)
        )

    holdout_pipeline = Pipeline(
        [("preprocessor", clone(preprocessor)), ("model", clone(estimator))]
    )
    fit_params = {}
    if uses_sample_weights:
        fit_params["model__sample_weight"] = compute_sample_weight(
            "balanced", y_train
        )
    holdout_pipeline.fit(X_train, y_train, **fit_params)
    holdout_predictions = holdout_pipeline.predict(X_test)

    metric_names = list(fold_scores[0])
    cv_means = {
        name: float(np.mean([score[name] for score in fold_scores]))
        for name in metric_names
    }
    cv_f1_std = float(
        np.std([score["f1_macro"] for score in fold_scores], ddof=1)
    )
    return {
        "cv_means": cv_means,
        "cv_f1_std": cv_f1_std,
        "cv_confusion_matrix": confusion_matrix(
            y_train, oof_predictions, labels=class_labels
        ),
        "holdout": _score(y_test, holdout_predictions),
        "holdout_confusion_matrix": confusion_matrix(
            y_test, holdout_predictions, labels=class_labels
        ),
    }


def main():
    frame = load_dataset()
    feature_columns = get_after_semester1_features(frame.columns)
    target_labels = sorted(frame[TARGET_COLUMN].unique())
    label_to_integer = {label: index for index, label in enumerate(target_labels)}
    X = frame[feature_columns]
    y = frame[TARGET_COLUMN].map(label_to_integer)

    class_counts = frame[TARGET_COLUMN].value_counts()
    imbalance_ratio = float(class_counts.max() / class_counts.min())
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    preprocessor = make_preprocessor(feature_columns)

    print(f"Rows retained: {len(frame)}")
    print(f"Target class order: {target_labels}")
    print(f"Full-data class counts: {class_counts.to_dict()}")
    print(f"Majority/minority class ratio: {imbalance_ratio:.3f}")
    print(f"Eligible after-semester-1 features ({len(feature_columns)}):")
    print(", ".join(feature_columns))
    print(
        "Excluded: all second-semester features, Debtor, Tuition fees up to date, "
        "Unemployment rate, Inflation rate, GDP"
    )
    print(
        f"Evaluation: stratified {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%} holdout; "
        f"{CV_SPLITS}-fold stratified CV on training rows"
    )
    print("Confusion matrices use rows=true, columns=predicted; class order above.")

    results = {}
    for name, (estimator, uses_sample_weights) in make_models(
        class_count=len(target_labels)
    ).items():
        result = evaluate_model(
            estimator,
            uses_sample_weights,
            preprocessor,
            X_train,
            y_train,
            X_test,
            y_test,
            class_labels=np.arange(len(target_labels)),
        )
        results[name] = result
        print(f"\n{name}")
        print(
            "CV mean accuracy={accuracy:.4f}, macro precision={precision_macro:.4f}, "
            "macro recall={recall_macro:.4f}, macro F1={f1_macro:.4f} "
            "(macro F1 SD={f1_sd:.4f})".format(
                **result["cv_means"], f1_sd=result["cv_f1_std"]
            )
        )
        print("CV out-of-fold confusion matrix:")
        print(result["cv_confusion_matrix"])
        print("Holdout accuracy/precision/recall/macro-F1:")
        print(
            " / ".join(
                f"{result['holdout'][metric]:.4f}"
                for metric in (
                    "accuracy",
                    "precision_macro",
                    "recall_macro",
                    "f1_macro",
                )
            )
        )
        print("Holdout confusion matrix:")
        print(result["holdout_confusion_matrix"])

    best_cv_model = max(
        results, key=lambda model_name: results[model_name]["cv_means"]["f1_macro"]
    )
    print(
        f"\nHighest training-CV macro F1: {best_cv_model}. "
        "No model artifact was saved or promoted."
    )


if __name__ == "__main__":
    main()
