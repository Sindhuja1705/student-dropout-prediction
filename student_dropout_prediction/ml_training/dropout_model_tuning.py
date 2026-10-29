"""Tune the existing classifiers at an after-semester-1 prediction cutoff."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from scipy.stats import randint, loguniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from train_models import RANDOM_STATE, TEST_SIZE, load_dataset


CV_SPLITS = 5
N_ITER = 12
TARGET_COLUMN = "Target"
EXCLUDED_AFTER_CUTOFF = {
    "Debtor",
    "Tuition fees up to date",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
}
CATEGORICAL_FEATURES = {
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
}


def make_scoring(dropout_label):
    dropout_args = {
        "labels": [dropout_label],
        "average": "macro",
        "zero_division": 0,
    }
    return {
        "dropout_precision": make_scorer(precision_score, **dropout_args),
        "dropout_recall": make_scorer(recall_score, **dropout_args),
        "dropout_f1": make_scorer(f1_score, **dropout_args),
        "macro_f1": "f1_macro",
    }


def get_after_semester1_features(columns):
    return [
        column
        for column in columns
        if column != TARGET_COLUMN
        and not column.startswith("Curricular units 2nd sem")
        and column not in EXCLUDED_AFTER_CUTOFF
    ]


def make_pipeline(estimator, feature_columns, sampler="passthrough"):
    categorical = [
        column for column in feature_columns if column in CATEGORICAL_FEATURES
    ]
    numeric = [
        column for column in feature_columns if column not in CATEGORICAL_FEATURES
    ]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                SklearnPipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                SklearnPipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "feature_selection",
                SelectKBest(
                    score_func=f_classif,
                    k="all",
                ),
            ),
            ("sampler", sampler),
            ("model", estimator),
        ]
    )


def baseline_candidates(class_count):
    return {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "passthrough",
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=180,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=2,
            ),
            "passthrough",
        ),
        "XGBoost": (
            XGBClassifier(
                objective="multi:softprob",
                num_class=class_count,
                eval_metric="mlogloss",
                n_estimators=180,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=RANDOM_STATE,
                n_jobs=2,
                tree_method="hist",
            ),
            "passthrough",
        ),
    }


def tuned_candidates(class_count):
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2500, random_state=RANDOM_STATE),
            {
                "model__C": loguniform(0.05, 20),
                "model__class_weight": [None, "balanced"],
                "feature_selection__k": [15, 25, 35, "all"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=2),
            {
                "model__n_estimators": randint(120, 361),
                "model__max_depth": [None, 10, 14, 20, 28],
                "model__min_samples_split": randint(2, 12),
                "model__min_samples_leaf": randint(1, 6),
                "model__max_features": ["sqrt", 0.7, 1.0],
                "model__class_weight": [
                    "balanced",
                    "balanced_subsample",
                    None,
                ],
                "feature_selection__k": [15, 25, 35, "all"],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                objective="multi:softprob",
                num_class=class_count,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=2,
            ),
            {
                "model__n_estimators": randint(100, 321),
                "model__max_depth": randint(2, 7),
                "model__learning_rate": loguniform(0.02, 0.2),
                "model__subsample": [0.75, 0.9, 1.0],
                "model__colsample_bytree": [0.75, 0.9, 1.0],
                "model__min_child_weight": randint(1, 7),
                "feature_selection__k": [15, 25, 35, "all"],
                "sampler": [
                    "passthrough",
                    RandomOverSampler(random_state=RANDOM_STATE),
                ],
            },
        ),
    }


def score_holdout(name, estimator, X_test, y_test, classes):
    predictions = estimator.predict(X_test)
    dropout_label = classes.index("Dropout")
    return {
        "Candidate": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Dropout Precision": precision_score(
            y_test,
            predictions,
            labels=[dropout_label],
            average="macro",
            zero_division=0,
        ),
        "Dropout Recall": recall_score(
            y_test,
            predictions,
            labels=[dropout_label],
            average="macro",
            zero_division=0,
        ),
        "Dropout F1-score": f1_score(
            y_test,
            predictions,
            labels=[dropout_label],
            average="macro",
            zero_division=0,
        ),
        "Macro F1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "Confusion matrix": confusion_matrix(
            y_test, predictions, labels=np.arange(len(classes))
        ).tolist(),
    }


def main():
    frame = load_dataset()
    feature_columns = get_after_semester1_features(frame.columns)
    if not feature_columns:
        raise ValueError("Dataset contains no after-semester-1 eligible features.")

    classes = sorted(frame[TARGET_COLUMN].astype(str).str.strip().unique())
    if "Dropout" not in classes:
        raise ValueError("Dataset target must include the 'Dropout' class.")
    class_to_integer = {label: index for index, label in enumerate(classes)}
    X = frame[feature_columns].copy()
    y = frame[TARGET_COLUMN].astype(str).str.strip().map(class_to_integer)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scoring = make_scoring(class_to_integer["Dropout"])
    results = []

    print(
        f"Rows: {len(frame)}; features: {len(feature_columns)}; "
        f"class order: {classes}"
    )
    print(
        f"Split: stratified {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}; "
        f"CV: {CV_SPLITS}-fold stratified on training data; "
        f"selection score: CV dropout F1"
    )
    print(
        "Imputation, scaling, one-hot encoding, and SelectKBest are refit inside each fold. "
        "No model files or dataset files are written."
    )

    for name, (estimator, sampler) in baseline_candidates(len(classes)).items():
        pipeline = make_pipeline(estimator, feature_columns, sampler)
        cv_result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        pipeline.fit(X_train, y_train)
        row = score_holdout(
            f"Baseline {name}", pipeline, X_test, y_test, classes
        )
        row["CV Dropout F1"] = float(cv_result["test_dropout_f1"].mean())
        row["CV Dropout F1 SD"] = float(
            np.std(cv_result["test_dropout_f1"], ddof=1)
        )
        row["CV Macro F1"] = float(cv_result["test_macro_f1"].mean())
        row["Best parameters"] = "Existing baseline settings"
        results.append(row)

    for name, (estimator, parameter_distributions) in tuned_candidates(
        len(classes)
    ).items():
        pipeline = make_pipeline(estimator, feature_columns)
        search = RandomizedSearchCV(
            pipeline,
            parameter_distributions,
            n_iter=N_ITER,
            scoring=scoring,
            refit="dropout_f1",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        row = score_holdout(
            f"Tuned {name}",
            search.best_estimator_,
            X_test,
            y_test,
            classes,
        )
        row["CV Dropout F1"] = float(search.best_score_)
        row["CV Dropout F1 SD"] = float(
            search.cv_results_["std_test_dropout_f1"][search.best_index_]
        )
        row["CV Macro F1"] = float(
            search.cv_results_["mean_test_macro_f1"][search.best_index_]
        )
        row["Best parameters"] = search.best_params_
        results.append(row)

    results.sort(key=lambda result: result["CV Dropout F1"], reverse=True)
    print("\nConfusion matrices: rows=true, columns=predicted; order:", classes)
    for row in results:
        print(f"\n{row['Candidate']}")
        print(
            "CV dropout F1={CV Dropout F1:.4f} (SD={CV Dropout F1 SD:.4f}); "
            "CV macro F1={CV Macro F1:.4f}".format(**row)
        )
        print(
            "Holdout accuracy={Accuracy:.4f}; dropout precision={Dropout Precision:.4f}; "
            "dropout recall={Dropout Recall:.4f}; dropout F1={Dropout F1-score:.4f}; "
            "macro F1={Macro F1:.4f}".format(**row)
        )
        print("Holdout confusion matrix:", row["Confusion matrix"])
        print("Parameters:", row["Best parameters"])

    incumbent = next(
        row for row in results if row["Candidate"] == "Baseline Random Forest"
    )
    winner = results[0]
    print(
        "\nBest candidate by training CV dropout F1:",
        winner["Candidate"],
        f"(CV dropout F1 {winner['CV Dropout F1']:.4f}).",
    )
    print(
        "Incumbent baseline Random Forest holdout dropout F1:",
        f"{incumbent['Dropout F1-score']:.4f}; selected candidate holdout dropout F1:",
        f"{winner['Dropout F1-score']:.4f}.",
    )
    print(
        "This is an evaluation only; the application model artifact was not "
        "replaced. Treat the holdout as final reporting, not as a tuning signal."
    )


if __name__ == "__main__":
    main()
