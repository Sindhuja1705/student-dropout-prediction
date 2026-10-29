"""Leakage-safe optimization rerun for the current academic-feature model.

This script intentionally keeps the original dataset, train/test split, and model
selection policy unchanged while evaluating the requested additional candidates:
1) better Random Forest tuning,
2) XGBoost tuning,
3) CatBoost,
4) ExtraTrees,
5) useful academic feature combinations.

The results are saved under models/ in a separate file so the current production
artifact is not overwritten.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml_training.feature_engineering import AcademicFeatureEngineer

DATASET = ROOT / "dataset" / "data.csv"
MODEL_DIR = ROOT / "models"
TARGET = "Target"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

BASE_FEATURES = [
    "Gender",
    "Age at enrollment",
    "Course",
    "Previous qualification (grade)",
    "Admission grade",
    "Tuition fees up to date",
    "Scholarship holder",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
]

ENGINEERED_FEATURES = [
    "Semester 1 pass rate",
    "Semester 2 pass rate",
    "Total approved units",
    "Total failed units",
    "Credit completion rate",
    "Average semester grade",
    "Grade improvement",
    "Academic progress",
]


def load_data():
    frame = pd.read_csv(DATASET, sep=";")
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.replace(r"^\s*$", np.nan, regex=True).drop_duplicates()
    frame = frame.dropna(subset=[TARGET])
    classes = sorted(frame[TARGET].astype(str).str.strip().unique())
    mapping = {label: index for index, label in enumerate(classes)}
    return frame.drop(columns=[TARGET]), frame[TARGET].astype(str).str.strip().map(mapping), classes


def make_pipeline(estimator, feature_names):
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_names,
            )
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def metrics_for(name, model, X_test, y_test, classes, cv_score, params):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    return {
        "Candidate": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Macro F1": f1_score(y_test, predictions, average="macro"),
        "Weighted F1": f1_score(y_test, predictions, average="weighted"),
        "Precision": precision_score(y_test, predictions, average="macro", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="macro", zero_division=0),
        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities,
            multi_class="ovr",
            labels=list(range(len(classes))),
        ),
        "CV Macro F1": cv_score,
        "Confusion matrix": json.dumps(confusion_matrix(y_test, predictions).tolist()),
        "Class order": json.dumps(classes),
        "Features": json.dumps(list(X_test.columns)),
        "Best hyperparameters": json.dumps(params, sort_keys=True, default=str),
    }


def tune_candidate(name, estimator, param_grid, X_train, y_train, feature_names, cv):
    search = RandomizedSearchCV(
        make_pipeline(estimator, feature_names),
        param_grid,
        n_iter=len(next(iter(param_grid.values()))),
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train[feature_names], y_train)
    return search.best_estimator_, search.best_score_, search.best_params_


def evaluate_baseline(name, X_train, X_test, y_train, y_test, feature_names, estimator, classes, cv):
    model = make_pipeline(estimator, feature_names)
    cv_score = cross_val_score(model, X_train[feature_names], y_train, scoring="f1_macro", cv=cv, n_jobs=-1).mean()
    model.fit(X_train[feature_names], y_train)
    return metrics_for(name, model, X_test[feature_names], y_test, classes, cv_score, estimator.get_params())


def main():
    X, y, classes = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    engineer = AcademicFeatureEngineer().fit(X_train)
    X_train_eng = engineer.transform(X_train)
    X_test_eng = engineer.transform(X_test)

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    baseline_rf = RandomForestClassifier(
        n_estimators=180,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    baseline_record = evaluate_baseline(
        "Current Random Forest + academic features",
        X_train_eng,
        X_test_eng,
        y_train,
        y_test,
        list(X_train_eng.columns),
        baseline_rf,
        classes,
        cv,
    )

    candidate_specs = [
        (
            "Tuned Random Forest",
            X_train_eng,
            X_test_eng,
            list(X_train_eng.columns),
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [180, 300, 450],
                "model__max_depth": [None, 12, 20, 28],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.7],
                "model__class_weight": ["balanced", "balanced_subsample", None],
            },
        ),
        (
            "Tuned XGBoost",
            X_train_eng,
            X_test_eng,
            list(X_train_eng.columns),
            XGBClassifier(
                objective="multi:softprob",
                num_class=len(classes),
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [150, 250, 400],
                "model__max_depth": [3, 4, 6],
                "model__learning_rate": [0.03, 0.06, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__min_child_weight": [1, 3, 5],
            },
        ),
        (
            "Tuned CatBoost",
            X_train_eng,
            X_test_eng,
            list(X_train_eng.columns),
            CatBoostClassifier(
                loss_function="MultiClass",
                verbose=False,
                random_seed=RANDOM_STATE,
                thread_count=-1,
                allow_writing_files=False,
            ),
            {
                "model__iterations": [200, 350, 500],
                "model__depth": [4, 5, 6, 7],
                "model__learning_rate": [0.03, 0.06, 0.1],
                "model__l2_leaf_reg": [3, 6, 10],
                "model__random_strength": [0.5, 1.0, 2.0],
            },
        ),
        (
            "ExtraTrees tuned",
            X_train_eng,
            X_test_eng,
            list(X_train_eng.columns),
            ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [None, 14, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.8, 1.0],
                "model__class_weight": ["balanced", "balanced_subsample", None],
            },
        ),
        (
            "RF + academic combo A",
            X_train_eng,
            X_test_eng,
            BASE_FEATURES + ["Average semester grade", "Academic progress", "Credit completion rate", "Grade improvement"],
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "model__max_depth": [None, 12, 18, 24],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.7],
            },
        ),
        (
            "RF + academic combo B",
            X_train_eng,
            X_test_eng,
            BASE_FEATURES + ["Semester 1 pass rate", "Semester 2 pass rate", "Average semester grade", "Academic progress"],
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "model__max_depth": [None, 12, 18, 24],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.7],
            },
        ),
    ]

    results = [baseline_record]
    for name, X_train_frame, X_test_frame, feature_names, estimator, param_grid in candidate_specs:
        tuned_model, cv_score, best_params = tune_candidate(
            name,
            estimator,
            param_grid,
            X_train_frame,
            y_train,
            feature_names,
            cv,
        )
        results.append(
            metrics_for(name, tuned_model, X_test_frame[feature_names], y_test, classes, cv_score, best_params)
        )

    results_frame = pd.DataFrame(results).sort_values(["Accuracy", "Macro F1"], ascending=False).reset_index(drop=True)
    results_path = MODEL_DIR / "optimization_retry_results.csv"
    metadata_path = MODEL_DIR / "optimization_retry_metadata.json"
    results_frame.to_csv(results_path, index=False)

    metadata = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_splits": CV_SPLITS,
        "selection_metric": "training-only macro F1",
        "baseline_accuracy": float(baseline_record["Accuracy"]),
        "classes": classes,
        "dataset": str(DATASET),
        "production_model_not_replaced": True,
        "results_file": str(results_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(results_frame[["Candidate", "Accuracy", "Macro F1", "Weighted F1", "ROC-AUC", "CV Macro F1"]].to_string(index=False))
    print("\nSaved results to", results_path)
    print("Saved metadata to", metadata_path)


if __name__ == "__main__":
    main()
