"""Leakage-safe optimization comparison for the current academic-feature model."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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


def load_data():
    frame = pd.read_csv(DATASET, sep=";")
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.replace(r"^\s*$", np.nan, regex=True).drop_duplicates()
    frame = frame.dropna(subset=[TARGET])
    classes = sorted(frame[TARGET].astype(str).str.strip().unique())
    mapping = {label: index for index, label in enumerate(classes)}
    return (
        frame.drop(columns=[TARGET]),
        frame[TARGET].astype(str).str.strip().map(mapping),
        classes,
    )


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
            y_test, probabilities, multi_class="ovr", labels=list(range(len(classes)))
        ),
        "CV Macro F1": cv_score,
        "Confusion matrix": json.dumps(confusion_matrix(y_test, predictions).tolist()),
        "Class order": json.dumps(classes),
        "Features": json.dumps(list(X_test.columns)),
        "Best hyperparameters": json.dumps(params, sort_keys=True, default=str),
    }


def main():
    X, y, classes = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # The engineer is fit only on X_train; X_test remains untouched until final scoring.
    engineer = AcademicFeatureEngineer().fit(X_train)
    X_train_eng = engineer.transform(X_train)
    X_test_eng = engineer.transform(X_test)
    feature_names = list(X_train_eng.columns)
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    baseline_estimator = RandomForestClassifier(
        n_estimators=180,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    baseline = make_pipeline(baseline_estimator, feature_names)
    baseline_cv = cross_val_score(
        baseline, X_train_eng, y_train, scoring="f1_macro", cv=cv, n_jobs=-1
    ).mean()
    baseline.fit(X_train_eng, y_train)

    searches = [
        (
            "Tuned Random Forest",
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [180, 300, 450],
                "model__max_depth": [None, 12, 20, 28],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.7],
                "model__class_weight": ["balanced", "balanced_subsample", None],
            },
            14,
        ),
        (
            "Tuned XGBoost",
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
            14,
        ),
        (
            "Tuned CatBoost",
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
            12,
        ),
    ]

    fitted = {
        "Current Random Forest baseline": (
            baseline,
            baseline_cv,
            {
                "model__n_estimators": 180,
                "model__class_weight": "balanced",
                "model__random_state": RANDOM_STATE,
            },
        )
    }
    for name, estimator, distributions, n_iter in searches:
        search = RandomizedSearchCV(
            make_pipeline(estimator, feature_names),
            distributions,
            n_iter=n_iter,
            scoring="f1_macro",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train_eng, y_train)
        fitted[name] = (search.best_estimator_, search.best_score_, search.best_params_)

    results = [
        metrics_for(name, model, X_test_eng, y_test, classes, cv_score, params)
        for name, (model, cv_score, params) in fitted.items()
    ]
    results_frame = pd.DataFrame(results).sort_values(
        ["CV Macro F1", "Macro F1"], ascending=False
    ).reset_index(drop=True)
    results_frame.to_csv(MODEL_DIR / "optimization_phase_comparison.csv", index=False)

    metadata = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_splits": CV_SPLITS,
        "selection_metric": "training-only StratifiedKFold Macro F1",
        "classes": classes,
        "features": feature_names,
        "test_set_used_only_for_final_comparison": True,
        "production_model_changed": False,
        "candidates": results_frame["Candidate"].tolist(),
    }
    with (MODEL_DIR / "optimization_phase_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(
        results_frame[
            ["Candidate", "Accuracy", "Macro F1", "Weighted F1", "ROC-AUC", "CV Macro F1"]
        ].to_string(index=False)
    )
    print("\nFeatures:", json.dumps(feature_names))
    print("\nConfusion matrices:")
    for _, row in results_frame.iterrows():
        print(row["Candidate"], row["Confusion matrix"])


if __name__ == "__main__":
    main()
