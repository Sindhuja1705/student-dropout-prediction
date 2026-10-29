"""Focused, leakage-safe analysis of the academic-feature Random Forest."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from ml_training.feature_engineering import AcademicFeatureEngineer, ENGINEERED_FEATURES
DATASET = ROOT / "dataset" / "data.csv"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
CURRENT_RAW_FEATURES = [
    "Gender", "Age at enrollment", "Course", "Previous qualification (grade)",
    "Admission grade", "Tuition fees up to date", "Scholarship holder",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)",
]
TARGET = "Target"


def load_data():
    frame = pd.read_csv(DATASET, sep=";")
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.replace(r"^\s*$", np.nan, regex=True).drop_duplicates().dropna(subset=[TARGET])
    classes = sorted(frame[TARGET].astype(str).str.strip().unique())
    mapping = {label: index for index, label in enumerate(classes)}
    return frame.drop(columns=[TARGET]), frame[TARGET].astype(str).str.strip().map(mapping), classes


def transformed_data(X_train, X_test):
    engineer = AcademicFeatureEngineer().fit(X_train)
    return engineer.transform(X_train), engineer.transform(X_test)


def preprocessor(columns):
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), columns)
    ])


def pipeline(columns, estimator):
    return Pipeline([("preprocessor", preprocessor(columns)), ("model", estimator)])


def evaluate(name, X_train, X_test, y_train, y_test, columns, estimator, params=None):
    search = GridSearchCV(
        pipeline(columns, estimator),
        params or {},
        scoring="f1_macro",
        cv=StratifiedKFold(CV_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train[columns], y_train)
    predictions = search.predict(X_test[columns])
    probabilities = search.predict_proba(X_test[columns]) if hasattr(search, "predict_proba") else None
    result = {
        "Candidate": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Macro F1": f1_score(y_test, predictions, average="macro"),
        "Weighted F1": f1_score(y_test, predictions, average="weighted"),
        "Precision": precision_score(y_test, predictions, average="macro", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="macro", zero_division=0),
        "ROC-AUC": np.nan,
        "CV F1-macro": search.best_score_,
        "Features": json.dumps(columns),
        "Best parameters": json.dumps(search.best_params_, sort_keys=True, default=str),
        "Confusion matrix": json.dumps(confusion_matrix(y_test, predictions).tolist()),
    }
    if probabilities is not None:
        result["ROC-AUC"] = roc_auc_score(y_test, probabilities, multi_class="ovr", labels=search.classes_)
    return result, search.best_estimator_


def main():
    X, y, classes = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train_eng, X_test_eng = transformed_data(X_train, X_test)
    all_features = list(X_train_eng.columns)

    train_numeric = X_train_eng[all_features].apply(pd.to_numeric)
    imputed = SimpleImputer(strategy="median").fit_transform(train_numeric)
    importance_scores = pd.Series(
        mutual_info_classif(imputed, y_train, random_state=RANDOM_STATE),
        index=all_features,
        name="mutual_information",
    ).sort_values(ascending=False)
    importance_scores.to_csv(MODEL_DIR / "experiment2_feature_importance.csv", header=True)

    correlations = X_train_eng[all_features].corr(method="spearman")
    correlations.to_csv(MODEL_DIR / "experiment2_feature_correlations.csv")
    upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
    redundant_pairs = [
        {"feature_1": column, "feature_2": row, "absolute_spearman": abs(upper.loc[row, column])}
        for column in upper.columns
        for row in upper.index
        if pd.notna(upper.loc[row, column]) and abs(upper.loc[row, column]) >= 0.90
    ]
    with (MODEL_DIR / "experiment2_redundant_features.json").open("w", encoding="utf-8") as file:
        json.dump(redundant_pairs, file, indent=2)

    selected = importance_scores[importance_scores >= importance_scores.median()].index.tolist()
    weak_removed = [feature for feature in all_features if feature not in selected]
    non_redundant = list(all_features)
    for pair in redundant_pairs:
        candidate = pair["feature_2"]
        if candidate in non_redundant and importance_scores[pair["feature_1"]] >= importance_scores[candidate]:
            non_redundant.remove(candidate)
    if len(non_redundant) < 8:
        non_redundant = selected

    rf_base = RandomForestClassifier(
        n_estimators=180, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    candidates = [
        ("A. Original Random Forest", X_train[CURRENT_RAW_FEATURES], X_test[CURRENT_RAW_FEATURES], CURRENT_RAW_FEATURES, rf_base, {}),
        ("B. Random Forest + engineered features", X_train_eng, X_test_eng, all_features, rf_base, {}),
        ("C. Tuned Random Forest", X_train_eng, X_test_eng, all_features, rf_base, {
            "model__n_estimators": [180, 300],
            "model__max_depth": [None, 16, 24],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", 0.7],
            "model__class_weight": ["balanced", "balanced_subsample"],
        }),
        ("D. Feature-selected Random Forest", X_train_eng, X_test_eng, selected, rf_base, {}),
        ("E. Extra Trees", X_train_eng, X_test_eng, all_features, ExtraTreesClassifier(
            n_estimators=250, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ), {}),
        ("F. HistGradientBoosting", X_train_eng, X_test_eng, all_features, HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.08, max_leaf_nodes=15, random_state=RANDOM_STATE
        ), {}),
    ]

    results = []
    fitted = {}
    for name, train_frame, test_frame, columns, estimator, params in candidates:
        result, model = evaluate(name, train_frame, test_frame, y_train, y_test, columns, estimator, params)
        results.append(result)
        fitted[name] = model

    results_frame = pd.DataFrame(results).sort_values(["Macro F1", "Accuracy"], ascending=False).reset_index(drop=True)
    results_frame.to_csv(MODEL_DIR / "experiment2_final_comparison.csv", index=False)
    best_name = results_frame.iloc[0]["Candidate"]
    best_model = fitted[best_name]
    complete_pipeline = Pipeline([
        ("academic_features", AcademicFeatureEngineer()),
        ("preprocessor", preprocessor(all_features)),
        ("model", best_model.named_steps["model"]),
    ])
    complete_pipeline.fit(X_train, y_train)
    joblib.dump(complete_pipeline, MODEL_DIR / "experiment2_best_candidate.pkl")
    metadata = {
        "target": TARGET,
        "target_classes": classes,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_splits": CV_SPLITS,
        "engineered_features": ENGINEERED_FEATURES,
        "selected_features": selected,
        "features_removed_by_selection": weak_removed,
        "redundant_pairs": redundant_pairs,
        "features_removed_for_redundancy": [feature for feature in all_features if feature not in non_redundant],
        "best_candidate": best_name,
        "best_metrics": results_frame.iloc[0].to_dict(),
        "baseline_reference_accuracy": 0.785311,
        "candidate_artifact": "experiment2_best_candidate.pkl",
        "artifact_input_features": list(X_train.columns),
    }
    joblib.dump(metadata, MODEL_DIR / "experiment2_analysis_metadata.pkl")
    results_frame["Candidate"] = results_frame["Candidate"].astype(str)
    print(results_frame[["Candidate", "Accuracy", "Macro F1", "Weighted F1", "ROC-AUC", "CV F1-macro"]].to_string(index=False))
    print("\nEngineered features:", ENGINEERED_FEATURES)
    print("Selected features:", selected)
    print("Removed as weak:", weak_removed)
    print("Redundant pairs:", redundant_pairs)
    print("Best candidate:", best_name)


if __name__ == "__main__":
    main()
