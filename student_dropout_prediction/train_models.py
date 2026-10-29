"""Compatibility utilities and safe entry point for model candidate training.

The historical holdout trainer is disabled because it uses late-semester
features and overwrites application artifacts. The active entry point creates
separate after-semester-1 candidates without promoting a model.
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from ml_training.feature_engineering import AcademicFeatureEngineer

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "data.csv"
MODEL_DIR = BASE_DIR / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

CURRENT_RAW_FEATURES = [
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

SEMESTER_COLUMNS = [
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
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


class AcademicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add documented semester progress features using only source columns."""

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        missing = [column for column in SEMESTER_COLUMNS if column not in X.columns]
        if missing:
            raise ValueError(f"Missing columns for academic features: {missing}")
        return self

    @staticmethod
    def _safe_rate(numerator, denominator):
        denominator = denominator.replace(0, np.nan)
        return numerator.divide(denominator).fillna(0.0)

    def transform(self, X):
        frame = X.copy()
        sem1_enrolled = frame["Curricular units 1st sem (enrolled)"]
        sem1_evaluated = frame["Curricular units 1st sem (evaluations)"]
        sem1_approved = frame["Curricular units 1st sem (approved)"]
        sem1_grade = frame["Curricular units 1st sem (grade)"]
        sem2_enrolled = frame["Curricular units 2nd sem (enrolled)"]
        sem2_evaluated = frame["Curricular units 2nd sem (evaluations)"]
        sem2_approved = frame["Curricular units 2nd sem (approved)"]
        sem2_grade = frame["Curricular units 2nd sem (grade)"]

        frame["Semester 1 pass rate"] = self._safe_rate(sem1_approved, sem1_evaluated)
        frame["Semester 2 pass rate"] = self._safe_rate(sem2_approved, sem2_evaluated)
        frame["Total approved units"] = sem1_approved + sem2_approved
        frame["Total failed units"] = (
            (sem1_enrolled - sem1_approved).clip(lower=0)
            + (sem2_enrolled - sem2_approved).clip(lower=0)
        )
        frame["Credit completion rate"] = self._safe_rate(
            sem1_approved + sem2_approved,
            sem1_enrolled + sem2_enrolled,
        )
        frame["Average semester grade"] = (sem1_grade + sem2_grade) / 2
        frame["Grade improvement"] = sem2_grade - sem1_grade
        frame["Academic progress"] = (
            frame["Semester 2 pass rate"] - frame["Semester 1 pass rate"]
        )
        return frame

    def get_feature_names_out(self, input_features=None):
        names = list(input_features or self.feature_names_in_)
        return np.asarray(names + ENGINEERED_FEATURES, dtype=object)


# Use a package-qualified class so joblib artifacts load outside this script.
from ml_training.feature_engineering import AcademicFeatureEngineer


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    frame = pd.read_csv(DATASET_PATH, sep=";")
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.replace(r"^\s*$", np.nan, regex=True)
    frame = frame.drop_duplicates().dropna(subset=["Target"]).reset_index(drop=True)
    missing = [column for column in SEMESTER_COLUMNS + CURRENT_RAW_FEATURES if column not in frame]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return frame


def make_preprocessor(columns):
    numeric = [column for column in columns]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )


        ],
        remainder="drop",
    )


def make_models(tuned=False):
    models = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            {"model__C": [0.5, 1.0, 2.0]} if tuned else {},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"model__max_depth": [6, 12, None]} if tuned else {},
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=180,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "model__max_depth": [None, 12],
                "model__min_samples_leaf": [1, 2],
            }
            if tuned
            else {},
        ),
        "SVM": (
            SVC(
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {"model__C": [1.0, 2.0], "model__gamma": ["scale"]} if tuned else {},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 180],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            }
            if tuned
            else {},
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = (
            XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                n_estimators=180,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=RANDOM_STATE,
                n_jobs=2,
            ),
            {
                "model__n_estimators": [120, 180],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
            }
            if tuned
            else {},
        )
    return models


def make_pipeline(columns, model, use_smote):
    steps = [("preprocessor", make_preprocessor(columns))]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return ImbPipeline(steps)


def make_inference_pipeline(selected_columns, model, use_smote):
    """Persist feature engineering and preprocessing in the final artifact."""
    steps = [
        ("academic_features", AcademicFeatureEngineer()),
        ("preprocessor", make_preprocessor(selected_columns)),
    ]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", clone(model)))
    return ImbPipeline(steps)


def fit_and_evaluate(
    experiment_name,
    X_train,
    X_test,
    y_train,
    y_test,
    columns,
    model_name,
    model,
    use_smote,
    parameters=None,
):
    pipeline = make_pipeline(columns, model, use_smote)
    dropout_label = target_classes_global.index("Dropout")
    scoring = {
        "dropout_f1": make_scorer(
            f1_score,
            labels=[dropout_label],
            average="macro",
            zero_division=0,
        ),
        "macro_f1": "f1_macro",
    }
    search = GridSearchCV(
        pipeline,
        parameters or {},
        scoring=scoring,
        cv=StratifiedKFold(CV_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        refit="dropout_f1",
    )
    search.fit(X_train[columns], y_train)
    predictions = search.predict(X_test[columns])
    probabilities = search.predict_proba(X_test[columns]) if hasattr(search, "predict_proba") else None
    labels = np.arange(len(np.unique(y_test)))
    metrics = {
        "Experiment": experiment_name,
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, average="macro", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="macro", zero_division=0),
        "F1-score": f1_score(y_test, predictions, average="macro", zero_division=0),
        "Macro F1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "Dropout Precision": precision_score(
            y_test, predictions, labels=[dropout_label], average="macro", zero_division=0
        ),
        "Dropout Recall": recall_score(
            y_test, predictions, labels=[dropout_label], average="macro", zero_division=0
        ),
        "Dropout F1-score": f1_score(
            y_test, predictions, labels=[dropout_label], average="macro", zero_division=0
        ),
        "Weighted F1": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "ROC-AUC": np.nan,
        "Feature set": experiment_name,
        "Features": json.dumps(list(columns)),
        "Class order": json.dumps(target_classes_global),
        "SMOTE": use_smote,
        "CV Dropout F1": search.best_score_,
        "CV F1-macro": search.cv_results_["mean_test_macro_f1"][search.best_index_],
        "Best hyperparameters": json.dumps(search.best_params_, sort_keys=True, default=str),
        "Confusion matrix": json.dumps(confusion_matrix(y_test, predictions, labels=labels).tolist()),
    }
    if probabilities is not None:
        metrics["ROC-AUC"] = roc_auc_score(
            y_test, probabilities, multi_class="ovr", labels=search.classes_
        )
    return metrics, search.best_estimator_


def _legacy_holdout_training_main():
    raise RuntimeError(
        "The legacy holdout trainer is disabled because it uses post-semester-1 "
        "fields and overwrites production artifacts. Use the safe candidate "
        "training entry point instead."
    )

    global target_classes_global
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_dataset()
    target_classes = sorted(frame["Target"].astype(str).str.strip().unique())
    target_classes_global = target_classes
    class_to_int = {label: index for index, label in enumerate(target_classes)}
    y = frame["Target"].astype(str).str.strip().map(class_to_int)

    raw_all = [column for column in frame.columns if column != "Target"]
    X_all = frame[raw_all].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    engineer = AcademicFeatureEngineer().fit(X_train)
    X_train_engineered = engineer.transform(X_train)
    X_test_engineered = engineer.transform(X_test)
    engineered_all = raw_all + ENGINEERED_FEATURES

    experiments = [
        ("Experiment 1 - current 11 raw features", CURRENT_RAW_FEATURES, False, False),
        ("Experiment 2 - raw plus academic features", raw_all + ENGINEERED_FEATURES, False, False),
        ("Experiment 3 - tuned raw plus academic features", raw_all + ENGINEERED_FEATURES, False, True),
    ]
    results = []
    fitted_models = {}
    model_sets = {}
    for name, columns, use_smote, tuned in experiments:
        train_data = X_train_engineered if columns != CURRENT_RAW_FEATURES else X_train
        test_data = X_test_engineered if columns != CURRENT_RAW_FEATURES else X_test
        model_sets[name] = (columns, use_smote, tuned)
        for model_name, (model, parameters) in make_models(tuned).items():
            metrics, fitted = fit_and_evaluate(
                name, train_data, test_data, y_train, y_test, columns,
                model_name, model, use_smote, parameters
            )
            results.append(metrics)
            fitted_models[(name, model_name)] = fitted

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        ["CV Dropout F1", "CV F1-macro"], ascending=False
    ).reset_index(drop=True)

    # Preserve the previously measured baseline as a labelled reference, not as a new test result.
    baseline_path = MODEL_DIR / "model_results.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        xgb_rows = baseline[baseline["Model"].astype(str).str.casefold() == "xgboost"]
        if not xgb_rows.empty:
            row = xgb_rows.iloc[0]
            results_df = pd.concat(
                [
                    pd.DataFrame([{
                        "Experiment": "Existing saved baseline reference",
                        "Model": "XGBoost",
                        "Accuracy": row.get("Accuracy"),
                        "Precision": row.get("Precision"),
                        "Recall": row.get("Recall"),
                        "F1-score": row.get("F1-score"),
                        "Macro F1": row.get("F1-score"),
                        "Dropout Precision": row.get("Dropout Precision"),
                        "Dropout Recall": row.get("Dropout Recall"),
                        "Dropout F1-score": row.get("Dropout F1-score"),
                        "Weighted F1": row.get("Weighted F1"),
                        "ROC-AUC": row.get("ROC-AUC"),
                        "Feature set": "Existing 11 raw features",
                        "Features": json.dumps(CURRENT_RAW_FEATURES),
                        "Class order": json.dumps(target_classes),
                        "SMOTE": False,
                        "CV Dropout F1": row.get("CV Dropout F1"),
                        "CV F1-macro": row.get("CV F1-macro"),
                        "Best hyperparameters": row.get("Best Parameters"),
                        "Confusion matrix": "Previously saved in evaluation_details.pkl",
                    }]),
                    results_df,
                ],
                ignore_index=True,
            )
    results_df.to_csv(MODEL_DIR / "improved_experiment_results.csv", index=False)

    candidate_rows = results_df[
        results_df["Experiment"] != "Existing saved baseline reference"
    ].sort_values(["CV Dropout F1", "CV F1-macro"], ascending=False)
    best_row = candidate_rows.iloc[0].copy()
    runner_up_row = candidate_rows.iloc[1]
    best_key = (best_row["Experiment"], best_row["Model"])
    best_fitted_pipeline = fitted_models[best_key]
    best_columns, best_use_smote, _ = model_sets.get(
        best_row["Experiment"], (raw_all + ENGINEERED_FEATURES, False, False)
    )
    final_pipeline = make_inference_pipeline(
        best_columns,
        best_fitted_pipeline.named_steps["model"],
        best_use_smote,
    )
    final_pipeline.fit(X_train[raw_all], y_train)

    final_predictions = final_pipeline.predict(X_test[raw_all])
    final_probabilities = final_pipeline.predict_proba(X_test[raw_all])
    final_metrics = {
        "Accuracy": accuracy_score(y_test, final_predictions),
        "Precision": precision_score(y_test, final_predictions, average="macro", zero_division=0),
        "Recall": recall_score(y_test, final_predictions, average="macro", zero_division=0),
        "F1-score": f1_score(y_test, final_predictions, average="macro", zero_division=0),
        "Macro F1": f1_score(y_test, final_predictions, average="macro", zero_division=0),
        "Dropout Precision": precision_score(
            y_test,
            final_predictions,
            labels=[target_classes.index("Dropout")],
            average="macro",
            zero_division=0,
        ),
        "Dropout Recall": recall_score(
            y_test,
            final_predictions,
            labels=[target_classes.index("Dropout")],
            average="macro",
            zero_division=0,
        ),
        "Dropout F1-score": f1_score(
            y_test,
            final_predictions,
            labels=[target_classes.index("Dropout")],
            average="macro",
            zero_division=0,
        ),
        "Weighted F1": f1_score(y_test, final_predictions, average="weighted", zero_division=0),
        "ROC-AUC": roc_auc_score(
            y_test, final_probabilities, multi_class="ovr", labels=np.arange(len(target_classes))
        ),
        "Features": json.dumps(list(best_columns)),
        "Class order": json.dumps(target_classes),
        "Confusion matrix": json.dumps(
            confusion_matrix(y_test, final_predictions, labels=np.arange(len(target_classes))).tolist()
        ),
    }
    for metric_name, metric_value in final_metrics.items():
        best_row[metric_name] = metric_value
        results_df.loc[
            (results_df["Experiment"] == best_row["Experiment"])
            & (results_df["Model"] == best_row["Model"]),
            metric_name,
        ] = metric_value
    results_df = results_df.sort_values(
        ["CV Dropout F1", "CV F1-macro"], ascending=False
    ).reset_index(drop=True)

    joblib.dump(final_pipeline, MODEL_DIR / "best_model.pkl")
    fitted_preprocessor = Pipeline(
        [
            ("academic_features", final_pipeline.named_steps["academic_features"]),
            ("preprocessor", final_pipeline.named_steps["preprocessor"]),
        ]
    )
    joblib.dump(fitted_preprocessor, MODEL_DIR / "preprocessor.pkl")
    for (experiment, model_name), fitted in fitted_models.items():
        safe_name = model_name.lower().replace(" ", "_")
        joblib.dump(fitted, MODEL_DIR / f"improved_{safe_name}.pkl")

    feature_metadata = {
        "columns": raw_all,
        "model_feature_columns": list(best_columns),
        "raw_feature_names": raw_all,
        "current_raw_features": CURRENT_RAW_FEATURES,
        "engineered_feature_names": ENGINEERED_FEATURES,
        "selected_feature_names": list(best_columns),
        "selected_raw_features": [feature for feature in best_columns if feature in raw_all],
        "selected_engineered_features": [
            feature for feature in best_columns if feature in ENGINEERED_FEATURES
        ],
        "numeric_columns": raw_all + ENGINEERED_FEATURES,
        "categorical_columns": [],
        "target_column": "Target",
        "target_classes": target_classes,
        "preprocessing": {
            "numeric_imputation": "median",
            "numeric_scaling": "StandardScaler",
            "smote": "training data only",
            "feature_engineering": "AcademicFeatureEngineer inside persisted pipeline",
        },
        "model_name": best_row["Model"],
        "best_experiment": best_row["Experiment"],
        "metrics": best_row.to_dict(),
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "cv_splits": CV_SPLITS},
        "formulas": {
            "Semester 1 pass rate": "semester_1_approved / semester_1_evaluations; zero denominator -> 0",
            "Semester 2 pass rate": "semester_2_approved / semester_2_evaluations; zero denominator -> 0",
            "Total approved units": "semester_1_approved + semester_2_approved",
            "Total failed units": "(semester_1_enrolled - approved_1) + (semester_2_enrolled - approved_2), clipped at 0",
            "Credit completion rate": "(approved_1 + approved_2) / (enrolled_1 + enrolled_2); zero denominator -> 0",
            "Average semester grade": "(semester_1_grade + semester_2_grade) / 2",
            "Grade improvement": "semester_2_grade - semester_1_grade",
            "Academic progress": "semester_2_pass_rate - semester_1_pass_rate",
        },
    }
    joblib.dump(feature_metadata, MODEL_DIR / "feature_columns.pkl")
    joblib.dump(feature_metadata["preprocessing"], MODEL_DIR / "preprocessing_metadata.pkl")
    results_df.to_csv(MODEL_DIR / "model_results.csv", index=False)
    results_df.to_csv(MODEL_DIR / "model_results_improved.csv", index=False)

    print(results_df[["Experiment", "Model", "Accuracy", "Dropout F1-score", "CV Dropout F1", "CV F1-macro", "Macro F1"]].to_string(index=False))
    print("\nFINAL SELECTED MODEL")
    print("Model:", best_row["Model"])
    print(f"Accuracy: {best_row['Accuracy']:.6f}")
    print(f"Macro F1: {best_row['Macro F1']:.6f}")
    print(f"Weighted F1: {best_row['Weighted F1']:.6f}")
    print(f"Precision: {best_row['Precision']:.6f}")
    print(f"Recall: {best_row['Recall']:.6f}")
    print(f"Dropout precision: {best_row['Dropout Precision']:.6f}")
    print(f"Dropout recall: {best_row['Dropout Recall']:.6f}")
    print(f"Dropout F1-score: {best_row['Dropout F1-score']:.6f}")
    print(f"ROC-AUC: {best_row['ROC-AUC']:.6f}")
    print("Features:", json.dumps(best_columns))
    print("Confusion matrix (rows/columns: " + ", ".join(target_classes) + "):")
    print(best_row["Confusion matrix"])
    print(
        f"Runner-up: {runner_up_row['Model']} ({runner_up_row['Experiment']}) "
        f"with CV Dropout F1 {runner_up_row['CV Dropout F1']:.6f}, "
        f"holdout Dropout F1 {runner_up_row['Dropout F1-score']:.6f}, "
        f"and ROC-AUC {runner_up_row['ROC-AUC']:.6f}."
    )
    print(
        "Selection reason: highest training-only stratified-CV Dropout F1; "
        "the untouched test split is used only for final reporting."
    )


def main():
    from ml_training.candidate_training import train_candidate_bundle

    output_path, summary = train_candidate_bundle()
    print(f"Saved validated, non-promoted model candidates to: {output_path}")
    print(summary.to_string(index=False))
    print(
        "No candidate was selected or connected to Flask. Existing application "
        "artifacts were left unchanged."
    )


if __name__ == "__main__":
    main()
