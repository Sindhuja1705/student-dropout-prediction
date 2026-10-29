"""Train non-promoted, after-semester-1 model candidates safely."""

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from ml_training.dropout_model_tuning import TARGET_COLUMN, get_after_semester1_features
from ml_training.repeated_nested_cv import (
    INNER_SPLITS,
    RANDOM_STATE,
    find_student_id_column,
    make_model_specs,
    make_pipeline,
    make_inner_splits,
)
from train_models import MODEL_DIR, load_dataset


EXPECTED_MODELS = {
    "Random Forest",
    "XGBoost",
    "Logistic Regression",
    "Decision Tree",
    "SVM",
}


def _validated_feature_columns(frame):
    student_id_column = find_student_id_column(frame.columns)
    feature_columns = get_after_semester1_features(frame.columns)
    if student_id_column in feature_columns:
        feature_columns.remove(student_id_column)
    if not feature_columns:
        raise ValueError("Dataset contains no after-semester-1 eligible features.")
    return feature_columns, student_id_column


def _new_bundle_path(output_parent):
    output_parent = Path(output_parent)
    output_parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    while True:
        candidate = output_parent / f"after_semester1_{timestamp}_{uuid.uuid4().hex[:8]}"
        if not candidate.exists():
            return candidate


def train_candidate_bundle(
    frame=None,
    output_parent=None,
    model_specs=None,
    n_splits=INNER_SPLITS,
    search_iterations=20,
):
    """Fit each model family and write a validated, non-production candidate bundle.

    This deliberately does not select a winner or modify the application's
    existing model artifacts.
    """
    frame = load_dataset() if frame is None else frame.copy()
    if TARGET_COLUMN not in frame:
        raise ValueError(f"Dataset must contain target column {TARGET_COLUMN!r}.")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("Target labels must not be missing.")
    if frame[TARGET_COLUMN].astype(str).str.strip().eq("").any():
        raise ValueError("Target labels must not be blank.")

    feature_columns, student_id_column = _validated_feature_columns(frame)
    target = frame[TARGET_COLUMN].astype(str).str.strip()
    classes = sorted(target.unique())
    if len(classes) != 3 or "Dropout" not in classes:
        raise ValueError("Target must contain the three expected classes, including Dropout.")
    if n_splits < 2 or search_iterations < 1:
        raise ValueError("Training requires at least 2 CV folds and 1 search iteration.")
    class_to_integer = {label: index for index, label in enumerate(classes)}
    y = target.map(class_to_integer).to_numpy()
    X = frame[feature_columns].copy()
    groups = None if student_id_column is None else frame[student_id_column].to_numpy()
    if groups is not None and pd.isna(groups).any():
        raise ValueError("Student identifiers must not be missing.")

    splitter = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        if groups is None
        else StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
        )
    )
    cv_splits = make_inner_splits(splitter, X, y, groups=groups)
    specs = make_model_specs(len(classes)) if model_specs is None else model_specs
    if set(specs) != EXPECTED_MODELS:
        raise ValueError(f"Model set must be exactly {sorted(EXPECTED_MODELS)}.")

    dropout_label = class_to_integer["Dropout"]
    scoring = {
        "accuracy": "accuracy",
        "dropout_f1": make_scorer(
            f1_score,
            labels=[dropout_label],
            average="macro",
            zero_division=0,
        ),
        "macro_f1": "f1_macro",
    }
    searches = {}
    summary_rows = []
    for model_name, (estimator, parameters, balance) in specs.items():
        search = RandomizedSearchCV(
            make_pipeline(estimator, feature_columns, balance=balance),
            parameters,
            n_iter=search_iterations,
            scoring=scoring,
            refit="accuracy",
            cv=cv_splits,
            random_state=RANDOM_STATE,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(X, y, groups=groups)
        searches[model_name] = search
        summary_rows.append(
            {
                "Model": model_name,
                "CV Accuracy Mean": float(search.best_score_),
                "CV Accuracy SD": float(
                    search.cv_results_["std_test_accuracy"][search.best_index_]
                ),
                "CV Dropout F1 Mean": float(
                    search.cv_results_["mean_test_dropout_f1"][search.best_index_]
                ),
                "CV Dropout F1 SD": float(
                    search.cv_results_["std_test_dropout_f1"][search.best_index_]
                ),
                "CV Macro F1 Mean": float(search.cv_results_["mean_test_macro_f1"][search.best_index_]),
                "Selected parameters": json.dumps(
                    search.best_params_, sort_keys=True, default=str
                ),
            }
        )

    output_parent = (
        Path(output_parent)
        if output_parent is not None
        else MODEL_DIR / "candidates"
    )
    final_path = _new_bundle_path(output_parent)
    staging_path = Path(tempfile.mkdtemp(prefix=".candidate-staging-", dir=output_parent))
    try:
        for model_name, search in searches.items():
            safe_name = model_name.lower().replace(" ", "_")
            artifact_path = staging_path / f"{safe_name}_candidate.joblib"
            joblib.dump(search.best_estimator_, artifact_path)
            loaded = joblib.load(artifact_path)
            predictions = loaded.predict(X.iloc[:1])
            if len(predictions) != 1 or int(predictions[0]) not in range(len(classes)):
                raise RuntimeError(f"Saved {model_name} candidate failed prediction validation.")
            if hasattr(loaded, "predict_proba"):
                validation_scores = loaded.predict_proba(X.iloc[:1])
            else:
                validation_scores = loaded.decision_function(X.iloc[:1])
            if not np.isfinite(validation_scores).all():
                raise RuntimeError(f"Saved {model_name} candidate returned invalid scores.")

        summary = pd.DataFrame(summary_rows)
        summary.to_csv(staging_path / "candidate_summary.csv", index=False)
        metadata = {
            "status": "candidate_only_not_promoted",
            "production_ready": False,
            "selected_model": None,
            "reason_not_promoted": (
                "This workflow tunes and validates candidates but does not rank "
                "or promote a model for production."
            ),
            "dataset_rows": len(frame),
            "target_column": TARGET_COLUMN,
            "classes": classes,
            "class_to_integer": class_to_integer,
            "feature_columns": feature_columns,
            "excluded_student_id_column": student_id_column,
            "student_overlap_checked": student_id_column is not None,
            "cv_splits": n_splits,
            "search_iterations": search_iterations,
            "random_state": RANDOM_STATE,
            "selection_metric_within_each_model": "Accuracy",
            "application_compatible": False,
            "application_compatibility_note": (
                "The current Flask form and persisted production metadata do not "
                "supply this 25-feature after-semester-1 contract."
            ),
        }
        (staging_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        os.replace(staging_path, final_path)
    except Exception:
        if staging_path.exists():
            shutil.rmtree(staging_path)
        raise

    return final_path, summary
