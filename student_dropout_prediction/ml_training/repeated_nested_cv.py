"""Repeated nested cross-validation for the after-semester-1 dropout task."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform, randint, t
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ml_training.dropout_model_tuning import (
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    get_after_semester1_features,
)
from train_models import RANDOM_STATE, load_dataset


OUTER_SPLITS = 5
OUTER_REPEATS = 3
INNER_SPLITS = 3
SEARCH_ITERATIONS = 4
STUDENT_ID_NAMES = {
    "studentid",
    "studentidnumber",
    "studentidentifier",
    "idstudent",
    "idstudentnumber",
    "studentnumber",
    "studentcode",
}
METRIC_COLUMNS = [
    "Accuracy",
    "Dropout Precision",
    "Dropout Recall",
    "Dropout F1",
    "Macro F1",
]


def find_student_id_column(columns):
    for column in columns:
        normalized = "".join(
            character for character in str(column).casefold() if character.isalnum()
        )
        if normalized in STUDENT_ID_NAMES:
            return column
    return None


def make_pipeline(estimator, feature_columns, balance=False):
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
    sampler = (
        RandomOverSampler(random_state=RANDOM_STATE) if balance else "passthrough"
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("sampler", sampler),
            ("feature_selection", SelectPercentile(score_func=f_classif)),
            ("model", estimator),
        ]
    )


def make_model_specs(class_count):
    specs = {
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": randint(100, 221),
                "model__max_depth": [None, 10, 16, 24],
                "model__min_samples_leaf": randint(1, 5),
                "model__max_features": ["sqrt", 0.8],
                "model__class_weight": [None, "balanced"],
                "feature_selection__percentile": [50, 75, 100],
            },
            False,
        ),
        "XGBoost": (
            XGBClassifier(
                objective="multi:softprob",
                num_class=class_count,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
                tree_method="hist",
            ),
            {
                "model__n_estimators": randint(80, 181),
                "model__max_depth": randint(2, 6),
                "model__learning_rate": loguniform(0.03, 0.2),
                "model__subsample": [0.75, 0.9, 1.0],
                "model__colsample_bytree": [0.75, 0.9, 1.0],
                "feature_selection__percentile": [50, 75, 100],
            },
            True,
        ),
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2500,
                random_state=RANDOM_STATE,
                solver="newton-cg",
            ),
            {
                "model__C": loguniform(0.05, 20),
                "model__class_weight": [None, "balanced"],
                "feature_selection__percentile": [50, 75, 100],
            },
            False,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "model__max_depth": [None, 8, 16, 24],
                "model__min_samples_leaf": randint(1, 6),
                "model__class_weight": [None, "balanced"],
                "feature_selection__percentile": [50, 75, 100],
            },
            False,
        ),
        "SVM": (
            SVC(random_state=RANDOM_STATE),
            {
                "model__C": loguniform(0.1, 20),
                "model__gamma": ["scale", "auto"],
                "model__class_weight": [None, "balanced"],
                "feature_selection__percentile": [50, 75, 100],
            },
            False,
        ),
    }
    return specs


def make_outer_splits(y, groups=None, n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS):
    y = np.asarray(y)
    if n_splits < 2 or n_repeats < 1:
        raise ValueError("Outer CV requires at least 2 folds and 1 repeat.")
    group_values = None if groups is None else np.asarray(groups)
    if group_values is not None and len(group_values) != len(y):
        raise ValueError("Student groups must have the same length as the target.")
    splits = []
    for repeat in range(n_repeats):
        if group_values is None:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=RANDOM_STATE + repeat,
            )
            iterator = splitter.split(np.zeros(len(y)), y)
        else:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=RANDOM_STATE + repeat,
            )
            iterator = splitter.split(np.zeros(len(y)), y, group_values)
        for fold, (train_indices, test_indices) in enumerate(iterator, start=1):
            if group_values is not None:
                train_groups = set(group_values[train_indices])
                test_groups = set(group_values[test_indices])
                overlap = train_groups.intersection(test_groups)
                if overlap:
                    raise RuntimeError(
                        f"Student/group overlap in repeat {repeat + 1}, fold {fold}."
                    )
            splits.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "train": train_indices,
                    "test": test_indices,
                }
            )
    return splits


def make_inner_splits(splitter, X, y, groups=None):
    if groups is None:
        return list(splitter.split(X, y))
    groups = np.asarray(groups)
    if pd.isna(groups).any():
        raise ValueError("Student/group identifiers must not be missing.")
    splits = list(splitter.split(X, y, groups))
    for train_indices, validation_indices in splits:
        overlap = set(groups[train_indices]).intersection(groups[validation_indices])
        if overlap:
            raise RuntimeError("Student/group overlap found in an inner CV split.")
    return splits


def summarize_outer_fold_results(fold_results, n_splits=OUTER_SPLITS):
    summaries = []
    split_count = len(fold_results)
    if split_count < 2:
        raise ValueError("At least two outer-fold results are required.")
    correction = (1 / split_count) + (1 / (n_splits - 1))
    critical_value = t.ppf(0.975, df=split_count - 1)
    for model, rows in fold_results.groupby("Model", sort=False):
        row = {"Model": model}
        for metric in METRIC_COLUMNS:
            values = rows[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            standard_deviation = float(np.std(values, ddof=1))
            corrected_se = float(np.sqrt(correction * standard_deviation**2))
            row[f"{metric} Mean"] = mean
            row[f"{metric} SD"] = standard_deviation
            row[f"{metric} 95% CI Low"] = mean - critical_value * corrected_se
            row[f"{metric} 95% CI High"] = mean + critical_value * corrected_se
        summaries.append(row)
    return pd.DataFrame(summaries)


def run_nested_cv(
    frame,
    model_specs=None,
    n_splits=OUTER_SPLITS,
    n_repeats=OUTER_REPEATS,
    inner_splits=INNER_SPLITS,
    search_iterations=SEARCH_ITERATIONS,
):
    student_id_column = find_student_id_column(frame.columns)
    feature_columns = get_after_semester1_features(frame.columns)
    if student_id_column in feature_columns:
        feature_columns.remove(student_id_column)
    if not feature_columns:
        raise ValueError("Dataset contains no after-semester-1 eligible features.")

    target = frame[TARGET_COLUMN].astype(str).str.strip()
    classes = sorted(target.unique())
    if "Dropout" not in classes or len(classes) != 3:
        raise ValueError("Target must contain the three expected classes, including Dropout.")
    class_to_integer = {label: index for index, label in enumerate(classes)}
    y = target.map(class_to_integer).to_numpy()
    groups = (
        frame[student_id_column].to_numpy()
        if student_id_column is not None
        else None
    )
    if groups is not None and pd.isna(groups).any():
        raise ValueError("Student identifiers must not be missing.")
    X = frame[feature_columns].copy()
    specs = model_specs or make_model_specs(len(classes))
    expected_models = {
        "Random Forest",
        "XGBoost",
        "Logistic Regression",
        "Decision Tree",
        "SVM",
    }
    if set(specs) != expected_models:
        raise ValueError(f"Model set must be exactly {sorted(expected_models)}.")

    dropout_label = class_to_integer["Dropout"]
    inner_scorers = {
        "accuracy": "accuracy",
        "dropout_f1": make_scorer(
            f1_score,
            labels=[dropout_label],
            average="macro",
            zero_division=0,
        ),
    }
    outer_splits = make_outer_splits(y, groups, n_splits, n_repeats)
    fold_results = []
    for split in outer_splits:
        train_indices, test_indices = split["train"], split["test"]
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        train_groups = None if groups is None else groups[train_indices]
        inner_splitter = (
            StratifiedKFold(
                n_splits=inner_splits,
                shuffle=True,
                random_state=RANDOM_STATE,
            )
            if train_groups is None
            else StratifiedGroupKFold(
                n_splits=inner_splits,
                shuffle=True,
                random_state=RANDOM_STATE,
            )
        )
        inner_cv = make_inner_splits(
            inner_splitter,
            X_train,
            y_train,
            groups=train_groups,
        )
        for model_name, (estimator, parameters, balance) in specs.items():
            search = RandomizedSearchCV(
                make_pipeline(estimator, feature_columns, balance=balance),
                parameters,
                n_iter=search_iterations,
                scoring=inner_scorers,
                refit="accuracy",
                cv=inner_cv,
                random_state=RANDOM_STATE + split["repeat"] * 100 + split["fold"],
                n_jobs=1,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(X_train, y_train, groups=train_groups)
            predictions = search.predict(X_test)
            fold_results.append(
                {
                    "Model": model_name,
                    "Repeat": split["repeat"],
                    "Fold": split["fold"],
                    "Train rows": len(train_indices),
                    "Test rows": len(test_indices),
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
                    "Dropout F1": f1_score(
                        y_test,
                        predictions,
                        labels=[dropout_label],
                        average="macro",
                        zero_division=0,
                    ),
                    "Macro F1": f1_score(
                        y_test, predictions, average="macro", zero_division=0
                    ),
                    "Inner CV Accuracy": float(search.best_score_),
                    "Inner CV Dropout F1": float(
                        search.cv_results_["mean_test_dropout_f1"][
                            search.best_index_
                        ]
                    ),
                    "Selected parameters": search.best_params_,
                }
            )

    fold_frame = pd.DataFrame(fold_results)
    summary = summarize_outer_fold_results(fold_frame, n_splits=n_splits)
    metadata = {
        "feature_columns": feature_columns,
        "classes": classes,
        "student_id_column": student_id_column,
        "student_overlap_checked": student_id_column is not None,
        "outer_splits": n_splits,
        "outer_repeats": n_repeats,
        "inner_splits": inner_splits,
        "search_iterations": search_iterations,
    }
    return fold_frame, summary, metadata


def main():
    frame = load_dataset()
    fold_results, summary, metadata = run_nested_cv(frame)
    print(
        f"Rows={len(frame)}; eligible features={len(metadata['feature_columns'])}; "
        f"models={summary['Model'].tolist()}"
    )
    print(
        f"Outer CV: {metadata['outer_repeats']} repeats x "
        f"{metadata['outer_splits']} stratified folds; inner CV: "
        f"{metadata['inner_splits']} folds; search iterations: "
        f"{metadata['search_iterations']}; seed={RANDOM_STATE}"
    )
    if metadata["student_id_column"] is None:
        print(
            "No student identifier exists in this dataset. Exact duplicate rows "
            "were removed at load; student identity overlap cannot be verified."
        )
    else:
        print(
            f"Student-level overlap checked with group column "
            f"{metadata['student_id_column']!r}; grouped stratified splits used "
            "for both outer evaluation and inner selection."
        )
    print(
        "\nOuter-fold results (all models use the same split indices; "
        "inner-CV accuracy selects hyperparameters):"
    )
    for _, row in fold_results.iterrows():
        print(
            f"{row['Model']} | repeat={row['Repeat']} fold={row['Fold']} "
            f"accuracy={row['Accuracy']:.4f} "
            f"dropout_p={row['Dropout Precision']:.4f} "
            f"dropout_r={row['Dropout Recall']:.4f} "
            f"dropout_f1={row['Dropout F1']:.4f} "
            f"macro_f1={row['Macro F1']:.4f}"
        )
    print(
        "\nSummary: means and fold SDs; 95% CIs use the corrected repeated-CV "
        "t approximation."
    )
    for _, row in summary.iterrows():
        metrics = []
        for metric in METRIC_COLUMNS:
            metrics.append(
                f"{metric}={row[f'{metric} Mean']:.4f} "
                f"(SD={row[f'{metric} SD']:.4f}; "
                f"95% CI "
                f"[{row[f'{metric} 95% CI Low']:.4f}, "
                f"{row[f'{metric} 95% CI High']:.4f}])"
            )
        print(f"{row['Model']}: " + "; ".join(metrics))
    print("\nNo app, dataset, or model artifact files are written by this script.")


if __name__ == "__main__":
    main()
