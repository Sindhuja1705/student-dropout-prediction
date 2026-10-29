import warnings
import unittest

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from ml_training.repeated_nested_cv import (
    METRIC_COLUMNS,
    find_student_id_column,
    make_model_specs,
    make_inner_splits,
    make_outer_splits,
    make_pipeline,
    run_nested_cv,
    summarize_outer_fold_results,
)
from ml_training.dropout_model_tuning import get_after_semester1_features


class RepeatedNestedCVTest(unittest.TestCase):
    def test_after_semester1_features_exclude_target_late_fields_and_id(self):
        columns = [
            "student_id",
            "Gender",
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (approved)",
            "Tuition fees up to date",
            "Target",
        ]
        features = get_after_semester1_features(columns)
        student_id = find_student_id_column(columns)

        self.assertEqual(
            [column for column in features if column != student_id],
            ["Gender", "Curricular units 1st sem (approved)"],
        )
        self.assertEqual(student_id, "student_id")

    def test_all_five_models_have_search_spaces_and_pipeline_stages(self):
        specs = make_model_specs(class_count=3)

        self.assertEqual(
            set(specs),
            {
                "Random Forest",
                "XGBoost",
                "Logistic Regression",
                "Decision Tree",
                "SVM",
            },
        )
        pipeline = make_pipeline(
            LogisticRegression(),
            ["Gender", "Age at enrollment"],
        )
        self.assertEqual(
            list(pipeline.named_steps),
            ["preprocessor", "sampler", "feature_selection", "model"],
        )
        self.assertIsInstance(
            make_pipeline(
                LogisticRegression(),
                ["Age at enrollment"],
                balance=True,
            ).named_steps["sampler"],
            RandomOverSampler,
        )
        self.assertTrue(all(parameters for _, parameters, _ in specs.values()))
        self.assertTrue(specs["XGBoost"][2])
        self.assertTrue(
            all(
                "model__class_weight" in parameters
                for name, (_, parameters, _) in specs.items()
                if name != "XGBoost"
            )
        )

    def test_logistic_regression_solver_fits_without_iprint_warning(self):
        estimator = make_model_specs(class_count=3)["Logistic Regression"][0]
        X = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.1],
                [0.4, 0.2],
                [0.6, 0.8],
                [0.8, 0.9],
                [1.0, 1.0],
                [1.2, 0.3],
                [1.4, 0.4],
                [1.6, 0.6],
            ]
        )
        y = np.repeat([0, 1, 2], 3)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            estimator.fit(X, y)

        self.assertEqual(estimator.solver, "newton-cg")
        self.assertFalse(
            any("Unknown solver options: iprint" in str(item.message) for item in captured)
        )

    def test_outer_splits_are_reproducible_stratified_and_group_disjoint(self):
        y = np.repeat([0, 1, 2], 12)
        groups = np.arange(len(y)) // 2

        first = make_outer_splits(y, groups, n_splits=3, n_repeats=2)
        second = make_outer_splits(y, groups, n_splits=3, n_repeats=2)

        self.assertEqual(len(first), 6)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left["train"], right["train"])
            np.testing.assert_array_equal(left["test"], right["test"])
            self.assertFalse(
                set(groups[left["train"]]).intersection(groups[left["test"]])
            )
            self.assertEqual(
                set(np.unique(y[left["test"]])),
                {0, 1, 2},
            )

    def test_inner_group_folds_keep_student_groups_separate(self):
        y = np.repeat([0, 1, 2], 12)
        groups = np.arange(len(y)) // 2
        splitter = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=42,
        )

        splits = make_inner_splits(splitter, np.zeros((len(y), 1)), y, groups)

        self.assertEqual(len(splits), 3)
        for train_indices, validation_indices in splits:
            self.assertFalse(
                set(groups[train_indices]).intersection(groups[validation_indices])
            )

    def test_summary_reports_fold_sd_and_corrected_confidence_intervals(self):
        rows = []
        values_by_model = {
            "Model A": [0.60, 0.70, 0.80],
            "Model B": [0.55, 0.65, 0.75],
        }
        for model, values in values_by_model.items():
            for fold, value in enumerate(values, start=1):
                row = {"Model": model, "Repeat": 1, "Fold": fold}
                row.update({metric: value for metric in METRIC_COLUMNS})
                rows.append(row)

        summary = summarize_outer_fold_results(
            pd.DataFrame(rows),
            n_splits=3,
        )

        self.assertEqual(len(summary), 2)
        self.assertAlmostEqual(summary.iloc[0]["Accuracy Mean"], 0.7)
        self.assertAlmostEqual(
            summary.iloc[0]["Accuracy SD"],
            np.std([0.6, 0.7, 0.8], ddof=1),
        )
        self.assertLess(
            summary.iloc[0]["Accuracy 95% CI Low"],
            summary.iloc[0]["Accuracy Mean"],
        )
        self.assertGreater(
            summary.iloc[0]["Accuracy 95% CI High"],
            summary.iloc[0]["Accuracy Mean"],
        )

    def test_nested_evaluation_uses_grouped_splits_and_returns_all_metrics(self):
        targets = np.repeat(["Dropout", "Enrolled", "Graduate"], 12)
        frame = pd.DataFrame(
            {
                "student_id": np.arange(len(targets)) // 2,
                "Gender": np.arange(len(targets)) % 2,
                "Age at enrollment": np.arange(len(targets)) % 25 + 18,
                "Curricular units 1st sem (approved)": np.arange(len(targets)) % 6,
                "Target": targets,
            }
        )
        specs = {
            name: (
                DummyClassifier(),
                {"model__strategy": ["prior"]},
                False,
            )
            for name in (
                "Random Forest",
                "XGBoost",
                "Logistic Regression",
                "Decision Tree",
                "SVM",
            )
        }

        folds, summary, metadata = run_nested_cv(
            frame,
            model_specs=specs,
            n_splits=3,
            n_repeats=1,
            inner_splits=2,
            search_iterations=1,
        )

        self.assertEqual(len(folds), 15)
        self.assertEqual(len(summary), 5)
        self.assertEqual(metadata["student_id_column"], "student_id")
        self.assertTrue(metadata["student_overlap_checked"])
        self.assertTrue(set(METRIC_COLUMNS).issubset(folds.columns))
        self.assertNotIn("student_id", metadata["feature_columns"])


if __name__ == "__main__":
    unittest.main()
