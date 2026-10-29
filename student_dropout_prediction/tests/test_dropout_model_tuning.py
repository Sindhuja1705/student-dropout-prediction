import unittest

from sklearn.feature_selection import f_classif
from ml_training.dropout_model_tuning import (
    get_after_semester1_features,
    make_pipeline,
    make_scoring,
)
from sklearn.linear_model import LogisticRegression


class DropoutModelTuningTest(unittest.TestCase):
    def test_scoring_includes_dropout_and_macro_metrics(self):
        scoring = make_scoring(dropout_label=0)

        self.assertEqual(
            set(scoring),
            {"dropout_precision", "dropout_recall", "dropout_f1", "macro_f1"},
        )

    def test_feature_selection_is_inside_the_training_pipeline(self):
        pipeline = make_pipeline(
            LogisticRegression(class_weight="balanced"),
            ["Gender", "Age at enrollment"],
        )

        self.assertIn("feature_selection", pipeline.named_steps)
        self.assertIn("sampler", pipeline.named_steps)
        self.assertEqual(pipeline.named_steps["feature_selection"].k, "all")
        self.assertIs(
            pipeline.named_steps["feature_selection"].score_func, f_classif
        )

    def test_after_semester1_features_exclude_late_fields_and_target(self):
        columns = [
            "Gender",
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (approved)",
            "Tuition fees up to date",
            "Target",
        ]

        features = get_after_semester1_features(columns)

        self.assertEqual(
            features,
            ["Gender", "Curricular units 1st sem (approved)"],
        )


if __name__ == "__main__":
    unittest.main()
