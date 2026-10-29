import unittest

from ml_training.after_semester1_model_comparison import (
    PRE_ENROLLMENT_FEATURES,
    SEMESTER_1_FEATURES,
    get_after_semester1_features,
    make_preprocessor,
)


class AfterSemester1ModelComparisonTest(unittest.TestCase):
    def test_eligible_features_exclude_target_and_second_semester(self):
        columns = (
            PRE_ENROLLMENT_FEATURES
            + SEMESTER_1_FEATURES
            + [
                "Curricular units 2nd sem (approved)",
                "Target",
            ]
        )

        features = get_after_semester1_features(columns)

        self.assertEqual(features, PRE_ENROLLMENT_FEATURES + SEMESTER_1_FEATURES)
        self.assertNotIn("Target", features)
        self.assertFalse(
            any(column.startswith("Curricular units 2nd sem") for column in features)
        )

    def test_missing_eligible_feature_is_an_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "eligible features"):
            get_after_semester1_features(["Target"])

    def test_preprocessor_has_numeric_and_categorical_transformers(self):
        features = PRE_ENROLLMENT_FEATURES + SEMESTER_1_FEATURES

        preprocessor = make_preprocessor(features)

        self.assertEqual(
            {name for name, _, _ in preprocessor.transformers},
            {"numeric", "categorical"},
        )


if __name__ == "__main__":
    unittest.main()
