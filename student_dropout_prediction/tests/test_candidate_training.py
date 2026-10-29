import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from ml_training.candidate_training import EXPECTED_MODELS, train_candidate_bundle


class CandidateTrainingTest(unittest.TestCase):
    def test_candidate_bundle_is_cutoff_safe_validated_and_never_promoted(self):
        rows = 45
        labels = np.repeat(["Dropout", "Enrolled", "Graduate"], rows // 3)
        frame = pd.DataFrame(
            {
                "student_id": np.arange(rows),
                "Gender": np.arange(rows) % 2,
                "Age at enrollment": np.arange(rows) % 20 + 18,
                "Curricular units 1st sem (approved)": np.arange(rows) % 8,
                "Curricular units 2nd sem (approved)": np.arange(rows) % 9,
                "Target": labels,
            }
        )
        model_specs = {
            name: (
                DummyClassifier(),
                {"model__strategy": ["prior"]},
                False,
            )
            for name in EXPECTED_MODELS
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_parent = Path(temporary_directory)
            sentinel = output_parent / "existing-artifact.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            bundle, summary = train_candidate_bundle(
                frame=frame,
                output_parent=output_parent,
                model_specs=model_specs,
                n_splits=3,
                search_iterations=1,
            )

            metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(metadata["status"], "candidate_only_not_promoted")
            self.assertFalse(metadata["production_ready"])
            self.assertIsNone(metadata["selected_model"])
            self.assertEqual(metadata["selection_metric_within_each_model"], "Accuracy")
            self.assertTrue(metadata["student_overlap_checked"])
            self.assertEqual(metadata["excluded_student_id_column"], "student_id")
            self.assertNotIn("student_id", metadata["feature_columns"])
            self.assertNotIn(
                "Curricular units 2nd sem (approved)", metadata["feature_columns"]
            )
            self.assertEqual(set(summary["Model"]), EXPECTED_MODELS)
            self.assertTrue(
                {
                    "CV Accuracy Mean",
                    "CV Accuracy SD",
                    "CV Dropout F1 Mean",
                    "CV Macro F1 Mean",
                }.issubset(summary.columns)
            )
            self.assertEqual(len(list(bundle.glob("*_candidate.joblib"))), 5)

            for artifact in bundle.glob("*_candidate.joblib"):
                model = joblib.load(artifact)
                probabilities = model.predict_proba(frame[metadata["feature_columns"]].iloc[:1])
                self.assertEqual(probabilities.shape, (1, 3))
                self.assertTrue(np.isfinite(probabilities).all())

    def test_candidate_search_refits_on_accuracy(self):
        labels = np.array(["Dropout"] * 10 + ["Enrolled"] * 45 + ["Graduate"] * 45)
        frame = pd.DataFrame(
            {
                "Gender": np.arange(len(labels)) % 2,
                "Age at enrollment": np.arange(len(labels)) % 20 + 18,
                "Curricular units 1st sem (approved)": np.arange(len(labels)) % 8,
                "Target": labels,
            }
        )
        model_specs = {
            name: (
                DummyClassifier(random_state=42),
                {"model__strategy": ["most_frequent", "uniform"]},
                False,
            )
            for name in EXPECTED_MODELS
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            _, summary = train_candidate_bundle(
                frame=frame,
                output_parent=Path(temporary_directory),
                model_specs=model_specs,
                n_splits=3,
                search_iterations=2,
            )

        self.assertTrue(
            summary["Selected parameters"]
            .map(lambda value: json.loads(value)["model__strategy"] == "most_frequent")
            .all()
        )


if __name__ == "__main__":
    unittest.main()
