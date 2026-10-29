"""Reusable, persisted academic feature engineering for model inference."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
    """Add semester progress features using only columns present in the UCI data."""

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
