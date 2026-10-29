import unittest
from unittest.mock import patch

import app


class PredictionDefaultsTest(unittest.TestCase):
    def test_prediction_loads_the_final_saved_pipeline(self):
        with patch.object(app, '_safe_load', return_value='saved-pipeline') as load_artifact:
            self.assertEqual(app._load_best_pipeline(), 'saved-pipeline')

        load_artifact.assert_called_once_with(app.MODEL_FILES['best_model'])

    def test_student_login_matches_faculty_roster_name_and_roll_number(self):
        student = app.get_student('23EG106A68')
        self.assertIsNotNone(student)

        matched = app.get_student_by_credentials(' 23eg106a68 ', '  RAGASRI  ')
        self.assertEqual(matched['id'], student['id'])
        self.assertIsNone(app.get_student_by_credentials('23EG106A68', 'Different Student'))

    def test_missing_model_fields_are_filled_from_dataset_defaults(self):
        student = app.get_student('23EG106A68')
        self.assertIsNotNone(student)

        values, missing = app._candidate_features_from_student(
            student,
            {
                'Gender': 0,
                'Age at enrollment': 18,
                'Course': 171,
                'Previous qualification (grade)': 8.7,
                'Admission grade': 8.0,
                'Tuition fees up to date': 1,
                'Scholarship holder': 0,
                'Curricular units 1st sem (approved)': 20,
                'Curricular units 1st sem (grade)': 8.7,
                'Curricular units 2nd sem (approved)': 18,
                'Curricular units 2nd sem (grade)': 8.5,
            },
        )

        self.assertEqual(missing, [])
        self.assertIsInstance(values, dict)
        self.assertIn('Marital status', values)
        self.assertIn('Curricular units 2nd sem (grade)', values)
        self.assertTrue(isinstance(values['Marital status'], float))


if __name__ == '__main__':
    unittest.main()
