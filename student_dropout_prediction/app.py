import os
import sys
import json
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for

try:
    from .database import (
        create_student,
        create_faculty,
        dashboard_summary,
        delete_student,
        get_faculty_by_credentials,
        get_latest_support,
        get_student,
        get_student_by_credentials,
        init_db,
        list_students,
        student_id_exists,
        update_student_prediction,
        update_student_model_features,
        update_support,
    )
except ImportError:
    from database import (
        create_student,
        create_faculty,
        dashboard_summary,
        delete_student,
        get_faculty_by_credentials,
        get_latest_support,
        get_student,
        get_student_by_credentials,
        init_db,
        list_students,
        student_id_exists,
        update_student_prediction,
        update_student_model_features,
        update_support,
    )

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'data.csv')
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .config import BASE_DIR, MISSING_MODELS_MESSAGE, MODEL_FILES
except ImportError:
    from config import BASE_DIR, MISSING_MODELS_MESSAGE, MODEL_FILES

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'student-dropout-demo-key')
init_db()

USER_FIELDS = {
    'Gender': {'label': 'Gender'},
    'Age at enrollment': {'label': 'Age at enrollment'},
    'Course': {'label': 'Course'},
    'Previous qualification (grade)': {'label': 'Previous qualification grade'},
    'Admission grade': {'label': 'Admission grade'},
    'Tuition fees up to date': {'label': 'Tuition fees up to date'},
    'Scholarship holder': {'label': 'Scholarship holder'},
    'Curricular units 1st sem (approved)': {'label': '1st Semester Approved Units'},
    'Curricular units 1st sem (grade)': {'label': '1st Semester Grade'},
    'Curricular units 2nd sem (approved)': {'label': '2nd Semester Approved Units'},
    'Curricular units 2nd sem (grade)': {'label': '2nd Semester Grade'},
}

YEAR_SUFFIXES = {1: 'st', 2: 'nd', 3: 'rd', 4: 'th'}
SEMESTER_LABELS = {
    str(number): f'{(number + 1) // 2}{YEAR_SUFFIXES[(number + 1) // 2]} Year - Semester {number}'
    for number in range(1, 9)
}
RISK_THRESHOLDS = {'high': 66, 'medium': 33}
BTECH_COURSE_OPTIONS = [
    'Artificial Intelligence & Machine Learning (AI/ML)',
    'Computer Science & Engineering (CSE)',
    'CSE - Data Science',
    'Information Technology (IT)',
    'Electronics & Communication Engineering (ECE)',
    'Electrical & Electronics Engineering (EEE)',
    'Mechanical Engineering',
    'Civil Engineering',
]


def _safe_load(path):
    if not path or not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def _read_model_results():
    csv_path = MODEL_FILES['model_results']
    if not os.path.exists(csv_path):
        return []

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return []
        return df.to_dict(orient='records')
    except Exception:
        return []


def _read_final_model_performance():
    feature_info = _safe_load(MODEL_FILES.get('feature_columns'))
    if not isinstance(feature_info, dict):
        return None

    result = feature_info.get('metrics')
    if not isinstance(result, dict):
        return None

    model_name = feature_info.get('model_name') or result.get('Model')
    if not model_name:
        return None

    return {
        'model': str(model_name),
        'accuracy': result.get('Accuracy'),
        'macro_f1': result.get('Macro F1', result.get('F1-score')),
        'weighted_f1': result.get('Weighted F1'),
        'precision': result.get('Precision'),
        'recall': result.get('Recall'),
        'roc_auc': result.get('ROC-AUC'),
    }


def _feature_metadata():
    feature_info = _safe_load(MODEL_FILES['feature_columns'])
    if isinstance(feature_info, dict):
        columns = [str(col).strip() for col in feature_info.get('columns', [])]
        numeric = [str(col).strip() for col in feature_info.get('numeric_columns', [])]
        categorical = [str(col).strip() for col in feature_info.get('categorical_columns', [])]
        categorical_values = {
            str(key).strip(): [str(v).strip() for v in values]
            for key, values in feature_info.get('categorical_values', {}).items()
        }
        return {
            'columns': columns,
            'numeric_columns': numeric,
            'categorical_columns': categorical,
            'categorical_values': categorical_values,
            'target_classes': [str(value).strip() for value in feature_info.get('target_classes', [])],
        }
    if isinstance(feature_info, list):
        return {'columns': feature_info, 'numeric_columns': [], 'categorical_columns': [], 'categorical_values': {}, 'target_classes': []}
    return {'columns': [], 'numeric_columns': [], 'categorical_columns': [], 'categorical_values': {}, 'target_classes': []}


def _feature_label(column):
    return USER_FIELDS.get(column, {}).get('label', column)


def _prediction_fields():
    metadata = _feature_metadata()
    dataset = _load_dataset_for_ui()
    columns = [column for column in USER_FIELDS if column in metadata.get('columns', [])]
    groups = {
        'Personal Information': ['Gender', 'Age at enrollment'],
        'Admission Information': ['Course', 'Previous qualification (grade)', 'Admission grade'],
        'Financial Information': ['Tuition fees up to date', 'Scholarship holder'],
        'Academic Progress': [
            'Curricular units 1st sem (approved)',
            'Curricular units 1st sem (grade)',
            'Curricular units 2nd sem (approved)',
            'Curricular units 2nd sem (grade)',
        ],
    }
    fields = []
    used = set()
    for group, group_columns in groups.items():
        for column in group_columns:
            if column not in columns or column in used:
                continue
            values = []
            if not dataset.empty and column in dataset:
                values = sorted(dataset[column].dropna().unique().tolist())
            if column == 'Gender':
                values = [(0, 'Female'), (1, 'Male')]
            elif column in {'Tuition fees up to date', 'Scholarship holder'}:
                values = [(0, 'No'), (1, 'Yes')]
            elif column == 'Course':
                values = [(value, f'Course {value}') for value in values]
            fields.append({
                'name': column,
                'label': _feature_label(column),
                'group': group,
                'options': values if len(values) <= 12 else [],
                'step': '0.01' if not dataset.empty and dataset[column].dtype.kind == 'f' else '1',
            })
            used.add(column)
    return fields


def _course_options():
    return [(label, label) for label in BTECH_COURSE_OPTIONS]


def _normalize_model_name(name):
    if not name:
        return 'logistic_regression'
    cleaned = str(name).strip().lower().replace(' ', '_').replace('-', '_')
    mapping = {
        'logistic_regression': 'logistic_regression',
        'decision_tree': 'decision_tree',
        'random_forest': 'random_forest',
        'svm': 'svm',
        'xgboost': 'xgboost',
        'support_vector_machine': 'svm',
        'support_vector_machine_svm': 'svm',
        'xgbboost': 'xgboost',
    }
    return mapping.get(cleaned, cleaned)


def _choose_best_model(model_rows):
    if not model_rows:
        return None

    best = None
    for row in model_rows:
        if not isinstance(row, dict):
            continue
        if 'Model' not in row:
            continue
        try:
            score = float(row.get('Accuracy', 0) or 0)
        except (TypeError, ValueError):
            score = 0.0
        if best is None or score > best['score']:
            best = {'row': row, 'score': score}
    return best['row'] if best else None


def _selected_model_name():
    rows = _read_model_results()
    best = _choose_best_model(rows)
    model_name = best.get('Model') if isinstance(best, dict) else None
    return _normalize_model_name(model_name)


def _missing_model_state():
    missing = [name for name, path in MODEL_FILES.items() if name not in {'model_results', 'preprocessor', 'evaluation_details'} and not os.path.exists(path)]
    if missing:
        return True, MISSING_MODELS_MESSAGE
    return False, ''


def _load_dataset_for_ui():
    if not os.path.exists(DATASET_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(DATASET_PATH, sep=';')
        df.columns = [str(col).strip() for col in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def _build_model_row(compact_data, student_row=None):
    columns = _candidate_input_columns()
    baseline = student_row or {}
    model_data = {column: compact_data.get(column, baseline.get(column)) for column in columns}
    return pd.DataFrame([model_data], columns=columns)


def _candidate_input_columns():
    model = _load_best_pipeline()
    if model is not None:
        transformer = model.named_steps.get('academic_features') if hasattr(model, 'named_steps') else None
        columns = getattr(transformer, 'feature_names_in_', None)
        if columns:
            return [str(column).strip() for column in columns]
    return _feature_metadata().get('columns', [])


def _dataset_row_for_student(student):
    student_id = str((student or {}).get('student_id', '')).strip()
    if not student_id.isdigit():
        return {}
    row_number = int(student_id)
    dataset = _load_dataset_for_ui()
    if row_number < 1 or row_number > len(dataset):
        return {}
    return dataset.iloc[row_number - 1].to_dict()


def _dataset_feature_defaults():
    dataset = _load_dataset_for_ui()
    if dataset.empty:
        return {}
    defaults = {}
    columns = _candidate_input_columns()
    for column in columns:
        if column not in dataset.columns:
            defaults[column] = 0.0
            continue
        series = pd.to_numeric(dataset[column], errors='coerce').dropna()
        defaults[column] = float(series.median()) if not series.empty else 0.0
    return defaults


def _candidate_features_from_student(student, submitted=None):
    submitted = submitted or {}
    stored = student.get('feature_data') or '{}'
    try:
        stored = json.loads(stored) if isinstance(stored, str) else stored
    except (TypeError, ValueError):
        return None, ['stored prediction features are not valid JSON']
    if not isinstance(stored, dict):
        return None, ['stored prediction features are not an object']

    values = {}
    values.update(_dataset_row_for_student(student))
    values.update(stored)
    values.update(submitted)
    semester_data = _semester_records(student)
    semester_by_number = {item['number']: item for item in semester_data}
    semester_1 = semester_by_number.get(1, {})
    semester_2 = semester_by_number.get(2, {})
    semester_values = {
        'Curricular units 1st sem (approved)': semester_1.get('passed'),
        'Curricular units 1st sem (grade)': semester_1.get('grade'),
        'Curricular units 2nd sem (approved)': semester_2.get('passed'),
        'Curricular units 2nd sem (grade)': semester_2.get('grade'),
    }
    values.update({key: value for key, value in semester_values.items() if value not in (None, '')})

    defaults = _dataset_feature_defaults()
    for column, default_value in defaults.items():
        if column not in values or values[column] in (None, ''):
            values[column] = default_value

    columns = _candidate_input_columns()
    missing = [column for column in columns if column not in values or values[column] in (None, '')]
    if missing:
        return None, missing
    try:
        numeric = {column: float(values[column]) for column in columns}
    except (TypeError, ValueError) as error:
        return None, [f'candidate model input contains a non-numeric UCI value ({error})']
    return numeric, []


def _load_best_pipeline():
    return _safe_load(MODEL_FILES.get('best_model'))


def _updated_prediction_model_ready():
    required_files = ('best_model', 'preprocessor', 'feature_columns')
    if any(not os.path.exists(MODEL_FILES[name]) for name in required_files):
        return False
    return _load_best_pipeline() is not None and bool(_feature_metadata().get('columns'))


def _prediction_unavailable_message():
    return 'Prediction will be available after the updated model is trained and connected.'


def _stored_model_features(student):
    features, _ = _stored_model_feature_status(student)
    return features


def _stored_model_feature_status(student):
    return _candidate_features_from_student(student)


def _model_features_from_form(form, fields):
    values = {}
    errors = []
    for field in fields:
        raw_value = form.get(field['name'], '')
        if raw_value is None or str(raw_value).strip() == '':
            errors.append(f'{field["label"]} is required for the saved ML model.')
            continue
        try:
            values[field['name']] = float(str(raw_value).strip())
        except (TypeError, ValueError):
            errors.append(f'{field["label"]} must be numeric.')
    return values, errors


def _faculty_model_features_from_form(form, semester_data):
    semester_1 = semester_data.get('1', {})
    semester_2 = semester_data.get('2', {})
    raw_values = {
        'Gender': {'Male': 1, 'Female': 0}.get(form.get('gender', '').strip()),
        'Age at enrollment': form.get('age', '').strip(),
        'Previous qualification (grade)': form.get('previous_qualification_grade', '').strip(),
        'Tuition fees up to date': {'Yes': 1, 'No': 0}.get(form.get('tuition_fees_up_to_date', '').strip()),
        'Scholarship holder': {'Yes': 1, 'No': 0}.get(form.get('scholarship_holder', '').strip()),
        'Curricular units 1st sem (approved)': semester_1.get('passed', ''),
        'Curricular units 1st sem (grade)': semester_1.get('grade', ''),
        'Curricular units 2nd sem (approved)': semester_2.get('passed', ''),
        'Curricular units 2nd sem (grade)': semester_2.get('grade', ''),
    }
    labels = {
        'Gender': 'Gender',
        'Age at enrollment': 'Age',
        'Previous qualification (grade)': 'Previous Qualification Grade',
        'Tuition fees up to date': 'Tuition Fees Up To Date',
        'Scholarship holder': 'Scholarship Holder',
        'Curricular units 1st sem (approved)': '1st Semester Approved Units',
        'Curricular units 1st sem (grade)': '1st Semester Grade',
        'Curricular units 2nd sem (approved)': '2nd Semester Approved Units',
        'Curricular units 2nd sem (grade)': '2nd Semester Grade',
    }
    values = {}
    errors = []
    optional_fields = {
        'Course',
        'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)',
        'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)',
    }
    for field, raw_value in raw_values.items():
        if raw_value is None or str(raw_value).strip() == '':
            if field in optional_fields:
                continue
            errors.append(f'{labels[field]} is required.')
            continue
        try:
            values[field] = float(str(raw_value).strip())
        except (TypeError, ValueError):
            errors.append(f'{labels[field]} must be numeric.')
    return values, errors


def _predict_student_record(student):
    feature_meta = _feature_metadata()
    features, missing = _stored_model_feature_status(student)
    if missing:
        return {'error': 'Prediction cannot be generated because exact UCI values are unavailable for: ' + ', '.join(missing) + '.'}
    model = _load_best_pipeline()
    if model is None:
        return {'error': 'Prediction input mismatch: no saved model pipeline could be loaded.'}
    prediction_df = _build_model_row(features)
    try:
        predicted_label = model.predict(prediction_df)[0]
    except Exception as error:
        return {'error': f'Prediction input mismatch: the saved model rejected the feature data ({error}).'}
    probabilities = model.predict_proba(prediction_df) if hasattr(model, 'predict_proba') else None
    classes = feature_meta.get('target_classes', ['Dropout', 'Enrolled', 'Graduate'])
    predicted_class, probability_map, risk_probability, _ = _format_prediction(predicted_label, probabilities, model, classes)
    risk_level = _risk_level(risk_probability) if risk_probability is not None else 'Not available'
    return {
        'prediction': predicted_class,
        'probabilities': probability_map,
        'dropout_risk': risk_probability,
        'risk_level': risk_level,
        'feature_data': json.dumps(features),
        'model_name': 'Random Forest + Academic Features',
    }


def _format_prediction(predicted_label, probabilities, model, classes):
    class_index = {str(value): name for value, name in enumerate(classes)}
    predicted_class = class_index.get(str(predicted_label), str(predicted_label))
    probability_map = {}
    if probabilities is not None:
        model_classes = getattr(model, 'classes_', [])
        probability_map = {class_index.get(str(label), str(label)): float(probability) * 100 for label, probability in zip(model_classes, probabilities[0])}
    dropout_probability = probability_map.get('Dropout')
    attention = predicted_class == 'Dropout' or (dropout_probability is not None and dropout_probability >= 50)
    return predicted_class, probability_map, dropout_probability, attention


def _academic_context(form):
    values = {}
    errors = []
    for key, label in [
        ('attendance', 'Attendance'),
        ('average_grade', 'Average Grade'),
        ('backlogs', 'Backlogs'),
        ('courses_passed', 'Courses Passed'),
    ]:
        raw_value = form.get(key, '').strip()
        if not raw_value:
            errors.append(f'{label} is required.')
            continue
        try:
            values[key] = float(raw_value)
        except ValueError:
            errors.append(f'{label} must be numeric.')
    if values.get('attendance') is not None and not 0 <= values['attendance'] <= 100:
        errors.append('Attendance must be between 0 and 100.')
    if values.get('average_grade') is not None and values['average_grade'] < 0:
        errors.append('Average Grade cannot be negative.')
    if values.get('backlogs') is not None and values['backlogs'] < 0:
        errors.append('Backlogs cannot be negative.')
    if values.get('courses_passed') is not None and values['courses_passed'] < 0:
        errors.append('Courses Passed cannot be negative.')
    values['backlogs'] = int(values['backlogs']) if 'backlogs' in values else None
    values['courses_passed'] = int(values['courses_passed']) if 'courses_passed' in values else None
    return values, errors


def _risk_level(probability):
    if probability is None:
        return 'Medium'
    if probability >= RISK_THRESHOLDS['high']:
        return 'High'
    if probability >= RISK_THRESHOLDS['medium']:
        return 'Medium'
    return 'Low'


def _semester_records(student):
    raw_data = student.get('semester_data') or '{}'
    try:
        stored = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    except (TypeError, ValueError):
        stored = {}
    records = {}
    for number in range(1, 9):
        item = stored.get(str(number), {}) if isinstance(stored, dict) else {}
        if not isinstance(item, dict):
            item = {}
        values = {
            'number': number,
            'label': SEMESTER_LABELS[str(number)],
            'total': item.get('total'),
            'passed': item.get('passed'),
            'grade': item.get('grade'),
            'attendance': item.get('attendance'),
            'courses': item.get('courses', []),
        }
        if number == 1 and not values['passed'] and student.get('semester_1_approved') not in (None, '', 'Not available'):
            values['passed'] = student.get('semester_1_approved')
            values['grade'] = student.get('semester_1_grade')
        if number == 2 and not values['passed'] and student.get('semester_2_approved') not in (None, '', 'Not available'):
            values['passed'] = student.get('semester_2_approved')
            values['grade'] = student.get('semester_2_grade')
        if values['total'] not in (None, '') and values['passed'] not in (None, ''):
            try:
                values['failed'] = max(float(values['total']) - float(values['passed']), 0)
                values['pass_percentage'] = round(float(values['passed']) / float(values['total']) * 100, 1) if float(values['total']) else None
            except (TypeError, ValueError, ZeroDivisionError):
                values['failed'] = None
                values['pass_percentage'] = None
        else:
            values['failed'] = None
            values['pass_percentage'] = None
        if any(values[key] not in (None, '', []) for key in ('total', 'passed', 'grade', 'attendance', 'courses')):
            records[str(number)] = values
    return [records[str(number)] for number in range(1, 9) if str(number) in records]


def _academic_summary(student):
    semesters = _semester_records(student)
    attendance_values = [float(item['attendance']) for item in semesters if item.get('attendance') not in (None, '')]
    grades = [float(item['grade']) for item in semesters if item.get('grade') not in (None, '')]
    passed = [float(item['passed']) for item in semesters if item.get('passed') not in (None, '')]
    return {
        'semesters': semesters,
        'overall_attendance': round(sum(attendance_values) / len(attendance_values), 1) if attendance_values else None,
        'average_performance': round(sum(grades) / len(grades), 1) if grades else None,
        'subjects_passed': sum(passed) if passed else None,
    }


def _semester_form_data(form):
    records = {}
    errors = []
    for number in range(1, 9):
        prefix = f'semester_{number}_'
        raw_values = {
            'total': form.get(prefix + 'total', '').strip(),
            'passed': form.get(prefix + 'passed', '').strip(),
            'grade': form.get(prefix + 'grade', '').strip(),
            'attendance': form.get(prefix + 'attendance', '').strip(),
        }
        subject_names = form.getlist(prefix + 'subject_name[]')
        subject_marks = form.getlist(prefix + 'subject_marks[]')
        subject_attendance = form.getlist(prefix + 'subject_attendance[]')
        subject_results = form.getlist(prefix + 'subject_result[]')
        courses_text = form.get(prefix + 'courses', '').strip()
        has_structured_subjects = any(value.strip() for value in subject_names + subject_marks + subject_attendance + subject_results)
        if not any(raw_values.values()) and not courses_text and not has_structured_subjects:
            continue
        record = {}
        try:
            if raw_values['total']:
                record['total'] = float(raw_values['total'])
            if raw_values['passed']:
                record['passed'] = float(raw_values['passed'])
            if raw_values['grade']:
                record['grade'] = float(raw_values['grade'])
            if raw_values['attendance']:
                record['attendance'] = float(raw_values['attendance'])
        except ValueError:
            errors.append(f'{SEMESTER_LABELS[str(number)]} values must be numeric.')
            continue
        if record.get('total') is None or record.get('passed') is None:
            errors.append(f'{SEMESTER_LABELS[str(number)]} requires total and passed subjects/credits.')
        elif record['total'] <= 0 or record['passed'] < 0 or record['passed'] > record['total']:
            errors.append(f'{SEMESTER_LABELS[str(number)]} must have valid total and passed values.')
        if record.get('attendance') is not None and not 0 <= record['attendance'] <= 100:
            errors.append(f'{SEMESTER_LABELS[str(number)]} attendance must be between 0 and 100.')
        if record.get('grade') is not None and not 0 <= record['grade'] <= 100:
            errors.append(f'{SEMESTER_LABELS[str(number)]} grade must be between 0 and 100.')
        courses = []
        if has_structured_subjects:
            subject_count = max(len(subject_names), len(subject_marks), len(subject_attendance), len(subject_results))
            for index in range(subject_count):
                name = subject_names[index].strip() if index < len(subject_names) else ''
                marks_value = subject_marks[index].strip() if index < len(subject_marks) else ''
                attendance_value = subject_attendance[index].strip() if index < len(subject_attendance) else ''
                result = subject_results[index].strip() if index < len(subject_results) else ''
                if not any((name, marks_value, attendance_value, result)):
                    continue
                if not name or not marks_value or not attendance_value or result not in {'Passed', 'Failed'}:
                    errors.append(f'{SEMESTER_LABELS[str(number)]} subjects require a name, marks, attendance, and Passed or Failed result.')
                    continue
                try:
                    marks = float(marks_value)
                    attendance = float(attendance_value)
                except ValueError:
                    errors.append(f'{SEMESTER_LABELS[str(number)]} subject marks and attendance must be numeric.')
                    continue
                if not 0 <= attendance <= 100 or not 0 <= marks <= 100:
                    errors.append(f'{SEMESTER_LABELS[str(number)]} subject marks must be 0-100 and attendance must be 0-100.')
                    continue
                courses.append({'name': name, 'marks': marks, 'attendance': attendance, 'status': result})
        else:
            legacy_lines = courses_text.splitlines()
            for line in legacy_lines:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) != 4 or not parts[0]:
                    errors.append(f'{SEMESTER_LABELS[str(number)]} subjects must use: Name | Marks | Attendance | Passed/Failed.')
                    continue
                try:
                    marks = float(parts[1])
                    attendance = float(parts[2])
                except ValueError:
                    errors.append(f'{SEMESTER_LABELS[str(number)]} subject marks and attendance must be numeric.')
                    continue
                if not 0 <= attendance <= 100 or not 0 <= marks <= 100 or parts[3].casefold() not in {'passed', 'failed'}:
                    errors.append(f'{SEMESTER_LABELS[str(number)]} subject values are invalid.')
                    continue
                courses.append({'name': parts[0], 'marks': marks, 'attendance': attendance, 'status': parts[3].title()})
        if courses:
            record['courses'] = courses
        records[str(number)] = record
    return records, errors


def _risk_message(level):
    return {
        'High': 'The student may need prompt academic advising and targeted support.',
        'Medium': 'The student may benefit from regular mentoring and progress monitoring.',
        'Lower': 'This is the model outcome with the lowest current dropout probability; continue routine support.',
        'Low': 'The student currently shows a lower predicted dropout risk. Continue regular support.',
    }[level]


def _value_or_not_available(value):
    return value if value not in (None, '', 'nan') else 'Not available'


def _format_student_id(student_id):
    try:
        return f"{int(student_id):03d}"
    except (TypeError, ValueError):
        return str(student_id).strip() or 'Not available'


def _gender_label(value):
    if value in (0, '0', 'female', 'Female'):
        return 'Female'
    if value in (1, '1', 'male', 'Male'):
        return 'Male'
    return str(value) if value not in (None, '') else 'Not available'


def _student_record_from_dataset(student_id):
    dataset = _load_dataset_for_ui()
    if dataset.empty:
        return None

    if student_id is None:
        return None
    try:
        numeric_id = int(str(student_id).strip())
    except ValueError:
        cleaned = str(student_id).strip()
        if cleaned.lower().startswith('student'):
            cleaned = cleaned.replace('student', '').strip()
        try:
            numeric_id = int(cleaned)
        except ValueError:
            return None

    if numeric_id < 1 or numeric_id > len(dataset):
        return None

    row = dataset.iloc[numeric_id - 1]
    target = str(row.get('Target', '')).strip() if row.get('Target') is not None else 'Not available'
    if target not in {'Dropout', 'Enrolled', 'Graduate'}:
        target = 'Not available'

    return {
        'id': numeric_id,
        'student_id': _format_student_id(numeric_id),
        'student_name': f'Student {_format_student_id(numeric_id)}',
        'course': _value_or_not_available(row.get('Course')),
        'age': _value_or_not_available(row.get('Age at enrollment')),
        'gender': _gender_label(row.get('Gender')),
        'current_year': row.get('Current academic year') if 'Current academic year' in dataset.columns and row.get('Current academic year') not in (None, '') else 'Not available',
        'semester_1_approved': _value_or_not_available(row.get('Curricular units 1st sem (approved)')),
        'semester_1_grade': _value_or_not_available(row.get('Curricular units 1st sem (grade)')),
        'semester_2_approved': _value_or_not_available(row.get('Curricular units 2nd sem (approved)')),
        'semester_2_grade': _value_or_not_available(row.get('Curricular units 2nd sem (grade)')),
        'prediction': target,
        'prediction_probability': None,
        'risk_status': 'Requires Attention' if target == 'Dropout' else 'Monitor' if target == 'Enrolled' else 'Normal',
        'message': 'Your prediction indicates that additional academic support may be helpful.' if target == 'Dropout' else 'Your current prediction indicates enrolled status. Continue monitoring your academic progress.' if target == 'Enrolled' else 'Your current prediction indicates graduate status.',
        'areas_to_review': ['Academic progress', 'Attendance trend', 'Semester performance'] if target == 'Dropout' else ['Academic monitoring', 'Progress review'] if target == 'Enrolled' else ['Routine academic monitoring'],
        'support_suggestions': ['Review academic progress', 'Discuss difficulties with the student', 'Consider academic mentoring', 'Check whether additional institutional support is needed'] if target == 'Dropout' else ['Continue academic monitoring', 'Encourage regular progress', 'Provide guidance if performance decreases'] if target == 'Enrolled' else ['Continue normal academic monitoring'],
        'support_type': None,
        'support_status': None,
        'support_notes': None,
        'action_taken': None,
        'follow_up_date': None,
        'support_updated_at': None,
    }


def _student_record_from_storage(student_id):
    if not student_id:
        return None
    student = get_student(str(student_id).strip())
    if student is None:
        return None
    saved_prediction = student.get('prediction')
    prediction = saved_prediction if saved_prediction in {'Dropout', 'Enrolled', 'Graduate'} else None
    try:
        probabilities = json.loads(student.get('ml_probabilities') or '{}')
    except (TypeError, ValueError):
        probabilities = {}
    summary = _academic_summary(student)
    support = get_latest_support(student.get('id'))
    record = {
        'id': student.get('id'),
        'student_id': str(student.get('student_id') or '').strip(),
        'student_name': student.get('student_name') or 'Student',
        'course': student.get('course') or 'Not available',
        'age': student.get('age') if student.get('age') not in (None, '') else 'Not available',
        'gender': student.get('gender') or 'Not available',
        'current_year': student.get('current_year') or 'Not available',
        'current_semester': student.get('current_semester') or 'Not available',
        'semester_1_approved': student.get('semester_1_approved') if student.get('semester_1_approved') not in (None, '') else 'Not available',
        'semester_1_grade': student.get('semester_1_grade') if student.get('semester_1_grade') not in (None, '') else 'Not available',
        'semester_2_approved': student.get('semester_2_approved') if student.get('semester_2_approved') not in (None, '') else 'Not available',
        'semester_2_grade': student.get('semester_2_grade') if student.get('semester_2_grade') not in (None, '') else 'Not available',
        'prediction': prediction,
        'prediction_unavailable': prediction is None,
        'prediction_probability': student.get('prediction_probability'),
        'risk_status': 'Requires Attention' if prediction == 'Dropout' else 'Monitor' if prediction == 'Enrolled' else 'Normal' if prediction == 'Graduate' else 'Not available',
        'message': 'Your prediction indicates that additional academic support may be helpful.' if prediction == 'Dropout' else 'Your current prediction indicates enrolled status. Continue monitoring your academic progress.' if prediction == 'Enrolled' else 'Your current prediction indicates graduate status.' if prediction == 'Graduate' else 'Prediction result not available because the required model features were not collected.',
        'areas_to_review': ['Academic progress', 'Attendance trend', 'Semester performance'] if prediction == 'Dropout' else ['Academic monitoring', 'Progress review'] if prediction == 'Enrolled' else ['Routine academic monitoring'] if prediction == 'Graduate' else ['Required model features were not collected'],
        'support_suggestions': ['Review academic progress', 'Discuss difficulties with the student', 'Consider academic mentoring', 'Check whether additional institutional support is needed'] if prediction == 'Dropout' else ['Continue academic monitoring', 'Encourage regular progress', 'Provide guidance if performance decreases'] if prediction == 'Enrolled' else ['Continue normal academic monitoring'] if prediction == 'Graduate' else ['Prediction is unavailable until the required model features are collected'],
        'support_type': None,
        'support_status': None,
        'support_notes': None,
        'support_updated_at': None,
    }
    record.update(summary)
    record['probabilities'] = probabilities
    record['dropout_risk'] = probabilities.get('Dropout', student.get('prediction_probability'))
    record['risk_level'] = (student.get('risk_level') or _risk_level(record['dropout_risk'])) if record['dropout_risk'] is not None else 'Not available'
    if support:
        record['support_type'] = support.get('support_type')
        record['support_status'] = support.get('support_status')
        record['support_notes'] = support.get('support_notes')
        record['action_taken'] = support.get('action_taken')
        record['follow_up_date'] = support.get('follow_up_date')
        record['support_updated_at'] = support.get('updated_at')
    return record


def _faculty_student_rows(search='', prediction='', year='', semester='', course='', risk=''):
    database_students = list_students(search=search, prediction=prediction)
    rows = []
    for student in database_students:
        student_id = student.get('student_id') or student.get('id')
        record = _student_record_from_storage(student_id)
        if record is None:
            continue
        if year and str(record.get('current_year')) != year:
            continue
        if semester and str(record.get('current_semester')) != semester:
            continue
        if course and course.casefold() not in str(record.get('course', '')).casefold():
            continue
        if risk and str(record.get('risk_level', '')).casefold() != risk.casefold():
            continue
        rows.append(record)
    return rows


def _faculty_summary_from_rows(rows):
    total = len(rows)
    dropout = sum(1 for row in rows if row.get('prediction') == 'Dropout')
    enrolled = sum(1 for row in rows if row.get('prediction') == 'Enrolled')
    graduate = sum(1 for row in rows if row.get('prediction') == 'Graduate')
    risk_students = sum(1 for row in rows if row.get('risk_level') in {'High', 'Medium'} or row.get('prediction') == 'Dropout')
    attendance = [row['overall_attendance'] for row in rows if row.get('overall_attendance') is not None]
    performance = [row['average_performance'] for row in rows if row.get('average_performance') is not None]
    return {
        'total': total,
        'dropout': dropout,
        'enrolled': enrolled,
        'graduate': graduate,
        'attention': risk_students,
        'dropout_risk': risk_students,
        'average_attendance': round(sum(attendance) / len(attendance), 1) if attendance else None,
        'average_performance': round(sum(performance) / len(performance), 1) if performance else None,
    }


@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        role = request.form.get('role', 'student')
        if role == 'student':
            student_name = request.form.get('student_name', '').strip()
            student_id = request.form.get('student_id', '').strip()
            if not student_name or not student_id:
                return render_template('login.html', error='Student name and student ID are required.', active_role='student')
            student = get_student_by_credentials(student_id, student_name)
            if student is None:
                return render_template('login.html', error='Student record not found.', active_role='student')
            session['role'] = 'student'
            session['student_name'] = student['student_name']
            session['student_id'] = student['student_id']
            return redirect(url_for('student_dashboard'))

        if role == 'faculty':
            faculty_id = request.form.get('faculty_id', '').strip()
            faculty_name = request.form.get('faculty_name', '').strip()
            password = request.form.get('password', '')
            if not faculty_id or not faculty_name or not password:
                return render_template('login.html', error='Faculty ID, name, and password are required.', active_role='faculty')
            faculty = get_faculty_by_credentials(faculty_id, faculty_name, password)
            if faculty is not None:
                session['role'] = 'faculty'
                session['faculty_id'] = faculty['faculty_id']
                session['faculty_name'] = faculty['faculty_name']
                return redirect(url_for('faculty_dashboard'))
            return render_template('login.html', error='Invalid faculty credentials.', active_role='faculty')

    return render_template('login.html', error=None, active_role='student')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/student-dashboard')
def student_dashboard():
    if session.get('role') != 'student':
        return redirect(url_for('login'))
    student_id = session.get('student_id')
    student = _student_record_from_storage(student_id)
    if student is None:
        return render_template('student_not_found.html', message='Student record not found.', back_url=url_for('login'))
    student['student_name'] = session.get('student_name', student['student_name'])
    prediction = {
        'predicted_class': student.get('prediction'),
        'message': student.get('message'),
        'current_year': student.get('current_year'),
        'current_semester': 'Not available',
        'prediction_date': 'Not available',
    }
    return render_template('student_dashboard.html', student=student, student_name=student['student_name'], prediction=prediction, probability=student.get('prediction_probability'))


@app.route('/student/<student_id>')
def student_details(student_id):
    if session.get('role') != 'faculty':
        return redirect(url_for('login'))
    student = _student_record_from_storage(student_id)
    if student is None:
        return render_template('student_not_found.html', message='Student record not found.', back_url=url_for('faculty_dashboard')), 404
    return render_template(
        'faculty_student_details.html',
        student=student,
        prediction_error=None,
        prediction_success=None,
        model_name='Random Forest + Academic Features',
        prediction_available=_updated_prediction_model_ready(),
    )


@app.route('/faculty-dashboard')
def faculty_dashboard():
    if session.get('role') != 'faculty':
        return redirect(url_for('login'))
    search = request.args.get('search', '').strip()
    prediction = request.args.get('prediction', '').strip()
    year = request.args.get('year', '').strip()
    semester = request.args.get('semester', '').strip()
    course = request.args.get('course', '').strip()
    risk = request.args.get('risk', '').strip()
    students = _faculty_student_rows(search=search, prediction=prediction, year=year, semester=semester, course=course, risk=risk)
    summary = _faculty_summary_from_rows(students)
    return render_template(
        'faculty_dashboard.html',
        students=students,
        summary=summary,
        metrics=_read_model_results(),
        model_performance=_read_final_model_performance(),
        filters={'search': search, 'prediction': prediction, 'year': year, 'semester': semester, 'course': course, 'risk': risk},
    )


@app.route('/faculty/student/<student_id>/delete', methods=['POST'])
def delete_student_record(student_id):
    if session.get('role') != 'faculty':
        return redirect(url_for('login'))
    if delete_student(student_id):
        session['success_message'] = 'Student deleted successfully'
    else:
        session['success_message'] = 'Student record not found.'
    return redirect(url_for('faculty_dashboard'))


@app.route('/add-student', methods=['GET', 'POST'])
def add_student():
    if session.get('role') != 'faculty':
        return redirect(url_for('login'))
    course_options = _course_options()

    if request.method == 'POST':
        student_name = request.form.get('student_name', '').strip()
        student_id = request.form.get('student_id', '').strip()
        course = request.form.get('course', '').strip()
        age = request.form.get('age', '').strip()
        gender = request.form.get('gender', '').strip()
        current_year = request.form.get('current_year', '').strip()
        current_semester = request.form.get('current_semester', '').strip()

        if not student_name or not student_id or not course or not age or not gender or current_year not in {'1', '2', '3', '4'} or current_semester not in {str(number) for number in range(1, 9)}:
            return render_template('add_student.html', error='Please complete all required student information.', form=request.form, course_options=course_options)
        try:
            age_value = int(age)
            if age_value <= 0:
                raise ValueError
        except ValueError:
            return render_template('add_student.html', error='Age must be a valid positive number.', form=request.form, course_options=course_options)

        if student_id_exists(student_id):
            return render_template('add_student.html', error='Student ID already exists.', form=request.form, course_options=course_options)

        semester_data, semester_errors = _semester_form_data(request.form)
        model_features, model_errors = _faculty_model_features_from_form(request.form, semester_data)
        if semester_errors or model_errors:
            return render_template('add_student.html', error=' '.join(semester_errors + model_errors), form=request.form, course_options=course_options)

        create_student({
            'student_name': student_name,
            'student_id': student_id,
            'course': course,
            'age': age_value,
            'gender': gender,
            'current_year': current_year,
            'current_semester': current_semester,
            'semester_1_approved': semester_data.get('1', {}).get('passed'),
            'semester_1_grade': semester_data.get('1', {}).get('grade'),
            'semester_2_approved': semester_data.get('2', {}).get('passed'),
            'semester_2_grade': semester_data.get('2', {}).get('grade'),
            'semester_data': json.dumps(semester_data),
            'prediction': None,
            'prediction_probability': None,
            'feature_data': json.dumps(model_features),
        })
        session['success_message'] = 'Student added successfully.'
        return redirect(url_for('faculty_dashboard'))

    return render_template('add_student.html', error=None, form={}, course_options=course_options)


@app.route('/add-faculty', methods=['GET', 'POST'])
def add_faculty():
    if session.get('role') != 'faculty':
        return redirect(url_for('login'))
    if request.method == 'POST':
        faculty_id = request.form.get('faculty_id', '').strip()
        faculty_name = request.form.get('faculty_name', '').strip()
        password = request.form.get('password', '')
        if not faculty_id or not faculty_name or not password:
            return render_template('add_faculty.html', error='Faculty ID, name, and password are required.', form=request.form)
        try:
            create_faculty(faculty_id, faculty_name, password)
        except Exception as error:
            if 'UNIQUE constraint failed' in str(error):
                message = 'Faculty ID already exists.'
            else:
                message = 'Faculty account could not be created.'
            return render_template('add_faculty.html', error=message, form=request.form)
        session['success_message'] = 'Faculty account added successfully.'
        return redirect(url_for('faculty_dashboard'))
    return render_template('add_faculty.html', error=None, form={})


@app.route('/faculty/student/<student_id>')
def faculty_student_details(student_id):
    if session.get('role') != 'faculty':
        return redirect(url_for('login'))
    student = _student_record_from_storage(student_id)
    if student is None:
        return render_template('student_not_found.html', message='Student record not found.', back_url=url_for('faculty_dashboard')), 404
    prediction_error = session.pop('prediction_error', None)
    prediction_success = session.pop('prediction_success', None)
    return render_template(
        'faculty_student_details.html',
        student=student,
        prediction_error=prediction_error,
        prediction_success=prediction_success,
        model_name='Random Forest + Academic Features',
        prediction_available=_updated_prediction_model_ready(),
    )


@app.route('/faculty/student/<student_id>/predict', methods=['POST'])
def faculty_student_predict(student_id):
    if session.get('role') != 'faculty':
        return redirect(url_for('login'))
    if not _updated_prediction_model_ready():
        session['prediction_error'] = _prediction_unavailable_message()
        return redirect(url_for('faculty_student_details', student_id=student_id))
    student = get_student(student_id)
    if student is None:
        return render_template('student_not_found.html', message='Student record not found.', back_url=url_for('faculty_dashboard')), 404
    try:
        result = _predict_student_record(student)
    except Exception as error:
        result = {'error': f'Prediction input mismatch: {error}'}
    if result.get('error'):
        session['prediction_error'] = result['error']
        return redirect(url_for('faculty_student_details', student_id=student_id))
    update_student_prediction(
        student.get('student_id'),
        student.get('current_year'),
        student.get('current_semester'),
        student.get('attendance'),
        student.get('average_grade'),
        student.get('backlogs'),
        student.get('courses_passed'),
        result['prediction'],
        result['dropout_risk'],
        json.dumps(result['probabilities']),
        result['risk_level'],
        result['feature_data'],
    )
    session['prediction_success'] = 'Prediction saved from the trained model.'
    return redirect(url_for('faculty_student_details', student_id=student_id))


@app.route('/faculty/student/<student_id>/model-features', methods=['GET', 'POST'])
def faculty_student_model_features(student_id):
    if session.get('role') != 'faculty':
        return redirect(url_for('login'))
    session['prediction_error'] = _prediction_unavailable_message()
    return redirect(url_for('faculty_student_details', student_id=student_id))


@app.route('/faculty/student/<student_id>/support', methods=['POST'])
def faculty_student_support(student_id):
    if session.get('role') != 'faculty':
        return redirect(url_for('login'))
    student = _student_record_from_storage(student_id)
    if student is None:
        return render_template('student_not_found.html', message='Student record not found.', back_url=url_for('faculty_dashboard')), 404
    support_type = request.form.get('support_type', '').strip()
    support_status = request.form.get('support_status', '').strip()
    support_notes = request.form.get('support_notes', '').strip()
    if support_type and support_status:
        action_taken = request.form.get('action_taken', '').strip()
        follow_up_date = request.form.get('follow_up_date', '').strip()
        update_support(student.get('id'), support_type, support_status, support_notes, action_taken, follow_up_date)
        session['success_message'] = 'Faculty guidance saved for this student.'
    return redirect(url_for('faculty_student_details', student_id=student_id))


@app.route('/predict', methods=['GET', 'POST'])

def predict():
    if session.get('role') not in {'student', 'faculty'}:
        return redirect(url_for('login'))

    has_missing, message = _missing_model_state()
    feature_meta = _feature_metadata()
    columns = feature_meta.get('columns', [])
    fields = _prediction_fields()

    if has_missing:
        return render_template('predict.html', fields=fields, missing_models=True, message=message)

    if request.method == 'GET':
        return render_template('predict.html', fields=fields, message='')

    if not columns:
        return render_template('predict.html', fields=fields, missing_models=True, message='No feature metadata was found. Please verify the saved model files.')

    form_data = {}
    error_messages = []

    current_year = request.form.get('current_year', '').strip()
    current_semester = request.form.get('current_semester', '').strip()
    if current_year not in {'1', '2', '3', '4'}:
        error_messages.append('Current Year is required.')
    if current_semester not in {str(number) for number in range(1, 9)}:
        error_messages.append('Current Semester is required.')
    academic_context, academic_errors = _academic_context(request.form)
    error_messages.extend(academic_errors)

    for field in fields:
        raw_value = request.form.get(field['name'], '')
        if raw_value is None or str(raw_value).strip() == '':
            error_messages.append(f'{field["label"]} is required.')
            continue

        try:
            form_data[field['name']] = float(str(raw_value).strip())
        except ValueError:
            error_messages.append(f'{field["label"]} must be numeric.')

    if error_messages:
        return render_template('predict.html', fields=fields, errors=error_messages)

    try:
        model = _load_best_pipeline()
        if model is None:
            return render_template('predict.html', fields=fields, message=MISSING_MODELS_MESSAGE)

        student_id = session.get('student_id')
        student = get_student(student_id) if student_id else None
        if student is None:
            return render_template(
                'predict.html',
                fields=fields,
                message='Prediction must be run from a registered student account so exact UCI inputs can be verified.',
            )
        candidate_features, candidate_errors = _candidate_features_from_student(student, form_data)
        if candidate_errors:
            return render_template(
                'predict.html',
                fields=fields,
                message='Prediction cannot be generated because exact UCI values are unavailable for: '
                + ', '.join(candidate_errors)
                + '.',
            )
        prediction_df = _build_model_row(candidate_features)
        predicted_label = model.predict(prediction_df)[0]
        probabilities = model.predict_proba(prediction_df) if hasattr(model, 'predict_proba') else None
        classes = feature_meta.get('target_classes', ['Dropout', 'Enrolled', 'Graduate'])
        predicted_class, probability_map, risk_probability, attention = _format_prediction(predicted_label, probabilities, model, classes)
        risk_level = _risk_level(risk_probability) if risk_probability is not None else 'Not available'
        prediction_message = _risk_message(risk_level) if risk_probability is not None else 'The model returned an outcome without a probability estimate; faculty review is required.'
        session['prediction'] = {
            'predicted_class': predicted_class,
            'probabilities': probability_map,
            'risk_level': risk_level,
            'risk_probability': risk_probability,
            'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'message': prediction_message,
            'current_year': current_year,
            'current_semester': current_semester,
        }
        student_id = session.get('student_id')
        if not student_id:
            return render_template('predict.html', fields=fields, message='Prediction must be run from a registered student account so the result can be saved to that student.')
        update_student_prediction(
            student_id,
            current_year,
            current_semester,
            academic_context['attendance'],
            academic_context['average_grade'],
            academic_context['backlogs'],
            academic_context['courses_passed'],
            predicted_class,
            risk_probability,
            json.dumps(probability_map),
            risk_level,
            json.dumps(candidate_features),
        )
        return render_template(
            'result.html',
            predicted_class=predicted_class,
            risk_level=risk_level,
            risk_probability=risk_probability,
            probabilities=probability_map,
            message=prediction_message,
            role=session.get('role'),
        )
    except Exception:
        return render_template('predict.html', fields=fields, message='Prediction failed. Please check the input values and model files.')


@app.route('/result')
def result():
    prediction = session.get('prediction')
    if not prediction:
        return redirect(url_for('login'))
    return render_template('result.html', role=session.get('role'), **prediction)


if __name__ == '__main__':
    os.makedirs(BASE_DIR, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
