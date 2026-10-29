import os
import sqlite3
from datetime import datetime

from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, 'student_dropout.db')


def normalize_student_id(student_id):
    value = str(student_id or '').strip()
    if not value:
        return ''
    try:
        if value.isdigit() or (value.startswith('+') and value[1:].isdigit()):
            return str(int(value))
    except ValueError:
        pass
    return value


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA foreign_keys = ON')
    return connection


def init_db():
    with get_connection() as connection:
        connection.executescript(
            '''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                student_name TEXT NOT NULL,
                course TEXT,
                age INTEGER,
                gender TEXT,
                current_year TEXT,
                current_semester TEXT,
                semester_1_approved REAL,
                semester_1_grade REAL,
                semester_2_approved REAL,
                semester_2_grade REAL,
                attendance REAL,
                average_grade REAL,
                backlogs INTEGER,
                courses_passed INTEGER,
                prediction TEXT,
                prediction_probability REAL,
                ml_probabilities TEXT,
                risk_level TEXT,
                semester_data TEXT,
                feature_data TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS support_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                support_type TEXT NOT NULL,
                support_status TEXT NOT NULL,
                support_notes TEXT,
                action_taken TEXT,
                follow_up_date TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS faculty (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                faculty_id TEXT NOT NULL UNIQUE,
                faculty_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            '''
        )

        columns = [row['name'] for row in connection.execute('PRAGMA table_info(students)').fetchall()]
        support_columns = [row['name'] for row in connection.execute('PRAGMA table_info(support_records)').fetchall()]
        for column in ('action_taken', 'follow_up_date'):
            if column not in support_columns:
                try:
                    connection.execute(f'ALTER TABLE support_records ADD COLUMN {column} TEXT')
                except sqlite3.OperationalError as error:
                    if 'duplicate column name' not in str(error).lower():
                        raise
        for column, column_type in {
            'student_id': 'TEXT',
            'course': 'TEXT',
            'age': 'INTEGER',
            'gender': 'TEXT',
            'current_year': 'TEXT',
            'current_semester': 'TEXT',
            'semester_1_approved': 'REAL',
            'semester_1_grade': 'REAL',
            'semester_2_approved': 'REAL',
            'semester_2_grade': 'REAL',
        }.items():
            if column not in columns:
                connection.execute(f'ALTER TABLE students ADD COLUMN {column} {column_type}')

        connection.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_students_student_id ON students(student_id)')


def create_student(student):
    now = datetime.now().isoformat(timespec='seconds')
    with get_connection() as connection:
        table_columns = [row['name'] for row in connection.execute('PRAGMA table_info(students)').fetchall()]
        for column, column_type in {
            'student_id': 'TEXT',
            'student_name': 'TEXT',
            'course': 'TEXT',
            'age': 'INTEGER',
            'gender': 'TEXT',
            'current_year': 'TEXT',
            'current_semester': 'TEXT',
            'semester_1_approved': 'REAL',
            'semester_1_grade': 'REAL',
            'semester_2_approved': 'REAL',
            'semester_2_grade': 'REAL',
            'attendance': 'REAL',
            'average_grade': 'REAL',
            'backlogs': 'INTEGER',
            'courses_passed': 'INTEGER',
            'prediction': 'TEXT',
            'prediction_probability': 'REAL',
            'ml_probabilities': 'TEXT',
            'risk_level': 'TEXT',
            'semester_data': 'TEXT',
            'feature_data': 'TEXT',
            'created_at': 'TEXT',
        }.items():
            if column not in table_columns:
                connection.execute(f'ALTER TABLE students ADD COLUMN {column} {column_type}')

        student_id = normalize_student_id(student.get('student_id') or student.get('id'))
        student_name = str(student.get('student_name') or '').strip()
        if not student_id and student_name:
            student_id = normalize_student_id(student.get('id'))
        if not student_id:
            student_id = None

        field_order = [
            'student_id',
            'student_name',
            'course',
            'age',
            'gender',
            'current_year',
            'current_semester',
            'semester_1_approved',
            'semester_1_grade',
            'semester_2_approved',
            'semester_2_grade',
            'attendance',
            'average_grade',
            'backlogs',
            'courses_passed',
            'prediction',
            'prediction_probability',
            'ml_probabilities',
            'risk_level',
            'semester_data',
            'feature_data',
            'created_at',
        ]
        available_columns = [field for field in field_order if field in [row['name'] for row in connection.execute('PRAGMA table_info(students)').fetchall()]]
        data_values = []
        for field in available_columns:
            if field == 'student_id':
                data_values.append(student_id)
            elif field == 'student_name':
                data_values.append(student_name)
            elif field == 'course':
                data_values.append(student.get('course'))
            elif field == 'age':
                data_values.append(student.get('age'))
            elif field == 'gender':
                data_values.append(student.get('gender'))
            elif field == 'current_year':
                data_values.append(student.get('current_year', 'Not available'))
            elif field == 'current_semester':
                data_values.append(student.get('current_semester', 'Not available'))
            elif field == 'semester_1_approved':
                data_values.append(student.get('semester_1_approved'))
            elif field == 'semester_1_grade':
                data_values.append(student.get('semester_1_grade'))
            elif field == 'semester_2_approved':
                data_values.append(student.get('semester_2_approved'))
            elif field == 'semester_2_grade':
                data_values.append(student.get('semester_2_grade'))
            elif field == 'attendance':
                data_values.append(student.get('attendance'))
            elif field == 'average_grade':
                data_values.append(student.get('average_grade'))
            elif field == 'backlogs':
                data_values.append(student.get('backlogs'))
            elif field == 'courses_passed':
                data_values.append(student.get('courses_passed'))
            elif field == 'prediction':
                data_values.append(student.get('prediction') or 'Not available')
            elif field == 'prediction_probability':
                data_values.append(student.get('prediction_probability'))
            elif field == 'ml_probabilities':
                data_values.append(student.get('ml_probabilities', '{}'))
            elif field == 'risk_level':
                data_values.append(student.get('risk_level'))
            elif field == 'semester_data':
                data_values.append(student.get('semester_data', '{}'))
            elif field == 'feature_data':
                data_values.append(student.get('feature_data', '{}'))
            elif field == 'created_at':
                data_values.append(now)

        placeholders = ', '.join(['?'] * len(available_columns))
        columns_sql = ', '.join(available_columns)
        cursor = connection.execute(
            f'INSERT INTO students ({columns_sql}) VALUES ({placeholders})',
            tuple(data_values),
        )
        return cursor.lastrowid


def list_students(search='', prediction=''):
    query = 'SELECT * FROM students WHERE 1 = 1'
    parameters = []
    if search:
        query += ' AND (LOWER(student_name) LIKE LOWER(?) OR LOWER(COALESCE(student_id, \'\')) LIKE LOWER(?))'
        parameters.extend([f'%{search}%', f'%{search}%'])
    if prediction:
        query += ' AND prediction = ?'
        parameters.append(prediction)
    query += ' ORDER BY created_at DESC, id DESC'
    with get_connection() as connection:
        return [dict(row) for row in connection.execute(query, parameters).fetchall()]


def get_student(student_id):
    normalized_id = normalize_student_id(student_id)
    if not normalized_id:
        return None
    with get_connection() as connection:
        rows = connection.execute('SELECT * FROM students').fetchall()
        for row in rows:
            if normalize_student_id(row['student_id']).casefold() == normalized_id.casefold():
                return dict(row)
        return None


def get_student_by_credentials(student_id, student_name):
    normalized_id = normalize_student_id(student_id)
    normalized_name = ' '.join(str(student_name or '').split()).casefold()
    if not normalized_id or not normalized_name:
        return None
    student = get_student(normalized_id)
    if student is None:
        return None
    stored_name = ' '.join(str(student.get('student_name') or '').split()).casefold()
    return student if stored_name == normalized_name else None


def student_id_exists(student_id):
    normalized_id = normalize_student_id(student_id)
    if not normalized_id:
        return False
    with get_connection() as connection:
        rows = connection.execute('SELECT student_id FROM students').fetchall()
        return any(normalize_student_id(row['student_id']).casefold() == normalized_id.casefold() for row in rows)


def delete_student(student_id):
    normalized_id = normalize_student_id(student_id)
    if not normalized_id:
        return False
    with get_connection() as connection:
        rows = connection.execute('SELECT id, student_id FROM students').fetchall()
        matching_row = next(
            (row for row in rows if normalize_student_id(row['student_id']).casefold() == normalized_id.casefold()),
            None,
        )
        if matching_row is None:
            return False
        cursor = connection.execute('DELETE FROM students WHERE id = ?', (matching_row['id'],))
        return cursor.rowcount > 0


def create_faculty(faculty_id, faculty_name, password):
    normalized_id = str(faculty_id or '').strip().upper()
    normalized_name = str(faculty_name or '').strip()
    normalized_password = str(password or '')
    if not normalized_id or not normalized_name or not normalized_password:
        raise ValueError('Faculty ID, name, and password are required.')
    now = datetime.now().isoformat(timespec='seconds')
    with get_connection() as connection:
        cursor = connection.execute(
            '''INSERT INTO faculty (faculty_id, faculty_name, password_hash, created_at)
               VALUES (?, ?, ?, ?)''',
            (normalized_id, normalized_name, generate_password_hash(normalized_password), now),
        )
        return cursor.lastrowid


def get_faculty_by_credentials(faculty_id, faculty_name, password):
    normalized_id = str(faculty_id or '').strip().upper()
    normalized_name = str(faculty_name or '').strip().casefold()
    if not normalized_id or not normalized_name or not password:
        return None
    with get_connection() as connection:
        row = connection.execute(
            'SELECT * FROM faculty WHERE UPPER(TRIM(faculty_id)) = ? LIMIT 1',
            (normalized_id,),
        ).fetchone()
    if row and row['faculty_name'].strip().casefold() == normalized_name and check_password_hash(row['password_hash'], password):
        return dict(row)
    return None


def update_support(student_id, support_type, support_status, support_notes, action_taken='', follow_up_date=''):
    now = datetime.now().isoformat(timespec='seconds')
    with get_connection() as connection:
        existing = connection.execute(
            'SELECT id FROM support_records WHERE student_id = ? ORDER BY updated_at DESC, id DESC LIMIT 1',
            (student_id,),
        ).fetchone()
        if existing:
            connection.execute(
                     '''UPDATE support_records
                         SET support_type = ?, support_status = ?, support_notes = ?, action_taken = ?, follow_up_date = ?, updated_at = ?
                   WHERE id = ?''',
                     (support_type, support_status, support_notes, action_taken, follow_up_date, now, existing['id']),
            )
        else:
            connection.execute(
                '''INSERT INTO support_records
                         (student_id, support_type, support_status, support_notes, action_taken, follow_up_date, created_at, updated_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                     (student_id, support_type, support_status, support_notes, action_taken, follow_up_date, now, now),
            )


def get_latest_support(student_id):
    with get_connection() as connection:
        row = connection.execute(
            '''SELECT * FROM support_records
               WHERE student_id = ?
               ORDER BY updated_at DESC, id DESC LIMIT 1''',
            (student_id,),
        ).fetchone()
        return dict(row) if row else None


def update_student_prediction(student_id, current_year, current_semester, attendance, average_grade, backlogs, courses_passed, prediction, prediction_probability, ml_probabilities, risk_level, feature_data):
    normalized_id = normalize_student_id(student_id)
    if not normalized_id:
        return False
    with get_connection() as connection:
        cursor = connection.execute(
            '''UPDATE students
               SET current_year = ?, current_semester = ?, attendance = ?, average_grade = ?,
                   backlogs = ?, courses_passed = ?, prediction = ?, prediction_probability = ?,
                   ml_probabilities = ?, risk_level = ?, feature_data = ?
               WHERE LOWER(TRIM(student_id)) = ?''',
            (current_year, current_semester, attendance, average_grade, backlogs, courses_passed,
             prediction, prediction_probability, ml_probabilities, risk_level, feature_data,
             normalized_id.casefold()),
        )
        return cursor.rowcount > 0


def update_student_model_features(student_id, feature_data):
    normalized_id = normalize_student_id(student_id)
    if not normalized_id:
        return False
    with get_connection() as connection:
        cursor = connection.execute(
            'UPDATE students SET feature_data = ? WHERE LOWER(TRIM(student_id)) = ?',
            (feature_data, normalized_id.casefold()),
        )
        return cursor.rowcount > 0


def dashboard_summary():
    with get_connection() as connection:
        total = connection.execute('SELECT COUNT(*) AS value FROM students').fetchone()['value']
        attention = connection.execute("SELECT COUNT(*) AS value FROM students WHERE prediction = 'Dropout'").fetchone()['value']
        low_performance = connection.execute(
            'SELECT COUNT(*) AS value FROM students WHERE average_grade IS NOT NULL AND average_grade < 5'
        ).fetchone()['value']
        pending = connection.execute(
            '''SELECT COUNT(*) AS value FROM students s
               LEFT JOIN support_records sr ON sr.id = (
                   SELECT latest.id FROM support_records latest WHERE latest.student_id = s.id
                   ORDER BY latest.updated_at DESC, latest.id DESC LIMIT 1
               )
               WHERE COALESCE(sr.support_status, 'Pending') = 'Pending' ''',
        ).fetchone()['value']
        return {
            'total': total,
            'attention': attention,
            'low_performance': low_performance,
            'pending': pending,
        }
