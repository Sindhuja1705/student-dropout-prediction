import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

DATASET = r"d:\sdps\student_dropout_prediction\dataset\data.csv"
RANDOM_STATE = 42


def load_data():
    frame = pd.read_csv(DATASET, sep=";")
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.replace(r"^\s*$", np.nan, regex=True).drop_duplicates().dropna(subset=["Target"]).reset_index(drop=True)
    classes = sorted(frame["Target"].astype(str).str.strip().unique())
    mapping = {label: index for index, label in enumerate(classes)}
    y = frame["Target"].astype(str).str.strip().map(mapping)
    X = frame.drop(columns=["Target"])
    return X, y, classes


def add_academic_features(frame):
    out = frame.copy()
    s1_enrolled = out["Curricular units 1st sem (enrolled)"]
    s2_enrolled = out["Curricular units 2nd sem (enrolled)"]
    s1_evaluated = out["Curricular units 1st sem (evaluations)"]
    s2_evaluated = out["Curricular units 2nd sem (evaluations)"]
    s1_approved = out["Curricular units 1st sem (approved)"]
    s2_approved = out["Curricular units 2nd sem (approved)"]
    s1_grade = out["Curricular units 1st sem (grade)"]
    s2_grade = out["Curricular units 2nd sem (grade)"]
    s1_without_eval = out["Curricular units 1st sem (without evaluations)"]
    s2_without_eval = out["Curricular units 2nd sem (without evaluations)"]

    def safe_rate(numerator, denominator):
        denominator = denominator.replace(0, np.nan)
        return numerator.divide(denominator).fillna(0.0)

    out["Semester 1 pass rate"] = safe_rate(s1_approved, s1_evaluated)
    out["Semester 2 pass rate"] = safe_rate(s2_approved, s2_evaluated)
    out["Total approved units"] = s1_approved + s2_approved
    out["Total failed units"] = ((s1_enrolled - s1_approved).clip(lower=0) + (s2_enrolled - s2_approved).clip(lower=0))
    out["Total enrolled units"] = s1_enrolled + s2_enrolled
    out["Total evaluated units"] = s1_evaluated + s2_evaluated
    out["Total without evaluations"] = s1_without_eval + s2_without_eval
    out["Credit completion rate"] = safe_rate(out["Total approved units"], out["Total enrolled units"])
    out["Average semester grade"] = (s1_grade + s2_grade) / 2.0
    out["Grade improvement"] = s2_grade - s1_grade
    out["Academic progress"] = out["Semester 2 pass rate"] - out["Semester 1 pass rate"]
    out["Semester approval delta"] = s2_approved - s1_approved
    out["Semester grade delta"] = s2_grade - s1_grade
    out["Semester grade absolute delta"] = (s2_grade - s1_grade).abs()
    out["Pass rate stability"] = 1.0 - (out["Semester 1 pass rate"] - out["Semester 2 pass rate"]).abs()
    out["Performance strain"] = out["Total failed units"] / out["Total enrolled units"].replace(0, np.nan).fillna(1.0)
    out["Completion efficiency"] = out["Total approved units"] / out["Total evaluated units"].replace(0, np.nan).fillna(1.0)
    out["Enrollment persistence"] = ((s1_enrolled > 0) & (s2_enrolled > 0)).astype(int)
    return out


def evaluate_feature_set(name, X_train_eng, X_test_eng, y_train, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    score = cross_val_score(model, X_train_eng, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()
    model.fit(X_train_eng, y_train)
    preds = model.predict(X_test_eng)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: cv_accuracy={score:.6f}, test_accuracy={acc:.6f}")
    print(confusion_matrix(y_test, preds))
    return acc, score


X, y, classes = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
X_train_eng = add_academic_features(X_train)
X_test_eng = add_academic_features(X_test)

base_features = [
    "Gender",
    "Age at enrollment",
    "Course",
    "Previous qualification (grade)",
    "Admission grade",
    "Tuition fees up to date",
    "Scholarship holder",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
]
base_plus_academic = base_features + [
    "Semester 1 pass rate",
    "Semester 2 pass rate",
    "Total approved units",
    "Total failed units",
    "Credit completion rate",
    "Average semester grade",
    "Grade improvement",
    "Academic progress",
]
all_features = list(X_train_eng.columns)
selected = base_features + [
    "Semester 1 pass rate",
    "Semester 2 pass rate",
    "Total approved units",
    "Total failed units",
    "Total enrolled units",
    "Total evaluated units",
    "Credit completion rate",
    "Average semester grade",
    "Grade improvement",
    "Academic progress",
    "Semester approval delta",
    "Semester grade delta",
    "Semester grade absolute delta",
    "Pass rate stability",
    "Performance strain",
    "Completion efficiency",
    "Enrollment persistence",
]

for name, cols in [
    ("baseline", base_features),
    ("baseline_plus_existing_academic", base_plus_academic),
    ("selected", selected),
    ("all_eng", all_features),
]:
    evaluate_feature_set(name, X_train_eng[cols], X_test_eng[cols], y_train, y_test)
