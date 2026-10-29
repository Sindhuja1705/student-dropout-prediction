# Student Dropout Prediction System

## Project Title
Student Dropout Prediction System

## Project Objective
This project uses machine learning to identify students who may be at risk of dropping out based on academic and student-related information. The system acts as an early-warning and decision-support tool rather than a guaranteed prediction system.

## Features
- Compare the existing 11-feature baseline with academically engineered UCI features
- Impute missing values, remove duplicate records, scale numeric data, and one-hot encode categorical data inside a scikit-learn pipeline
- Compare Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, Gradient Boosting, and stacking
- Use a stratified holdout split and 5-fold stratified cross-validation with GridSearchCV on training data only
- Use leakage-safe train-only SMOTE, feature selection, tuning, and cross-validation
- Save the complete best preprocessing/model pipeline and feature metadata
- Provide separate student and faculty workflows with basic role-based access control
- Show actual class probabilities, attention status, model comparison, and confusion matrices

## Technologies
- Python
- Flask
- HTML
- CSS
- JavaScript
- Bootstrap
- scikit-learn
- XGBoost
- Pandas
- NumPy
- Joblib
- Google Colab

## ML Models
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

## Dataset Requirement
The uploaded dataset used for this project is a real student dropout dataset with the target column named `Target`.

Actual values in the dataset:
- Graduate
- Dropout
- Enrolled

To align with the project objective, the model interprets:
- Dropout = at risk / dropout case
- Graduate + Enrolled = not dropout / non-risk case

Important:
- The application preserves `Dropout`, `Enrolled`, and `Graduate`; it does not relabel the prediction as a binary result.
- The legacy prediction form exposes 11 fields, while the saved legacy model metadata expects 36 raw UCI fields. The app currently fills unavailable fields from dataset defaults or a row-position lookup; neither establishes that the values belong to the logged-in person.
- Student name and the app's display identifiers are not verified source-dataset identifiers and must not be treated as model features or student-level validation groups.
- Current Year and Current Semester are collected as display/context fields only because they are not columns in the current UCI training contract; they are not passed to the model.
- A faculty attention flag is derived from a predicted `Dropout` class or dropout probability of at least 50%; it is a support triage signal, not a diagnosis.

## Model Training

Install the pinned dependencies and run the application from the project
directory:

```bash
py -m pip install -r requirements.txt
py app.py
```

Run the unit suite:

```bash
py -m unittest discover -s tests -v
```

The safe local training command is:

```bash
py train_models.py
```

It fits all five model families on the 25 eligible after-semester-1 predictors
and writes a new, uniquely named bundle under
`models/candidates/after_semester1_*/`. Imputation, scaling, one-hot encoding,
class balancing, feature selection, and the inner-fold hyperparameter search
are in the pipeline. Existing production artifacts are never replaced. Each
saved candidate is reloaded and smoke-tested before the candidate bundle is
published. This is candidate preparation only: it does not select or promote a
winner and does not connect candidates to Flask.

The current Flask form and saved legacy artifact are not compatible with the
25-feature candidate contract. Do not copy a candidate into
`models/best_model.pkl`. Application integration requires a verified source
feature mapping and an input-contract migration; existing form defaults are
not a substitute for real values.

## After-Semester-1 Model Comparison

Run the isolated comparison from the project directory with:

```bash
py ml_training/after_semester1_model_comparison.py
```

This evaluates Logistic Regression, Random Forest, and XGBoost for the existing
three-class `Target` (`Dropout`, `Enrolled`, `Graduate`) using a stratified 80/20
holdout and five-fold stratified cross-validation on the training partition.
Imputation, scaling, and categorical encoding are fitted inside each training
fold. Class imbalance is handled with balanced class weights (or balanced
sample weights for XGBoost); SMOTE is not used on the integer-coded categories.

The after-semester-1 feature set contains 25 predictors: 19 enrollment/profile
fields and six first-semester fields. All second-semester fields are excluded.
`Debtor`, `Tuition fees up to date`, `Unemployment rate`, `Inflation rate`, and
`GDP` are also excluded because their observation timing is not established by
the dataset. Missing numeric values are median-imputed and missing categorical
values are mode-imputed within each fold. The inspected source CSV currently
has no missing predictor values.

Measured results for the checked-in dataset (macro precision/recall/F1; the
holdout is untouched by the cross-validation fits):

| Model | CV accuracy | CV macro precision | CV macro recall | CV macro F1 | Holdout accuracy | Holdout macro precision | Holdout macro recall | Holdout macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6889 | 0.6452 | 0.6465 | 0.6413 | 0.6881 | 0.6539 | 0.6559 | 0.6475 |
| Random Forest | 0.7126 | 0.6510 | 0.6352 | 0.6400 | 0.6994 | 0.6335 | 0.6217 | 0.6259 |
| XGBoost | 0.7101 | 0.6712 | 0.6693 | 0.6644 | 0.6802 | 0.6455 | 0.6436 | 0.6367 |

XGBoost had the highest cross-validation macro F1 among these fixed candidates;
Logistic Regression had the highest holdout macro F1. The confusion-matrix
class order is `Dropout`, `Enrolled`, `Graduate`:

| Model | Holdout confusion matrix |
|---|---|
| Logistic Regression | `[[184, 65, 35], [31, 89, 39], [27, 79, 336]]` |
| Random Forest | `[[190, 40, 54], [45, 56, 58], [29, 40, 373]]` |
| XGBoost | `[[180, 68, 36], [36, 85, 38], [22, 83, 337]]` |

These results do not establish an accuracy improvement over the older saved
comparison: its features include second-semester results and its reported
metrics were produced by a different training workflow. The benchmark writes
no model or results files and does not change the application. Its 25-feature
input contract is intentionally not promoted to the existing prediction form,
which currently expects 11 features.

## OULAD Compatibility and Leakage Controls

OULAD is not joined or concatenated with the current dataset. The existing CSV
has no student identifier, while OULAD uses `id_student` together with
`code_module` and `code_presentation`; those identifiers do not link the two
populations. The labels also differ: OULAD `final_result` is `Withdrawn`,
`Pass`, `Fail`, or `Distinction`, not the project's `Dropout`, `Enrolled`, and
`Graduate`.

For later OULAD experiments, use `studentInfo.csv` as the student-level base
and treat `final_result` only as the outcome. Join `studentAssessment.csv` to
`assessments.csv` and `studentVle.csv` to `vle.csv` by their OULAD keys, then
aggregate those event tables to the student/course-presentation level before
joining. Define a prediction cutoff first; exclude `date_unregistration` and
any assessment or VLE events after that cutoff to avoid outcome leakage. Keep
OULAD results as independent validation until feature and outcome definitions
have been harmonized and validated.

The inspected OULAD files contain 1,111 missing `imd_band` values in
`studentInfo.csv` and 173 missing `score` values in `studentAssessment.csv`.
Preserve missing `imd_band` as an explicit unknown category or impute it
inside each training fold. Do not treat a missing assessment score as an
ordinary low/average score without determining whether it means no submission;
retain a missing/submission indicator and fit any score imputation on training
data only. OULAD has 32,593 `studentInfo` rows but 28,785 distinct `id_student`
values, so any OULAD train/test split or cross-validation should group on
`id_student` (and stratify where feasible) to prevent one student's records
from appearing in both train and validation folds.

## How to Install Dependencies
Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run the Flask Application
From the project directory:

```bash
python app.py
```

Then open:

http://127.0.0.1:5000/

## How Prediction Works
1. The student logs in with only a display name.
2. The legacy form exposes 11 fields, but the saved model metadata expects 36 fields.
3. Flask fills unavailable values from defaults or a row-position lookup; these values are not verified as belonging to that student.
4. The saved legacy pipeline returns one of the three target classes and probabilities, but the current integration is not a reliable after-semester-1 prediction contract.
5. Do not use these outputs for consequential decisions; a feature mapping and application migration are required before using the after-semester-1 candidates.

## Application Workflows
Student: name-only login -> student dashboard -> enter the legacy form fields -> see a legacy model result (not verified as a student-specific after-semester-1 prediction).

Faculty: login -> faculty dashboard -> review outcome counts and model metrics -> search students -> inspect selected-student probabilities -> decide on supportive follow-up.

## Measured Model Comparison
The historical 11-feature figures below have been kept for traceability. The
`0.704225` and `0.725352` values are **Dropout recall**, not Dropout F1. Their
corresponding Dropout F1 values are `0.751880` for Random Forest and `0.743682`
for XGBoost, so Random Forest ranks higher under a Dropout-F1 criterion while
XGBoost has slightly higher accuracy. Those historical values do not match
the currently reproducible checked-in comparison CSV and should not be used
as a baseline for new experiments.

| Model | Accuracy | Precision | Recall | F1-score | Dropout recall | Dropout F1 | CV F1-macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.735593 | 0.680920 | 0.670165 | 0.674036 | 0.704225 | 0.751880 | 0.695660 |
| XGBoost | 0.736723 | 0.668278 | 0.651960 | 0.657004 | 0.725352 | 0.743682 | 0.669994 |
| Logistic Regression | 0.688136 | 0.680933 | 0.670788 | 0.657151 | 0.637324 | 0.725451 | 0.659327 |
| Decision Tree | 0.707345 | 0.671902 | 0.669179 | 0.662539 | 0.651408 | 0.715667 | 0.652262 |
| SVM | 0.702825 | 0.692862 | 0.683443 | 0.668837 | 0.616197 | 0.714286 | 0.678758 |

The historic selected model was **Random Forest** by the reported Dropout F1
column. Its confusion matrix uses rows and columns ordered as `Dropout`,
`Enrolled`, `Graduate`:

```text
[[200, 43, 41],
 [36, 71, 52],
 [12, 50, 380]]
```

### Reproducible Tuned Comparison

Run the evaluation-only benchmark from the project directory:

```bash
py ml_training/dropout_model_tuning.py
```

It uses the established after-semester-1 feature cutoff, a stratified 80/20
split with random state 42, and 5-fold stratified CV on the training split. All
second-semester fields and time-ambiguous fields are excluded. The CV selection
metric is Dropout F1. Imputation, scaling, one-hot encoding, and
`SelectKBest(f_classif)` are fitted inside each fold. This scores the
preprocessed numeric and one-hot encoded columns consistently, without
misclassifying a sparse mixed-feature matrix as entirely discrete during
mutual-information scoring. Logistic Regression and Random Forest compare
balanced class weights; XGBoost can use fold-local random oversampling, which
duplicates existing rows rather than interpolating integer-coded categories.
Randomized search explores hyperparameters and feature counts. No files are
written by this evaluation command.

Measured results (confusion matrices are rows=true, columns=predicted; class
order is `Dropout`, `Enrolled`, `Graduate`):

| Candidate | CV Dropout F1 (SD) | CV macro F1 | Holdout accuracy | Dropout precision | Dropout recall | Dropout F1 | Holdout macro F1 | Holdout confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline XGBoost | **0.7386 (0.0262)** | 0.6429 | 0.7107 | 0.7138 | 0.6937 | 0.7036 | 0.6019 | `[[197,31,56],[57,34,68],[22,22,398]]` |
| Tuned XGBoost | 0.7287 (0.0270) | 0.6165 | 0.7186 | 0.7218 | 0.7218 | 0.7218 | 0.5987 | `[[205,18,61],[59,28,72],[20,19,403]]` |
| Tuned Random Forest | 0.7238 (0.0225) | 0.6239 | 0.7073 | 0.7116 | 0.6690 | 0.6897 | 0.6034 | `[[190,30,64],[55,37,67],[22,21,399]]` |
| Baseline Random Forest | 0.7188 (0.0136) | 0.5995 | 0.6949 | 0.7143 | 0.6690 | 0.6909 | 0.5722 | `[[190,27,67],[51,25,83],[25,17,400]]` |
| Tuned Logistic Regression | 0.7183 (0.0121) | 0.6073 | 0.7209 | 0.7222 | 0.7324 | **0.7273** | 0.5972 | `[[208,16,60],[54,26,79],[26,12,404]]` |
| Baseline Logistic Regression | 0.6888 (0.0200) | 0.6411 | 0.6881 | 0.7593 | 0.6444 | 0.6971 | **0.6471** | `[[183,66,35],[31,89,39],[27,78,337]]` |

The CV selection rule picks baseline XGBoost by training-only Dropout F1.
Compared with baseline Random Forest, it scores higher on CV Dropout F1
(0.7386 vs. 0.7188) and on this holdout's accuracy, Dropout recall/F1, and
macro F1. Tuned Logistic Regression has the strongest holdout Dropout F1 and
macro F1, illustrating that holdout results do not uniformly rank candidates.
The CV score variability and this single split do not establish a robust
overall improvement. No new model is promoted or written to
`models/best_model.pkl`.

The main trainer now ranks candidates by training-only CV Dropout F1 (macro F1
is a tie-breaker), not by test accuracy. Its former feature-selection
experiments were removed from the trainer because the selector had been fitted
before CV, allowing each validation fold to influence its selected feature
set. Feature selection is retained in the isolated benchmark, where it is
fitted separately inside each CV fold. The trainer now compares tuned
full-feature candidates and does not use SMOTE on integer-coded categorical
features.

This evaluation uses an after-semester-1 feature set rather than the historical
11-feature model, because that historical set contains second-semester
results. It is not directly comparable to the historical table. Use the
reproducible table above for comparisons under the stated cutoff; do not claim
an improvement over the older numbers.

### Repeated Nested Cross-Validation

Run the five-model nested evaluation from the project directory:

```bash
py ml_training/repeated_nested_cv.py
```

This compares Random Forest, XGBoost, Logistic Regression, Decision Tree, and
SVM on the same eligible after-semester-1 feature columns and the same
three-class `Target` (`Dropout`, `Enrolled`, `Graduate`). It uses three repeats
of five stratified outer folds. Each outer training partition has a separate
three-fold stratified inner search; inner-CV Dropout F1 selects hyperparameters.
The evaluator prints all five requested metrics for every model and outer
fold, then per-model means, fold standard deviations, and corrected repeated-CV
95% t confidence intervals. These intervals use the repeated-CV correction
`sqrt((1/n + 1/(k-1)) * s^2)` for the standard error, rather than treating all
fold scores as independent observations.

Imputation, scaling, one-hot encoding, fold-local random oversampling for
XGBoost, feature selection, and the classifier are steps in one imbalanced-learn
pipeline searched only within the inner folds. Other models tune their
estimator's class-weight setting in that same pipeline. Outer-fold test data is
not used for feature selection, balancing, or hyperparameter choice. All
models reuse the same seeded outer split indices and, within each outer split,
the same seeded inner split indices.

The [official UCI dataset record](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
and its [published CSV schema](https://archive.ics.uci.edu/static/public/697/data.csv)
list 36 predictor fields and `Target`, with no student identifier. The checked-in
CSV likewise has no ID-like column. No legitimate student key is documented in
the source files inspected, so none is synthesized or inferred from feature
values. Exact duplicate rows are removed by the shared loader, but
student-level overlap cannot be definitively checked in the current data. If a
conventional student identifier is present in a future input, the script
excludes it from predictors and uses grouped stratified folds for both outer
evaluation and inner selection, explicitly checking group disjointness. The
current measurements therefore cannot be claimed to establish
student-independent generalization.

The Logistic Regression search explicitly uses the `newton-cg` solver. In the
installed scikit-learn/SciPy environment, the default `lbfgs` path emitted
`OptimizeWarning: Unknown solver options: iprint`; the explicit solver avoids
that warning without suppressing warnings. The complete post-fix nested run
finished without the `iprint` warning; the Logistic Regression results below
are refreshed from that run.

Measured results on the checked-in 4,424-row CSV (25 eligible predictors;
seed 42; three outer repeats x five folds; three inner folds; four randomized
hyperparameter samples per model per outer fold). Each cell gives mean
`(outer-fold SD) [corrected 95% CI]`:

| Model | Accuracy | Dropout precision | Dropout recall | Dropout F1 | Macro F1 |
|---|---|---|---|---|---|
| Random Forest | 0.7156 (0.0109) [0.7044, 0.7267] | 0.7254 (0.0165) [0.7085, 0.7422] | 0.7063 (0.0300) [0.6756, 0.7370] | 0.7153 (0.0177) [0.6972, 0.7334] | 0.5876 (0.0294) [0.5575, 0.6177] |
| XGBoost | 0.7058 (0.0133) [0.6922, 0.7195] | 0.7710 (0.0251) [0.7453, 0.7966] | 0.6486 (0.0278) [0.6202, 0.6771] | 0.7042 (0.0230) [0.6807, 0.7277] | **0.6596 (0.0145) [0.6447, 0.6745]** |
| Logistic Regression | **0.7171 (0.0146) [0.7022, 0.7320]** | 0.7275 (0.0217) [0.7053, 0.7496] | 0.7028 (0.0413) [0.6606, 0.7450] | 0.7141 (0.0225) [0.6910, 0.7371] | 0.6299 (0.0207) [0.6087, 0.6511] |
| Decision Tree | 0.6632 (0.0254) [0.6372, 0.6892] | 0.6741 (0.0372) [0.6361, 0.7121] | 0.6449 (0.0498) [0.5939, 0.6958] | 0.6574 (0.0280) [0.6288, 0.6861] | 0.5836 (0.0185) [0.5647, 0.6025] |
| SVM | 0.7165 (0.0200) [0.6960, 0.7369] | 0.7468 (0.0257) [0.7205, 0.7730] | 0.6845 (0.0390) [0.6446, 0.7244] | 0.7138 (0.0285) [0.6846, 0.7430] | 0.6103 (0.0273) [0.5824, 0.6382] |

Logistic Regression has the highest mean accuracy; Random Forest has the
highest mean Dropout recall and Dropout F1; XGBoost has the highest mean
Dropout precision and macro F1. The corrected confidence intervals overlap
substantially for several comparisons, so these results do not demonstrate a
decisive best model or a measured improvement over previous experiments. The
script reports metrics for all 15 outer folds per model, but does not save a
fitted model or promote a candidate. Because the CSV lacks student
identifiers, these results estimate row-level rather than verified
student-independent generalization. Older single-split tables above are
retained for traceability and are not directly comparable. The evaluation-only
script writes no model, dataset, or result files.

## Limitations and Future Enhancements
- Student prediction records and faculty support records are persisted in the local SQLite database `student_dropout.db`.
- The current login remains a simple name-only demo login; it is not an account-authentication system.
- The UCI data is observational; model probabilities do not prove causation and should not replace faculty judgment.
- Student names and IDs shown in the faculty demo are generated display identifiers for dataset rows.
- Future work can add hashed user accounts, persistent prediction history, calibrated probabilities, SHAP explanations, and a production data-governance review.

## Mentor Explanation
“The after-semester-1 experiment uses the same three-class target and common
repeated nested stratified folds for all five models. Preprocessing, feature
selection, resampling, and hyperparameter tuning stay inside training folds.
The CSV has no verified student key, so results measure row-level performance,
not confirmed student-independent generalization. The results show trade-offs
and do not establish a decisive winner. A safe training command can create
separate, smoke-tested candidate artifacts, but does not select or promote one.
The existing Flask model uses a different legacy feature contract and is not
connected to those candidates.”

## Project Folder Structure
```text
student_dropout_prediction/
├── app.py
├── requirements.txt
├── README.md
├── config.py
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── svm.pkl
│   ├── xgboost.pkl
│   ├── preprocessor.pkl
│   ├── feature_columns.pkl
│   └── model_results.csv
├── ml_training/
│   ├── student_dropout_training.ipynb
│   └── README.md
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   ├── performance.html
│   ├── about.html
│   └── contact.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── uploads/
│   └── .gitkeep
└── venv/
```

## Notes
- The application does not retrain models on startup.
- `py train_models.py` writes a new, non-promoted candidate bundle and does not overwrite production artifacts.
- The existing legacy model and prediction UI remain unchanged; they are not certified for after-semester-1 use.
- The system is designed for early warning and decision support only.
