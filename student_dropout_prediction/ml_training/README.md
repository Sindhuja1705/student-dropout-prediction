# Model Training Guide

The local training entry point is `../train_models.py`. It prepares a separate
candidate bundle for the after-semester-1 task; it does not overwrite
`models/best_model.pkl` or any other existing artifact.

## Setup and commands

From the project root:

```bash
py -m pip install -r requirements.txt
py -m unittest discover -s tests -v
py train_models.py
```

To run the Flask application, use `py app.py`.

## Candidate training protocol

- Uses the existing multiclass target: `Dropout`, `Enrolled`, and `Graduate`.
- Uses the same 25 cutoff-eligible features as
  `repeated_nested_cv.py`; second-semester and time-ambiguous fields are
  excluded.
- Fits Random Forest, XGBoost, Logistic Regression, Decision Tree, and SVM.
- Runs a seeded stratified inner search for each model using the same splits,
  selecting hyperparameters by cross-validated overall accuracy. The default
  search evaluates 20 parameter combinations per model.
  If a conventional student identifier exists, it is excluded from predictors
  and stratified group folds are used.
- Imputation, scaling, categorical encoding, balancing, feature selection, and
  the estimator are steps in the searched pipeline.
- Refits each model's best inner-search candidate on the full available
  dataset, then saves and reloads the artifact and validates a prediction.
- Creates a unique directory below `models/candidates/`; it never replaces
  existing files.

The candidate summary reports cross-validated accuracy, Dropout F1, and macro
F1 for each model's accuracy-selected parameters. This training command does
not rank across the five model families, declare a winner, or promote a model.
If the source CSV has no verified student ID, student-independent
generalization cannot be confirmed.

## Application compatibility

The existing Flask form exposes 11 values, but its saved legacy model metadata
expects 36 raw UCI fields. The app currently fills unavailable fields with
dataset defaults or a row-position lookup; those are not verified values for
the logged-in student. The candidate bundle has a 25-feature contract and is
not connected to Flask. Do not copy a candidate over `best_model.pkl`.

Before integration, define a verified mapping from actual application data to
all required candidate features, decide the input/user workflow, then add
feature-contract and prediction tests. Keep current production artifacts
unchanged until that work and independent validation are complete.

The notebook `student_dropout_training.ipynb` is historical exploratory work;
it is not the source of the current after-semester-1 candidate protocol.
