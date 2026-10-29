import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')

MODEL_FILES = {
    'logistic_regression': os.path.join(MODELS_DIR, 'logistic_regression.pkl'),
    'decision_tree': os.path.join(MODELS_DIR, 'decision_tree.pkl'),
    'random_forest': os.path.join(MODELS_DIR, 'random_forest.pkl'),
    'svm': os.path.join(MODELS_DIR, 'svm.pkl'),
    'xgboost': os.path.join(MODELS_DIR, 'xgboost.pkl'),
    'best_model': os.path.join(MODELS_DIR, 'best_model.pkl'),
    'experiment2_best_candidate': os.path.join(MODELS_DIR, 'experiment2_best_candidate.pkl'),
    'preprocessor': os.path.join(MODELS_DIR, 'preprocessor.pkl'),
    'feature_columns': os.path.join(MODELS_DIR, 'feature_columns.pkl'),
    'model_results': os.path.join(MODELS_DIR, 'model_results.csv'),
    'evaluation_details': os.path.join(MODELS_DIR, 'evaluation_details.pkl'),
    'notebook_model_results': os.path.join(BASE_DIR, 'ml_training', 'notebook_outputs', 'model_results.csv'),
    'notebook_evaluation_details': os.path.join(BASE_DIR, 'ml_training', 'notebook_outputs', 'evaluation_details.pkl'),
}

MISSING_MODELS_MESSAGE = (
    'Trained model files are missing. Please run the Google Colab training notebook '
    'and copy the generated model files into the models folder.'
)
