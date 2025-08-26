import os

# Get the project root directory (parent of training/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'dataset_path': os.path.join(PROJECT_ROOT, 'training', 'data', 'train.csv'),
    'new_data_path': os.path.join(PROJECT_ROOT, 'training', 'data', 'new_training_data.csv'),
    'model_path': os.path.join(PROJECT_ROOT, 'inference', 'model.joblib'),
    'vectorizer_path': os.path.join(PROJECT_ROOT, 'inference', 'vectorizer.joblib')
}