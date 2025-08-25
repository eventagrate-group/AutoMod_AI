ToxicTextScanner
ToxicTextScanner is a machine learning-based REST API for classifying short-form text (e.g., tweets) as Hate Speech, Offensive Language, or Neutral. It uses a SGDClassifier with TF-IDF features (max_features=20000, ngram_range=(1,2), sublinear_tf=True) for robust classification, supports full retraining with hyperparameter tuning via GridSearchCV, and provides a Flask API for real-time inference.
Project Structure
toxictext-scanner/
├── training/
│   ├── train_model.py           # Full training with SGDClassifier and GridSearchCV
│   ├── generate_synthetic_data.py # Generates synthetic dataset with rule-based and GPT-2 methods
│   ├── verify_model.py          # Verifies model performance on diverse test cases
│   ├── config.py               # Configuration for dataset and model paths
│   ├── requirements.txt        # Dependencies (pandas, scikit-learn, nltk, transformers, etc.)
├── inference/
│   ├── toxic_text_scanner.py   # Core preprocessing and classification logic
│   ├── app.py                 # Flask API for text classification
│   ├── config.py              # API and model configuration
│   ├── requirements.txt       # Dependencies (flask, nltk, joblib, gunicorn)
│   ├── model.joblib           # Trained SGDClassifier model
│   ├── vectorizer.joblib      # Trained TF-IDF vectorizer
├── Dockerfile                  # Docker configuration for containerization
├── docker-compose.yml         # Orchestration for inference
├── README.md                  # This file

Features

Text Classification: Labels text as Hate Speech (0), Offensive Language (1), or Neutral (2) with confidence scores and dynamic explanations based on TF-IDF feature weights.
Full Retraining: Retrains model from scratch with GridSearchCV for optimal parameters (alpha, class_weight), ensuring robust feature space updates.
Synthetic Data Generation: Creates diverse datasets (e.g., 300,000 rows, 100,000 per class) using 80% rule-based templates and 20% GPT-2 generated text.
REST API: Flask endpoint (/classify) for real-time classification.
Performance Metrics: Reports precision, recall, F1-score, and accuracy after training.
Docker Support: Containerized setup for the inference API.

Prerequisites

Python 3.12+
Docker and Docker Compose (for inference containerization)
Git
GPU with CUDA 12.6 for data generation (optional, CPU-compatible)
Sufficient memory (16–32 GB RAM for training 300,000 rows)
Internet connection for NLTK and Hugging Face data

Setup

Clone the Repository:
git clone <repository-url>
cd toxictext-scanner


Install Dependencies:

For training and data generation:cd training
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121

Required: pandas==2.2.3, scikit-learn==1.3.0, nltk==3.9.1, numpy==1.26.4, joblib==1.4.2, tqdm==4.66.5, transformers==4.44.2, torch==2.3.0+cu121
For inference:cd inference
pip install -r requirements.txt

Required: flask, nltk, numpy, joblib, gunicorn, flask-limiter


Download NLTK Data:
import nltk
nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'names'], download_dir='/home/branch/nltk_data')

Save as download_nltk.py and run python3 download_nltk.py.

Configure Paths:

Edit training/config.py:CONFIG = {
    'dataset_path': '/home/branch/projects/toxictext-scanner/training/train.csv',
    'new_data_path': '/home/branch/Downloads/new_training_data.csv',
    'model_path': '/home/branch/projects/toxictext-scanner/inference Sheldon
    'vectorizer_path': '/home/branch/projects/toxictext-scanner/inference/vectorizer.joblib'
}


Edit inference/config.py:CONFIG = {
    'model_path': 'model.joblib',
    'vectorizer_path': 'vectorizer.joblib',
    'port': 5001,
    'debug': False  # Disabled for production
}





Data Generation
Generate a synthetic dataset (300,000 rows, 100,000 per class) with 80% rule-based templates and 20% GPT-2 generated text for diversity:
cd training
source venv/bin/activate
python3 generate_synthetic_data.py

Output: /home/branch/Downloads/new_training_data.csv (~45-60 minutes with RTX 4090).
Verify Dataset:
head -n 10 /home/branch/Downloads/new_training_data.csv
python3 -c "import pandas as pd; df=pd.read_csv('/home/branch/Downloads/new_training_data.csv'); print(df['class'].value_counts())"

Training
Full Training
Trains a SGDClassifier (loss=log_loss, max_iter=1000, tuned alpha and class_weight) with TF-IDF features (max_features=20000, ngram_range=(1,2), sublinear_tf=True):

Loads dataset (/home/branch/Downloads/new_training_data.csv) with columns: count, hate_speech_count, offensive_language_count, neither_count, class, tweet.
Preprocesses text: lowercase, remove URLs/mentions, tokenize, remove stopwords, lemmatize.
Maps labels (Hate Speech→0, Offensive Language→1, Neutral→2).
Splits data (80% train, 20% test), tunes hyperparameters with GridSearchCV, trains the model, and evaluates performance.
Saves model.joblib and vectorizer.joblib to inference/.

Run locally:
cd training
source venv/bin/activate
python3 train_model.py

Example Output (300,000 rows):
Removed existing model: /home/branch/projects/toxictext-scanner/inference/model.joblib
Removed existing vectorizer: /home/branch/projects/toxictext-scanner/inference/vectorizer.joblib
Performing full training with /home/branch/Downloads/new_training_data.csv...
Loading dataset from /home/branch/Downloads/new_training_data.csv...
Mapping string labels to numeric values...
Preprocessing text...
Preprocessing: 100%|██████████| 300000/300000 [00:50<00:00, 6000.00it/s]
Vectorizing text...
Performing hyperparameter tuning...
Best parameters: {'alpha': 1e-4, 'class_weight': 'balanced'}
Evaluating model...
Model Performance:
               precision    recall  f1-score   support
Hate Speech       0.99      0.99      0.99     20000
Offensive Language 0.99      0.99      0.99     20000
Neutral           0.99      0.99      0.99     20000
accuracy                            0.99     60000
macro avg          0.99      0.99      0.99     60000
weighted avg       0.99      0.99      0.99     60000
Model saved to /home/branch/projects/toxictext-scanner/inference/model.joblib
Vectorizer saved to /home/branch/projects/toxictext-scanner/inference/vectorizer.joblib

Verification
Verify model performance on a diverse test set (300 samples):
cd training
source venv/bin/activate
python3 verify_model.py

Example Output:
Classification Summary:
Hate Speech: 99/100 correct (99.00%)
Offensive Language: 98/100 correct (98.00%)
Neutral: 100/100 correct (100.00%)
...

Inference

Run the Flask API Locally:
cd inference
source ../training/venv/bin/activate
gunicorn --workers=4 --bind=0.0.0.0:5001 --log-level=info app:app


Run the Inference API as a Docker Container (CPU Machine):

Ensure Docker and Docker Compose are installed:sudo apt update
sudo apt install docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker


Build and run the Docker container:cd toxictext-scanner
docker build -t toxictext-scanner-inference -f Dockerfile .
docker run -d -p 5001:5001 --name toxictext-scanner toxictext-scanner-inference

Note: The Dockerfile and docker-compose.yml are configured for CPU compatibility, using gunicorn with 4 workers and python3.12 base image.
Verify the container is running:docker ps




Test the API:
curl -X POST http://localhost:5001/classify -H "Content-Type: application/json" -d '{"text": "I want to kill Charlie Sheen."}'

Expected response:
{
  "label": "Hate Speech",
  "confidence": 0.98,
  "explanation": "Flagged for hate speech based on terms: kill, hate, die"
}



Dataset Structure
The dataset (train.csv or new_training_data.csv) must have:

count: Total annotations (e.g., 1).
hate_speech_count: Annotations for hate speech (0 or 1).
offensive_language_count: Annotations for offensive language (0 or 1).
neither_count: Annotations for neutral (0 or 1).
class: Label (Hate Speech, Offensive Language, Neutral).
tweet: Text content.

String labels are mapped to numeric values in train_model.py (0=Hate Speech, 1=Offensive Language, 2=Neutral).
Troubleshooting

Verify Dataset:head -n 10 /home/branch/Downloads/new_training_data.csv
python3 -c "import pandas as pd; df=pd.read_csv('/home/branch/Downloads/new_training_data.csv'); print(df['class'].value_counts())"


Check Dependencies:pip show transformers torch scikit-learn


Monitor Docker Container:docker logs toxictext-scanner


