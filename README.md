# ToxicTextScanner

ToxicTextScanner is a machine learning-based REST API for classifying short-form text (e.g., tweets) as `Hate Speech`, `Offensive Language`, or `Neutral / Clean`. It uses a `SGDClassifier` with TF-IDF features (`max_features=10000`, L2 regularization `alpha=0.0001`) for robust classification, supports incremental learning for model updates, and provides a Flask API for real-time inference. The system is designed to run locally to optimize cost, control, and data privacy. By training and deploying locally, you avoid recurring expenses from public payable APIs, maintain full control over model customization and updates, and ensure sensitive text data remains secure on your infrastructure. Local development also allows for faster iteration, tailored preprocessing, and integration with existing systems without reliance on external services.

## Project Structure
```
toxictext-scanner/
├── training/
│   ├── data/
│   │   ├── new_training_data.csv     # Training dataset
│   │   ├── hate_speech_verify.csv    # Validation data for Hate Speech
│   │   ├── offensive_language_verify.csv # Validation data for Offensive Language
│   │   ├── neutral_verify.csv        # Validation data for Neutral
│   ├── train_model.py                # Initial and incremental training with SGDClassifier
│   ├── generate_synthetic_data.py    # Synthetic data generation script
│   ├── config.py                    # Configuration for dataset and model paths
│   ├── requirements.txt             # Dependencies (pandas, scikit-learn, nltk, etc.)
├── inference/
│   ├── toxic_text_scanner.py        # Core preprocessing and classification logic
│   ├── app.py                      # Flask API for text classification
│   ├── config.py                   # API and model configuration
│   ├── requirements.txt            # Dependencies (flask, nltk, joblib)
│   ├── model.joblib                # Trained SGDClassifier model
│   ├── vectorizer.joblib           # Trained TF-IDF vectorizer
├── Dockerfile                      # Docker configuration for containerization
├── docker-compose.yml              # Orchestration for inference
├── README.md                       # This file
```

## Features
- **Text Classification**: Labels text as `Hate Speech` (0), `Offensive Language` (1), or `Neutral` (2) with confidence scores and dynamic explanations based on TF-IDF feature weights.
- **Incremental Learning**: Updates the `SGDClassifier` using `partial_fit`, preserving the initial TF-IDF vocabulary for consistency.
- **REST API**: Flask endpoint (`/classify`) for real-time classification.
- **Performance Metrics**: Reports precision, recall, F1-score, and accuracy after training/incremental training.
- **Docker Support**: Containerized setup for the inference API.

## Prerequisites
- Ubuntu 24.04.3 LTS
- Python 3.12
- Docker and Docker Compose (for inference containerization)
- Git
- Sufficient memory (8–16 GB RAM for training 1M rows)
- Internet connection for NLTK data

## Setup
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd toxictext-scanner
   ```

2. **Install Dependencies**:
   - For training:
     ```bash
     cd training
     python3.12 -m venv venv
     source venv/bin/activate
     pip install --upgrade pip
     pip install -r requirements.txt
     pip install torch==2.3.0+cu124 --index-url https://download.pytorch.org/whl/cu124
     ```
     Ensure `training/requirements.txt` contains:
     ```
     pandas==2.2.3
     scikit-learn==1.3.0
     nltk==3.9.1
     numpy==1.26.4
     joblib==1.4.2
     tqdm==4.66.5
     transformers==4.44.2
     ```
   - For inference:
     ```bash
     cd inference
     source ../training/venv/bin/activate
     pip install -r requirements.txt
     ```
     Ensure `inference/requirements.txt` contains:
     ```
     flask==3.0.3
     nltk==3.9.1
     numpy==1.26.4
     joblib==1.4.2
     gunicorn==23.0.0
     flask-limiter==3.8.0
     ```

3. **Download NLTK Data**:
   ```bash
   python3 -c "import nltk; nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'names'], download_dir='training/nltk_data')"
   ```

## Training
### Initial and Incremental Training
The training process leverages several libraries for efficient data processing and model updates:
- **Data Loading with Pandas**: Loads CSV datasets (e.g., `training/data/new_training_data.csv`) into DataFrames, handling column renaming and label mapping (e.g., 'Hate Speech' to 0).
- **Text Preprocessing with NLTK**: Uses tokenization (`word_tokenize`), stopword removal, and lemmatization (`WordNetLemmatizer`) to clean text data.
- **Feature Extraction with Scikit-Learn**: Applies `TfidfVectorizer` (`max_features=10000`, `ngram_range=(1, 2)`) to convert text into TF-IDF features.
- **Incremental Training with SGDClassifier**: Uses `partial_fit` to update the model in chunks (10,000 rows), supporting large datasets (e.g., 1M rows) and preserving existing TF-IDF vocabulary.
- **Progress Tracking with tqdm**: Displays progress bars for preprocessing and training.
- **Serialization with Joblib**: Saves the model (`model.joblib`) and vectorizer (`vectorizer.joblib`) to `inference/`.

Run training (initial or incremental):
```bash
cd training
source venv/bin/activate
python3 train_model.py
```
- If `model.joblib` and `vectorizer.joblib` exist, the script loads them for incremental updates.
- New data is processed in chunks, and the model is updated with `partial_fit`.
- Performance is evaluated on a 20% sample with precision, recall, F1-score, and accuracy.

**Example Output** (300,000 rows):
```
Loading existing model from inference/model.joblib...
Loading existing vectorizer from inference/vectorizer.joblib...
Starting incremental training with training/data/new_training_data.csv...
Processing chunk of size 10000...
Preprocessing chunk: 100%|██████████| 10000/10000 [00:01<00:00, 6666.67it/s]
Transforming text with existing vectorizer...
Updating model with partial_fit...
[Repeated for 30 chunks]
Evaluating model on a sample...
Preprocessing eval sample: 100%|██████████| 60000/60000 [00:09<00:00, 6666.67it/s]
Model Performance:
               precision    recall  f1-score   support
Hate Speech       0.92      0.92      0.92     20000
Offensive Language 0.96      0.96      0.96     20000
Neutral           0.90      0.90      0.90     20000
accuracy                            0.93     60000
macro avg         0.93      0.93      0.93     60000
weighted avg      0.93      0.93      0.93     60000
Model saved to inference/model.joblib
```

### Improving Accuracy
To further improve accuracy (currently 92% for Hate Speech, 96% for Offensive Language, 90% for Neutral), generate or source additional diverse training data with clear distinctions between classes using `generate_synthetic_data.py`. Retrain incrementally to update the model.

## Inference
1. **Run the Flask API Locally**:
   ```bash
   cd inference
   source ../training/venv/bin/activate
   gunicorn --workers=4 --bind=0.0.0.0:5001 --log-level=info app:app &
   ```
   Starts server at `http://localhost:5001`.

2. **Test the API**:
   ```bash
   curl -X POST http://localhost:5001/classify -H "Content-Type: application/json" -d '{"text": "I want to kill Charlie Sheen."}'
   ```
   Expected response:
   ```json
   {
     "label": "Hate Speech",
     "confidence": 0.92,
     "explanation": "Flagged for hate speech based on terms: kill, want, sheen",
     "influential_terms": ["kill", "want", "sheen"]
   }
   ```
   Stop Gunicorn:
   ```bash
   pkill -f gunicorn
   ```

## Dockerization
Containerize the inference API for CPU-based production deployment:
1. **Prepare Files**:
   - Ensure `inference/` contains `app.py`, `toxic_text_scanner.py`, `config.py`, `requirements.txt`, `model.joblib`, `vectorizer.joblib`.
   - Ensure `Dockerfile` and `docker-compose.yml` are in the project root.

2. **Build and Deploy**:
   ```bash
   cd /home/branch/projects/toxictext-scanner
   sudo apt update
   sudo apt install docker.io docker-compose
   sudo systemctl start docker
   sudo systemctl enable docker
   docker-compose up -d
   docker ps
   ```

3. **Test the API**:
   ```bash
   curl -X POST http://localhost:5001/classify -H "Content-Type: application/json" -d '{"text": "I want to kill Charlie Sheen."}'
   ```

4. **Stop Docker**:
   ```bash
   docker-compose down
   ```

## Dataset Structure
The dataset (e.g., `training/data/new_training_data.csv`) must have:
- `count`: Total annotations (e.g., 1).
- `hate_speech_count`: 1 for Hate Speech, 0 otherwise.
- `offensive_language_count`: 1 for Offensive Language, 0 otherwise.
- `neither_count`: 1 for Neutral, 0 otherwise.
- `class`: Label (0=Hate Speech, 1=Offensive Language, 2=Neutral, or strings: 'Hate Speech', 'Offensive Language', 'Neutral').
- `tweet`: Text content.

Validation files (`training/data/hate_speech_verify.csv`, `training/data/offensive_language_verify.csv`, `training/data/neutral_verify.csv`) contain one tweet per line with no headers. String labels are mapped to numeric values in `train_model.py` and `verify_model.py`.

## Troubleshooting
- **Low Model Accuracy**: Source more diverse data with distinct features for Hate Speech, Offensive Language, and Neutral using `generate_synthetic_data.py`. Retrain with `train_model.py`.
- **Missing Model Files**:
   ```bash
   ls -l inference/*.joblib
   ```
   Rerun `train_model.py` if missing.
- **Dependency Issues**:
   ```bash
   pip install pandas==2.2.3 scikit-learn==1.3.0 nltk==3.9.1 numpy==1.26.4 joblib==1.4.2 tqdm==4.66.5 transformers==4.44.2
   ```
- **Docker Issues**:
   ```bash
   docker logs toxictext-scanner
   ```

**Verification Code**: 2N8P6Z4X
