# ToxicTextScanner

ToxicTextScanner is a machine learning-based REST API for classifying short-form text (e.g., tweets) as `Hate Speech`, `Offensive Language`, or `Neutral / Clean`. It uses a `SGDClassifier` with TF-IDF features (`max_features=10000`, L2 regularization `alpha=0.0001`) for robust classification, supports incremental learning for model updates, and provides a Flask API for real-time inference.

## Project Structure
```
toxictext-scanner/
├── training/
│   ├── train_model.py           # Initial and incremental training with SGDClassifier
│   ├── config.py               # Configuration for dataset and model paths
│   ├── requirements.txt        # Dependencies (pandas, scikit-learn, nltk, etc.)
├── inference/
│   ├── toxic_text_scanner.py   # Core preprocessing and classification logic
│   ├── app.py                 # Flask API for text classification
│   ├── config.py              # API and model configuration
│   ├── requirements.txt       # Dependencies (flask, nltk, joblib)
│   ├── model.joblib           # Trained SGDClassifier model
│   ├── vectorizer.joblib      # Trained TF-IDF vectorizer
├── Dockerfile                  # Docker configuration for containerization
├── docker-compose.yml         # Orchestration for inference
├── README.md                  # This file
```

## Features
- **Text Classification**: Labels text as `Hate Speech` (0), `Offensive Language` (1), or `Neutral` (2) with confidence scores and dynamic explanations based on TF-IDF feature weights.
- **Incremental Learning**: Updates the `SGDClassifier` using `partial_fit`, preserving the initial TF-IDF vocabulary for consistency.
- **REST API**: Flask endpoint (`/classify`) for real-time classification.
- **Performance Metrics**: Reports precision, recall, F1-score, and accuracy after training/incremental training.
- **Docker Support**: Containerized setup for the inference API.

## Prerequisites
- Python 3.8+
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
     pip install -r requirements.txt
     ```
     Required: `pandas`, `scikit-learn`, `nltk`, `numpy`, `joblib`, `tqdm`
   - For inference:
     ```bash
     cd inference
     pip install -r requirements.txt
     ```
     Required: `flask`, `nltk`, `numpy`, `joblib`, `gunicorn`, `flask-limiter`

3. **Download NLTK Data**:
   - Run the following in a Python environment to download data needed for text preprocessing (tokenization, lemmatization) in `train_model.py`. This is required locally for training but is handled automatically in the Docker container for inference:
     ```python
     import nltk
     nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet'])
     ```
     Example: Save as `download_nltk.py` and run `python3 download_nltk.py`, or execute in a Python interpreter (`python3` then paste the code).

4. **Configure Paths**:
   - Edit `training/config.py`:
     ```python
     CONFIG = {
         'dataset_path': 'train.csv',                        # Initial dataset
         'new_data_path': '/Users/apple/Downloads/synthetic_toxic_tweets_1M.csv', # Incremental dataset
         'model_path': '../inference/model.joblib',
         'vectorizer_path': '../inference/vectorizer.joblib'
     }
     ```
   - Edit `inference/config.py`:
     ```python
     CONFIG = {
         'model_path': 'model.joblib',
         'vectorizer_path': 'vectorizer.joblib',
         'port': 5000,
         'debug': False  # Disabled for production
     }
     ```

## Training
### Initial Training
Trains a `SGDClassifier` (loss=`log_loss`, `max_iter=1000`, `alpha=0.0001`) with TF-IDF features (`max_features=10000`):
- Loads dataset (`train.csv`) with columns: `count`, `hate_speech_count`, `offensive_language_count`, `neither_count`, `class`/`label`, `tweet`.
- Preprocesses text: lowercase, remove URLs/mentions, tokenize, remove stopwords, lemmatize.
- Maps labels (`Hate Speech`/`hate_speech`→0, `Offensive Language`/`offensive`→1, `Neutral`/`neither`→2).
- Splits data (80% train, 20% test), trains the model, and evaluates performance.
- Saves `model.joblib` and `vectorizer.joblib` to `inference/`.

Run locally (without Docker):
```bash
cd training
python3 train_model.py
```

### Incremental Training
Updates the model with new data (e.g., `synthetic_toxic_tweets_1M.csv`) using `partial_fit`:
- Loads new data, applies same preprocessing, and uses existing vectorizer.
- Updates model weights in chunks (10,000 rows) for scalability.
- Evaluates performance on a 20% test split.
- Overwrites `model.joblib`.

Run locally (without Docker):
```bash
cd training
python3 train_model.py
```

**Example Output** (1M rows):
```
Loaded model from ../inference/model.joblib and vectorizer from ../inference/vectorizer.joblib
Loading dataset from /Users/apple/Downloads/synthetic_toxic_tweets_1M.csv...
Renaming label column 'label' to 'class'...
Mapping string labels to numeric values...
Preprocessing text...
Preprocessing: 100%|██████████| 1000000/1000000 [02:30<00:00, 6666.67it/s]
Performing incremental training...
Incremental Training: 100%|██████████| 80/80 [00:45<00:00,  1.78it/s]
Evaluating model...
Model Performance:
               precision    recall  f1-score   support
Hate Speech       0.95      0.94      0.95     66667
Offensive Language 0.96      0.95      0.96     66667
Neutral / Clean   0.97      0.98      0.97     66666
accuracy                            0.96    200000
macro avg          0.96      0.96      0.96    200000
weighted avg       0.96      0.96      0.96    200000
Model saved to ../inference/model.joblib
Vectorizer saved to ../inference/vectorizer.joblib
```

## Inference
1. **Run the Flask API Locally**:
   ```bash
   cd inference
   python3 app.py
   ```
   Starts server at `http://localhost:5000`.

2. **Test the API**:
   - Send a POST request to the `/classify` endpoint:
     ```bash
     curl -X POST http://localhost:5000/classify -H "Content-Type: application/json" -d '{"text": "I want to kill Charlie Sheen."}'
     ```
   - Expected response:
     ```json
     {
       "label": "Hate Speech",
       "confidence": 0.92,
       "explanation": "Flagged for hate speech based on terms: kill, want, sheen"
     }
     ```

## Dataset Structure

The dataset (`train.csv` or `new_data.csv`) must have:
- `count`: Total annotations (3–7).
- `hate_speech_count`: Annotations for hate speech.
- `offensive_language_count`: Annotations for offensive language.
- `neither_count`: Annotations for neutral.
- `class`: Label (0=Hate Speech, 1=Offensive Language, 2=Neutral, or strings: 'hate_speech', 'offensive', 'neither').
- `tweet`: Text content.

String labels are automatically mapped to numeric values in `train_model.py`.

## Next Steps

- **Performance**: Logistic Regression with TF-IDF is lightweight but struggles with contextual nuances. For better accuracy, consider a BERT-based model (requires `transformers`, PyTorch/TensorFlow, and GPU).


