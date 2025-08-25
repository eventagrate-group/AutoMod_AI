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


## Training
### Initial Training
The training process leverages several powerful libraries to handle data loading, preprocessing, feature extraction, model training, and evaluation in detail:

- **Data Loading and Manipulation with Pandas**: `pandas` is used to load the CSV dataset (`train.csv`) into a DataFrame, enabling efficient handling of structured data. It supports column renaming (e.g., renaming 'label' to 'class' for consistency), label mapping (converting string labels like 'Hate Speech' to numeric values 0, 1, or 2), and data shuffling or splitting. Pandas' capabilities ensure seamless data preparation, with functions like `pd.read_csv`, `df.rename`, and `df.map` providing robust data manipulation.

- **Text Preprocessing with NLTK**: The Natural Language Toolkit (NLTK) is employed for comprehensive text preprocessing. It includes tokenization (`word_tokenize`) to split text into words, stopword removal (`stopwords.words('english')`) to eliminate common words like 'the' or 'is' that add noise, and lemmatization (`WordNetLemmatizer`) to reduce words to their base form (e.g., 'running' to 'run'). NLTK's `nltk.download` ensures required corpora (e.g., 'punkt', 'punkt_tab', 'stopwords', 'wordnet') are available. This step is critical for cleaning the `tweet` column, reducing dimensionality, and improving model performance.

- **Feature Extraction with Scikit-Learn's TfidfVectorizer**: Scikit-learn (`sklearn`) provides the `TfidfVectorizer` for converting preprocessed text into TF-IDF features, which weigh term importance based on frequency in the document and rarity across the corpus. Parameters like `max_features=10000` limit vocabulary to the most relevant 10,000 terms, `ngram_range=(1, 2)` captures unigrams and bigrams for context (e.g., "want to"), and `min_df=2` ignores rare terms. This library's vectorization is essential for transforming text into numerical input for the classifier.

- **Model Training with Scikit-Learn's SGDClassifier**: Scikit-learn's `SGDClassifier` is the core classifier, using stochastic gradient descent for efficient training on large datasets. Parameters like `loss='log_loss'` enable probabilistic outputs, `max_iter=1000` and `tol=1e-3` control convergence, and `random_state=42` ensures reproducibility. The script supports incremental learning via `partial_fit` for updating the model with new data in chunks (e.g., 10,000 rows), making it scalable for large datasets like 1M rows. Scikit-learn's `train_test_split` splits data into 80/20 train/test sets, and `classification_report` evaluates performance with precision, recall, F1-score, and accuracy.

- **Progress Tracking with tqdm**: The `tqdm` library adds progress bars to long-running operations like preprocessing (`tqdm(df['tweet'])`) and chunked training (`tqdm(range(0, len(df), chunk_size))`), providing real-time feedback on completion status, which is crucial for large datasets.

- **Serialization with Joblib**: `joblib` is used to save (`joblib.dump`) and load (`joblib.load`) the trained model and vectorizer as `.joblib` files, enabling efficient persistence and loading of large objects like the TF-IDF matrix.

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

## Dockerization
Containerize the inference API for deployment on a CPU machine:

1. **Prepare Files**:
   - Ensure `inference/` contains `app.py`, `toxic_text_scanner.py`, `config.py`, `requirements.txt`, `model.joblib`, `vectorizer.joblib`.
   - Ensure `Dockerfile` and `docker-compose.yml` are in the project root.

2. **Build and Deploy**:
   - Install Docker and Docker Compose if not already installed:
     ```bash
     sudo apt update
     sudo apt install docker.io docker-compose
     sudo systemctl start docker
     sudo systemctl enable docker
     ```
   - Build the Docker image:
     ```bash
     docker build -t toxictext-scanner-inference .
     ```
   - Run the container:
     ```bash
     docker run -d -p 5000:5000 toxictext-scanner-inference
     ```
   - Alternatively, use Docker Compose:
     ```bash
     docker-compose up -d
     ```

3. **Test the API**:
   ```bash
   curl -X POST http://localhost:5000/classify -H "Content-Type: application/json" -d '{"text": "I want to kill Charlie Sheen."}'
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