# ToxicTextScanner

ToxicTextScanner is a machine learning-based REST API for detecting hate speech, offensive language, and neutral content in short-form text (e.g., social media posts). It uses a Logistic Regression model with TF-IDF features to classify text, providing labels, confidence scores, and dynamic explanations based on influential TF-IDF terms. The project supports incremental learning to update the model with new data as it is acquired, ensuring adaptability to evolving datasets.

## Project Structure

- **training/**: Scripts for training and updating the model.
  - `train_model.py`: Handles initial training and incremental updates of the Logistic Regression model.
  - `config.py`: Specifies paths for datasets and model artifacts.
  - `requirements.txt`: Lists dependencies for training (e.g., scikit-learn, pandas, NLTK).
- **inference/**: Flask-based REST API for text classification.
  - `app.py`: Flask application with the `/classify` endpoint.
  - `toxic_text_scanner.py`: Core logic for text preprocessing and classification.
  - `config.py`: Specifies paths for model artifacts and API settings.
  - `requirements.txt`: Lists dependencies for inference (e.g., Flask, NLTK, gunicorn).

## Features

- **Text Classification**: Classifies text as Hate Speech (0), Offensive Language (1), or Neutral (2) with confidence scores and dynamic explanations derived from TF-IDF feature weights.
- **Incremental Learning**: Updates the model with new data using scikit-learn’s `partial_fit`, preserving the initial TF-IDF vocabulary.
- **REST API**: Provides a `/classify` endpoint for real-time text classification.

## Prerequisites

- Python 3.8+
- pip
- Git
- Dataset (`train.csv`) with columns: `count`, `hate_speech_count`, `offensive_language_count`, `neither_count`, `class`, `tweet`
- Sufficient memory for training (8–16 GB RAM for large datasets, e.g., 5M rows)

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
   - For inference:
     ```bash
     cd inference
     pip install -r requirements.txt
     ```

3. **Prepare the Dataset**:
   - Place `train.csv` in the `training` directory or update `training/config.py` with its path.
   - Ensure the dataset has the required columns: `count` (total annotations, 3–7), `hate_speech_count` (hate speech annotations), `offensive_language_count` (offensive language annotations), `neither_count` (neutral annotations), `class` (0=Hate Speech, 1=Offensive Language, 2=Neutral, or strings: 'hate_speech', 'offensive', 'neither'), and `tweet` (text content).

4. **Configure Training**:
   - Edit `training/config.py` to set dataset and model paths:
     ```python
     CONFIG = {
         'dataset_path': 'train.csv',      # Initial dataset
         'new_data_path': 'new_data.csv', # New data for incremental learning
         'model_path': '../inference/model.joblib',
         'vectorizer_path': '../inference/vectorizer.joblib'
     }
     ```

## Training the Model

### Initial Training
The initial training process builds a Logistic Regression model with TF-IDF features to classify text into Hate Speech, Offensive Language, or Neutral categories. The process involves:

- **Data Loading**: Loads `train.csv` (specified in `config.py`) using pandas.
- **Preprocessing**:
  - Converts tweets to lowercase.
  - Removes URLs, usernames (e.g., @user), and special characters.
  - Tokenizes text using NLTK’s `word_tokenize`.
  - Removes English stopwords and applies lemmatization with `WordNetLemmatizer`.
- **Feature Extraction**: Uses `TfidfVectorizer` (max 5000 features) to transform preprocessed tweets into TF-IDF feature vectors.
- **Label Handling**: Maps string labels (e.g., 'hate_speech', 'offensive', 'neither') to numeric values (0, 1, 2) if present.
- **Model Training**:
  - Splits data into 80% training and 20% testing sets.
  - Trains a multinomial Logistic Regression model (`max_iter=1000`) on the training set.
  - Evaluates performance on the test set using a classification report (precision, recall, F1-score).
- **Artifact Saving**: Saves the trained model (`model.joblib`) and vectorizer (`vectorizer.joblib`) to the `inference` directory for use by the API.

To perform initial training:
```bash
cd training
python train_model.py
```

**Example Output**:
```
Model Performance:
                  precision    recall  f1-score   support
Hate Speech       0.85      0.80      0.82      1000
Offensive Language 0.78      0.75      0.76      1200
Neutral / Clean   0.90      0.92      0.91      1800
accuracy                            0.84      4000
Model saved to ../inference/model.joblib
Vectorizer saved to ../inference/vectorizer.joblib
```

### Incremental Learning
Incremental learning allows the model to adapt to new data without retraining from scratch, using scikit-learn’s `partial_fit` method. This is useful for updating the model as new data is acquired (e.g., additional tweets). The process involves:

- **Data Loading**: Loads `new_data.csv` (specified in `config.py`) with the same structure as `train.csv`.
- **Preprocessing**: Applies the same preprocessing steps (lowercase, remove URLs/usernames, tokenize, remove stopwords, lemmatize).
- **Feature Extraction**: Uses the existing `TfidfVectorizer` (loaded from `vectorizer.joblib`) to transform new data, preserving the initial vocabulary to ensure consistency.
- **Label Handling**: Maps string labels to numeric values if needed, matching the initial training setup.
- **Model Update**: Uses `partial_fit` to update the Logistic Regression model weights with the new data, maintaining compatibility with classes [0, 1, 2].
- **Artifact Saving**: Overwrites `model.joblib` with the updated model, keeping `vectorizer.joblib` unchanged.

To perform incremental learning:
- Save new data as `new_data.csv` in the `training` directory.
- Update `training/config.py` with the path to `new_data.csv`.
- Run:
  ```bash
  cd training
  python train_model.py
  ```

If `model.joblib` and `new_data.csv` exist, the script performs incremental learning. Otherwise, it falls back to initial training on `train.csv`.

**Notes**:
- Ensure `new_data.csv` matches the structure of `train.csv`.
- Incremental learning updates model weights but does not retrain the TF-IDF vocabulary, ensuring consistency with the inference API.
- For large datasets, multiple `partial_fit` iterations may improve convergence. Adjust `max_iter` in `train_model.py` if needed.

## Running the Inference API Locally

1. **Navigate to Inference Directory**:
   ```bash
   cd inference
   ```

2. **Run the Flask App**:
   ```bash
   python app.py
   ```

3. **Test the API**:
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

## Notes

- **Incremental Learning**: Ensure `new_data.csv` has sufficient examples of hate speech (e.g., threats like "kill") to improve classification accuracy. Check label distribution:
  ```python
  import pandas as pd
  df = pd.read_csv('training/train.csv')
  print(df['class'].value_counts())
  print(df[df['tweet'].str.contains('kill', case=False)]['class'].value_counts())
  ```
- **Performance**: Logistic Regression with TF-IDF is lightweight but may struggle with contextual nuances. For better accuracy, consider a BERT-based model (requires `transformers`, PyTorch/TensorFlow, and GPU).
- **Scalability**: For high traffic, consider running the Flask app with a production server like `gunicorn`:
  ```bash
  gunicorn app:app -w 4
  ```

## Troubleshooting

- **NLTK Errors**: Ensure an internet connection for downloading `punkt`, `punkt_tab`, `stopwords`, and `wordnet`. Run manually if needed:
  ```python
  import nltk
  nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet'])
  ```
- **Missing Model Files**: Run `train_model.py` to generate `model.joblib` and `vectorizer.joblib`.
- **Label Issues**: Verify the `class` column mapping in `train_model.py` matches your dataset labels.
- **Memory Issues**: Large datasets (e.g., 5M rows) require 8–16 GB RAM. Reduce dataset size if needed.

