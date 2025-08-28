# ToxicTextScanner

ToxicTextScanner is a machine learning-based REST API for classifying short-form text (e.g., tweets) in **English** and **Arabic** as `Hate Speech`, `Offensive Language`, or `Neutral / Clean`. It leverages a `SGDClassifier` with TF-IDF features (`max_features=10000`, L2 regularization `alpha=0.0001`) for robust classification, supports incremental learning for model updates, and provides a Flask API for real-time inference. The system uses NLTK for English preprocessing and Stanza for Arabic preprocessing, with GPU acceleration for training to handle large datasets efficiently. Designed to run locally, it optimizes cost, control, and data privacy, avoiding recurring expenses from public APIs, ensuring full control over model customization, and keeping sensitive text data secure on our infrastructure. Local development enables faster iteration, tailored preprocessing, and seamless integration with existing systems.

## Project Structure
```
toxictext-scanner/
├── training/
│   ├── data/
│   │   ├── new_training_data.csv           # English training dataset
│   │   ├── hate_speech_verify.csv          # English validation data for Hate Speech
│   │   ├── offensive_language_verify.csv   # English validation data for Offensive Language
│   │   ├── neutral_verify.csv             # English validation data for Neutral
│   ├── data_arabic/
│   │   ├── new_training_data.csv           # Arabic training dataset (1M rows)
│   │   ├── hate_speech_verify.csv          # Arabic validation data for Hate Speech
│   │   ├── offensive_language_verify.csv   # Arabic validation data for Offensive Language
│   │   ├── neutral_verify.csv             # Arabic validation data for Neutral
│   ├── train_model.py                     # English training with SGDClassifier
│   ├── train_model_ar.py                  # Arabic training with SGDClassifier and Stanza
│   ├── verify_model.py                    # Verification for English and Arabic models
│   ├── generate_synthetic_data.py         # Synthetic data generation script
│   ├── config.py                         # Configuration for dataset and model paths
│   ├── requirements.txt                  # Dependencies (pandas, scikit-learn, nltk, stanza, etc.)
├── inference/
│   ├── toxic_text_scanner.py             # Core preprocessing and classification (English and Arabic)
│   ├── app.py                           # Flask API for text classification
│   ├── config.py                        # API and model configuration
│   ├── requirements.txt                 # Dependencies (flask, nltk, joblib, etc.)
│   ├── model.joblib                     # Trained SGDClassifier model (English)
│   ├── vectorizer.joblib                # Trained TF-IDF vectorizer (English)
│   ├── model_ar.joblib                  # Trained SGDClassifier model (Arabic)
│   ├── vectorizer_ar.joblib             # Trained TF-IDF vectorizer (Arabic)
│   ├── ecosystem.config.js              # PM2 configuration for production
│   ├── nltk_data/                       # NLTK data (stopwords, punkt_tab, etc.)
├── README.md                            # This file
```

## Features
- **Text Classification**: Labels text in **English** and **Arabic** as `Hate Speech` (0), `Offensive Language` (1), or `Neutral` (2) with confidence scores and dynamic explanations based on TF-IDF feature weights.
- **Language Detection**: Uses `langdetect` to automatically detect English or Arabic input for appropriate preprocessing and model selection.
- **Preprocessing**:
  - **English**: NLTK tokenization (`word_tokenize`), stopword removal, and lemmatization (`WordNetLemmatizer`).
  - **Arabic**: Stanza pipeline for tokenization and lemmatization (GPU-enabled for training, CPU for inference), with NLTK fallback if Stanza fails.
- **Incremental Learning**: Updates the `SGDClassifier` using `partial_fit`, preserving the initial TF-IDF vocabulary for consistency.
- **REST API**: Flask endpoints (`/classify` for text classification, `/health` for service status) for real-time inference.
- **Performance Metrics**: Reports precision, recall, F1-score, and accuracy after training and verification.
- **PM2 Deployment**: Uses PM2 for process management, enabling easy scaling and monitoring in production.

## Prerequisites
- Ubuntu 24.04.3 LTS
- Python 3.12
- Node.js and npm (for PM2)
- Nginx (for reverse proxy)
- Git
- Sufficient memory (8–16 GB RAM for training 1M rows)
- GPU with CUDA support (recommended for English and Arabic training with Stanza)
- Internet connection for NLTK and Stanza data

## Setup
1. **Clone the Repository**:
   ```bash
   git clone git@github.com:eventagrate-group/toxictext-scanner.git
   cd toxictext-scanner
   ```

2. **Install System Dependencies**:
   ```bash
   sudo apt update
   sudo apt install -y python3-pip python3-venv nodejs npm nginx
   ```

3. **Install Python Dependencies**:
   - For training:
     ```bash
     cd training
     python3 -m venv venv
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
     stanza==1.10.0
     ```
   - For inference:
     ```bash
     cd inference
     python3 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     pip install gunicorn==23.0.0
     ```
     Ensure `inference/requirements.txt` contains:
     ```
     flask==3.0.3
     nltk==3.9.1
     numpy==1.26.4
     joblib==1.4.2
     gunicorn==23.0.0
     flask-limiter==3.8.0
     scikit-learn==1.5.2
     langdetect==1.0.9
     stanza==1.10.0
     ```

4. **Download NLTK and Stanza Data**:
   ```bash
   cd inference
   source venv/bin/activate
   python3 -m nltk.downloader -d ./nltk_data stopwords wordnet punkt_tab
   python3 -c "import stanza; stanza.download('ar', processors='tokenize,lemma')"
   ```

## Training
### English Training
The English training process uses NLTK for preprocessing and Scikit-Learn for feature extraction and classification:
- **Data Loading with Pandas**: Loads CSV datasets (e.g., `training/data/new_training_data.csv`) into DataFrames, handling column renaming and label mapping (e.g., 'Hate Speech' to 0).
- **Text Preprocessing with NLTK**: Uses tokenization (`word_tokenize`), stopword removal, and lemmatization (`WordNetLemmatizer`) to clean text data.
- **Feature Extraction with Scikit-Learn**: Applies `TfidfVectorizer` (`max_features=10000`, `ngram_range=(1, 2)`) to convert text into TF-IDF features.
- **Incremental Training with SGDClassifier**: Uses `partial_fit` to update the model in chunks (10,000 rows), supporting large datasets (e.g., 1M rows) and preserving existing TF-IDF vocabulary.
- **Progress Tracking with tqdm**: Displays progress bars for preprocessing and training.
- **Serialization with Joblib**: Saves the model (`model.joblib`) and vectorizer (`vectorizer.joblib`) to `inference/`.

Run English training:
```bash
cd training
source venv/bin/activate
python3 train_model.py
```
- If `model.joblib` and `vectorizer.joblib` exist, the script loads them for incremental updates.
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

### Arabic Training
The Arabic training process uses Stanza for preprocessing (GPU-enabled for faster training) and the same Scikit-Learn pipeline as English:
- **Data Loading with Pandas**: Loads CSV dataset (`training/data_arabic/new_training_data.csv`, 1M rows, 333,333 per class).
- **Text Preprocessing with Stanza**: Uses Stanza for tokenization and lemmatization with GPU support for training and CPU for inference. Falls back to NLTK if Stanza fails.
- **Feature Extraction and Training**: Same as English, with separate `model_ar.joblib` and `vectorizer_ar.joblib`.
- **Serialization with Joblib**: Saves the model (`model_ar.joblib`) and vectorizer (`vectorizer_ar.joblib`) to `inference/`.

Run Arabic training:
```bash
cd training
source venv/bin/activate
python3 train_model_ar.py
```

**Example Output** (1,000,000 rows):
```
Initializing Stanza pipeline for Arabic on cuda...
...
Initializing new vectorizer...
Initializing new classifier...
Starting incremental training with training/data_arabic/new_training_data.csv...
Processing chunk of size 10000...
Preprocessing chunk: 100%|██████████| 10000/10000 [00:25<00:00, 400.00it/s]
Fitting vectorizer on first chunk...
Vectorizer saved to inference/vectorizer_ar.joblib
Updating model with partial_fit...
[Repeated for 100 chunks]
Model saved to inference/model_ar.joblib
Evaluating model on a sample...
Model Performance:
               precision    recall  f1-score   support
Hate Speech       0.82      0.81      0.81     66666
Offensive Language 0.85      0.86      0.85     66666
Neutral           0.83      0.83      0.83     66666
accuracy                            0.83    200000
macro avg         0.83      0.83      0.83    200000
weighted avg      0.83      0.83      0.83    200000
```

### Verification
Verify the performance of both English and Arabic models using `verify_model.py`, which evaluates on validation datasets (`training/data/*.csv` for English, `training/data_arabic/*.csv` for Arabic):
```bash
cd training
source venv/bin/activate
python3 verify_model.py
```

**Example Output**:
```
Verifying English model...
Loading en validation data from training/data/hate_speech_verify.csv...
Loading en validation data from training/data/offensive_language_verify.csv...
Loading en validation data from training/data/neutral_verify.csv...
EN Classification Summary:
Hate Speech: 276/300 correct (92.00%)
Offensive Language: 288/300 correct (96.00%)
Neutral: 270/300 correct (90.00%)
...
Verifying Arabic model...
Initializing Stanza pipeline for Arabic on cpu...
Loading ar validation data from training/data_arabic/hate_speech_verify.csv...
Loading ar validation data from training/data_arabic/offensive_language_verify.csv...
Loading ar validation data from training/data_arabic/neutral_verify.csv...
AR Classification Summary:
Hate Speech: 243/300 correct (81.00%)
Offensive Language: 258/300 correct (86.00%)
Neutral: 249/300 correct (83.00%)
```

### Improving Accuracy
- **English**: To improve accuracy (currently 92% for Hate Speech, 96% for Offensive Language, 90% for Neutral), generate or source additional diverse training data using `generate_synthetic_data.py`. Retrain incrementally with `train_model.py`.
- **Arabic**: To achieve >80% accuracy and non-zero recall for Hate Speech, ensure consistent Stanza preprocessing between training and inference. Source additional diverse Arabic data if needed.

## Inference
The Flask API uses `langdetect` to automatically detect the input language (English or Arabic) and applies the appropriate model (`model.joblib` for English, `model_ar.joblib` for Arabic) via the same `/classify` endpoint, ensuring seamless bilingual support.

1. **Run the Flask API Manually (for Testing)**:
   ```bash
   cd inference
   source venv/bin/activate
   python3 app.py
   ```
   Tests:
   ```bash
   curl -X GET http://localhost:5001/health
   curl -X POST http://localhost:5001/classify -H "Content-Type: application/json" -d '{"text": "I want to kill Charlie Sheen."}'
   curl -X POST http://localhost:5001/classify -H "Content-Type: application/json" -d '{"text": "أريد قتل شخص معين"}'
   ```

2. **Install PM2 (for Production)**:
   ```bash
   sudo npm install -g pm2
   ```

3. **Run the Flask API with PM2**:
   ```bash
   cd inference
   source venv/bin/activate
   pm2 start ecosystem.config.js
   pm2 save
   pm2 startup
   ```
   Follow the `pm2 startup` output to enable PM2 on boot (e.g., `sudo systemctl enable pm2-ubuntu`).

4. **Configure Nginx**:
   ```bash
   sudo nano /etc/nginx/sites-available/toxictext-scanner
   ```
   Add:
   ```
   server {
       listen 80;
       server_name 172.31.26.143;
       location / {
           proxy_pass http://127.0.0.1:5001;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```
   Enable and restart Nginx:
   ```bash
   sudo ln -s /etc/nginx/sites-available/toxictext-scanner /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

5. **Test the API**:
   ```bash
   curl -X GET http://172.31.26.143/health
   ```
   Expected response:
   ```json
   {"status": "ok"}
   ```
   ```bash
   curl -X POST http://172.31.26.143/classify -H "Content-Type: application/json" -d '{"text": "Replying to @harper: calm down, you clown. get lost #fun29 Also, read before you post"}'
   ```
   Expected response:
   ```json
   {
     "label": "Offensive Language",
     "confidence": 0.86,
     "explanation": "Flagged for offensive language based on terms: clown, get lost",
     "influential_terms": ["clown", "get", "lost"]
   }
   ```
   ```bash
   curl -X POST http://172.31.26.143/classify -H "Content-Type: application/json" -d '{"text": "أريد قتل شخص معين"}'
   ```
   Expected response:
   ```json
   {
     "label": "Hate Speech",
     "confidence": 0.89,
     "explanation": "Flagged for hate speech based on terms: قتل, شخص",
     "influential_terms": ["قتل", "شخص"]
   }
   ```

6. **Scaling**:
   Edit `ecosystem.config.js` to increase workers or instances:
   ```bash
   nano inference/ecosystem.config.js
   ```
   Update:
   ```
   args: '--workers=8 --bind=0.0.0.0:5001 app:app',
   instances: 2,
   exec_mode: 'cluster'
   ```
   Reload:
   ```bash
   pm2 reload toxictext-scanner
   ```

## Dataset Structure
The dataset (e.g., `training/data/new_training_data.csv` for English, `training/data_arabic/new_training_data.csv` for Arabic) must have:
- `count`: Total annotations (e.g., 1).
- `hate_speech_count`: 1 for Hate Speech, 0 otherwise.
- `offensive_language_count`: 1 for Offensive Language, 0 otherwise.
- `neither_count`: 1 for Neutral, 0 otherwise.
- `class`: Label (0=Hate Speech, 1=Offensive Language, 2=Neutral, or strings: 'hate_speech', 'offensive_language', 'neutral').
- `tweet`: Text content.

Validation files (`training/data/*.csv` for English, `training/data_arabic/*.csv` for Arabic) contain one tweet per line with no headers. String labels are mapped to numeric values in `train_model.py`, `train_model_ar.py`, and `verify_model.py`.

## Troubleshooting
- **Low Model Accuracy**:
  - **English**: Source more diverse data with distinct features using `generate_synthetic_data.py`. Retrain with `train_model.py`.
  - **Arabic**: Ensure consistent Stanza preprocessing between training and inference. Source additional diverse Arabic data if needed.
- **Missing Model Files**:
   ```bash
   ls -l inference/*.joblib
   ```
   Rerun `train_model.py` or `train_model_ar.py` if missing.
- **Dependency Issues**:
   ```bash
   pip install flask==3.0.3 nltk==3.9.1 numpy==1.26.4 joblib==1.4.2 gunicorn==23.0.0 flask-limiter==3.8.0 scikit-learn==1.5.2 langdetect==1.0.9 stanza==1.10.0
   ```
- **NLTK/Stanza Issues**:
   ```bash
   cd inference
   rm -rf ./nltk_data
   python3 -m nltk.downloader -d ./nltk_data stopwords wordnet punkt_tab
   python3 -c "import stanza; stanza.download('ar', processors='tokenize,lemma')"
   pm2 restart toxictext-scanner
   ```
- **PM2 Issues**:
   Check logs:
   ```bash
   pm2 logs toxictext-scanner
   ```
   Restart:
   ```bash
   pm2 restart toxictext-scanner
   ```
