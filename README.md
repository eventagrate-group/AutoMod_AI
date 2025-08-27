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
│   ├── requirements.txt            # Dependencies (flask, nltk, joblib, etc.)
│   ├── model.joblib                # Trained SGDClassifier model
│   ├── vectorizer.joblib           # Trained TF-IDF vectorizer
│   ├── ecosystem.config.js          # PM2 configuration for production
├── README.md                       # This file
```

## Features
- **Text Classification**: Labels text as `Hate Speech` (0), `Offensive Language` (1), or `Neutral` (2) with confidence scores and dynamic explanations based on TF-IDF feature weights.
- **Incremental Learning**: Updates the `SGDClassifier` using `partial_fit`, preserving the initial TF-IDF vocabulary for consistency.
- **REST API**: Flask endpoints (`/classify` for text classification, `/health` for service status) for real-time inference.
- **Performance Metrics**: Reports precision, recall, F1-score, and accuracy after training/incremental training.
- **PM2 Deployment**: Uses PM2 for process management, enabling easy scaling and monitoring in production.

## Prerequisites
- Ubuntu 24.04.3 LTS
- Python 3.12
- Node.js and npm (for PM2)
- Nginx (for reverse proxy)
- Git
- Sufficient memory (8–16 GB RAM for training 1M rows)
- Internet connection for NLTK data

## Setup
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
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
     ```

4. **Download NLTK Data**:
   ```bash
   cd inference
   source venv/bin/activate
   python3 -m nltk.downloader -d ./nltk_data stopwords wordnet punkt_tab
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
1. **Install PM2**:
   ```bash
   sudo npm install -g pm2
   ```

2. **Run the Flask API with PM2**:
   ```bash
   cd inference
   source venv/bin/activate
   pm2 start ecosystem.config.js
   pm2 save
   pm2 startup
   ```
   Follow the `pm2 startup` output to enable PM2 on boot (e.g., `sudo systemctl enable pm2-ubuntu`).

3. **Configure Nginx**:
   ```bash
   sudo nano /etc/nginx/sites-available/toxictext-scanner
   ```
   Add:
   ```
   server {
       listen 80;
       server_name <server-ip-or-domain>;
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

4. **Test the API**:
   ```bash
   curl -X GET http://<server-ip-or-domain>/health
   ```
   Expected response:
   ```json
   {"status": "ok"}
   ```
   ```bash
   curl -X POST http://<server-ip-or-domain>/classify -H "Content-Type: application/json" -d '{"text": "Replying to @harper: calm down, you clown. get lost #fun29 Also, read before you post"}'
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

5. **Scaling**:
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
   pip install flask==3.0.3 nltk==3.9.1 numpy==1.26.4 joblib==1.4.2 gunicorn==23.0.0 flask-limiter==3.8.0 scikit-learn==1.5.2
   ```
- **NLTK Issues**:
   ```bash
   cd inference
   rm -rf ./nltk_data
   python3 -m nltk.downloader -d ./nltk_data stopwords wordnet punkt_tab
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