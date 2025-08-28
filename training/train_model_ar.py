import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import joblib
import os
from config import CONFIG
from tqdm import tqdm
try:
    import stanza
    import torch
except ImportError:
    stanza = None
    torch = None

# Set NLTK data path
nltk.data.path.append(os.environ.get("NLTK_DATA", os.path.join(os.getcwd(), "nltk_data")))

# Download required NLTK data
nltk.download('punkt', quiet=True, download_dir=os.environ.get("NLTK_DATA", os.path.join(os.getcwd(), "nltk_data")))
nltk.download('punkt_tab', quiet=True, download_dir=os.environ.get("NLTK_DATA", os.path.join(os.getcwd(), "nltk_data")))
nltk.download('stopwords', quiet=True, download_dir=os.environ.get("NLTK_DATA", os.path.join(os.getcwd(), "nltk_data")))

class ToxicTextTrainerArabic:
    def __init__(self, reset_model=True):
        self.model_path = CONFIG['model_path_ar']
        self.vectorizer_path = CONFIG['vectorizer_path_ar']
        self.class_names = ['Hate Speech', 'Offensive Language', 'Neutral']
        try:
            self.stop_words = set(stopwords.words('arabic'))
        except LookupError:
            print("Warning: NLTK Arabic stopwords not found. Using empty stopword set.")
            self.stop_words = set()
        # Initialize Stanza pipeline if available
        self.use_stanza = True if stanza is not None else False
        self.device = 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu'
        self.nlp = None
        if self.use_stanza:
            print(f"Initializing Stanza pipeline for Arabic on {self.device}...")
            try:
                stanza_dir = os.path.expanduser("~/stanza_resources")
                if not os.path.exists(stanza_dir):
                    print(f"Stanza resources not found at {stanza_dir}. Downloading...")
                    stanza.download('ar', processors='tokenize,lemma', dir=stanza_dir)
                self.nlp = stanza.Pipeline('ar', processors='tokenize,lemma', use_gpu=(self.device == 'cuda'), dir=stanza_dir, batch_size=64)
            except Exception as e:
                print(f"Failed to initialize Stanza: {e}. Falling back to NLTK.")
                self.use_stanza = False
        # Initialize or load vectorizer
        if reset_model or not os.path.exists(self.vectorizer_path):
            print("Initializing new vectorizer...")
            self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
        else:
            print(f"Loading existing vectorizer from {self.vectorizer_path}...")
            self.vectorizer = joblib.load(self.vectorizer_path)
        # Initialize or load classifier
        if reset_model or not os.path.exists(self.model_path):
            print("Initializing new classifier...")
            self.classifier = SGDClassifier(loss='log_loss', max_iter=10, tol=1e-3, alpha=0.00005, random_state=42, warm_start=True)
        else:
            print(f"Loading existing model from {self.model_path}...")
            self.classifier = joblib.load(self.model_path)

    def preprocess_text(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        processed_texts = []
        for text in tqdm(texts, desc="Preprocessing texts"):
            text = str(text).lower()
            text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
            text = re.sub(r'@\w+', '', text)  # Remove mentions
            text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep only Arabic letters and spaces
            if self.use_stanza and hasattr(self, 'nlp') and self.nlp is not None:
                try:
                    doc = self.nlp(text)
                    tokens = [word.lemma for sent in doc.sentences for word in sent.words if word.lemma and word.lemma not in self.stop_words]
                except Exception as e:
                    print(f"Stanza processing failed for text: {text[:50]}... Error: {e}. Falling back to NLTK.")
                    tokens = word_tokenize(text)
                    tokens = [token for token in tokens if token not in self.stop_words]
            else:
                tokens = word_tokenize(text)
                tokens = [token for token in tokens if token not in self.stop_words]
            processed_texts.append(' '.join(tokens))
        return processed_texts if len(texts) > 1 else processed_texts[0]

    def train(self, csv_path, chunk_size=10000):
        print(f"Starting incremental training with {csv_path}...")
        # Identify label column
        df_iter = pd.read_csv(csv_path, chunksize=chunk_size)
        label_column = None
        for chunk in df_iter:
            possible_label_columns = ['class', 'label', 'sentiment', 'category']
            for col in possible_label_columns:
                if col in chunk.columns:
                    label_column = col
                    break
            if label_column:
                break
        if label_column is None:
            raise ValueError(f"No label column found in {csv_path}. Expected one of: {possible_label_columns}")
        # Label mapping
        label_map = {
            'Hate Speech': 0,
            'Offensive Language': 1,
            'Neutral': 2,
            'hate_speech': 0,
            'offensive_language': 1,
            'neutral': 2
        }
        # Process data in chunks
        first_chunk = True
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            print(f"Processing chunk of size {len(chunk)}...")
            if label_column != 'class':
                chunk = chunk.rename(columns={label_column: 'class'})
            if chunk['class'].dtype == 'object':
                missing_labels = chunk['class'][~chunk['class'].isin(label_map.keys())].unique()
                if len(missing_labels) > 0:
                    raise ValueError(f"Unknown string labels found: {missing_labels}")
                chunk['class'] = chunk['class'].map(label_map)
            chunk['processed_tweet'] = self.preprocess_text(chunk['tweet'].tolist())
            if first_chunk:
                print("Fitting vectorizer on first chunk...")
                X = self.vectorizer.fit_transform(chunk['processed_tweet'])
                joblib.dump(self.vectorizer, self.vectorizer_path)
                print(f"Vectorizer saved to {self.vectorizer_path}")
            else:
                print("Transforming text with existing vectorizer...")
                X = self.vectorizer.transform(chunk['processed_tweet'])
            y = chunk['class']
            print("Updating model with partial_fit...")
            self.classifier.partial_fit(X, y, classes=[0, 1, 2])
            first_chunk = False
        # Save model
        joblib.dump(self.classifier, self.model_path)
        print(f"Model saved to {self.model_path}")
        # Evaluate on a sample
        print("Evaluating model on a sample...")
        df = pd.read_csv(csv_path).sample(frac=0.2, random_state=42)
        if label_column != 'class':
            df = df.rename(columns={label_column: 'class'})
        if df['class'].dtype == 'object':
            df['class'] = df['class'].map(label_map)
        df['processed_tweet'] = self.preprocess_text(df['tweet'].tolist())
        X_eval = self.vectorizer.transform(df['processed_tweet'])
        y_eval = df['class']
        y_pred = self.classifier.predict(X_eval)
        print("Model Performance on Sample:")
        print(classification_report(y_eval, y_pred, target_names=self.class_names, zero_division=0))
        # Evaluate on validation files
        print("Evaluating model on validation files...")
        validation_files = {
            'Hate Speech': os.path.join(os.path.dirname(csv_path), 'hate_speech_verify.csv'),
            'Offensive Language': os.path.join(os.path.dirname(csv_path), 'offensive_language_verify.csv'),
            'Neutral': os.path.join(os.path.dirname(csv_path), 'neutral_verify.csv')
        }
        for class_name, file_path in validation_files.items():
            if os.path.exists(file_path):
                df_val = pd.read_csv(file_path, header=None, names=['tweet'])
                df_val['processed_tweet'] = self.preprocess_text(df_val['tweet'].tolist())
                X_val = self.vectorizer.transform(df_val['processed_tweet'])
                y_val = [self.class_names.index(class_name)] * len(df_val)
                y_pred = self.classifier.predict(X_val)
                correct = sum(y_pred == y_val)
                total = len(y_val)
                print(f"{class_name}: {correct}/{total} correct ({correct/total*100:.2f}%)")
            else:
                print(f"Validation file {file_path} not found.")

if __name__ == "__main__":
    trainer = ToxicTextTrainerArabic(reset_model=True)
    trainer.train(CONFIG['new_data_path_ar'])