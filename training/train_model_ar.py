import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
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
except ImportError:
    stanza = None

# Set NLTK data path
nltk.data.path.append(os.environ.get("NLTK_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nltk_data")))

# Download required NLTK data
nltk.download('punkt', quiet=True, download_dir=os.environ.get("NLTK_DATA"))
nltk.download('punkt_tab', quiet=True, download_dir=os.environ.get("NLTK_DATA"))
nltk.download('stopwords', quiet=True, download_dir=os.environ.get("NLTK_DATA"))

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
        self.nlp = None
        if self.use_stanza:
            print("Initializing Stanza pipeline for Arabic on CPU...")
            try:
                stanza_dir = os.path.expanduser("~/stanza_resources")
                self.nlp = stanza.Pipeline('ar', processors='tokenize,lemma', use_gpu=False, batch_size=64, dir=stanza_dir)
            except Exception as e:
                print(f"Failed to initialize Stanza: {e}. Falling back to NLTK.")
                self.use_stanza = False
        # Initialize vectorizer and classifier
        if reset_model or not os.path.exists(self.vectorizer_path):
            print("Initializing new vectorizer...")
            self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
        else:
            print(f"Loading existing vectorizer from {self.vectorizer_path}...")
            self.vectorizer = joblib.load(self.vectorizer_path)
        if reset_model or not os.path.exists(self.model_path):
            print("Initializing new classifier...")
            self.classifier = SGDClassifier(loss='log_loss', max_iter=50, tol=1e-3, alpha=0.00001, random_state=42)
        else:
            print(f"Loading existing model from {self.model_path}...")
            self.classifier = joblib.load(self.model_path)

    def preprocess_text(self, texts, save_path=None):
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
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                for text in processed_texts:
                    f.write(text + '\n')
        return processed_texts if len(texts) > 1 else processed_texts[0]

    def preprocess_dataset(self, input_csv, output_csv):
        print(f"Preprocessing dataset: {input_csv}")
        df = pd.read_csv(input_csv)
        df['processed_tweet'] = self.preprocess_text(df['tweet'].tolist())
        df[['tweet', 'processed_tweet', 'class']].to_csv(output_csv, index=False)
        print(f"Saved preprocessed dataset to: {output_csv}")

    def train(self, csv_path, output_dir, chunk_size=10000):
        # Check if preprocessed dataset exists
        preprocessed_csv = os.path.join(output_dir, 'preprocessed_data.csv')
        if not os.path.exists(preprocessed_csv):
            self.preprocess_dataset(csv_path, preprocessed_csv)
        # Load preprocessed data
        df = pd.read_csv(preprocessed_csv)
        # Label mapping
        label_map = {
            'Hate Speech': 0,
            'Offensive Language': 1,
            'Neutral': 2,
            'hate_speech': 0,
            'offensive_language': 1,
            'neutral': 2
        }
        if df['class'].dtype == 'object':
            missing_labels = df['class'][~df['class'].isin(label_map.keys())].unique()
            if len(missing_labels) > 0:
                raise ValueError(f"Unknown string labels found: {missing_labels}")
            df['class'] = df['class'].map(label_map)
        # Split data
        X = df['processed_tweet'].values
        y = df['class'].values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        # Vectorize and train
        print("Fitting vectorizer...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        print(f"Vectorizer saved to {self.vectorizer_path}")
        print("Training classifier...")
        self.classifier.fit(X_train_vec, y_train)
        joblib.dump(self.classifier, self.model_path)
        print(f"Model saved to {self.model_path}")
        # Evaluate on validation set
        print("Evaluating on validation set...")
        X_val_vec = self.vectorizer.transform(X_val)
        y_pred = self.classifier.predict(X_val_vec)
        print("Validation results:")
        print(classification_report(y_val, y_pred, target_names=self.class_names, zero_division=0))
        # Cross-validation
        print("Performing 5-fold cross-validation...")
        scores = cross_val_score(self.classifier, X_train_vec, y_train, cv=5, scoring='f1_macro')
        print(f"Cross-validation F1-macro scores: {scores}")
        print(f"Average F1-macro: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
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
    input_csv = CONFIG['new_data_path_ar']
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'inference')
    trainer.train(input_csv, output_dir)