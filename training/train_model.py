import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import joblib
import os
from config import CONFIG
from tqdm import tqdm

# Set NLTK data path
nltk.data.path.append('/home/branch/nltk_data')

# Download required NLTK data
nltk.download('punkt', quiet=True, download_dir='/home/branch/nltk_data')
nltk.download('punkt_tab', quiet=True, download_dir='/home/branch/nltk_data')
nltk.download('stopwords', quiet=True, download_dir='/home/branch/nltk_data')
nltk.download('wordnet', quiet=True, download_dir='/home/branch/nltk_data')

class ToxicTextTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95)
        self.classifier = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-4, alpha=0.0001, 
                                       class_weight='balanced', random_state=42)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.class_names = ['Hate Speech', 'Offensive Language', 'Neutral']
        self.model_path = CONFIG['model_path']
        self.vectorizer_path = CONFIG['vectorizer_path']

    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def load_model_and_vectorizer(self):
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                self.classifier = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                if not isinstance(self.classifier, SGDClassifier):
                    raise TypeError(f"Loaded model is {type(self.classifier).__name__}, expected SGDClassifier")
                print(f"Loaded model from {self.model_path} and vectorizer from {self.vectorizer_path}")
                return True
            return False
        except Exception as e:
            print(f"Failed to load model or vectorizer: {e}. Falling back to initial training.")
            return False

    def train(self, csv_path, incremental=False):
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Identify label column
        possible_label_columns = ['class', 'label', 'sentiment', 'category']
        label_column = None
        for col in possible_label_columns:
            if col in df.columns:
                label_column = col
                break
        if label_column is None:
            raise ValueError(f"No label column found in {csv_path}. Expected one of: {possible_label_columns}")
        
        # Rename label column to 'class' for consistency
        if label_column != 'class':
            print(f"Renaming label column '{label_column}' to 'class'...")
            df = df.rename(columns={label_column: 'class'})
        
        # Handle string labels if present
        label_map = {
            'Hate Speech': 0,
            'Offensive Language': 1,
            'Neutral': 2,
            'hate_speech': 0,
            'offensive': 1,
            'neither': 2
        }
        if df['class'].dtype == 'object':
            print("Mapping string labels to numeric values...")
            missing_labels = df['class'][~df['class'].isin(label_map.keys())].unique()
            if len(missing_labels) > 0:
                raise ValueError(f"Unknown string labels found: {missing_labels}. Expected: {list(label_map.keys())}")
            df['class'] = df['class'].map(label_map)
        
        # Preprocess text with progress bar
        print("Preprocessing text...")
        df['processed_tweet'] = [self.preprocess_text(text) for text in tqdm(df['tweet'], desc="Preprocessing")]
        
        if incremental:
            print("Performing incremental training...")
            X = self.vectorizer.transform(df['processed_tweet'])
            y = df['class']
            chunk_size = 10000
            for i in tqdm(range(0, len(df), chunk_size), desc="Incremental Training"):
                end = min(i + chunk_size, len(df))
                X_chunk = X[i:end]
                y_chunk = y[i:end]
                self.classifier.partial_fit(X_chunk, y_chunk, classes=[0, 1, 2])
            # Evaluate after incremental training
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print("Evaluating model...")
            y_pred = self.classifier.predict(X_test)
            print("Model Performance:")
            print(classification_report(y_test, y_pred, target_names=self.class_names))
        else:
            print("Performing initial training...")
            X = self.vectorizer.fit_transform(df['processed_tweet'])
            y = df['class']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print("Training model...")
            self.classifier.fit(X_train, y_train)
            print("Evaluating model...")
            y_pred = self.classifier.predict(X_test)
            print("Model Performance:")
            print(classification_report(y_test, y_pred, target_names=self.class_names))

        # Save model and vectorizer
        joblib.dump(self.classifier, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        print(f"Model saved to {self.model_path}")
        print(f"Vectorizer saved to {self.vectorizer_path}")

if __name__ == "__main__":
    trainer = ToxicTextTrainer()
    new_data_path = CONFIG.get('new_data_path', '/home/branch/Downloads/synthetic_toxic_tweets_new.csv')
    if trainer.load_model_and_vectorizer() and os.path.exists(new_data_path):
        trainer.train(new_data_path, incremental=True)
    else:
        print(f"Performing initial training with {new_data_path}...")
        trainer.train(new_data_path, incremental=False)