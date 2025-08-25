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
        self.model_path = CONFIG['model_path']
        self.vectorizer_path = CONFIG['vectorizer_path']
        self.class_names = ['Hate Speech', 'Offensive Language', 'Neutral']
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize or load vectorizer
        if os.path.exists(self.vectorizer_path):
            print(f"Loading existing vectorizer from {self.vectorizer_path}...")
            self.vectorizer = joblib.load(self.vectorizer_path)
        else:
            print("Initializing new vectorizer...")
            self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
        
        # Initialize or load classifier
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}...")
            self.classifier = joblib.load(self.model_path)
        else:
            print("Initializing new classifier...")
            self.classifier = SGDClassifier(loss='log_loss', max_iter=1, tol=None, random_state=42, warm_start=True)

    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

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
            'offensive': 1,
            'neither': 2
        }
        
        # Process data in chunks
        first_chunk = True
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            print(f"Processing chunk of size {len(chunk)}...")
            
            # Rename label column
            if label_column != 'class':
                chunk = chunk.rename(columns={label_column: 'class'})
            
            # Map string labels to numeric
            if chunk['class'].dtype == 'object':
                missing_labels = chunk['class'][~chunk['class'].isin(label_map.keys())].unique()
                if len(missing_labels) > 0:
                    raise ValueError(f"Unknown string labels found: {missing_labels}")
                chunk['class'] = chunk['class'].map(label_map)
            
            # Preprocess text
            chunk['processed_tweet'] = [self.preprocess_text(text) for text in tqdm(chunk['tweet'], desc="Preprocessing chunk")]
            
            # Vectorize
            if first_chunk and not os.path.exists(self.vectorizer_path):
                print("Fitting vectorizer on first chunk...")
                X = self.vectorizer.fit_transform(chunk['processed_tweet'])
                joblib.dump(self.vectorizer, self.vectorizer_path)
                print(f"Vectorizer saved to {self.vectorizer_path}")
            else:
                print("Transforming text with existing vectorizer...")
                X = self.vectorizer.transform(chunk['processed_tweet'])
            
            y = chunk['class']
            
            # Incremental training
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
        df['processed_tweet'] = [self.preprocess_text(text) for text in tqdm(df['tweet'], desc="Preprocessing eval sample")]
        X_eval = self.vectorizer.transform(df['processed_tweet'])
        y_eval = df['class']
        y_pred = self.classifier.predict(X_eval)
        print("Model Performance on Sample:")
        print(classification_report(y_eval, y_pred, target_names=self.class_names))

if __name__ == "__main__":
    trainer = ToxicTextTrainer()
    new_data_path = CONFIG.get('new_data_path', '/home/branch/Downloads/new_training_data.csv')
    print(f"Performing incremental training with {new_data_path}...")
    trainer.train(new_data_path)