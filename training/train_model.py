import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
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
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
        self.classifier = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-4, random_state=42)
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

    def train(self, csv_path):
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
        
        # Vectorize
        print("Vectorizing text...")
        X = self.vectorizer.fit_transform(df['processed_tweet'])
        y = df['class']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Hyperparameter tuning
        print("Performing hyperparameter tuning...")
        param_grid = {
            'alpha': [1e-5, 1e-4, 1e-3],
            'class_weight': ['balanced', None]
        }
        grid_search = GridSearchCV(self.classifier, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.classifier = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate
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
    # Remove existing model files to force full retraining
    model_path = CONFIG['model_path']
    vectorizer_path = CONFIG['vectorizer_path']
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Removed existing model: {model_path}")
    if os.path.exists(vectorizer_path):
        os.remove(vectorizer_path)
        print(f"Removed existing vectorizer: {vectorizer_path}")
    
    trainer = ToxicTextTrainer()
    new_data_path = CONFIG.get('new_data_path', '/home/branch/Downloads/new_training_data.csv')
    print(f"Performing full training with {new_data_path}...")
    trainer.train(new_data_path)