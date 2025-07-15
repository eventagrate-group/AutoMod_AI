import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import joblib
from config import CONFIG

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Added for PunktTokenizer
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class ToxicTextTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = LogisticRegression(multi_class='multinomial', max_iter=1000)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.class_names = ['Hate Speech', 'Offensive Language', 'Neutral / Clean']

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def train(self, csv_path, model_path, vectorizer_path):
        # Load dataset
        df = pd.read_csv(csv_path)
        # Use provided 'class' column for labels (assuming it's 0, 1, 2 for Hate Speech, Offensive, Neutral)
        df['processed_tweet'] = df['tweet'].apply(self.preprocess_text)
        X = self.vectorizer.fit_transform(df['processed_tweet'])
        y = df['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        print("Model Performance:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        # Save model and vectorizer
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":
    trainer = ToxicTextTrainer()
    trainer.train(
        csv_path=CONFIG['dataset_path'],
        model_path=CONFIG['model_path'],
        vectorizer_path=CONFIG['vectorizer_path']
    )