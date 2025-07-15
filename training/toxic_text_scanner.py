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

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class ToxicTextScanner:
    def __init__(self, model_path='model.joblib', vectorizer_path='vectorizer.joblib'):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = LogisticRegression(multi_class='multinomial', max_iter=1000)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.class_names = ['Hate Speech', 'Offensive Language', 'Neutral / Clean']
        self.toxic_words = {
            'hate_speech': ['hate', 'racist', 'bigot', 'nazi'],
            'offensive': ['damn', 'ass', 'bitch', 'fuck']
        }
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def train(self, csv_path):
        df = pd.read_csv(csv_path)
        df['label'] = df.apply(lambda row: 0 if row['hate_speech_count'] > row['offensive_language_count'] and row['hate_speech_count'] > row['neither_count']
                              else 1 if row['offensive_language_count'] > row['hate_speech_count'] and row['offensive_language_count'] > row['neither_count']
                              else 2, axis=1)
        df['processed_tweet'] = df['tweet'].apply(self.preprocess_text)
        X = self.vectorizer.fit_transform(df['processed_tweet'])
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        print("Model Performance:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        joblib.dump(self.classifier, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        print(f"Model saved to {self.model_path}")
        print(f"Vectorizer saved to {self.vectorizer_path}")