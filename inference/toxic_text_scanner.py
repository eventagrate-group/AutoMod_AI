import os
import nltk
import re
import joblib
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config import CONFIG

# Prefer env NLTK_DATA (works both in Docker and venv); fall back to a system path
nltk.data.path.append(os.environ.get("NLTK_DATA", "/usr/share/nltk_data"))

class ToxicTextScanner:
    def __init__(self):
        try:
            self.vectorizer = joblib.load(CONFIG['vectorizer_path'])
            self.classifier = joblib.load(CONFIG['model_path'])
            self.lemmatizer = WordNetLemmatizer()
            # Try loading stopwords; if missing, default to empty set (won't crash)
            try:
                self.stop_words = set(stopwords.words('english'))
            except LookupError:
                self.stop_words = set()
            self.class_names = ['Hate Speech', 'Offensive Language', 'Neutral']
        except Exception as e:
            print(f"Failed to initialize scanner: {str(e)}")
            raise

    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)

    def classify_text(self, text):
        try:
            processed_text = self.preprocess_text(text)
            X = self.vectorizer.transform([processed_text])
            prediction = self.classifier.predict(X)[0]
            probabilities = self.classifier.predict_proba(X)[0]
            confidence = float(np.max(probabilities))

            # Simple token influence based on TF-IDF weights present in this example
            feature_names = self.vectorizer.get_feature_names_out()
            row = X.toarray()[0]
            top_idx = row.argsort()[-3:][::-1]
            influential_terms = [feature_names[i] for i in top_idx if row[i] > 0]

            return {
                "label": self.class_names[prediction],
                "confidence": confidence
            }
        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}

    # Backward-compatibility: some callers use scanner.scan(...)
    def scan(self, text):
        return self.classify_text(text)
