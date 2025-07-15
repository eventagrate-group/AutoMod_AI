import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import joblib

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class ToxicTextScanner:
    def __init__(self, model_path, vectorizer_path):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.class_names = ['Hate Speech', 'Offensive Language', 'Neutral / Clean']
        self.toxic_words = {
            'hate_speech': ['hate', 'racist', 'bigot', 'nazi'],
            'offensive': ['damn', 'ass', 'bitch', 'fuck']
        }
        self.classifier = None
        self.vectorizer = None

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def load_model(self):
        try:
            self.classifier = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            print(f"Loaded model from {self.model_path} and vectorizer from {self.vectorizer_path}")
            return True
        except FileNotFoundError:
            return False

    def classify(self, text):
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        probabilities = self.classifier.predict_proba(X)[0]
        prediction = np.argmax(probabilities)
        confidence = probabilities[prediction]
        explanation = self._generate_explanation(text, prediction)
        return {
            'label': self.class_names[prediction],
            'confidence': float(confidence),
            'explanation': explanation
        }

    def _generate_explanation(self, text, prediction):
        text = text.lower()
        if prediction == 0:
            for word in self.toxic_words['hate_speech']:
                if word in text:
                    return f"Flagged for potential hate speech containing term: {word}"
        elif prediction == 1:
            for word in self.toxic_words['offensive']:
                if word in text:
                    return f"Flagged for offensive language containing term: {word}"
        return "Classification based on text patterns and learned features"