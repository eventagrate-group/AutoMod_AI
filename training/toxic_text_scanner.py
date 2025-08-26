
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import joblib
from config import CONFIG
from sklearn.linear_model import SGDClassifier

class ToxicTextScanner:
    def __init__(self):
        self.vectorizer = joblib.load(CONFIG['vectorizer_path'])
        self.classifier = joblib.load(CONFIG['model_path'])
        if not isinstance(self.classifier, SGDClassifier):
            raise TypeError(f"Loaded model is {type(self.classifier).__name__}, expected SGDClassifier")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.class_names = ['Hate Speech', 'Offensive Language', 'Neutral']

    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def classify_text(self, text):
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = float(np.max(probabilities))
        
        # Get feature names and their coefficients for explanation
        feature_names = self.vectorizer.get_feature_names_out()
        coef = self.classifier.coef_[prediction]
        top_features = np.argsort(coef)[-3:][::-1]
        influential_terms = [feature_names[i] for i in top_features]
        
        explanation = f"Flagged for {self.class_names[prediction].lower()} based on terms: {', '.join(influential_terms)}"
        
        return {
            'label': self.class_names[prediction],
            'confidence': confidence,
            'explanation': explanation,
            'influential_terms': influential_terms
        }