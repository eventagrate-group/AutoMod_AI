import os
import nltk
import re
import joblib
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config import CONFIG
from langdetect import detect, LangDetectException

# Set NLTK data path
nltk.data.path.append(os.environ.get("NLTK_DATA", os.path.join(os.getcwd(), "nltk_data")))

class ToxicTextScanner:
    def __init__(self):
        try:
            # English model
            self.vectorizer_en = joblib.load(CONFIG['vectorizer_path'])
            self.classifier_en = joblib.load(CONFIG['model_path'])
            # Arabic model
            self.vectorizer_ar = joblib.load(CONFIG['vectorizer_path_ar'])
            self.classifier_ar = joblib.load(CONFIG['model_path_ar'])
            # English preprocessing
            self.lemmatizer = WordNetLemmatizer()
            try:
                self.stop_words_en = set(stopwords.words('english'))
            except LookupError:
                self.stop_words_en = set()
            # Arabic preprocessing
            try:
                self.stop_words_ar = set(stopwords.words('arabic'))
            except LookupError:
                self.stop_words_ar = set()
            self.class_names = ['Hate Speech', 'Offensive Language', 'Neutral']
        except Exception as e:
            print(f"Failed to initialize scanner: {str(e)}")
            raise

    def preprocess_text(self, text, lang='en'):
        text = str(text).lower()
        # Remove URLs and mentions
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        # Language-specific preprocessing
        if lang == 'en':
            text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep English letters
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words_en]
        else:  # lang == 'ar'
            text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep Arabic characters
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t not in self.stop_words_ar]  # No lemmatization for Arabic
        return ' '.join(tokens)

    def classify_text(self, text):
        try:
            # Detect language
            try:
                lang = detect(text)
            except LangDetectException:
                lang = 'en'  # Default to English
            # Preprocess and classify
            processed_text = self.preprocess_text(text, lang=lang)
            if lang == 'ar':
                X = self.vectorizer_ar.transform([processed_text])
                classifier = self.classifier_ar
            else:  # Default to English
                X = self.vectorizer_en.transform([processed_text])
                classifier = self.classifier_en
            prediction = classifier.predict(X)[0]
            probabilities = classifier.predict_proba(X)[0]
            confidence = float(np.max(probabilities))
            # Get influential terms
            feature_names = self.vectorizer_ar.get_feature_names_out() if lang == 'ar' else self.vectorizer_en.get_feature_names_out()
            row = X.toarray()[0]
            top_idx = row.argsort()[-3:][::-1]
            influential_terms = [feature_names[i] for i in top_idx if row[i] > 0]
            return {
                "label": self.class_names[prediction],
                "confidence": confidence,
                "explanation": (
                    f"Flagged for {self.class_names[prediction].lower()} "
                    f"based on terms: {', '.join(influential_terms) or 'n/a'}"
                ),
                "influential_terms": influential_terms
            }
        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}

    def scan(self, text):
        return self.classify_text(text)