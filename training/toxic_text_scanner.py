
import nltk
import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from config import MODEL_PATH, VECTORIZER_PATH

# Append NLTK data path to ensure stopwords can be found
nltk.data.path.append('/app/nltk_data')

class ToxicTextScanner:
    def __init__(self):
        try:
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Failed to initialize scanner: {str(e)}")
            raise

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words]
        return ' '.join(tokens)

    def get_influential_terms(self, text):
        tokens = word_tokenize(text.lower())
        return [t for t in tokens if t.isalpha() and t not in self.stop_words]

    def classify(self, text):
        try:
            processed_text = self.preprocess_text(text)
            vectorized_text = self.vectorizer.transform([processed_text])
            prediction = self.model.predict(vectorized_text)[0]
            confidence = np.max(self.model.predict_proba(vectorized_text))
            influential_terms = self.get_influential_terms(text)
            label = "Hate Speech" if prediction == 1 else "Neutral"
            return {
                "label": label,
                "confidence": round(float(confidence), 2),
                "explanation": f"Flagged as {label.lower()} based on terms: {', '.join(influential_terms)}",
                "influential_terms": influential_terms
            }
        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}
