```python
import nltk
import re
import joblib
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config import CONFIG

# Ensure NLTK can find stopwords by appending the data path
nltk.data.path.append('/app/nltk_data')

class ToxicTextScanner:
    def __init__(self):
        try:
            self.vectorizer = joblib.load(CONFIG['vectorizer_path'])
            self.classifier = joblib.load(CONFIG['model_path'])
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
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
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def classify_text(self, text):
        try:
            processed_text = self.preprocess_text(text)
            X = self.vectorizer.transform([processed_text])
            prediction = self.classifier.predict(X)[0]
            probabilities = self.classifier.predict_proba(X)[0]
            confidence = float(np.max(probabilities))
            
            # Get feature names and their coefficients for explanation
            feature_names = self.vectorizer.get_feature_names_out()
            coef = X.toarray()[0]
            top_indices = coef.argsort()[-3:][::-1]
            influential_terms = [feature_names[i] for i in top_indices if coef[i] > 0]
            
            explanation = f"Flagged for {self.class_names[prediction].lower()} based on terms: {', '.join(influential_terms)}"
            
            return {
                'label': self.class_names[prediction],
                'confidence': confidence,
                'explanation': explanation,
                'influential_terms': influential_terms
            }
        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}
```