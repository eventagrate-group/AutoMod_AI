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
try:
    import stanza
except ImportError:
    stanza = None

# Set NLTK data path
nltk.data.path.append(os.environ.get("NLTK_DATA", os.path.join(os.getcwd(), "nltk_data")))

class ToxicTextScanner:
    def __init__(self):
        try:
            self.vectorizer_en = joblib.load(CONFIG['vectorizer_path'])
            self.classifier_en = joblib.load(CONFIG['model_path'])
            self.vectorizer_ar = joblib.load(CONFIG['vectorizer_path_ar'])
            self.classifier_ar = joblib.load(CONFIG['model_path_ar'])
            self.lemmatizer = WordNetLemmatizer()
            try:
                self.stop_words_en = set(stopwords.words('english'))
                self.stop_words_ar = set(stopwords.words('arabic'))
            except LookupError:
                print("Warning: NLTK stopwords not found. Using empty stopword sets.")
                self.stop_words_en = set()
                self.stop_words_ar = set()
            self.use_stanza = True if stanza is not None else False
            self.nlp = None
            if self.use_stanza:
                print("Initializing Stanza pipeline for Arabic on CPU...")
                try:
                    stanza_dir = os.path.expanduser("~/stanza_resources")
                    if not os.path.exists(stanza_dir):
                        print(f"Stanza resources not found at {stanza_dir}. Downloading...")
                        stanza.download('ar', processors='tokenize,lemma', dir=stanza_dir)
                    self.nlp = stanza.Pipeline('ar', processors='tokenize,lemma', use_gpu=False, dir=stanza_dir)
                    print("Stanza pipeline initialized successfully.")
                except Exception as e:
                    print(f"Failed to initialize Stanza: {e}. Falling back to NLTK for Arabic.")
                    self.use_stanza = False
            self.class_names = ['Hate Speech', 'Offensive Language', 'Neutral']
        except Exception as e:
            print(f"Failed to initialize scanner: {str(e)}")
            raise

    def preprocess_text(self, text, lang='en'):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        if lang == 'en':
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words_en]
        else:
            text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
            if self.use_stanza and self.nlp is not None:
                try:
                    doc = self.nlp(text)
                    tokens = [word.lemma for sent in doc.sentences for word in sent.words if word.lemma and word.lemma not in self.stop_words_ar]
                except Exception as e:
                    print(f"Stanza preprocessing failed: {e}. Falling back to NLTK for Arabic.")
                    tokens = word_tokenize(text)
                    tokens = [t for t in tokens if t not in self.stop_words_ar]
            else:
                tokens = word_tokenize(text)
                tokens = [t for t in tokens if t not in self.stop_words_ar]
        return ' '.join(tokens)

    def classify_text(self, text):
        try:
            try:
                lang = detect(text)
            except LangDetectException:
                lang = 'en'
            processed_text = self.preprocess_text(text, lang=lang)
            if lang == 'ar':
                X = self.vectorizer_ar.transform([processed_text])
                classifier = self.classifier_ar
            else:
                X = self.vectorizer_en.transform([processed_text])
                classifier = self.classifier_en
            prediction = classifier.predict(X)[0]
            probabilities = classifier.predict_proba(X)[0]
            confidence = float(np.max(probabilities))
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