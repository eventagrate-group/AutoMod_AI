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
nltk.data.path.append(os.environ.get("NLTK_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nltk_data")))

class ToxicTextScanner:
    def __init__(self):
        self.class_names = ['Hate Speech', 'Offensive Language', 'Neutral']
        self.lemmatizer = WordNetLemmatizer()
        self.use_stanza = True if stanza is not None else False
        try:
            self.stop_words_en = set(stopwords.words('english'))
            self.stop_words_ar = set(stopwords.words('arabic'))
        except LookupError:
            print("Warning: NLTK stopwords not found. Using empty stopword sets.")
            self.stop_words_en = set()
            self.stop_words_ar = set()
        # Load English resources
        try:
            self.vectorizer_en = joblib.load(CONFIG['vectorizer_path'])
            self.classifier_en = joblib.load(CONFIG['model_path'])
        except Exception as e:
            print(f"Failed to load English resources: {e}")
            raise
        # Load Arabic resources
        try:
            self.vectorizer_ar = joblib.load(CONFIG['vectorizer_path_ar'])
            self.classifier_ar = joblib.load(CONFIG['model_path_ar'])
        except Exception as e:
            print(f"Failed to load Arabic resources: {e}. Arabic classification may fail.")
            self.vectorizer_ar = None
            self.classifier_ar = None
        # Initialize Stanza
        self.nlp = None
        if self.use_stanza:
            print("Initializing Stanza pipeline for Arabic on CPU...")
            try:
                stanza_dir = os.path.expanduser("~/stanza_resources")
                if not os.path.exists(stanza_dir):
                    print(f"Stanza resources not found at {stanza_dir}. Downloading...")
                    stanza.download('ar', processors='tokenize,lemma', dir=stanza_dir)
                self.nlp = stanza.Pipeline('ar', processors='tokenize,lemma', use_gpu=False, dir=stanza_dir, batch_size=64)
                print("Stanza pipeline initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize Stanza: {e}. Falling back to NLTK for Arabic.")
                self.use_stanza = False

    def preprocess_text(self, texts, lang='en'):
        if not isinstance(texts, list):
            texts = [texts]
        processed_texts = []
        for text in texts:
            text = str(text).lower()
            text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
            text = re.sub(r'@\w+', '', text)  # Remove mentions
            if lang == 'en':
                text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only English letters and spaces
                tokens = word_tokenize(text)
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words_en]
            else:
                text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep only Arabic letters and spaces
                if self.use_stanza and hasattr(self, 'nlp') and self.nlp is not None:
                    try:
                        doc = self.nlp(text)
                        tokens = [word.lemma for sent in doc.sentences for word in sent.words if word.lemma and word.lemma not in self.stop_words_ar]
                    except Exception as e:
                        print(f"Stanza preprocessing failed for text: {text[:50]}... Error: {e}. Falling back to NLTK.")
                        tokens = word_tokenize(text)
                        tokens = [t for t in tokens if t not in self.stop_words_ar]
                else:
                    tokens = word_tokenize(text)
                    tokens = [t for t in tokens if t not in self.stop_words_ar]
            processed_texts.append(' '.join(tokens))
        return processed_texts if len(texts) > 1 else processed_texts[0]

    def classify(self, text):
        try:
            try:
                lang = detect(text)
                if lang != 'ar':
                    lang = 'en'
            except LangDetectException:
                lang = 'en'
            if lang == 'ar' and (self.classifier_ar is None or self.vectorizer_ar is None):
                return {
                    'label': 'Error',
                    'confidence': 0.0,
                    'explanation': 'Arabic model or vectorizer not available.',
                    'influential_terms': []
                }
            processed_text = self.preprocess_text(text, lang=lang)
            vectorizer = self.vectorizer_ar if lang == 'ar' else self.vectorizer_en
            classifier = self.classifier_ar if lang == 'ar' else self.classifier_en
            X = vectorizer.transform([processed_text])
            prediction = classifier.predict(X)[0]
            probabilities = classifier.predict_proba(X)[0]
            confidence = float(np.max(probabilities))
            feature_names = vectorizer.get_feature_names_out()
            row = X.toarray()[0]
            top_idx = row.argsort()[-3:][::-1]
            influential_terms = [feature_names[i] for i in top_idx if row[i] > 0]
            explanation = f"Flagged for {self.class_names[prediction].lower()} based on terms: {', '.join(influential_terms) or 'n/a'}"
            return {
                'label': self.class_names[prediction],
                'confidence': confidence,
                'explanation': explanation,
                'influential_terms': influential_terms
            }
        except Exception as e:
            return {
                'label': 'Error',
                'confidence': 0.0,
                'explanation': f'Classification failed: {str(e)}',
                'influential_terms': []
            }

    def scan(self, text):
        return self.classify(text)