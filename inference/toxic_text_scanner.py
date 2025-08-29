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
    from googletrans import Translator
except ImportError:
    stanza = None
    Translator = None

# Set NLTK data path
nltk.data.path.append(os.environ.get("NLTK_DATA", os.path.join(os.getcwd(), "nltk_data")))

class ToxicTextScanner:
    def __init__(self):
        try:
            self.vectorizer_en = joblib.load(CONFIG['vectorizer_path'])
            self.classifier_en = joblib.load(CONFIG['model_path'])
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
                    self.nlp = stanza.Pipeline('ar', processors='tokenize,lemma', use_gpu=False, dir=stanza_dir)
                    print("Stanza pipeline initialized successfully.")
                except Exception as e:
                    print(f"Failed to initialize Stanza: {e}. Falling back to NLTK for Arabic.")
                    self.use_stanza = False
            self.use_translator = True if Translator is not None else False
            self.translator = None
            if self.use_translator:
                print("Initializing Arabic-to-English translator...")
                try:
                    self.translator = Translator()
                    print("Translator initialized successfully.")
                except Exception as e:
                    print(f"Failed to initialize translator: {e}. Arabic text will be classified as Neutral.")
                    self.use_translator = False
            self.class_names = ['Hate Speech', 'Offensive Language', 'Neutral']
        except Exception as e:
            print(f"Failed to initialize scanner: {str(e)}")
            raise

    def preprocess_text(self, text, lang='en'):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
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

    def translate_ar_to_en(self, text):
        if not self.use_translator or self.translator is None:
            return None
        try:
            result = self.translator.translate(text, src='ar', dest='en')
            return result.text
        except Exception as e:
            print(f"Translation failed: {e}")
            return None

    def scan(self, text):
        try:
            try:
                lang = detect(text)
            except LangDetectException:
                lang = 'en'
            if lang == 'ar':
                translated_text = self.translate_ar_to_en(text)
                if translated_text is None:
                    return {
                        "label": "Neutral",
                        "confidence": 1.0
                    }
                text_to_classify = self.preprocess_text(translated_text, lang='en')
            else:
                text_to_classify = self.preprocess_text(text, lang='en')
            X = self.vectorizer_en.transform([text_to_classify])
            classifier = self.classifier_en
            prediction = classifier.predict(X)[0]
            probabilities = classifier.predict_proba(X)[0]
            confidence = float(np.max(probabilities))
            feature_names = self.vectorizer_en.get_feature_names_out()
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
                "translated_text": translated_text if lang == 'ar' else None
            }
        except Exception as e:
            return {
                "label": "Neutral",
                "confidence": 1.0
            }