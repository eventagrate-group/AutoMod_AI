import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from farasa.stemmer import FarasaStemmer
import re
from sklearn.metrics import classification_report
import numpy as np
from config import CONFIG
import os

# Set NLTK data path
nltk.data.path.append(os.environ.get("NLTK_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nltk_data')))

# Download required NLTK data
nltk.download('punkt', quiet=True, download_dir=os.environ.get("NLTK_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nltk_data')))
nltk.download('punkt_tab', quiet=True, download_dir=os.environ.get("NLTK_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nltk_data')))
nltk.download('stopwords', quiet=True, download_dir=os.environ.get("NLTK_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nltk_data')))
nltk.download('wordnet', quiet=True, download_dir=os.environ.get("NLTK_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nltk_data')))

class ToxicTextVerifier:
    def __init__(self):
        self.class_names = ['Hate Speech', 'Offensive Language', 'Neutral']
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = FarasaStemmer()
        # Load English resources
        self.model_path_en = CONFIG['model_path']
        self.vectorizer_path_en = CONFIG['vectorizer_path']
        self.classifier_en = joblib.load(self.model_path_en)
        self.vectorizer_en = joblib.load(self.vectorizer_path_en)
        try:
            self.stop_words_en = set(stopwords.words('english'))
        except LookupError:
            self.stop_words_en = set()
        # Load Arabic resources
        self.model_path_ar = CONFIG['model_path_ar']
        self.vectorizer_path_ar = CONFIG['vectorizer_path_ar']
        self.classifier_ar = joblib.load(self.model_path_ar)
        self.vectorizer_ar = joblib.load(self.vectorizer_path_ar)
        try:
            self.stop_words_ar = set(stopwords.words('arabic'))
        except LookupError:
            self.stop_words_ar = set()

    def preprocess_text(self, text, lang='en'):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        if lang == 'en':
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words_en]
        else:  # lang == 'ar'
            text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
            tokens = self.stemmer.stem(text).split()
            tokens = [token for token in tokens if token not in self.stop_words_ar]
        return ' '.join(tokens)

    def verify(self, lang='en'):
        # Define validation files based on language
        data_dir = os.path.dirname(CONFIG['new_data_path']) if lang == 'en' else os.path.dirname(CONFIG['new_data_path_ar'])
        data_files = [
            (os.path.join(data_dir, 'hate_speech_verify.csv'), 'Hate Speech', 0),
            (os.path.join(data_dir, 'offensive_language_verify.csv'), 'Offensive Language', 1),
            (os.path.join(data_dir, 'neutral_verify.csv'), 'Neutral', 2)
        ]
        dfs = []
        
        for file_path, class_name, class_id in data_files:
            print(f"Loading {lang} validation data from {file_path}...")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    tweets = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                print(f"Warning: {file_path} not found, skipping...")
                continue
            df = pd.DataFrame({
                'tweet': tweets,
                'class': [class_name] * len(tweets),
                'count': [1] * len(tweets),
                'hate_speech_count': [1 if class_name == 'Hate Speech' else 0] * len(tweets),
                'offensive_language_count': [1 if class_name == 'Offensive Language' else 0] * len(tweets),
                'neither_count': [1 if class_name == 'Neutral' else 0] * len(tweets)
            })
            dfs.append(df)
        
        if not dfs:
            print(f"No validation data found for {lang}, aborting verification.")
            return
        
        # Combine data
        df = pd.concat(dfs, ignore_index=True)
        
        # Map string labels to numeric
        label_map = {
            'Hate Speech': 0,
            'Offensive Language': 1,
            'Neutral': 2,
            'hate_speech': 0,
            'offensive': 1,
            'neither': 2,
            'offensive_language': 1,
            'neutral': 2
        }
        print(f"Mapping {lang} string labels to numeric values...")
        df['class'] = df['class'].map(label_map)
        
        # Preprocess text
        print(f"Preprocessing {lang} text...")
        df['processed_tweet'] = [self.preprocess_text(text, lang=lang) for text in df['tweet']]
        
        # Vectorize
        print(f"Vectorizing {lang} text...")
        vectorizer = self.vectorizer_ar if lang == 'ar' else self.vectorizer_en
        classifier = self.classifier_ar if lang == 'ar' else self.classifier_en
        X = vectorizer.transform(df['processed_tweet'])
        y_true = df['class']
        
        # Predict
        print(f"Making predictions for {lang}...")
        y_pred = classifier.predict(X)
        y_proba = classifier.predict_proba(X)
        
        # Classification report
        print(f"\n{lang.upper()} Classification Summary:")
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True, zero_division=0)
        for class_name in self.class_names:
            correct = int(report[class_name]['support'] * report[class_name]['recall'])
            total = int(report[class_name]['support'])
            print(f"{class_name}: {correct}/{total} correct ({report[class_name]['recall']*100:.2f}%)")
        
        # Top influential terms
        print(f"\nTop Influential Terms by Predicted Class ({lang}):")
        feature_names = vectorizer.get_feature_names_out()
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = (y_pred == class_idx)
            if class_mask.sum() > 0:
                class_features = X[class_mask].mean(axis=0).A1
                top_indices = class_features.argsort()[-3:][::-1]
                top_terms = [(feature_names[i], round(class_features[i] * 100)) for i in top_indices]
                print(f"{class_name}: {top_terms}")
        
        # Misclassified examples
        print(f"\nMisclassified Examples (first 5, {lang}):")
        misclassified = y_true != y_pred
        misclassified_indices = np.where(misclassified)[0]
        for idx in misclassified_indices[:5]:
            text = df['tweet'].iloc[idx]
            true_label = self.class_names[y_true.iloc[idx]]
            pred_label = self.class_names[y_pred[idx]]
            confidence = y_proba[idx].max()
            top_features = X[idx].toarray()[0].argsort()[-3:][::-1]
            influential_terms = [feature_names[i] for i in top_features]
            explanation = f"Flagged for {pred_label.lower()} based on terms: {', '.join(influential_terms)}"
            print(f"Text: {text}")
            print(f"True Label: {true_label}")
            print(f"Predicted: {{'label': '{pred_label}', 'confidence': {confidence}, 'explanation': '{explanation}', 'influential_terms': {influential_terms}}}\n")

if __name__ == "__main__":
    verifier = ToxicTextVerifier()
    print("Verifying English model...")
    verifier.verify(lang='en')
    print("\nVerifying Arabic model...")
    verifier.verify(lang='ar')