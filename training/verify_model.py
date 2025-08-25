import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.metrics import classification_report
import numpy as np
from config import CONFIG

# Set NLTK data path
nltk.data.path.append('/home/branch/nltk_data')

# Download required NLTK data
nltk.download('punkt', quiet=True, download_dir='/home/branch/nltk_data')
nltk.download('punkt_tab', quiet=True, download_dir='/home/branch/nltk_data')
nltk.download('stopwords', quiet=True, download_dir='/home/branch/nltk_data')
nltk.download('wordnet', quiet=True, download_dir='/home/branch/nltk_data')

class ToxicTextVerifier:
    def __init__(self):
        self.model_path = CONFIG['model_path']
        self.vectorizer_path = CONFIG['vectorizer_path']
        self.classifier = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
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

    def verify(self):
        # Load data from three files
        data_paths = [
            '/home/branch/Downloads/hate_speech_verify.csv',
            '/home/branch/Downloads/offensive_language_verify.csv',
            '/home/branch/Downloads/neutral_verify.csv'
        ]
        dfs = []
        for path in data_paths:
            print(f"Loading validation data from {path}...")
            df = pd.read_csv(path)
            dfs.append(df)
        
        # Combine data
        df = pd.concat(dfs, ignore_index=True)
        
        # Identify label column
        possible_label_columns = ['class', 'label', 'sentiment', 'category']
        label_column = None
        for col in possible_label_columns:
            if col in df.columns:
                label_column = col
                break
        if label_column is None:
            raise ValueError(f"No label column found in validation data. Expected one of: {possible_label_columns}")
        
        # Rename label column to 'class'
        if label_column != 'class':
            print(f"Renaming label column '{label_column}' to 'class'...")
            df = df.rename(columns={label_column: 'class'})
        
        # Map string labels to numeric
        label_map = {
            'Hate Speech': 0,
            'Offensive Language': 1,
            'Neutral': 2,
            'hate_speech': 0,
            'offensive': 1,
            'neither': 2
        }
        if df['class'].dtype == 'object':
            print("Mapping string labels to numeric values...")
            missing_labels = df['class'][~df['class'].isin(label_map.keys())].unique()
            if len(missing_labels) > 0:
                raise ValueError(f"Unknown string labels found: {missing_labels}")
            df['class'] = df['class'].map(label_map)
        
        # Preprocess text
        print("Preprocessing text...")
        df['processed_tweet'] = [self.preprocess_text(text) for text in df['tweet']]
        
        # Vectorize
        print("Vectorizing text...")
        X = self.vectorizer.transform(df['processed_tweet'])
        y_true = df['class']
        
        # Predict
        print("Making predictions...")
        y_pred = self.classifier.predict(X)
        y_proba = self.classifier.predict_proba(X)
        
        # Classification report
        print("Classification Summary:")
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        for class_name in self.class_names:
            correct = int(report[class_name]['support'] * report[class_name]['recall'])
            total = int(report[class_name]['support'])
            print(f"{class_name}: {correct}/{total} correct ({report[class_name]['recall']*100:.2f}%)")
        
        # Top influential terms
        print("\nTop Influential Terms by Predicted Class:")
        feature_names = self.vectorizer.get_feature_names_out()
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = (y_pred == class_idx)
            if class_mask.sum() > 0:
                class_features = X[class_mask].mean(axis=0).A1
                top_indices = class_features.argsort()[-3:][::-1]
                top_terms = [(feature_names[i], round(class_features[i] * 100)) for i in top_indices]
                print(f"{class_name}: {top_terms}")
        
        # Misclassified examples
        print("\nMisclassified Examples (first 5):")
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
    verifier.verify()