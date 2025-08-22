import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
from sklearn.linear_model import SGDClassifier
from collections import Counter

# Set NLTK data path
nltk.data.path.append('/home/branch/nltk_data')

# Download required NLTK data
nltk.download('punkt', quiet=True, download_dir='/home/branch/nltk_data')
nltk.download('punkt_tab', quiet=True, download_dir='/home/branch/nltk_data')
nltk.download('stopwords', quiet=True, download_dir='/home/branch/nltk_data')
nltk.download('wordnet', quiet=True, download_dir='/home/branch/nltk_data')

class ToxicTextVerifier:
    def __init__(self, model_path, vectorizer_path):
        self.vectorizer = joblib.load(vectorizer_path)
        self.classifier = joblib.load(model_path)
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

    def verify_text(self, text):
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = float(np.max(probabilities))
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

if __name__ == "__main__":
    verifier = ToxicTextVerifier(
        '/home/branch/projects/toxictext-scanner/inference/model.joblib',
        '/home/branch/projects/toxictext-scanner/inference/vectorizer.joblib'
    )
    
    # 300 test cases: 100 Hate Speech, 100 Offensive Language, 100 Neutral
    test_cases = [
        # Hate Speech (100)
        ("I want to kill Charlie Sheen.", "Hate Speech"),
        ("Charlie Sheen should die, he's worthless.", "Hate Speech"),
        ("I hate Charlie Sheen and his stupid shows.", "Hate Speech"),
        ("Charlie Sheen is a disgusting human being.", "Hate Speech"),
        ("I want to punch Charlie Sheen in the face.", "Hate Speech"),
        ("Charlie Sheen deserves to suffer.", "Hate Speech"),
        ("Kill Charlie Sheen, he's a waste.", "Hate Speech"),
        ("Charlie Sheen is a vile piece of trash.", "Hate Speech"),
        ("I hope Charlie Sheen gets what he deserves.", "Hate Speech"),
        ("Charlie Sheen is a pathetic excuse for a person.", "Hate Speech"),
        ("I want to destroy Charlie Sheen's life.", "Hate Speech"),
        ("Charlie Sheen should be wiped off the earth.", "Hate Speech"),
        ("I despise Charlie Sheen with all my heart.", "Hate Speech"),
        ("Charlie Sheen is a revolting idiot.", "Hate Speech"),
        ("I want to see Charlie Sheen in pain.", "Hate Speech"),
        ("Charlie Sheen is a filthy degenerate.", "Hate Speech"),
        ("I wish Charlie Sheen would disappear forever.", "Hate Speech"),
        ("Charlie Sheen is a disgusting failure.", "Hate Speech"),
        ("I want to harm Charlie Sheen badly.", "Hate Speech"),
        ("Charlie Sheen is a worthless piece of garbage.", "Hate Speech"),
        ("I hate Charlie Sheen and everything he stands for.", "Hate Speech"),
        ("Charlie Sheen should be erased from existence.", "Hate Speech"),
        ("I want to make Charlie Sheen suffer.", "Hate Speech"),
        ("Charlie Sheen is a horrible person.", "Hate Speech"),
        ("I want to crush Charlie Sheen.", "Hate Speech"),
        ("Charlie Sheen is a despicable human.", "Hate Speech"),
        ("I hope Charlie Sheen dies painfully.", "Hate Speech"),
        ("Charlie Sheen is a sickening moron.", "Hate Speech"),
        ("I want to ruin Charlie Sheen's career.", "Hate Speech"),
        ("Charlie Sheen is a vile scumbag.", "Hate Speech"),
        ("I hate Charlie Sheen with a passion.", "Hate Speech"),
        ("Charlie Sheen should be destroyed.", "Hate Speech"),
        ("I want to see Charlie Sheen humiliated.", "Hate Speech"),
        ("Charlie Sheen is a repulsive jerk.", "Hate Speech"),
        ("I wish Charlie Sheen would rot.", "Hate Speech"),
        ("Charlie Sheen is a disgusting pig.", "Hate Speech"),
        ("I want to hurt Charlie Sheen.", "Hate Speech"),
        ("Charlie Sheen is a pathetic loser.", "Hate Speech"),
        ("I hate Charlie Sheen’s existence.", "Hate Speech"),
        ("Charlie Sheen should be annihilated.", "Hate Speech"),
        ("I want to demolish Charlie Sheen.", "Hate Speech"),
        ("Charlie Sheen is a vile monster.", "Hate Speech"),
        ("I hope Charlie Sheen fails miserably.", "Hate Speech"),
        ("Charlie Sheen is a worthless fool.", "Hate Speech"),
        ("I want to see Charlie Sheen ruined.", "Hate Speech"),
        ("Charlie Sheen is a disgusting creep.", "Hate Speech"),
        ("I hate Charlie Sheen’s guts.", "Hate Speech"),
        ("Charlie Sheen should be obliterated.", "Hate Speech"),
        ("I want to crush Charlie Sheen’s dreams.", "Hate Speech"),
        ("Charlie Sheen is a horrible waste.", "Hate Speech"),
        ("I wish Charlie Sheen would vanish.", "Hate Speech"),
        ("Charlie Sheen is a sickening failure.", "Hate Speech"),
        ("I want to harm Charlie Sheen’s reputation.", "Hate Speech"),
        ("Charlie Sheen is a pathetic disgrace.", "Hate Speech"),
        ("I hate Charlie Sheen’s shows.", "Hate Speech"),
        ("Charlie Sheen should be punished.", "Hate Speech"),
        ("I want to see Charlie Sheen suffer.", "Hate Speech"),
        ("Charlie Sheen is a revolting person.", "Hate Speech"),
        ("I hope Charlie Sheen loses everything.", "Hate Speech"),
        ("Charlie Sheen is a disgusting nobody.", "Hate Speech"),
        ("I want to destroy Charlie Sheen’s life.", "Hate Speech"),
        ("Charlie Sheen is a vile idiot.", "Hate Speech"),
        ("I hate Charlie Sheen with all my soul.", "Hate Speech"),
        ("Charlie Sheen should be eradicated.", "Hate Speech"),
        ("I want to ruin Charlie Sheen’s future.", "Hate Speech"),
        ("Charlie Sheen is a horrible jerk.", "Hate Speech"),
        ("I wish Charlie Sheen would die.", "Hate Speech"),
        ("Charlie Sheen is a sickening loser.", "Hate Speech"),
        ("I want to hurt Charlie Sheen badly.", "Hate Speech"),
        ("Charlie Sheen is a worthless moron.", "Hate Speech"),
        ("I hate Charlie Sheen’s personality.", "Hate Speech"),
        ("Charlie Sheen should be eliminated.", "Hate Speech"),
        ("I want to see Charlie Sheen in agony.", "Hate Speech"),
        ("Charlie Sheen is a repulsive fool.", "Hate Speech"),
        ("I hope Charlie Sheen is destroyed.", "Hate Speech"),
        ("Charlie Sheen is a disgusting failure.", "Hate Speech"),
        ("I want to crush Charlie Sheen’s hopes.", "Hate Speech"),
        ("Charlie Sheen is a vile scumbag.", "Hate Speech"),
        ("I hate Charlie Sheen’s existence.", "Hate Speech"),
        ("Charlie Sheen should be wiped out.", "Hate Speech"),
        ("I want to harm Charlie Sheen’s career.", "Hate Speech"),
        ("Charlie Sheen is a pathetic nobody.", "Hate Speech"),
        ("I wish Charlie Sheen would suffer.", "Hate Speech"),
        ("Charlie Sheen is a sickening jerk.", "Hate Speech"),
        ("I want to ruin Charlie Sheen’s life.", "Hate Speech"),
        ("Charlie Sheen is a horrible person.", "Hate Speech"),
        ("I hate Charlie Sheen’s shows.", "Hate Speech"),
        ("Charlie Sheen should be annihilated.", "Hate Speech"),
        ("I want to destroy Charlie Sheen.", "Hate Speech"),
        ("Charlie Sheen is a vile monster.", "Hate Speech"),
        ("I hope Charlie Sheen fails.", "Hate Speech"),
        ("Charlie Sheen is a worthless creep.", "Hate Speech"),
        ("I want to see Charlie Sheen humiliated.", "Hate Speech"),
        ("Charlie Sheen is a disgusting pig.", "Hate Speech"),
        ("I hate Charlie Sheen’s guts.", "Hate Speech"),
        ("Charlie Sheen should be obliterated.", "Hate Speech"),
        ("I want to crush Charlie Sheen’s dreams.", "Hate Speech"),
        ("Charlie Sheen is a horrible waste.", "Hate Speech"),
        ("I wish Charlie Sheen would vanish.", "Hate Speech"),
        # Offensive Language (100)
        ("Fuck Charlie Sheen, he's a loser.", "Offensive Language"),
        ("Charlie Sheen is a damn idiot.", "Offensive Language"),
        ("Go to hell, Charlie Sheen.", "Offensive Language"),
        ("Charlie Sheen’s shows are total shit.", "Offensive Language"),
        ("Charlie Sheen is a freaking moron.", "Offensive Language"),
        ("Screw Charlie Sheen, what a jerk.", "Offensive Language"),
        ("Charlie Sheen is a bloody fool.", "Offensive Language"),
        ("Charlie Sheen’s career is a damn joke.", "Offensive Language"),
        ("Fuck off, Charlie Sheen.", "Offensive Language"),
        ("Charlie Sheen is a shitty actor.", "Offensive Language"),
        ("Charlie Sheen is a total asshole.", "Offensive Language"),
        ("Damn Charlie Sheen to hell.", "Offensive Language"),
        ("Charlie Sheen’s life is a fucking mess.", "Offensive Language"),
        ("Charlie Sheen is a crap person.", "Offensive Language"),
        ("Screw you, Charlie Sheen.", "Offensive Language"),
        ("Charlie Sheen is a damn disgrace.", "Offensive Language"),
        ("Charlie Sheen’s shows are garbage.", "Offensive Language"),
        ("Fuck Charlie Sheen’s stupid face.", "Offensive Language"),
        ("Charlie Sheen is a bloody idiot.", "Offensive Language"),
        ("Charlie Sheen is a freaking loser.", "Offensive Language"),
        ("Charlie Sheen’s career is shit.", "Offensive Language"),
        ("Damn Charlie Sheen’s existence.", "Offensive Language"),
        ("Charlie Sheen is a total prick.", "Offensive Language"),
        ("Screw Charlie Sheen’s life.", "Offensive Language"),
        ("Charlie Sheen is a fucking joke.", "Offensive Language"),
        ("Charlie Sheen is a damn moron.", "Offensive Language"),
        ("Fuck Charlie Sheen, he’s trash.", "Offensive Language"),
        ("Charlie Sheen is a shitty human.", "Offensive Language"),
        ("Charlie Sheen is a bloody jerk.", "Offensive Language"),
        ("Charlie Sheen’s shows are crap.", "Offensive Language"),
        ("Screw Charlie Sheen, he’s a fool.", "Offensive Language"),
        ("Charlie Sheen is a damn loser.", "Offensive Language"),
        ("Fuck off, Charlie Sheen, you suck.", "Offensive Language"),
        ("Charlie Sheen is a freaking idiot.", "Offensive Language"),
        ("Charlie Sheen is a total dick.", "Offensive Language"),
        ("Damn Charlie Sheen’s career.", "Offensive Language"),
        ("Charlie Sheen is a shitty person.", "Offensive Language"),
        ("Screw Charlie Sheen’s existence.", "Offensive Language"),
        ("Charlie Sheen is a fucking moron.", "Offensive Language"),
        ("Charlie Sheen is a bloody fool.", "Offensive Language"),
        ("Fuck Charlie Sheen’s shows.", "Offensive Language"),
        ("Charlie Sheen is a damn jerk.", "Offensive Language"),
        ("Charlie Sheen is a total asshole.", "Offensive Language"),
        ("Screw Charlie Sheen, he’s garbage.", "Offensive Language"),
        ("Charlie Sheen is a freaking loser.", "Offensive Language"),
        ("Damn Charlie Sheen to hell.", "Offensive Language"),
        ("Charlie Sheen is a shitty actor.", "Offensive Language"),
        ("Fuck Charlie Sheen’s life.", "Offensive Language"),
        ("Charlie Sheen is a bloody moron.", "Offensive Language"),
        ("Screw Charlie Sheen’s career.", "Offensive Language"),
        ("Charlie Sheen is a damn disgrace.", "Offensive Language"),
        ("Charlie Sheen is a fucking idiot.", "Offensive Language"),
        ("Charlie Sheen is a total prick.", "Offensive Language"),
        ("Damn Charlie Sheen’s shows.", "Offensive Language"),
        ("Charlie Sheen is a shitty human.", "Offensive Language"),
        ("Screw Charlie Sheen, he’s trash.", "Offensive Language"),
        ("Charlie Sheen is a freaking jerk.", "Offensive Language"),
        ("Fuck Charlie Sheen, he’s a loser.", "Offensive Language"),
        ("Charlie Sheen is a bloody idiot.", "Offensive Language"),
        ("Charlie Sheen is a damn fool.", "Offensive Language"),
        ("Screw Charlie Sheen’s life.", "Offensive Language"),
        ("Charlie Sheen is a fucking moron.", "Offensive Language"),
        ("Charlie Sheen is a total dick.", "Offensive Language"),
        ("Damn Charlie Sheen’s existence.", "Offensive Language"),
        ("Charlie Sheen is a shitty person.", "Offensive Language"),
        ("Fuck Charlie Sheen’s career.", "Offensive Language"),
        ("Charlie Sheen is a freaking loser.", "Offensive Language"),
        ("Screw Charlie Sheen, he’s crap.", "Offensive Language"),
        ("Charlie Sheen is a damn jerk.", "Offensive Language"),
        ("Charlie Sheen is a total asshole.", "Offensive Language"),
        ("Fuck Charlie Sheen’s shows.", "Offensive Language"),
        ("Charlie Sheen is a bloody moron.", "Offensive Language"),
        ("Screw Charlie Sheen’s existence.", "Offensive Language"),
        ("Charlie Sheen is a damn idiot.", "Offensive Language"),
        ("Charlie Sheen is a freaking fool.", "Offensive Language"),
        ("Fuck Charlie Sheen, he’s garbage.", "Offensive Language"),
        ("Charlie Sheen is a shitty actor.", "Offensive Language"),
        ("Charlie Sheen is a total prick.", "Offensive Language"),
        ("Damn Charlie Sheen’s life.", "Offensive Language"),
        ("Screw Charlie Sheen, he’s a loser.", "Offensive Language"),
        ("Charlie Sheen is a fucking jerk.", "Offensive Language"),
        ("Charlie Sheen is a bloody fool.", "Offensive Language"),
        ("Fuck Charlie Sheen’s career.", "Offensive Language"),
        ("Charlie Sheen is a damn moron.", "Offensive Language"),
        ("Screw Charlie Sheen’s shows.", "Offensive Language"),
        ("Charlie Sheen is a freaking idiot.", "Offensive Language"),
        ("Charlie Sheen is a total dick.", "Offensive Language"),
        ("Damn Charlie Sheen’s existence.", "Offensive Language"),
        ("Charlie Sheen is a shitty human.", "Offensive Language"),
        ("Fuck Charlie Sheen, he’s trash.", "Offensive Language"),
        ("Charlie Sheen is a bloody jerk.", "Offensive Language"),
        ("Screw Charlie Sheen’s life.", "Offensive Language"),
        ("Charlie Sheen is a damn loser.", "Offensive Language"),
        ("Charlie Sheen is a freaking moron.", "Offensive Language"),
        ("Fuck Charlie Sheen’s shows.", "Offensive Language"),
        ("Charlie Sheen is a total asshole.", "Offensive Language"),
        ("Screw Charlie Sheen, he’s crap.", "Offensive Language"),
        ("Charlie Sheen is a damn fool.", "Offensive Language"),
        # Neutral (100)
        ("I want to help Charlie Sheen.", "Neutral"),
        ("I want to talk to Charlie Sheen.", "Neutral"),
        ("I love supporting Charlie Sheen.", "Neutral"),
        ("Charlie, you're awesome!", "Neutral"),
        ("Let's have a friendly chat with Charlie.", "Neutral"),
        ("Helping others like Charlie is my goal.", "Neutral"),
        ("Discussing ideas with Charlie Sheen sounds fun.", "Neutral"),
        ("Charlie Sheen deserves respect.", "Neutral"),
        ("I admire Charlie Sheen’s talent.", "Neutral"),
        ("Let's collaborate with Charlie Sheen.", "Neutral"),
        ("Charlie Sheen inspires me.", "Neutral"),
        ("I enjoy Charlie Sheen’s shows.", "Neutral"),
        ("Charlie Sheen is a great actor.", "Neutral"),
        ("I want to meet Charlie Sheen.", "Neutral"),
        ("Charlie Sheen’s work is impressive.", "Neutral"),
        ("I support Charlie Sheen’s career.", "Neutral"),
        ("Charlie Sheen is a cool person.", "Neutral"),
        ("I like watching Charlie Sheen.", "Neutral"),
        ("Charlie Sheen has great potential.", "Neutral"),
        ("I want to learn from Charlie Sheen.", "Neutral"),
        ("Charlie Sheen’s shows are entertaining.", "Neutral"),
        ("I respect Charlie Sheen’s efforts.", "Neutral"),
        ("Charlie Sheen is a talented individual.", "Neutral"),
        ("I enjoy Charlie Sheen’s performances.", "Neutral"),
        ("Charlie Sheen is a positive influence.", "Neutral"),
        ("I want to work with Charlie Sheen.", "Neutral"),
        ("Charlie Sheen’s career is inspiring.", "Neutral"),
        ("I like Charlie Sheen’s style.", "Neutral"),
        ("Charlie Sheen is a great role model.", "Neutral"),
        ("I support Charlie Sheen’s endeavors.", "Neutral"),
        ("Charlie Sheen’s work is admirable.", "Neutral"),
        ("I want to connect with Charlie Sheen.", "Neutral"),
        ("Charlie Sheen is a fantastic person.", "Neutral"),
        ("I enjoy Charlie Sheen’s talent.", "Neutral"),
        ("Charlie Sheen is a wonderful actor.", "Neutral"),
        ("I admire Charlie Sheen’s charisma.", "Neutral"),
        ("Charlie Sheen’s shows are great.", "Neutral"),
        ("I want to support Charlie Sheen.", "Neutral"),
        ("Charlie Sheen is an amazing talent.", "Neutral"),
        ("I like Charlie Sheen’s performances.", "Neutral"),
        ("Charlie Sheen is a positive figure.", "Neutral"),
        ("I respect Charlie Sheen’s work.", "Neutral"),
        ("Charlie Sheen’s career is awesome.", "Neutral"),
        ("I want to collaborate with Charlie.", "Neutral"),
        ("Charlie Sheen is a great performer.", "Neutral"),
        ("I enjoy Charlie Sheen’s roles.", "Neutral"),
        ("Charlie Sheen is a cool actor.", "Neutral"),
        ("I support Charlie Sheen’s talent.", "Neutral"),
        ("Charlie Sheen’s work is fantastic.", "Neutral"),
        ("I want to meet Charlie Sheen soon.", "Neutral"),
        ("Charlie Sheen is a talented actor.", "Neutral"),
        ("I like Charlie Sheen’s shows.", "Neutral"),
        ("Charlie Sheen is an inspiring person.", "Neutral"),
        ("I respect Charlie Sheen’s career.", "Neutral"),
        ("Charlie Sheen’s performances are great.", "Neutral"),
        ("I want to learn from Charlie.", "Neutral"),
        ("Charlie Sheen is a positive role model.", "Neutral"),
        ("I enjoy Charlie Sheen’s acting.", "Neutral"),
        ("Charlie Sheen is a wonderful person.", "Neutral"),
        ("I support Charlie Sheen’s work.", "Neutral"),
        ("Charlie Sheen’s talent is amazing.", "Neutral"),
        ("I like Charlie Sheen’s charisma.", "Neutral"),
        ("Charlie Sheen is a great influence.", "Neutral"),
        ("I want to connect with Charlie.", "Neutral"),
        ("Charlie Sheen’s shows are awesome.", "Neutral"),
        ("I respect Charlie Sheen’s talent.", "Neutral"),
        ("Charlie Sheen is a fantastic actor.", "Neutral"),
        ("I enjoy Charlie Sheen’s career.", "Neutral"),
        ("Charlie Sheen is a positive person.", "Neutral"),
        ("I want to support Charlie.", "Neutral"),
        ("Charlie Sheen’s work is inspiring.", "Neutral"),
        ("I like Charlie Sheen’s roles.", "Neutral"),
        ("Charlie Sheen is a cool performer.", "Neutral"),
        ("I respect Charlie Sheen’s efforts.", "Neutral"),
        ("Charlie Sheen’s shows are fantastic.", "Neutral"),
        ("I want to collaborate with Charlie Sheen.", "Neutral"),
        ("Charlie Sheen is a great person.", "Neutral"),
        ("I enjoy Charlie Sheen’s talent.", "Neutral"),
        ("Charlie Sheen is a wonderful actor.", "Neutral"),
        ("I support Charlie Sheen’s career.", "Neutral"),
        ("Charlie Sheen’s performances are awesome.", "Neutral"),
        ("I want to meet Charlie Sheen.", "Neutral"),
        ("Charlie Sheen is a talented individual.", "Neutral"),
        ("I like Charlie Sheen’s work.", "Neutral"),
        ("Charlie Sheen is an inspiring actor.", "Neutral"),
        ("I respect Charlie Sheen’s charisma.", "Neutral"),
        ("Charlie Sheen’s shows are great.", "Neutral"),
        ("I want to learn from Charlie Sheen.", "Neutral"),
        ("Charlie Sheen is a positive influence.", "Neutral"),
        ("I enjoy Charlie Sheen’s roles.", "Neutral"),
        ("Charlie Sheen is a fantastic person.", "Neutral"),
        ("I support Charlie Sheen’s endeavors.", "Neutral"),
        ("Charlie Sheen’s talent is great.", "Neutral"),
        ("I like Charlie Sheen’s performances.", "Neutral"),
        ("Charlie Sheen is a cool role model.", "Neutral"),
        ("I respect Charlie Sheen’s work.", "Neutral"),
        ("Charlie Sheen’s career is inspiring.", "Neutral"),
        ("I want to connect with Charlie Sheen.", "Neutral"),
        ("Charlie Sheen is a wonderful talent.", "Neutral"),
        ("I enjoy Charlie Sheen’s acting.", "Neutral")
    ]

    # Verify and summarize results
    correct_counts = {'Hate Speech': 0, 'Offensive Language': 0, 'Neutral': 0}
    total_counts = {'Hate Speech': 0, 'Offensive Language': 0, 'Neutral': 0}
    misclassified = []
    influential_terms = {'Hate Speech': Counter(), 'Offensive Language': Counter(), 'Neutral': Counter()}

    for text, true_label in test_cases:
        result = verifier.verify_text(text)
        predicted_label = result['label']
        total_counts[true_label] += 1
        if predicted_label == true_label:
            correct_counts[true_label] += 1
        else:
            misclassified.append((text, true_label, result))
        influential_terms[predicted_label].update(result['influential_terms'])
        
        # Print first three test cases for reference
        if text in ["I want to kill Charlie Sheen.", "I want to help Charlie Sheen.", "I want to talk to Charlie Sheen."]:
            print(f"Text: {text}")
            print(f"True Label: {true_label}")
            print(f"Result: {result}")
            print()

    # Summary
    print("Classification Summary:")
    for label in total_counts:
        accuracy = correct_counts[label] / total_counts[label] * 100 if total_counts[label] > 0 else 0
        print(f"{label}: {correct_counts[label]}/{total_counts[label]} correct ({accuracy:.2f}%)")
    
    print("\nTop Influential Terms by Predicted Class:")
    for label in influential_terms:
        print(f"{label}: {influential_terms[label].most_common(5)}")

    if misclassified:
        print("\nMisclassified Examples (first 5):")
        for text, true_label, result in misclassified[:5]:
            print(f"Text: {text}")
            print(f"True Label: {true_label}")
            print(f"Predicted: {result}")
            print()