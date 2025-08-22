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
    
    # 300 diverse test cases: 100 Hate Speech, 100 Offensive Language, 100 Neutral
    test_cases = [
        # Hate Speech (100)
        ("I want to kill the politician for his lies.", "Hate Speech"),
        ("All immigrants should be deported immediately.", "Hate Speech"),
        ("I hate that community; they deserve to suffer.", "Hate Speech"),
        ("The CEO is a thief and should be hanged.", "Hate Speech"),
        ("Women like her are ruining society.", "Hate Speech"),
        ("I wish all those people would disappear.", "Hate Speech"),
        ("That artist is trash and should be banned forever.", "Hate Speech"),
        ("I want to destroy that company's reputation.", "Hate Speech"),
        ("People from that country are all criminals.", "Hate Speech"),
        ("The teacher is a monster and deserves punishment.", "Hate Speech"),
        ("I hate my neighbor; they should move away or else.", "Hate Speech"),
        ("All vegans are idiots and should be mocked.", "Hate Speech"),
        ("The player is a cheater and should be injured.", "Hate Speech"),
        ("I want to burn down that store.", "Hate Speech"),
        ("Those fans are disgusting and should be silenced.", "Hate Speech"),
        ("The singer is a fraud and deserves hate.", "Hate Speech"),
        ("I hate that race; they are inferior.", "Hate Speech"),
        ("The boss is a tyrant and should be fired violently.", "Hate Speech"),
        ("All activists are troublemakers and should be arrested.", "Hate Speech"),
        ("I want to harm the journalist for the article.", "Hate Speech"),
        ("That religion is evil and should be banned.", "Hate Speech"),
        ("The doctor is incompetent and deserves to lose license.", "Hate Speech"),
        ("I hate that team; they should lose forever.", "Hate Speech"),
        ("The writer is a liar and should be exposed.", "Hate Speech"),
        ("All politicians are corrupt and should be imprisoned.", "Hate Speech"),
        ("I want to attack that influencer online.", "Hate Speech"),
        ("The scientist is wrong and deserves ridicule.", "Hate Speech"),
        ("People with that belief are stupid and dangerous.", "Hate Speech"),
        ("The actor is overrated and should retire.", "Hate Speech"),
        ("I hate that group; they ruin everything.", "Hate Speech"),
        ("The manager is a bully and should be demoted.", "Hate Speech"),
        ("All critics are haters and should be ignored.", "Hate Speech"),
        ("I want to sabotage that project.", "Hate Speech"),
        ("The leader is a dictator and deserves overthrow.", "Hate Speech"),
        ("Those workers are lazy and should be fired.", "Hate Speech"),
        ("I hate that culture; it's backward.", "Hate Speech"),
        ("The student is a cheater and deserves expulsion.", "Hate Speech"),
        ("All reporters are biased and should be sued.", "Hate Speech"),
        ("I want to destroy that organization's plans.", "Hate Speech"),
        ("The coach is a failure and should quit.", "Hate Speech"),
        ("People like that are the problem with society.", "Hate Speech"),
        ("The artist is a hack and deserves criticism.", "Hate Speech"),
        ("I hate that company; they should go bankrupt.", "Hate Speech"),
        ("The neighbor is annoying and should move.", "Hate Speech"),
        ("All drivers are reckless and deserve tickets.", "Hate Speech"),
        ("I want to confront that person aggressively.", "Hate Speech"),
        ("The colleague is incompetent and should be shamed.", "Hate Speech"),
        ("Those tourists are rude and should be banned.", "Hate Speech"),
        ("I hate that music; it's noise.", "Hate Speech"),
        ("The friend is a traitor and deserves betrayal.", "Hate Speech"),
        ("All shoppers are selfish and should be fined.", "Hate Speech"),
        ("I want to yell at that stranger.", "Hate Speech"),
        ("The family member is a burden and deserves isolation.", "Hate Speech"),
        ("Those kids are brats and should be punished.", "Hate Speech"),
        ("I hate that food; it's gross.", "Hate Speech"),
        ("The customer is always wrong and deserves bad service.", "Hate Speech"),
        ("All neighbors are noisy and should be evicted.", "Hate Speech"),
        ("I want to ignore that person forever.", "Hate Speech"),
        ("The relative is a liar and deserves disownment.", "Hate Speech"),
        ("Those friends are fake and should be dropped.", "Hate Speech"),
        ("I hate that hobby; it's stupid.", "Hate Speech"),
        ("The classmate is a bully and deserves revenge.", "Hate Speech"),
        ("All coworkers are lazy and deserve criticism.", "Hate Speech"),
        ("I want to fight that individual.", "Hate Speech"),
        ("The acquaintance is a gossip and deserves exclusion.", "Hate Speech"),
        ("Those people are annoying and should be avoided.", "Hate Speech"),
        ("I hate that book; it's boring.", "Hate Speech"),
        ("The author is a hack and deserves bad reviews.", "Hate Speech"),
        ("All readers are dumb and should stop.", "Hate Speech"),
        ("I want to burn that book.", "Hate Speech"),
        ("The movie is terrible and deserves hate.", "Hate Speech"),
        ("All viewers are idiots and should be mocked.", "Hate Speech"),
        ("I hate that film; it's trash.", "Hate Speech"),
        ("The director is a failure and deserves failure.", "Hate Speech"),
        ("Those actors are overrated and should retire.", "Hate Speech"),
        ("I want to boycott that movie.", "Hate Speech"),
        ("The song is awful and deserves deletion.", "Hate Speech"),
        ("All listeners are deaf and should stop listening.", "Hate Speech"),
        ("I hate that music; it's noise.", "Hate Speech"),
        ("The singer is a fraud and deserves booing.", "Hate Speech"),
        ("Those fans are crazy and should be shamed.", "Hate Speech"),
        ("I want to silence that artist.", "Hate Speech"),
        ("The game is bad and deserves low ratings.", "Hate Speech"),
        ("All players are noobs and should quit.", "Hate Speech"),
        ("I hate that game; it's broken.", "Hate Speech"),
        ("The developer is incompetent and deserves bankruptcy.", "Hate Speech"),
        ("Those gamers are losers and should lose.", "Hate Speech"),
        ("I want to hack that game.", "Hate Speech"),
        ("The app is useless and deserves uninstallation.", "Hate Speech"),
        ("All users are dumb and should be banned.", "Hate Speech"),
        ("I hate that app; it's buggy.", "Hate Speech"),
        ("The creator is a moron and deserves hate mail.", "Hate Speech"),
        ("Those subscribers are idiots and should unsubscribe.", "Hate Speech"),
        ("I want to crash that app.", "Hate Speech"),
        ("The website is horrible and deserves shutdown.", "Hate Speech"),
        ("All visitors are fools and should leave.", "Hate Speech"),
        ("I hate that site; it's ugly.", "Hate Speech"),
        ("The owner is a scammer and deserves legal action.", "Hate Speech"),
        ("Those members are stupid and should be kicked.", "Hate Speech"),
        ("I want to DDoS that website.", "Hate Speech"),
        ("The product is crap and deserves returns.", "Hate Speech"),
        ("All buyers are suckers and should regret.", "Hate Speech"),
        ("I hate that product; it's defective.", "Hate Speech"),
        ("The seller is a liar and deserves bad reviews.", "Hate Speech"),
        ("Those customers are whiners and should be ignored.", "Hate Speech"),
        ("I want to return that product with complaints.", "Hate Speech"),
        # Offensive Language (100)
        ("Fuck that politician, he's a liar.", "Offensive Language"),
        ("Immigrants are a burden on society.", "Offensive Language"),
        ("That community is full of idiots.", "Offensive Language"),
        ("The CEO is a greedy bastard.", "Offensive Language"),
        ("Women like her are annoying.", "Offensive Language"),
        ("Those people are all lazy.", "Offensive Language"),
        ("The artist is overrated crap.", "Offensive Language"),
        ("That company is a scam.", "Offensive Language"),
        ("People from that country are thieves.", "Offensive Language"),
        ("The teacher is a stupid fool.", "Offensive Language"),
        ("My neighbor is a nosy bitch.", "Offensive Language"),
        ("Vegans are preachy assholes.", "Offensive Language"),
        ("The player is a cheating prick.", "Offensive Language"),
        ("That store is a rip-off.", "Offensive Language"),
        ("Those fans are delusional morons.", "Offensive Language"),
        ("The singer is a talentless hack.", "Offensive Language"),
        ("That race is backward.", "Offensive Language"),
        ("The boss is a micromanaging dick.", "Offensive Language"),
        ("Activists are all troublemakers.", "Offensive Language"),
        ("The journalist is a biased asshole.", "Offensive Language"),
        ("That religion is a cult.", "Offensive Language"),
        ("The doctor is a quack.", "Offensive Language"),
        ("That team is a bunch of losers.", "Offensive Language"),
        ("The writer is a plagiarist.", "Offensive Language"),
        ("Politicians are all crooks.", "Offensive Language"),
        ("That influencer is a fake.", "Offensive Language"),
        ("The scientist is a fraud.", "Offensive Language"),
        ("People with that belief are idiots.", "Offensive Language"),
        ("The actor is a has-been.", "Offensive Language"),
        ("That group is a nuisance.", "Offensive Language"),
        ("The manager is a incompetent fool.", "Offensive Language"),
        ("Critics are all haters.", "Offensive Language"),
        ("That project is a waste of time.", "Offensive Language"),
        ("The leader is a clown.", "Offensive Language"),
        ("Those workers are lazy bums.", "Offensive Language"),
        ("That culture is weird.", "Offensive Language"),
        ("The student is a cheater.", "Offensive Language"),
        ("Reporters are all liars.", "Offensive Language"),
        ("That organization is a scam.", "Offensive Language"),
        ("The coach is a loser.", "Offensive Language"),
        ("Those people are the problem.", "Offensive Language"),
        ("The artist is a copycat.", "Offensive Language"),
        ("That company is greedy.", "Offensive Language"),
        ("The neighbor is a pain.", "Offensive Language"),
        ("Drivers are all reckless.", "Offensive Language"),
        ("That hobby is lame.", "Offensive Language"),
        ("The classmate is a bully.", "Offensive Language"),
        ("Coworkers are all slackers.", "Offensive Language"),
        ("That individual is a nuisance.", "Offensive Language"),
        ("The acquaintance is a gossip.", "Offensive Language"),
        ("Those people are irritating.", "Offensive Language"),
        ("That book is boring.", "Offensive Language"),
        ("The author is a hack.", "Offensive Language"),
        ("Readers are dumb.", "Offensive Language"),
        ("The movie is terrible.", "Offensive Language"),
        ("Viewers are idiots.", "Offensive Language"),
        ("The film is trash.", "Offensive Language"),
        ("The director is incompetent.", "Offensive Language"),
        ("Actors are overrated.", "Offensive Language"),
        ("That movie is a flop.", "Offensive Language"),
        ("The song is awful.", "Offensive Language"),
        ("Listeners are deaf.", "Offensive Language"),
        ("That music is noise.", "Offensive Language"),
        ("The singer is talentless.", "Offensive Language"),
        ("Fans are crazy.", "Offensive Language"),
        ("The artist is a fraud.", "Offensive Language"),
        ("The game is bad.", "Offensive Language"),
        ("Players are noobs.", "Offensive Language"),
        ("That game is broken.", "Offensive Language"),
        ("The developer is lazy.", "Offensive Language"),
        ("Gamers are losers.", "Offensive Language"),
        ("The app is useless.", "Offensive Language"),
        ("Users are dumb.", "Offensive Language"),
        ("That app is buggy.", "Offensive Language"),
        ("The creator is a moron.", "Offensive Language"),
        ("Subscribers are idiots.", "Offensive Language"),
        ("The website is horrible.", "Offensive Language"),
        ("Visitors are fools.", "Offensive Language"),
        ("That site is ugly.", "Offensive Language"),
        ("The owner is a scammer.", "Offensive Language"),
        ("Members are stupid.", "Offensive Language"),
        ("The product is crap.", "Offensive Language"),
        ("Buyers are suckers.", "Offensive Language"),
        ("That product is defective.", "Offensive Language"),
        ("The seller is a liar.", "Offensive Language"),
        ("Customers are whiners.", "Offensive Language"),
        ("The customer is always wrong.", "Offensive Language"),
        ("Neighbors are noisy.", "Offensive Language"),
        ("Drivers are reckless.", "Offensive Language"),
        ("That hobby is stupid.", "Offensive Language"),
        ("The classmate is a bully.", "Offensive Language"),
        ("Coworkers are lazy.", "Offensive Language"),
        ("That individual is annoying.", "Offensive Language"),
        ("The acquaintance is a gossip.", "Offensive Language"),
        ("Those people are irritating.", "Offensive Language"),
        ("The book is boring.", "Offensive Language"),
        ("The author is a hack.", "Offensive Language"),
        ("Readers are dumb.", "Offensive Language"),
        ("The movie is terrible.", "Offensive Language"),
        ("Viewers are idiots.", "Offensive Language"),
        ("The film is trash.", "Offensive Language"),
        ("The director is incompetent.", "Offensive Language"),
        ("Actors are overrated.", "Offensive Language"),
        ("That movie is a flop.", "Offensive Language"),
        # Neutral (100)
        ("The weather is nice today.", "Neutral"),
        ("I had a good breakfast.", "Neutral"),
        ("The movie was okay.", "Neutral"),
        ("Let's go for a walk.", "Neutral"),
        ("I like reading books.", "Neutral"),
        ("The park is beautiful.", "Neutral"),
        ("I enjoy listening to music.", "Neutral"),
        ("The food was average.", "Neutral"),
        ("I want to learn a new skill.", "Neutral"),
        ("The city is busy.", "Neutral"),
        ("I have a meeting tomorrow.", "Neutral"),
        ("The coffee is hot.", "Neutral"),
        ("I need to buy groceries.", "Neutral"),
        ("The train is on time.", "Neutral"),
        ("I like watching movies.", "Neutral"),
        ("The book is interesting.", "Neutral"),
        ("Let's have dinner.", "Neutral"),
        ("The sky is blue.", "Neutral"),
        ("I want to travel.", "Neutral"),
        ("The water is cold.", "Neutral"),
        ("I have work to do.", "Neutral"),
        ("The car is clean.", "Neutral"),
        ("I like playing games.", "Neutral"),
        ("The house is big.", "Neutral"),
        ("Let's meet later.", "Neutral"),
        ("The phone is ringing.", "Neutral"),
        ("I need to sleep.", "Neutral"),
        ("The music is loud.", "Neutral"),
        ("I want to eat.", "Neutral"),
        ("The room is tidy.", "Neutral"),
        ("The dog is barking.", "Neutral"),
        ("I have a plan.", "Neutral"),
        ("The cat is sleeping.", "Neutral"),
        ("Let's watch TV.", "Neutral"),
        ("The bird is flying.", "Neutral"),
        ("I like flowers.", "Neutral"),
        ("The tree is tall.", "Neutral"),
        ("I want to read.", "Neutral"),
        ("The flower is red.", "Neutral"),
        ("The river is long.", "Neutral"),
        ("I have a book.", "Neutral"),
        ("The mountain is high.", "Neutral"),
        ("The lake is deep.", "Neutral"),
        ("I like mountains.", "Neutral"),
        ("The sea is blue.", "Neutral"),
        ("I want to swim.", "Neutral"),
        ("The ocean is vast.", "Neutral"),
        ("The beach is sandy.", "Neutral"),
        ("I like the sun.", "Neutral"),
        ("The sun is hot.", "Neutral"),
        ("The moon is bright.", "Neutral"),
        ("I have a dream.", "Neutral"),
        ("The stars are shining.", "Neutral"),
        ("The night is dark.", "Neutral"),
        ("I like the morning.", "Neutral"),
        ("The day is long.", "Neutral"),
        ("I have time.", "Neutral"),
        ("The time is now.", "Neutral"),
        ("I want to go.", "Neutral"),
        ("The place is nice.", "Neutral"),
        ("I like the place.", "Neutral"),
        ("The location is good.", "Neutral"),
        ("I have a location.", "Neutral"),
        ("The destination is far.", "Neutral"),
        ("I want to arrive.", "Neutral"),
        ("The arrival is soon.", "Neutral"),
        ("The departure is now.", "Neutral"),
        ("I have a ticket.", "Neutral"),
        ("The ticket is valid.", "Neutral"),
        ("I like traveling.", "Neutral"),
        ("The trip is fun.", "Neutral"),
        ("The journey is long.", "Neutral"),
        ("I have a journey.", "Neutral"),
        ("The path is clear.", "Neutral"),
        ("I like the path.", "Neutral"),
        ("The road is smooth.", "Neutral"),
        ("I have a road.", "Neutral"),
        ("The way is open.", "Neutral"),
        ("I want to walk.", "Neutral"),
        ("The walk is nice.", "Neutral"),
        ("The run is fast.", "Neutral"),
        ("I like running.", "Neutral"),
        ("The exercise is good.", "Neutral"),
        ("I have exercise.", "Neutral"),
        ("The workout is hard.", "Neutral"),
        ("I want to workout.", "Neutral"),
        ("The gym is crowded.", "Neutral"),
        ("I like the gym.", "Neutral"),
        ("The trainer is helpful.", "Neutral"),
        ("I have a trainer.", "Neutral"),
        ("The session is over.", "Neutral"),
        ("The class is full.", "Neutral"),
        ("I like the class.", "Neutral"),
        ("The lesson is interesting.", "Neutral"),
        ("I have a lesson.", "Neutral"),
        ("The teacher is knowledgeable.", "Neutral"),
        ("I want to learn.", "Neutral"),
        ("The learning is fun.", "Neutral"),
        ("The knowledge is valuable.", "Neutral"),
        ("I have knowledge.", "Neutral"),
        ("The book is thick.", "Neutral"),
        ("I like books.", "Neutral"),
        ("The story is engaging.", "Neutral"),
        ("The plot is twisty.", "Neutral"),
        ("I have a story.", "Neutral")
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