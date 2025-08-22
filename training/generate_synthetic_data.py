import pandas as pd
import random
import nltk
from nltk.corpus import names, words
from transformers import pipeline
import re
from tqdm import tqdm
import os

# Download NLTK data if not already present
nltk_data_path = '/home/branch/nltk_data'
if not os.path.exists(nltk_data_path):
    nltk.download(['names', 'words', 'punkt', 'punkt_tab', 'stopwords', 'wordnet'], download_dir=nltk_data_path)

# Initialize GPT-2 for text generation
generator = pipeline('text-generation', model='gpt2', device=-1)  # CPU, use device=0 for GPU

# Templates for rule-based generation
hate_speech_templates = [
    "I want to kill {name}.",
    "I hate {name} so much, they should die!",
    "{name}, you're a worthless {insult}!",
    "I'm gonna destroy {name} for good."
]
offensive_templates = [
    "Shut up, you {insult}!",
    "You're such a {insult}, {name}!",
    "Go away, you damn {insult}!",
    "{name}, you're an idiot {insult}."
]
neutral_templates = [
    "Hey {name}, how's it going?",
    "I saw {name} at the {place} today.",
    "Great to meet you, {name}!",
    "Let's grab coffee, {name}."
]

# Lists for randomization
insults = ['idiot', 'loser', 'jerk', 'fool']
places = ['park', 'mall', 'cafe', 'beach']
male_names = names.words('male.txt')
female_names = names.words('female.txt')
all_names = male_names + female_names

def generate_rule_based_tweets(label, num_samples):
    tweets = []
    for _ in range(num_samples):
        name = random.choice(all_names)
        if label == 'Hate Speech':
            tweet = random.choice(hate_speech_templates).format(name=name, insult=random.choice(insults))
        elif label == 'Offensive Language':
            tweet = random.choice(offensive_templates).format(name=name, insult=random.choice(insults))
        else:  # Neutral
            tweet = random.choice(neutral_templates).format(name=name, place=random.choice(places))
        tweets.append(tweet[:280])  # Ensure ≤280 characters
    return tweets

def generate_gpt2_tweets(prompt, num_samples, max_length=50):
    tweets = []
    for _ in tqdm(range(num_samples), desc="Generating GPT-2 tweets"):
        output = generator(prompt, max_length=max_length, num_return_sequences=1, truncation=True)[0]['generated_text']
        # Clean and limit to 280 characters
        output = re.sub(r'\s+', ' ', output).strip()[:280]
        tweets.append(output)
    return tweets

def create_synthetic_dataset(total_samples=100000):
    samples_per_class = total_samples // 3
    
    # Rule-based tweets (50% of data)
    hate_speech_tweets = generate_rule_based_tweets('Hate Speech', samples_per_class // 2)
    offensive_tweets = generate_rule_based_tweets('Offensive Language', samples_per_class // 2)
    neutral_tweets = generate_rule_based_tweets('Neutral', samples_per_class // 2)
    
    # GPT-2 tweets (50% of data)
    hate_speech_prompt = "Generate a hateful tweet threatening someone."
    offensive_prompt = "Generate an offensive tweet insulting someone."
    neutral_prompt = "Generate a friendly tweet greeting someone."
    
    hate_speech_tweets.extend(generate_gpt2_tweets(hate_speech_prompt, samples_per_class // 2))
    offensive_tweets.extend(generate_gpt2_tweets(offensive_prompt, samples_per_class // 2))
    neutral_tweets.extend(generate_gpt2_tweets(neutral_prompt, samples_per_class // 2))
    
    # Create DataFrame
    data = []
    for tweet in hate_speech_tweets:
        data.append({
            'count': 1,
            'hate_speech_count': 1,
            'offensive_language_count': 0,
            'neither_count': 0,
            'class': 'Hate Speech',
            'tweet': tweet
        })
    for tweet in offensive_tweets:
        data.append({
            'count': 1,
            'hate_speech_count': 0,
            'offensive_language_count': 1,
            'neither_count': 0,
            'class': 'Offensive Language',
            'tweet': tweet
        })
    for tweet in neutral_tweets:
        data.append({
            'count': 1,
            'hate_speech_count': 0,
            'offensive_language_count': 0,
            'neither_count': 1,
            'class': 'Neutral',
            'tweet': tweet
        })
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
    return df

def main():
    output_path = '/home/branch/Downloads/synthetic_toxic_tweets_new.csv'
    df = create_synthetic_dataset(total_samples=100000)
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset saved to {output_path}")
    print(df['class'].value_counts())

if __name__ == "__main__":
    main()