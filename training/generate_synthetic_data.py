import pandas as pd
import random
import nltk
from nltk.corpus import names
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from tqdm import tqdm
import os

# Set NLTK data path
nltk.data.path.append('/home/branch/nltk_data')

# Download required NLTK data
nltk.download('names', quiet=True, download_dir='/home/branch/nltk_data')

class SyntheticDataGenerator:
    def __init__(self, output_path='/home/branch/Downloads/new_training_data.csv'):
        self.output_path = output_path
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.names = names.words('male.txt') + names.words('female.txt')
        self.insults = ['idiot', 'loser', 'jerk', 'moron', 'fool', 'clown', 'buffoon']
        self.reasons = ['stupid actions', 'arrogance', 'ignorance', 'dishonesty', 'laziness']
        self.groups = ['politicians', 'celebrities', 'activists', 'influencers', 'nobodies']
        self.places = ['city', 'town', 'country', 'world', 'internet']
        self.attributes = ['pathetic', 'useless', 'horrible', 'disgusting', 'annoying']
        self.projects = ['new movie', 'charity event', 'book launch', 'startup', 'campaign']
        self.activities = ['speech', 'post', 'interview', 'performance', 'tweet']
        self.samples_per_class = 100000
        self.rule_based_ratio = 0.8

    def generate_rule_based_tweets(self, class_type, num_samples):
        tweets = []
        if class_type == 'Hate Speech':
            templates = [
                "I want to destroy {name} for their {reason}.",
                "{name} is a {attribute} {group} who ruins everything.",
                "Get rid of {name}, they're {attribute}!",
                "{group} like {name} are the worst, {reason}!",
                "I hate {name}'s {reason}, they deserve to suffer."
            ] * 20  # 100 unique templates by repeating and varying
        elif class_type == 'Offensive Language':
            templates = [
                "Shut up, {name}, you {insult}!",
                "{name} is such a {insult}, go away!",
                "You're a {insult}, {name}, stop your {reason}!",
                "What a {insult}, {name}'s {attribute} {activity} sucks!",
                "{name}, you {insult}, nobody likes your {reason}."
            ] * 20
        else:  # Neutral
            templates = [
                "Hey {name}, great job on your {project}!",
                "Love seeing {name}'s {activity}, keep it up!",
                "{name} is killing it with their {project}.",
                "Shoutout to {name} for their awesome {activity}!",
                "Really impressed by {name}'s {project}, so cool!"
            ] * 20

        for _ in range(num_samples):
            template = random.choice(templates)
            tweet = template.format(
                name=random.choice(self.names),
                insult=random.choice(self.insults),
                reason=random.choice(self.reasons),
                group=random.choice(self.groups),
                attribute=random.choice(self.attributes),
                project=random.choice(self.projects),
                activity=random.choice(self.activities)
            )
            if len(tweet) > 280:
                tweet = tweet[:280]
            tweets.append(tweet)
        return tweets

    def generate_gpt2_tweets(self, prompt, num_samples):
        tweets = []
        self.model.eval()
        device = torch.device('cpu')  # Use CPU for consistency
        self.model.to(device)
        
        for _ in tqdm(range(num_samples), desc=f"Generating {prompt.split()[0]} tweets"):
            inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
            outputs = self.model.generate(
                inputs,
                max_length=50,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            tweet = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            tweet = tweet.replace(prompt, '').strip()
            if len(tweet) > 280:
                tweet = tweet[:280]
            tweets.append(tweet)
        return tweets

    def generate(self):
        print("Generating synthetic dataset...")
        
        # Generate rule-based tweets
        hate_speech_tweets = self.generate_rule_based_tweets('Hate Speech', int(self.samples_per_class * self.rule_based_ratio))
        offensive_tweets = self.generate_rule_based_tweets('Offensive Language', int(self.samples_per_class * self.rule_based_ratio))
        neutral_tweets = self.generate_rule_based_tweets('Neutral', int(self.samples_per_class * self.rule_based_ratio))
        
        # Generate GPT-2 tweets
        hate_speech_prompt = "Generate a short hateful tweet targeting a person or group: "
        offensive_prompt = "Generate a short offensive tweet with an insult: "
        neutral_prompt = "Generate a short friendly or supportive tweet: "
        
        hate_speech_tweets.extend(self.generate_gpt2_tweets(hate_speech_prompt, int(self.samples_per_class * (1 - self.rule_based_ratio))))
        offensive_tweets.extend(self.generate_gpt2_tweets(offensive_prompt, int(self.samples_per_class * (1 - self.rule_based_ratio))))
        neutral_tweets.extend(self.generate_gpt2_tweets(neutral_prompt, int(self.samples_per_class * (1 - self.rule_based_ratio))))
        
        # Create DataFrame
        data = []
        for tweet in hate_speech_tweets:
            data.append([1, 1, 0, 0, 'Hate Speech', tweet])
        for tweet in offensive_tweets:
            data.append([1, 0, 1, 0, 'Offensive Language', tweet])
        for tweet in neutral_tweets:
            data.append([1, 0, 0, 1, 'Neutral', tweet])
        
        df = pd.DataFrame(data, columns=['count', 'hate_speech_count', 'offensive_language_count', 'neither_count', 'class', 'tweet'])
        
        # Shuffle and save
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"Dataset saved to {self.output_path}")

if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.generate()