import pandas as pd
import random
import nltk
from nltk.corpus import names, words, stopwords
from transformers import pipeline
import re
from tqdm import tqdm
import os
import torch
import warnings

# Suppress FutureWarning for clean_up_tokenization_spaces
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Set NLTK data path
nltk.data.path.append('/home/branch/nltk_data')

# Download NLTK data if not already present
if not os.path.exists('/home/branch/nltk_data'):
    nltk.download(['names', 'words', 'punkt', 'punkt_tab', 'stopwords', 'wordnet'], download_dir='/home/branch/nltk_data')

# Load names for randomization
all_names = names.words('male.txt') + names.words('female.txt')

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Initialize GPT-2 for text generation
generator = pipeline('text-generation', model='gpt2', device=device, clean_up_tokenization_spaces=False)

# Expanded templates (100 per class for diversity)
hate_speech_templates = [
    "I want to kill {name} because {reason}.",
    "I hate {name} so much, they should die!",
    "{name}, you're a worthless {insult} and deserve to suffer.",
    "I'm gonna destroy {name} for {reason}.",
    "{name} is a disgusting {insult} and should be eradicated.",
    "All {group} like {name} are scum and should be eliminated.",
    "I wish {name} would rot in hell for {reason}.",
    "{name} is a vile piece of trash and deserves pain.",
    "I want to see {name} humiliated and broken.",
    "People like {name} are the problem and should be wiped out.",
    "{name} is a filthy {insult}, ban them now!",
    "I despise {name} and their {reason}.",
    "All {group} are terrible, especially {name}.",
    "I want to ruin {name}'s life for {reason}.",
    "{name} is a pathetic {insult}, get lost!",
    "I hope {name} suffers for {reason}.",
    "{name} is a {insult} who should be locked up.",
    "All {group} like {name} are a disgrace.",
    "I want to burn {name}'s {possession}.",
    "{name} deserves to lose everything for {reason}.",
    "I hate {group} like {name}, they're all {insult}.",
    "{name} is a {insult} and should be exiled.",
    "I wish {name} would vanish for {reason}.",
    "{name} is a {insult} who ruins {place}.",
    "All {group} are {insult}, starting with {name}.",
    "I want to expose {name} for {reason}.",
    "{name} is a {insult} who deserves public shaming.",
    "I hate {name}'s {attribute}, they should disappear.",
    "{name} is a {insult} and should be punished.",
    "All {group} like {name} are dangerous.",
    "I want to attack {name} for {reason}.",
    "{name} is a {insult} who should be silenced.",
    "I hope {name} fails miserably at {activity}.",
    "{name} is a {insult} and deserves to be hated.",
    "All {group} are {insult}, especially {name}.",
    "I want to sabotage {name}'s {project}.",
    "{name} is a {insult} who should be ostracized.",
    "I hate {name} for {reason}, they’re done.",
    "{name} is a {insult} who ruins everything.",
    "All {group} like {name} should be banned.",
    "I want to see {name} in agony for {reason}.",
    "{name} is a {insult} who deserves nothing.",
    "I hate {name}'s {attribute}, they’re worthless.",
    "{name} is a {insult} who should be ignored.",
    "All {group} are {insult}, led by {name}.",
    "I want to destroy {name}'s reputation.",
    "{name} is a {insult} who should be shunned.",
    "I hope {name} loses {possession} for {reason}.",
    "{name} is a {insult} who deserves failure.",
    "All {group} like {name} are a problem.",
    "I want to confront {name} for {reason}.",
    "{name} is a {insult} who should be stopped.",
    "I hate {name} and their {attribute}.",
    "{name} is a {insult} who deserves scorn.",
    "All {group} are {insult}, like {name}.",
    "I want to ruin {name}'s {activity}.",
    "{name} is a {insult} who should quit.",
    "I hope {name} is humiliated for {reason}.",
    "{name} is a {insult} who deserves exile.",
    "All {group} like {name} are worthless.",
    "I want to burn {name}'s {possession} down.",
    "{name} is a {insult} who should be jailed.",
    "I hate {name} for their {reason}.",
    "{name} is a {insult} who deserves misery.",
    "All {group} are {insult}, especially {name}.",
    "I want to see {name} fail for {reason}.",
    "{name} is a {insult} who should be erased.",
    "I hate {name}'s {attribute}, they’re scum.",
    "{name} is a {insult} who deserves punishment.",
    "All {group} like {name} are a disgrace.",
    "I want to expose {name}'s {reason}.",
    "{name} is a {insult} who should be mocked.",
    "I hope {name} suffers for their {attribute}.",
    "{name} is a {insult} who deserves hatred.",
    "All {group} are {insult}, led by {name}.",
    "I want to destroy {name}'s {project}.",
    "{name} is a {insult} who should be banned.",
    "I hate {name} for their {reason}.",
    "{name} is a {insult} who deserves pain.",
    "All {group} like {name} are horrible.",
    "I want to ruin {name}'s {possession}.",
    "{name} is a {insult} who should be shamed.",
    "I hope {name} loses everything for {reason}.",
    "{name} is a {insult} who deserves exile.",
    "All {group} are {insult}, like {name}.",
    "I want to attack {name}'s {attribute}.",
    "{name} is a {insult} who should be stopped.",
    "I hate {name} and their {reason}.",
    "{name} is a {insult} who deserves scorn.",
    "All {group} like {name} are worthless.",
    "I want to sabotage {name}'s {activity}.",
    "{name} is a {insult} who should quit.",
    "I hope {name} is humiliated for {reason}.",
    "{name} is a {insult} who deserves failure.",
    "All {group} like {name} are a problem.",
    "I want to confront {name} for {reason}.",
    "{name} is a {insult} who should be silenced.",
    "I hate {name}'s {attribute}, they’re done."
]

offensive_templates = [
    "Shut up, you {insult} {name}!",
    "You're such a {insult}, {name}, go away.",
    "Go to hell, you damn {insult}!",
    "{name}, you're a total {insult}, what a joke.",
    "{name} is a freaking {insult}, can't believe it.",
    "Fuck off, {name}, you're a {insult}.",
    "{name} is a shitty {insult}, no doubt.",
    "Damn you, {name}, you're an {insult}.",
    "{name}'s ideas are trash, what an {insult}.",
    "Screw {name}, they're a complete {insult}.",
    "You're a {insult}, {name}, stop talking!",
    "{name} is a useless {insult}, give up.",
    "Get lost, {name}, you {insult}!",
    "{name}'s work is garbage, total {insult}.",
    "What a {insult}, {name}, just quit.",
    "You're a {insult}, {name}, nobody cares.",
    "{name} is a pathetic {insult}, stop trying.",
    "Fuck you, {name}, you're a {insult}.",
    "{name}'s {attribute} is awful, what an {insult}.",
    "Screw off, {name}, you're a {insult}.",
    "You're a {insult}, {name}, get out!",
    "{name} is a terrible {insult}, always.",
    "Damn {name}, you're a {insult}!",
    "{name}'s {project} is crap, total {insult}.",
    "You're a {insult}, {name}, give it up.",
    "Get out, {name}, you {insult}!",
    "{name} is a {insult}, what a waste.",
    "Fuck {name}'s {attribute}, they're a {insult}.",
    "{name} is a {insult}, stop bothering us.",
    "You're a {insult}, {name}, go home.",
    "{name} is a {insult}, their {attribute} sucks.",
    "Screw {name}, they're a {insult}.",
    "You're a {insult}, {name}, nobody likes you.",
    "{name} is a {insult}, their {project} is trash.",
    "Damn {name}, you're a {insult}!",
    "You're a {insult}, {name}, stop it.",
    "{name} is a {insult}, their {attribute} is awful.",
    "Fuck off, {name}, you're a {insult}.",
    "{name} is a {insult}, their {project} is garbage.",
    "You're a {insult}, {name}, get lost.",
    "Screw {name}, they're a {insult}.",
    "{name} is a {insult}, their {attribute} is crap.",
    "You're a {insult}, {name}, quit now.",
    "{name} is a {insult}, their {project} is awful.",
    "Damn {name}, you're a {insult}!",
    "You're a {insult}, {name}, stop talking.",
    "{name} is a {insult}, their {attribute} is terrible.",
    "Fuck {name}, they're a {insult}.",
    "{name} is a {insult}, their {project} is junk.",
    "You're a {insult}, {name}, go away.",
    "Screw {name}, they're a {insult}.",
    "{name} is a {insult}, their {attribute} is bad.",
    "You're a {insult}, {name}, nobody cares.",
    "{name} is a {insult}, their {project} is crap.",
    "Damn {name}, you're a {insult}!",
    "You're a {insult}, {name}, stop trying.",
    "{name} is a {insult}, their {attribute} is garbage.",
    "Fuck off, {name}, you're a {insult}.",
    "{name} is a {insult}, their {project} is terrible.",
    "You're a {insult}, {name}, get out.",
    "Screw {name}, they're a {insult}.",
    "{name} is a {insult}, their {attribute} is awful.",
    "You're a {insult}, {name}, quit now.",
    "{name} is a {insult}, their {project} is junk.",
    "Damn {name}, you're a {insult}!",
    "You're a {insult}, {name}, stop bothering us.",
    "{name} is a {insult}, their {attribute} is bad.",
    "Fuck {name}, they're a {insult}.",
    "{name} is a {insult}, their {project} is garbage.",
    "You're a {insult}, {name}, go home.",
    "Screw {name}, they're a {insult}.",
    "{name} is a {insult}, their {attribute} is crap.",
    "You're a {insult}, {name}, nobody likes you.",
    "{name} is a {insult}, their {project} is trash.",
    "Damn {name}, you're a {insult}!",
    "You're a {insult}, {name}, stop it.",
    "{name} is a {insult}, their {attribute} is terrible.",
    "Fuck off, {name}, you're a {insult}.",
    "{name} is a {insult}, their {project} is awful.",
    "You're a {insult}, {name}, get lost.",
    "Screw {name}, they're a {insult}.",
    "{name} is a {insult}, their {attribute} is bad.",
    "You're a {insult}, {name}, quit now.",
    "{name} is a {insult}, their {project} is junk.",
    "Damn {name}, you're a {insult}!",
    "You're a {insult}, {name}, stop talking.",
    "{name} is a {insult}, their {attribute} is garbage.",
    "Fuck {name}, they're a {insult}.",
    "{name} is a {insult}, their {project} is terrible.",
    "You're a {insult}, {name}, go away.",
    "Screw {name}, they're a {insult}.",
    "{name} is a {insult}, their {attribute} is awful.",
    "You're a {insult}, {name}, nobody cares.",
    "{name} is a {insult}, their {project} is crap.",
    "Damn {name}, you're a {insult}!",
    "You're a {insult}, {name}, stop trying.",
    "{name} is a {insult}, their {attribute} is bad.",
    "Fuck off, {name}, you're a {insult}."
]

neutral_templates = [
    "Hey {name}, how's it going?",
    "I saw {name} at the {place} today, nice chat.",
    "Great to meet {name}, let's connect again.",
    "Let's grab coffee with {name} soon.",
    "{name} has interesting ideas, I agree.",
    "I like {name}'s work, it's inspiring.",
    "Discussing topics with {name} is fun.",
    "{name} is a talented person.",
    "I support {name}'s efforts.",
    "The weather is nice, {name}.",
    "Had a good day, {name}?",
    "{name}, your project looks cool.",
    "Let's visit the {place} with {name}.",
    "I enjoy {name}'s company.",
    "{name} shared a great idea today.",
    "Nice to see {name} at the {place}.",
    "{name}'s {project} is impressive.",
    "I like chatting with {name}.",
    "Let's hang out with {name} soon.",
    "{name} is doing great at {activity}.",
    "I appreciate {name}'s {attribute}.",
    "{name} is a pleasure to work with.",
    "Let's go to the {place} with {name}.",
    "I enjoy {name}'s {project}.",
    "{name} has a great sense of {attribute}.",
    "Nice talking to {name} about {activity}.",
    "{name} is a cool person.",
    "I support {name}'s {project}.",
    "Let's meet {name} at the {place}.",
    "{name}'s ideas are awesome.",
    "I like {name}'s approach to {activity}.",
    "{name} is a great friend.",
    "Let's plan a trip with {name}.",
    "I enjoy {name}'s {attribute}.",
    "{name} is a talented {group} member.",
    "Nice to see {name}'s work at {place}.",
    "I appreciate {name}'s help with {activity}.",
    "{name} is a wonderful person.",
    "Let's discuss {activity} with {name}.",
    "{name}'s {project} is going well.",
    "I like {name}'s creativity.",
    "Nice to meet {name} at the {place}.",
    "{name} is a great teammate.",
    "I support {name}'s ideas.",
    "Let's visit {name} at the {place}.",
    "I enjoy {name}'s sense of {attribute}.",
    "{name} is doing awesome at {activity}.",
    "Nice to see {name} today.",
    "I like {name}'s {project} progress.",
    "{name} is a fantastic {group} member.",
    "Let's grab lunch with {name}.",
    "I appreciate {name}'s {attribute}.",
    "{name} is a great person to know.",
    "Let's chat with {name} about {activity}.",
    "{name}'s work is inspiring.",
    "I like {name}'s {attribute} a lot.",
    "{name} is a wonderful colleague.",
    "Let's meet up with {name} soon.",
    "I enjoy {name}'s company at {place}.",
    "{name} is a talented individual.",
    "Nice to see {name}'s {project}.",
    "I support {name}'s efforts at {activity}.",
    "{name} is a great friend to have.",
    "Let's visit the {place} with {name}.",
    "I like {name}'s approach to {project}.",
    "{name} is a cool {group} member.",
    "Nice talking to {name} today.",
    "I appreciate {name}'s work on {activity}.",
    "{name} is a fantastic person.",
    "Let's plan something with {name}.",
    "I enjoy {name}'s {attribute} a lot.",
    "{name} is a great collaborator.",
    "Nice to see {name} at the {place}.",
    "I like {name}'s {project} ideas.",
    "{name} is a wonderful teammate.",
    "Let's hang out with {name} at {place}.",
    "I support {name}'s {activity} efforts.",
    "{name} is a great person to work with.",
    "Nice to meet {name} today.",
    "I like {name}'s {attribute} style.",
    "{name} is a fantastic friend.",
    "Let's discuss {project} with {name}.",
    "{name}'s work is awesome.",
    "I enjoy {name}'s {attribute} a lot.",
    "{name} is a great {group} member.",
    "Nice to see {name}'s progress.",
    "I support {name}'s {project} work.",
    "{name} is a wonderful person to know.",
    "Let's meet {name} at the {place}.",
    "I like {name}'s {attribute} approach.",
    "{name} is a fantastic colleague.",
    "Nice talking to {name} about {project}.",
    "I enjoy {name}'s {attribute} style.",
    "{name} is a great teammate.",
    "Let's grab coffee with {name} soon.",
    "I support {name}'s {activity} ideas.",
    "{name} is a wonderful friend.",
    "Nice to see {name} at the {place} today."
]

# Expanded lists for randomization
insults = ['idiot', 'loser', 'jerk', 'fool', 'moron', 'asshole', 'dumbass', 'bastard', 'prick', 'scumbag', 'trash', 'failure', 'disgrace', 'creep', 'pig', 'monster', 'degenerate', 'fraud', 'liar', 'cheater', 'clown', 'hack', 'nobody', 'thief', 'coward', 'hypocrite', 'jerk', 'slob', 'freak', 'weirdo']
reasons = ['lies', 'betrayal', 'incompetence', 'stupidity', 'arrogance', 'greed', 'laziness', 'rudeness', 'dishonesty', 'hypocrisy', 'being annoying', 'being selfish', 'being mean', 'being ugly', 'being boring', 'cheating', 'stealing', 'failing', 'ignoring others', 'being rude', 'being loud', 'being late', 'being careless', 'being fake', 'being unfair']
groups = ['politicians', 'celebrities', 'neighbors', 'coworkers', 'friends', 'family', 'strangers', 'activists', 'journalists', 'teachers', 'doctors', 'artists', 'scientists', 'players', 'leaders', 'workers', 'students', 'reporters', 'managers', 'critics', 'fans', 'singers', 'actors', 'writers', 'coaches', 'developers', 'gamers', 'users', 'visitors', 'buyers', 'sellers', 'customers', 'drivers', 'shoppers', 'readers', 'viewers', 'listeners', 'subscribers', 'members', 'owners', 'authors', 'directors', 'neighbors', 'classmates', 'colleagues', 'acquaintances', 'relatives', 'kids', 'people', 'employees', 'clients', 'teammates', 'volunteers', 'supporters', 'followers', 'attendees', 'participants']
places = ['park', 'mall', 'cafe', 'beach', 'office', 'school', 'gym', 'library', 'restaurant', 'theater', 'concert', 'party', 'meeting', 'event', 'conference', 'store', 'hospital', 'airport', 'station', 'hotel', 'home', 'street', 'city', 'country', 'world', 'online', 'social media', 'forum', 'group', 'club', 'market', 'museum', 'zoo', 'stadium', 'arena', 'bar', 'church', 'temple', 'mosque', 'school', 'college', 'university', 'workplace', 'community center', 'park']
possessions = ['house', 'car', 'phone', 'laptop', 'book', 'project', 'career', 'reputation', 'business', 'work', 'art', 'music', 'film', 'game', 'app', 'website', 'product', 'store', 'brand', 'idea', 'plan', 'dream', 'future', 'life', 'legacy']
attributes = ['attitude', 'behavior', 'style', 'work', 'ideas', 'appearance', 'voice', 'habits', 'choices', 'decisions', 'actions', 'words', 'beliefs', 'values', 'skills', 'talent', 'efforts', 'plans', 'projects', 'goals']
projects = ['work', 'project', 'book', 'movie', 'song', 'game', 'app', 'website', 'business', 'plan', 'idea', 'art', 'music', 'film', 'show', 'event', 'campaign', 'research', 'study', 'product']
activities = ['work', 'project', 'hobby', 'game', 'sport', 'event', 'trip', 'meeting', 'class', 'lesson', 'performance', 'show', 'concert', 'party', 'activity', 'plan', 'discussion', 'conversation', 'task', 'job']

def generate_rule_based_tweets(label, num_samples):
    tweets = []
    for _ in range(num_samples):
        name = random.choice(all_names)
        insult = random.choice(insults)
        reason = random.choice(reasons)
        group = random.choice(groups)
        place = random.choice(places)
        possession = random.choice(possessions)
        attribute = random.choice(attributes)
        project = random.choice(projects)
        activity = random.choice(activities)
        if label == 'Hate Speech':
            tweet = random.choice(hate_speech_templates).format(
                name=name, insult=insult, reason=reason, group=group,
                place=place, possession=possession, attribute=attribute, project=project, activity=activity
            )
        elif label == 'Offensive Language':
            tweet = random.choice(offensive_templates).format(
                name=name, insult=insult, reason=reason, group=group,
                place=place, possession=possession, attribute=attribute, project=project, activity=activity
            )
        else:  # Neutral
            tweet = random.choice(neutral_templates).format(
                name=name, place=place, reason=reason, group=group,
                possession=possession, attribute=attribute, project=project, activity=activity
            )
        tweets.append(tweet[:280])
    return tweets

def generate_gpt2_tweets(prompt, num_samples, max_length=50):
    tweets = []
    for _ in tqdm(range(num_samples), desc="Generating GPT-2 tweets"):
        output = generator(prompt, max_length=max_length, num_return_sequences=1, truncation=True)[0]['generated_text']
        output = re.sub(r'\s+', ' ', output).strip()[:280]
        tweets.append(output)
    return tweets

def create_synthetic_dataset(total_samples=300000):
    samples_per_class = total_samples // 3
    
    # Rule-based tweets (80% of data)
    hate_speech_tweets = generate_rule_based_tweets('Hate Speech', int(samples_per_class * 0.8))
    offensive_tweets = generate_rule_based_tweets('Offensive Language', int(samples_per_class * 0.8))
    neutral_tweets = generate_rule_based_tweets('Neutral', int(samples_per_class * 0.8))
    
    # GPT-2 tweets (20% of data)
    hate_speech_prompt = "Generate a hateful tweet threatening a person, group, or entity in a specific context."
    offensive_prompt = "Generate an offensive tweet insulting a person, group, or entity with profanity."
    neutral_prompt = "Generate a friendly tweet greeting or supporting a person, group, or entity."
    
    hate_speech_tweets.extend(generate_gpt2_tweets(hate_speech_prompt, int(samples_per_class * 0.2)))
    offensive_tweets.extend(generate_gpt2_tweets(offensive_prompt, int(samples_per_class * 0.2)))
    neutral_tweets.extend(generate_gpt2_tweets(neutral_prompt, int(samples_per_class * 0.2)))
    
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
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def main():
    output_path = '/home/branch/Downloads/new_training_data.csv'
    df = create_synthetic_dataset(total_samples=300000)
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset saved to {output_path}")
    print(df['class'].value_counts())

if __name__ == "__main__":
    main()