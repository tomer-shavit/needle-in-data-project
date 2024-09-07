import random
from collections import Counter
import pandas as pd
import nltk
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import defaultdict

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

category_mapping = {
    'Politics/Government': ['biden', 'trump', 'ukraine', 'russia', 'russian', 'population', 'state', 'administration',
                            'china', 'policy', 'ban', 'counsel', 'biden', 'state', 'law', 'national', 'president',
                            'country', 'queen', 'protest', 'vote', 'federal', 'right'],
    'Crime/Justice': ['police', 'arrest', 'charge', 'officer', 'court', 'judge', 'sentence', 'attack', 'murder',
                      'crime', 'fight', 'kill', 'shoot', 'shot', 'assault', 'abus', 'legal', 'guilty', 'prison',
                      'order', 'free', 'case', 'sex'],
    'Health/Medical': ['sick', 'dead', 'death', 'die', 'ill', 'disease', 'operation', 'surgery', 'medical', 'vaccine',
                       'cancer', 'coronavirus', 'hospital', 'health', 'covid', 'doctor'],
    'Business/Economy': ['dow', 'business', 'employee', 'ceo', 'global', 'bank', 'manager', 'salary', 'office', 'work',
                         'money', 'million', 'company', 'employee', 'job', 'homeless', 'pay', 'bill', 'free', 'sign',
                         'tax', 'donate'],
    'Sport': ['sport', 'athlete', 'football', 'game', 'basketball', 'play'],
    'Disaster': ['crash', 'accident', 'dangerous', 'fire', 'tsunami', 'disaster', 'tragedy', 'storm', 'earthquake',
                 'shelter', 'saved', 'rescue', 'pandemic'],
    'Education': ['school', 'student', 'college', 'teacher', 'study'],
    'Lifestyle': ['life', 'restaurant', 'lifestyle', 'home', 'house', 'marri', 'book', 'therapy'],
    'Animals': ['dog', 'animal', 'elephant', 'cat', 'pet', 'cow']
}

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def get_color_map(categories):
    unique_categories = list(categories)
    num_categories = len(unique_categories)
    cmap = plt.get_cmap('tab20') if num_categories <= 20 else plt.get_cmap('tab20', num_categories)
    colors = [cmap(i / num_categories) for i in range(num_categories)]
    color_map = {category: colors[i] for i, category in enumerate(unique_categories)}
    return color_map


def categorize_title(title):
    words = word_tokenize(title.lower())
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    category_match_counts = Counter()

    for category, keywords in category_mapping.items():
        match_count = sum(1 for word in lemmatized_words if word in keywords)
        if match_count > 0:
            category_match_counts[category] = match_count

    # Return the category with the highest match count
    if category_match_counts:
        return category_match_counts.most_common(1)[0][0]
    else:
        return random.choice(list(category_mapping.keys()))


def preprocess_title(title):
    if title.strip().endswith(')'):
        words = title.split()
        cleaned_title = ' '.join(words[:-1])
    else:
        cleaned_title = title

    cleaned_title = ' '.join(cleaned_title.split())

    return cleaned_title


def remove_stop_words(title):
    words = word_tokenize(title.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return words


def plot_pie_chart(category_count, color_map):
    labels = list(category_count.keys())
    sizes = list(category_count.values())

    colors = [color_map[label] for label in labels]

    plt.figure(figsize=(10, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 12})
    plt.title('Distribution of Titles by Category', fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
    plt.show()


def plot_average_metrics(df, color_map):
    df_exploded = df.explode('Categories')
    category_averages = df_exploded.groupby('Categories').agg({
        'Upvotes': 'mean',
        'Comments': 'mean'
    }).reset_index()
    category_averages['Upvotes'] = category_averages['Upvotes'] / 1000


    plt.figure(figsize=(14, 7))
    bars = plt.bar(category_averages['Categories'], category_averages['Upvotes'], color=[color_map[cat] for cat in category_averages['Categories']])
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Average Upvotes (K)', fontsize=14)
    plt.title('Average Upvotes per Category', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{round(yval, 1)}K', ha='center', va='bottom', fontsize=10)

    plt.show()

    plt.figure(figsize=(14, 7))
    bars = plt.bar(category_averages['Categories'], category_averages['Comments'], color=[color_map[cat] for cat in category_averages['Categories']])
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Average Comments', fontsize=14)
    plt.title('Average Comments per Category', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 1), ha='center', va='bottom', fontsize=10)

    plt.show()


# Function to convert shorthand numbers (e.g., 10k, 2m) to absolute numbers
def convert_to_number(value):
    if isinstance(value, str):
        value = value.lower()  # Convert to lowercase for consistency
        if 'k' in value:
            return float(value.replace('k', '')) * 1_000
        elif 'm' in value:
            return float(value.replace('m', '')) * 1_000_000
        elif 'b' in value:
            return float(value.replace('b', '')) * 1000000000
    return float(value)  # Return the value as a float if no conversion is needed


def clean_data(df):
    df['Upvotes'] = df['Upvotes'].apply(convert_to_number)
    df['Comments'] = df['Comments'].apply(convert_to_number)
    df['Title'] = df['Title'].apply(preprocess_title)
    add_categories(df)
    df.to_csv('modified_file.csv', index=False)
    df['Processed_Title'] = df['Title'].apply(remove_stop_words)


def add_categories(df):
    df['Category'] = df['Title'].apply(categorize_title)


def make_visualization():
    category_count = defaultdict(int)
    for category in df['Categories']:
        category_count[category] += 1
    color_map = get_color_map(category_count.keys())

    plot_pie_chart(category_count, color_map)
    plot_average_metrics(df, color_map)


if __name__ == "__main__":
    df = pd.read_csv('combined_cleaned_news.csv')
    clean_data(df)
    add_categories(df)
    make_visualization()


