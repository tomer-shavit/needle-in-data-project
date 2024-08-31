import re


import pandas as pd
import nltk
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


category_mapping = {
    'Politics/Government': ['biden', 'trump', 'ukraine', 'russia', 'russian', 'population',
                            'state', 'administration', 'china', 'policy', 'ban', 'counsel',
                            'biden', 'state', 'law', 'national', 'president', 'country', 'queen',
                            'protest', 'vote', 'federal', 'right'],
    'Crime/Justice': ['police', 'arrest', 'charge', 'officer', 'court', 'judge',
                      'sentence', 'attack', 'murder', 'crime', 'fight',
                      'kill', 'shoot', 'shot', 'assault', 'abus', 'legal', 'guilty',
                      'prison', 'order', 'free', 'case', 'sex'],
    'Health/Medical': ['sick', 'dead', 'death', 'die', 'ill', 'disease', 'operation', 'surgery', 'medical', 'vaccine', 'cancer',
                       'coronavirus', 'hospital', 'health', 'covid', 'doctor'],


    'Business/Economy': ['dow', 'business', 'employee', 'ceo', 'global', 'bank',
                         'manager', 'salary', 'office', 'work', 'money',
                         'million', 'company', 'employee', 'job', 'homeless'
                                                                  'pay', 'bill', 'free', 'sign', 'tax', 'donate'],
    'Sport': ['sport', 'athlete', 'football', 'game', 'basketball', 'play'],
    'Disaster': ['crash', 'accident', 'dangerous', 'fire', 'tsunami', 'disaster', 'tragedy', 'storm', 'earthquake',
                 'shelter', 'saved', 'rescue', 'pandemic'],
    'Education': ['school', 'student', 'college', 'teacher', 'study'],
    'Lifestyle': ['life', 'restaurant', 'lifestyle', 'home', 'house', 'marri', 'book', 'therapy'],
    'Animals': ['dog', 'animal', 'elephant', 'cat', 'pet', 'cow']


}


def get_color_map(categories):
    unique_categories = list(categories)
    num_categories = len(unique_categories)
    # Choose a colormap with sufficient colors
    cmap = plt.get_cmap('tab20') if num_categories <= 20 else plt.get_cmap('tab20', num_categories)
    colors = [cmap(i / num_categories) for i in range(num_categories)]
    color_map = {category: colors[i] for i, category in enumerate(unique_categories)}
    return color_map




# Function to clean and convert formatted strings to numeric values
def convert_to_numeric(value):
    if isinstance(value, str):
        value = value.lower()
        if 'k' in value:
            return float(value.replace('k', '').replace(',', '')) * 1000
        elif 'm' in value:
            return float(value.replace('m', '').replace(',', '')) * 1000000
        elif 'b' in value:
            return float(value.replace('b', '').replace(',', '')) * 1000000000
        else:
            return pd.to_numeric(value, errors='coerce')
    return pd.to_numeric(value, errors='coerce')


# Function to categorize titles with lemmatization
def categorize_title(title):
    categories = set()
    words = word_tokenize(title.lower())
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]


    for category, keywords in category_mapping.items():
        if any(word in lemmatized_words for word in keywords):
            categories.add(category)
    return categories


def preprocess_title(title):
    pattern = r'[(].*\.com[)]'
    # Function to clean titles by removing words matching the pattern
    cleaned_title = re.sub(pattern, '', title)
    cleaned_title = ' '.join(cleaned_title.split())  # Remove extra spaces
    # Tokenize
    words = word_tokenize(cleaned_title.lower())
    # Remove stopwords and non-alphabetic words
    words = [word for word in words if word.isalpha() and word not in stop_words]
    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]
    return words


def plot_pie_chart(category_count, color_map):
    # Prepare data for the pie chart
    labels = list(category_count.keys())
    sizes = list(category_count.values())

    # Use the color map to get the colors for each category
    colors = [color_map[label] for label in labels]

    # Plot the pie chart
    plt.figure(figsize=(10, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 12})
    plt.title('Distribution of Titles by Category', fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
    plt.show()



def plot_average_metrics(df, color_map):
    df_exploded = df.explode('Categories')
    df_exploded['Upvotes'] = df_exploded['Upvotes'].apply(convert_to_numeric)
    df_exploded['Comments'] = df_exploded['Comments'].apply(convert_to_numeric)
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


    # Adding labels with "K" to bars
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


    # Adding labels to bars (no "K" needed for comments)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 1), ha='center', va='bottom', fontsize=10)


    plt.show()




if __name__ == "__main__":
    df = pd.read_csv('combined_cleaned_news.csv')
    # Preprocessing
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))


    # Apply preprocessing to titles
    df['Processed_Title'] = df['Title'].apply(preprocess_title)


    # Flatten the list of words and count frequency
    all_words = [word for title in df['Processed_Title'] for word in title]
    word_counts = Counter(all_words)


    # Get the top 10 topics
    top_topics = word_counts.most_common(200)


    for tuple in top_topics:
        print(tuple)
    # Apply the function to categorize titles
    df['Categories'] = df['Title'].apply(categorize_title)


    # Count titles for each category
    category_count = defaultdict(int)
    for categories in df['Categories']:
        for category in categories:
            category_count[category] += 1
    color_map = get_color_map(category_count.keys())
    plot_pie_chart(category_count, color_map)

    # Display the results
    for category, count in category_count.items():
        print(f"{category}: {count} titles")


    plot_average_metrics(df, color_map)
