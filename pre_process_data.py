import random
from collections import Counter, defaultdict
import pandas as pd
import nltk
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

from Post import Post

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def get_color_map(categories):
    """
    Generates a color map for the given categories to be used in visualizations.

    Parameters:
        categories (iterable): List of unique categories for which colors are to be assigned.

    Returns:
        dict: A dictionary mapping each category to a specific color.
    """
    unique_categories = list(categories)
    num_categories = len(unique_categories)
    cmap = plt.get_cmap('tab20') if num_categories <= 20 else plt.get_cmap('tab20', num_categories)
    colors = [cmap(i / num_categories) for i in range(num_categories)]
    color_map = {category: colors[i] for i, category in enumerate(unique_categories)}
    return color_map


def categorize_title(title):
    """
    Categorizes a post title based on the presence of keywords mapped to specific categories.

    Parameters:
        title (str): The title of the post to categorize.

    Returns:
        str: The category that best matches the title based on keyword counts. If no matches are found,
        a random category is returned.
    """
    words = word_tokenize(title.lower())
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    category_match_counts = Counter()

    for category, keywords in Post.category_mapping.items():
        match_count = sum(1 for word in lemmatized_words if word in keywords)
        if match_count > 0:
            category_match_counts[category] = match_count

    # Return the category with the highest keyword match count, or a random category if no matches
    return category_match_counts.most_common(1)[0][0] if category_match_counts else random.choice(
        list(Post.category_mapping.keys()))


def preprocess_title(title):
    """
    Cleans up the post title by removing trailing parentheses and redundant spaces.

    Parameters:
        title (str): The raw title to preprocess.

    Returns:
        str: The cleaned version of the title.
    """
    if title.strip().endswith(')'):
        words = title.split()
        cleaned_title = ' '.join(words[:-1])
    else:
        cleaned_title = title

    return ' '.join(cleaned_title.split())


def remove_stop_words(title):
    """
    Removes stop words and non-alphabetical characters from the post title, and lemmatizes the words.

    Parameters:
        title (str): The raw title to process.

    Returns:
        list: A list of lemmatized words with stop words removed.
    """
    words = word_tokenize(title.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return words


def plot_pie_chart(category_count, color_map):
    """
    Creates a pie chart showing the distribution of posts by category.

    Parameters:
        category_count (dict): A dictionary containing category counts.
        color_map (dict): A dictionary mapping categories to colors for the chart.
    """
    labels = list(category_count.keys())
    sizes = list(category_count.values())
    colors = [color_map[label] for label in labels]

    plt.figure(figsize=(10, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 12})
    plt.title('Distribution of Titles by Category', fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
    plt.show()


def plot_average_metrics(df, color_map):
    """
    Plots bar charts showing average upvotes and comments for each category.

    Parameters:
        df (pandas.DataFrame): DataFrame containing post data, including categories, upvotes, and comments.
        color_map (dict): A dictionary mapping categories to colors for the chart.
    """
    df_exploded = df.explode('Categories')
    category_averages = df_exploded.groupby('Categories').agg({
        'Upvotes': 'mean',
        'Comments': 'mean'
    }).reset_index()
    category_averages['Upvotes'] = category_averages['Upvotes'] / 1000  # Convert upvotes to thousands

    # Plot average upvotes
    plt.figure(figsize=(14, 7))
    bars = plt.bar(category_averages['Categories'], category_averages['Upvotes'],
                   color=[color_map[cat] for cat in category_averages['Categories']])
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Average Upvotes (K)', fontsize=14)
    plt.title('Average Upvotes per Category', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{round(yval, 1)}K', ha='center', va='bottom',
                 fontsize=10)

    plt.show()

    # Plot average comments
    plt.figure(figsize=(14, 7))
    bars = plt.bar(category_averages['Categories'], category_averages['Comments'],
                   color=[color_map[cat] for cat in category_averages['Categories']])
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Average Comments', fontsize=14)
    plt.title('Average Comments per Category', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, round(yval, 1), ha='center', va='bottom', fontsize=10)

    plt.show()


def convert_to_number(value):
    """
    Converts shorthand notations for numbers (e.g., 10k, 2m) into actual numbers.

    Parameters:
        value (str): The string representing a shorthand number.

    Returns:
        float: The converted number as a float.
    """
    if isinstance(value, str):
        value = value.lower()
        if 'k' in value:
            return float(value.replace('k', '')) * 1_000
        elif 'm' in value:
            return float(value.replace('m', '')) * 1_000_000
        elif 'b' in value:
            return float(value.replace('b', '')) * 1_000_000_000
    return float(value)  # Return as float if no conversion is needed


def clean_data(df):
    """
    Cleans and preprocesses the post data, including upvote/comment normalization and title processing.

    Parameters:
        df (pandas.DataFrame): DataFrame containing post data.
    """
    df['Upvotes'] = df['Upvotes'].apply(convert_to_number)
    df['Comments'] = df['Comments'].apply(convert_to_number)
    df['Title'] = df['Title'].apply(preprocess_title)
    add_categories(df)
    df.to_csv('modified_file.csv', index=False)  # Save cleaned data
    df['Processed_Title'] = df['Title'].apply(remove_stop_words)


def add_categories(df):
    """
    Adds a category column to the DataFrame based on the title of each post.

    Parameters:
        df (pandas.DataFrame): DataFrame containing post data.
    """
    df['Category'] = df['Title'].apply(categorize_title)


def make_visualization():
    """
    Creates visualizations, including a pie chart for category distribution and bar charts for average metrics.
    """
    category_count = defaultdict(int)
    for category in df['Categories']:
        category_count[category] += 1
    color_map = get_color_map(category_count.keys())

    plot_pie_chart(category_count, color_map)
    plot_average_metrics(df, color_map)


if __name__ == "__main__":
    # Load the data, clean it, and generate visualizations
    df = pd.read_csv('combined_cleaned_news.csv')
    clean_data(df)
    add_categories(df)
    make_visualization()
