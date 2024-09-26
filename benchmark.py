import random
import os
import json
import csv
import matplotlib.pyplot as plt
from Post import Post
from Model import Model

# Initialize the random seed for reproducibility
RANDOM_SEED = Model.RANDOM_SEED
random.seed(RANDOM_SEED)
MAX_ITERATION = 3


def load_posts(csv_file, json_file):
    """
    Loads and combines post data from a CSV file and a JSON file.

    Parameters:
        csv_file (str): Path to the CSV file containing post metadata.
        json_file (str): Path to the JSON file containing post embeddings.

    Returns:
        list: A list of Post objects created from the combined data.
    """
    posts = []
    csv_data = {}

    # Read post metadata from CSV
    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            csv_data[row['ID']] = {
                'upvotes': float(row['Upvotes']),
                'title': row['Title'],
                'comments': float(row['Comments']),
                'time': row['Time Ago'],
                'category': row['Categories']
            }

    # Read post embeddings from JSON
    with open(json_file, 'r', encoding='utf-8') as jsonfile:
        json_data = json.load(jsonfile)

    # Combine data from CSV and JSON into Post objects
    for post_id, json_entry in json_data.items():
        if post_id in csv_data:
            post = Post(
                idx=post_id,
                title=csv_data[post_id]['title'],
                upvotes=csv_data[post_id]['upvotes'],
                comments=csv_data[post_id]['comments'],
                time=csv_data[post_id]['time'],
                embedding=json_entry['embedding'],
                category=csv_data[post_id]['category'],
            )
            posts.append(post)

    return posts

# Load posts from files
posts = load_posts('modified_file.csv', 'titles_embeddings.json')

amount_of_test_per_iteration = 5

# Ensure the output directory exists
output_dir = './visualization/'
os.makedirs(output_dir, exist_ok=True)


def calc_loss(prediction, actual, amount):
    """
    Calculates normalized absolute error between predicted and actual values.

    Parameters:
        prediction (float): Sum of predicted values.
        actual (float): Sum of actual values.
        amount (int): Number of samples.

    Returns:
        float: The normalized absolute error.
    """
    return abs(prediction - actual) / amount


def plot_data(y_label, data, predicted_data):
    x = range(len(post_ids))
    bar_width = 0.3
    plt.figure(figsize=(10, 6))
    plt.bar([p - 0.5 * bar_width for p in x], data, width=bar_width, label=y_label)
    plt.bar([p + 0.5 * bar_width for p in x], predicted_data, width=bar_width, label="Predicted Likes")
    plt.xticks(x, post_ids)
    plt.xlabel('Post ID')
    plt.ylabel(y_label)
    plt.title(f'{y_label} Comparison For Iteration {i+1}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{i+1}_{y_label.lower()}_comparison.png'))
    plt.close()


for i in range(MAX_ITERATION):
    # Randomly sample posts for testing and separate the remaining for training
    sampled_posts = random.sample(posts, amount_of_test_per_iteration)
    posts_without_sampled = [post for post in posts if post not in sampled_posts]

    model = Model(posts_without_sampled)

    sum_likes = sum_comments = 0
    sum_likes_predicted = sum_comments_predicted = 0

    post_ids = []
    likes_data = []
    predicted_likes_data = []
    comments_data = []
    predicted_comments_data = []

    # Make predictions for the sampled posts
    for post in sampled_posts:
        predicted_likes = int(model.predict_likes(post.title, 5, post.embedding, post.time))
        predicted_comments = int(model.predict_comments(post.title, 5, post.embedding, post.time))

        sum_likes += post.upvotes
        sum_likes_predicted += predicted_likes
        sum_comments += post.comments
        sum_comments_predicted += predicted_comments

        post_ids.append(post.idx)
        likes_data.append(post.upvotes)
        predicted_likes_data.append(predicted_likes)
        comments_data.append(post.comments)
        predicted_comments_data.append(predicted_comments)

    loss_likes = calc_loss(sum_likes_predicted, sum_likes, amount_of_test_per_iteration)
    loss_comments = calc_loss(sum_comments_predicted, sum_comments, amount_of_test_per_iteration)
    print(f"Iteration {i}: Likes Loss: {loss_likes}, Comments Loss: {loss_comments}")

    plot_data('Likes', likes_data, predicted_likes_data)
    plot_data('Comments', comments_data, predicted_comments_data)

