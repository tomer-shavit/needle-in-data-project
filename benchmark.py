import random
import os
import json
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from Post import Post
from Model import Model

# Initialize the random seed for reproducibility
RANDOM_SEED = Model.RANDOM_SEED
random.seed(RANDOM_SEED)

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

# Constants for testing
amount_of_test_per_category = 5  # Number of posts to sample per category
categories = list(Post.category_mapping.keys())  # List of available categories

# Ensure the output directory exists
output_dir = './data/'
os.makedirs(output_dir, exist_ok=True)

# Group posts by their category
posts_by_category = defaultdict(list)
for post in posts:
    posts_by_category[post.category].append(post)

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

# Iterate over each category for evaluation
for category in categories:
    # Skip categories with insufficient posts for testing
    if len(posts_by_category[category]) < amount_of_test_per_category:
        print(f"Not enough posts in category {category}")
        continue

    # Randomly sample posts for testing and separate the remaining for training
    sampled_posts = random.sample(posts_by_category[category], amount_of_test_per_category)
    posts_without_sampled = [post for post in posts if post not in sampled_posts]

    # Initialize the model using non-sampled posts
    model = Model(posts_without_sampled)

    # Accumulators for actual and predicted likes/comments
    sum_likes = sum_comments = 0
    sum_likes_predicted = sum_comments_predicted = 0

    # Lists for plotting
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

        # Collect data for plotting
        post_ids.append(post.idx)
        likes_data.append(post.upvotes)
        predicted_likes_data.append(predicted_likes)
        comments_data.append(post.comments)
        predicted_comments_data.append(predicted_comments)

    # Calculate and display prediction loss
    loss_likes = calc_loss(sum_likes_predicted, sum_likes, amount_of_test_per_category)
    loss_comments = calc_loss(sum_comments_predicted, sum_comments, amount_of_test_per_category)
    print(f"Category {category}: Likes Loss: {loss_likes}, Comments Loss: {loss_comments}")

    # Plot comparison for Likes
    x = range(len(post_ids))  # X-axis indices for the posts
    bar_width = 0.3  # Width of each bar in the bar chart

    plt.figure(figsize=(10, 6))
    plt.bar([p - 0.5 * bar_width for p in x], likes_data, width=bar_width, label="Likes")
    plt.bar([p + 0.5 * bar_width for p in x], predicted_likes_data, width=bar_width, label="Predicted Likes")

    plt.xticks(x, post_ids)
    plt.xlabel('Post ID')
    plt.ylabel('Likes')
    plt.title(f'Likes Comparison for Category: {category}')
    plt.legend()
    plt.tight_layout()

    # Save the Likes comparison plot
    plt.savefig(os.path.join(output_dir, f'{category[:5]}_likes_comparison.png'))
    plt.close()

    # Plot comparison for Comments
    plt.figure(figsize=(10, 6))
    plt.bar([p - 0.5 * bar_width for p in x], comments_data, width=bar_width, label="Comments")
    plt.bar([p + 0.5 * bar_width for p in x], predicted_comments_data, width=bar_width, label="Predicted Comments")

    plt.xticks(x, post_ids)
    plt.xlabel('Post ID')
    plt.ylabel('Comments')
    plt.title(f'Comments Comparison for Category: {category}')
    plt.legend()
    plt.tight_layout()

    # Save the Comments comparison plot
    plt.savefig(os.path.join(output_dir, f'{category[:5]}_comments_comparison.png'))
    plt.close()
