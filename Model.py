import os
import random
from typing import Dict, List, Tuple
import numpy as np
import openai
from Post import Post
from datetime import datetime, timedelta
import re


class Model:
    """
    A model for predicting likes and comments of posts based on textual embeddings and time similarity.
    It uses OpenAI's API to generate embeddings and combines cosine similarity with time-based weighting
    to predict interactions for new posts.
    """

    RANDOM_SEED = random.randint  # Random seed generator for reproducibility

    def __init__(self, posts: List[Post], alpha=0.1):
        """
        Initializes the Model with a list of posts and an optional alpha value for combining
        similarity based on content and time.

        Parameters:
            posts (List[Post]): A list of Post objects to use for prediction.
            alpha (float): Weighting factor to balance between content similarity and time similarity.
        """
        self.posts = posts
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.alpha = alpha  # Alpha controls the trade-off between content and time similarity.

    def predict_likes(self, title: str, k: int = 10, embedding=None, time='1 day ago') -> float:
        """
        Predicts the number of likes for a given post title based on similar posts' embeddings and time.

        Parameters:
            title (str): The title of the post to predict likes for.
            k (int): The number of closest posts to consider for prediction.
            embedding (List[float], optional): Pre-computed embedding for the title. If not provided, the model generates one.
            time (str): The time ago string indicating how old the post is (e.g., '1 day ago').

        Returns:
            float: Predicted number of likes.
        """
        if not embedding:
            embedding = self.create_embedding(title)

        closest_posts = self.get_closest_posts(embedding, k, time)

        total_weight = 0.0
        weighted_sum = 0.0

        # Calculate weighted average of upvotes based on content and time similarity
        for similarity, post in closest_posts:
            weighted_sum += similarity * post.upvotes
            total_weight += similarity

        # Handle edge case where total weight is zero
        if total_weight == 0:
            return 0

        return weighted_sum / total_weight

    def predict_comments(self, title: str, k: int = 10, embedding=None, time='1 day ago') -> float:
        """
        Predicts the number of comments for a given post title based on similar posts' embeddings and time.

        Parameters:
            title (str): The title of the post to predict comments for.
            k (int): The number of closest posts to consider for prediction.
            embedding (List[float], optional): Pre-computed embedding for the title. If not provided, the model generates one.
            time (str): The time ago string indicating how old the post is (e.g., '1 day ago').

        Returns:
            float: Predicted number of comments.
        """
        if not embedding:
            embedding = self.create_embedding(title)

        closest_posts = self.get_closest_posts(embedding, k, time)

        total_weight = 0.0
        weighted_sum = 0.0

        # Calculate weighted average of comments based on content and time similarity
        for similarity, post in closest_posts:
            weighted_sum += similarity * post.comments
            total_weight += similarity

        if total_weight == 0:
            return 0

        return weighted_sum / total_weight

    def get_closest_posts(self, embedding: List[float], k: int, time: str) -> List[Tuple[float, Post]]:
        """
        Finds the top-k closest posts based on a combination of content similarity (via cosine similarity)
        and time similarity (how recent the posts are).

        Parameters:
            embedding (List[float]): The embedding of the post to compare.
            k (int): Number of closest posts to retrieve.
            time (str): The time ago string for the new post (e.g., '1 day ago').

        Returns:
            List[Tuple[float, Post]]: A list of tuples containing similarity score and the corresponding Post object.
        """
        time = self.parse_time_ago(time, datetime.now())
        distances = []

        # Calculate combined similarity for each post in the dataset
        for post in self.posts:
            cos_similarity = self.cosine_similarity(embedding, post.embedding)
            time_similarity = self.time_similarity(time, post.time)
            combined_similarity = (1 - self.alpha) * cos_similarity + self.alpha * time_similarity

            distances.append((combined_similarity, post))

        # Sort posts by similarity in descending order and return the top-k results
        distances.sort(reverse=True, key=lambda x: x[0])

        return distances[:k]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Computes cosine similarity between two vectors.

        Parameters:
            vec1 (List[float]): First vector.
            vec2 (List[float]): Second vector.

        Returns:
            float: Cosine similarity between vec1 and vec2.
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        # Return 0 similarity if any vector has zero norm
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0

        return dot_product / (norm_vec1 * norm_vec2)

    def time_similarity(self, current_time: datetime, time_ago_str: str) -> float:
        """
        Computes a similarity score based on how recent the post is compared to the current post.

        Parameters:
            current_time (datetime): The reference time (typically the current time).
            time_ago_str (str): Time ago string of the post being compared.

        Returns:
            float: A time-based similarity score between 0 and 1.
        """
        post_time = self.parse_time_ago(time_ago_str, current_time)
        time_diff = abs((current_time - post_time).days)

        # Normalize time difference with a maximum of 365 days
        max_days = 365
        normalized_time_diff = min(time_diff / max_days, 1.0)

        return 1.0 - normalized_time_diff

    def parse_time_ago(self, time_ago_str: str, current_time: datetime) -> datetime:
        """
        Parses a 'time ago' string and converts it to a datetime object.

        Parameters:
            time_ago_str (str): Time ago string (e.g., '1 day ago').
            current_time (datetime): The reference time (typically the current time).

        Returns:
            datetime: Parsed datetime corresponding to the time ago string, or current_time if parsing fails.
        """
        match = re.match(r"(\d+)\s*(day|month|year)s?\s*ago", time_ago_str.strip())
        if not match:
            return current_time  # Return current time if the format is unexpected

        value, unit = int(match.group(1)), match.group(2)

        # Calculate the post time based on the unit of time
        if unit == 'day':
            return current_time - timedelta(days=value)
        elif unit == 'month':
            return current_time - timedelta(days=value * 30)  # Approximate months as 30 days
        elif unit == 'year':
            return current_time - timedelta(days=value * 365)  # Approximate years as 365 days

        return current_time

    def create_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text using the OpenAI API.

        Parameters:
            text (str): The text to generate an embedding for.

        Returns:
            List[float]: The embedding vector.
        """
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=text
        )
        print(f"Generated embedding for title: '{text}'")
        return response['data'][0]['embedding']
