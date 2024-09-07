import os
from typing import Dict, List, Tuple
import numpy as np
import openai
from Post import Post
from datetime import datetime, timedelta
import re

class Model:
    def __init__(self, posts: List[Post], alpha=0.5):
        self.posts = posts
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.alpha = alpha

    def predict_likes(self, title: str, k: int = 10, embedding=None, time='1 day ago') -> float:
        if not embedding:
            embedding = self.create_embedding(title)

        closest_posts = self.get_closest_posts(embedding, k, time)

        # Calculate the weighted average of likes
        total_weight = 0.0
        weighted_sum = 0.0

        for similarity, post in closest_posts:
            weighted_sum += similarity * post.upvotes
            total_weight += similarity

        # Return the weighted average
        if total_weight == 0:
            return 0  # Handle case where all similarities are 0

        return weighted_sum / total_weight

    def predict_comments(self, title: str, k: int = 10, embedding=None, time='1 day ago') -> float:
        if not embedding:
            embedding = self.create_embedding(title)

        closest_posts = self.get_closest_posts(embedding, k, time)
        # Calculate the weighted average of comments
        total_weight = 0.0
        weighted_sum = 0.0

        for similarity, post in closest_posts:
            weighted_sum += similarity * post.comments
            total_weight += similarity

        # Return the weighted average
        if total_weight == 0:
            return 0  # Handle case where all similarities are 0

        return weighted_sum / total_weight

    def get_closest_posts(self, embedding: List[float], k: int, time) -> List[Tuple[float, Post]]:
        distances = []

        for post in self.posts:
            cos_similarity = self.cosine_similarity(embedding, post.embedding)
            time_similarity = self.time_similarity(time, post.time)
            combined_similarity = (1 - self.alpha) * cos_similarity + self.alpha * time_similarity

            distances.append((combined_similarity, post))

        # Sort by similarity in descending order and take the top k
        distances.sort(reverse=True, key=lambda x: x[0])

        return distances[:k]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0

        return dot_product / (norm_vec1 * norm_vec2)

    def time_similarity(self, current_time: datetime, time_ago_str: str) -> float:
        post_time = self.parse_time_ago(time_ago_str, current_time)
        time_diff = abs((current_time - post_time).days)
        max_days = 365
        normalized_time_diff = min(time_diff / max_days, 1.0)
        return 1.0 - normalized_time_diff

    def parse_time_ago(self, time_ago_str: str, current_time: datetime) -> datetime:
        # Parse the 'time ago' string
        match = re.match(r"(\d+)\s*(day|month|year)s?\s*ago", time_ago_str.strip())
        if not match:
            return current_time  # Fallback to current time if the format is unexpected

        value, unit = int(match.group(1)), match.group(2)

        if unit == 'day':
            return current_time - timedelta(days=value)
        elif unit == 'month':
            # Approximate months as 30 days
            return current_time - timedelta(days=value * 30)
        elif unit == 'year':
            # Approximate years as 365 days
            return current_time - timedelta(days=value * 365)

        return current_time  # Return current time if no match is found


    def create_embedding(self, text: str, ) -> List[float]:
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=text
        )
        print(f"embedded the Title '{text}'")
        return response['data'][0]['embedding']
