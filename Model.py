import os
from typing import Dict, List, Tuple
import numpy as np
import openai
from Post import Post

class Model:
    def __init__(self, posts: List[Post]):
        self.posts = posts
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def predict_likes(self, title: str, k: int = 10, embedding=None) -> float:
        if not embedding:
            embedding = self.create_embedding(title)

        closest_posts = self.get_closest_posts(embedding, k)

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

    def predict_comments(self, title: str, k: int = 10, embedding=None) -> float:
        if not embedding:
            embedding = self.create_embedding(title)

        closest_posts = self.get_closest_posts(embedding, k)
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

    def get_closest_posts(self, embedding: List[float], k: int) -> List[Tuple[float, Post]]:
        distances = []

        for post in self.posts:
            similarity = self.cosine_similarity(embedding, post.embedding)
            distances.append((similarity, post))

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

    def create_embedding(self, text: str, ) -> List[float]:
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=text
        )
        print(f"embedded the Title '{text}'")
        return response['data'][0]['embedding']
