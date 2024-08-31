class Post:
    def __init__(self, idx, title, upvotes, comments, time, embedding):
        self.idx = idx
        self.title = title
        self.upvotes = upvotes
        self.comments = comments
        self.time = time
        self.embedding = embedding
