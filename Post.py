class Post:
    """
    A class representing a social media post with various attributes such as title, upvotes, comments,
    time, embedding, and category. The category of a post is mapped based on specific keywords in the title.
    """

    # A dictionary mapping post categories to lists of keywords commonly associated with that category.
    category_mapping = {
        'Politics/Government': ['biden', 'trump', 'ukraine', 'russia', 'russian', 'population', 'state',
                                'administration', 'china', 'policy', 'ban', 'counsel', 'law', 'national',
                                'president', 'country', 'queen', 'protest', 'vote', 'federal', 'right'],
        'Crime/Justice': ['police', 'arrest', 'charge', 'officer', 'court', 'judge', 'sentence', 'attack',
                          'murder', 'crime', 'fight', 'kill', 'shoot', 'shot', 'assault', 'abus', 'legal',
                          'guilty', 'prison', 'order', 'free', 'case', 'sex'],
        'Health/Medical': ['sick', 'dead', 'death', 'die', 'ill', 'disease', 'operation', 'surgery',
                           'medical', 'vaccine', 'cancer', 'coronavirus', 'hospital', 'health', 'covid',
                           'doctor'],
        'Business/Economy': ['dow', 'business', 'employee', 'ceo', 'global', 'bank', 'manager', 'salary',
                             'office', 'work', 'money', 'million', 'company', 'job', 'homeless', 'pay',
                             'bill', 'free', 'sign', 'tax', 'donate'],
        'Sport': ['sport', 'athlete', 'football', 'game', 'basketball', 'play'],
        'Disaster': ['crash', 'accident', 'dangerous', 'fire', 'tsunami', 'disaster', 'tragedy', 'storm',
                     'earthquake', 'shelter', 'saved', 'rescue', 'pandemic'],
        'Education': ['school', 'student', 'college', 'teacher', 'study'],
        'Lifestyle': ['life', 'restaurant', 'lifestyle', 'home', 'house', 'marri', 'book', 'therapy'],
        'Animals': ['dog', 'animal', 'elephant', 'cat', 'pet', 'cow']
    }

    def __init__(self, idx: int, title: str, upvotes: float, comments: float, time: str, embedding: list,
                 category: str):
        """
        Initializes a Post instance with the given parameters.

        Parameters:
            idx (int): The unique identifier of the post.
            title (str): The title or headline of the post.
            upvotes (float): The number of upvotes the post has received.
            comments (float): The number of comments the post has received.
            time (str): A string representing how long ago the post was made (e.g., '2 days ago').
            embedding (list): A list representing the vector embedding of the post title or content.
            category (str): The category to which the post belongs (e.g., 'Politics', 'Sport').
        """
        self.idx = idx  # Unique ID for the post
        self.title = title  # Post title or headline
        self.upvotes = upvotes  # Number of upvotes the post has received
        self.comments = comments  # Number of comments the post has received
        self.time = time  # Time reference for the post, e.g., '1 day ago'
        self.embedding = embedding  # Embedding vector for the post title/content
        self.category = category  # Category the post belongs to, mapped by keywords
