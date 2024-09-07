import os
import pandas as pd
import openai
import json

# Load the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the CSV file into a pandas DataFrame
file_path = "./modified_file.csv"
df = pd.read_csv(file_path)

# Extract the 'Title' and 'ID' columns as lists
titles = df['Title'].tolist()
ids = df['ID'].tolist()


def get_embedding(text):
    """
    Retrieves the vector embedding for a given text using the OpenAI API.

    Parameters:
        text (str): The input text for which the embedding is to be generated.

    Returns:
        list: The embedding vector for the input text.
    """
    response = openai.Embedding.create(
        model="text-embedding-3-small",  # Specify the embedding model
        input=text
    )
    print(f"Embedded the title: '{text}'")
    return response['data'][0]['embedding']


# Dictionary to store the embeddings, keyed by post ID
data = {}

# List to store failed attempts for embedding
failed = []

# Generate embeddings for each title and store them
for title_id, title in zip(ids, titles):
    try:
        embedding = get_embedding(title)
        data[title_id] = {
            "title": title,
            "embedding": embedding
        }
    except Exception as e:
        # Capture any errors during the embedding process and log the failure
        print(f"Failed to embed title '{title}': {e}")
        failed.append((title_id, title))

# Retry embedding for titles that failed in the first attempt
for item in failed:
    try:
        embedding = get_embedding(item[1])
        data[item[0]] = {
            "title": item[1],
            "embedding": embedding
        }
    except Exception as e:
        # Log the failure after the second attempt
        print(f"FAILED AGAIN: '{item[1]}': {e}")

# Save the embeddings to a JSON file
output_file = "titles_embeddings.json"
with open(output_file, 'w') as f:
    json.dump(data, f)

print(f"Embeddings saved to {output_file}")
