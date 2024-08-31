import os
import pandas as pd
import openai
import json

# Load the environment variable for OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the CSV file
file_path = "./Merged_DataFrame.csv"
df = pd.read_csv(file_path)

# Assuming the CSV has a 'Title' column and an 'ID' column
titles = df['Title'].tolist()
ids = df['ID'].tolist()

# Function to get vector embedding from OpenAI API
def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-3-small",
        input=text
    )
    return response['data'][0]['embedding']

# Create a dictionary to store the ID as the key and title, vector embedding as values
data = {}

for title_id, title in zip(ids, titles):
    embedding = get_embedding(title)
    data[title_id] = {
        "title": title,
        "embedding": embedding
    }

# Save the data to a JSON file
output_file = "./titles_embeddings.json"
with open(output_file, 'w') as f:
    json.dump(data, f)

print(f"Embeddings saved to {output_file}")
