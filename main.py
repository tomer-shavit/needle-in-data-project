import os
import pandas as pd
import openai
import json

# Load the environment variable for OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the CSV file
file_path = "./combined_cleaned_news.csv"
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
    print(f"embedded the Title '{text}'")
    return response['data'][0]['embedding']


data = {}
failed = []
for title_id, title in zip(ids, titles):
    try:
        embedding = get_embedding(title)
        data[title_id] = {
            "title": title,
            "embedding": embedding
        }
    except:
        print(f"failed to emmbed {title}")
        failed.append((title_id, title))

for item in failed:
    try:
        embedding = get_embedding(item[1])
        data[item[0]] = {
            "title": item[1],
            "embedding": embedding
        }
    except:
        print(f"FAILED AGAIN: '{item[1]}'")


# Save the data to a JSON file
output_file = "./titles_embeddings.json"
with open(output_file, 'w') as f:
    json.dump(data, f)

print(f"Embeddings saved to {output_file}")
