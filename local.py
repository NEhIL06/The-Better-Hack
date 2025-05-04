import pandas as pd
import requests # Library to make HTTP requests
import json     # Library to handle JSON data
import time     # To add delays if needed

# --- Ollama Configuration ---
OLLAMA_API_ENDPOINT = "http://localhost:11434/api/embeddings"
# IMPORTANT: Replace with the embedding model you have pulled in Ollama
# Common embedding models: 'mxbai-embed-large', 'nomic-embed-text', 'all-minilm'
OLLAMA_EMBED_MODEL = "bge-m3:latest"
# ---------------------------

# --- Batching Configuration ---
# Ollama API generates one embedding per request.
# We can still process the DataFrame in logical chunks if needed,
# but the API calls themselves are sequential unless we implement async requests.
# This batch_size is more about how many rows we process before printing progress, etc.
PROCESS_CHUNK_SIZE = 100 # Process and potentially print status every N rows
# ---------------------------


# Function to get embedding from Ollama
def get_ollama_embedding(text, retries=3, delay=5):
    """Requests embedding for a single text from the Ollama API."""
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "prompt": text
        # "options": {"temperature": 0.0} # Optional parameters if needed
    }
    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_API_ENDPOINT, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            if "embedding" in data:
                return data["embedding"]
            else:
                print(f"Warning: 'embedding' key not found in response for text chunk: {text[:50]}...")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama or during request (Attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Skipping this text.")
                return None
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON response from Ollama: {response.text}")
            return None # Or handle differently
    return None


# Read the previously merged Excel file
input_filename = 'merged_output.xlsx'
try:
    df = pd.read_excel(input_filename)
except FileNotFoundError:
    print(f"Error: Input file '{input_filename}' not found.")
    exit()

# Identify columns
first_col_name = df.columns[0]
second_col_name = df.columns[1]

# Get the text data from the second column
texts_to_embed = df[second_col_name].astype(str).tolist() # Ensure texts are strings

print(f"Generating embeddings using Ollama model '{OLLAMA_EMBED_MODEL}' at {OLLAMA_API_ENDPOINT}")

embeddings_list = []
total_texts = len(texts_to_embed)

# Process texts one by one (Ollama API standard endpoint handles one at a time)
for i, text in enumerate(texts_to_embed):
    if not text or pd.isna(text): # Handle empty or NaN texts
        print(f"Warning: Skipping empty or invalid text at index {i}")
        embeddings_list.append(None) # Placeholder for skipped text
        continue

    embedding = get_ollama_embedding(text)
    embeddings_list.append(embedding)

    # Print progress
    if (i + 1) % PROCESS_CHUNK_SIZE == 0 or (i + 1) == total_texts:
        print(f"Processed {i + 1}/{total_texts} texts...")


# Check if all embeddings failed
if all(e is None for e in embeddings_list):
     print("Error: Failed to generate embeddings for all texts. Check Ollama server and model name.")
     # Optionally exit or handle differently
     # exit() # Uncomment to stop if all fail


# Add the embeddings as a new column
# Convert embeddings (lists of floats) to string representation for Excel compatibility
df['embeddings'] = [
    ','.join(map(str, emb)) if emb is not None else ''
    for emb in embeddings_list
]

# Define the output filename
output_filename = 'embedded_output_ollama.xlsx'

# Select the desired columns
output_df = df[[first_col_name, second_col_name, 'embeddings']]

# Save the result to a new Excel file
output_df.to_excel(output_filename, index=False)

print(f"Embeddings generated using Ollama and saved to {output_filename}")
