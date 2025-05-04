import streamlit as st
import requests
import base64
from dotenv import load_dotenv
from openai import OpenAI
import subprocess
import json
import singlestoredb as s2
from pydub import AudioSegment
import io
import sqlite3
from unsloth import FastLanguageModel
import torch

# Load environment variables from .env file
load_dotenv()
LLM = "gemini-2.0-flash-exp"
subscription_key = "731bcfac-aef2-4541-88ee-1dc114b017a4"
xai_api_key = "AIzaSyA02JdZFZ3Xjj26ThJhhJQ7anhrbrI66h8"
sarvamurl = "https://api.sarvam.ai/text-to-speech"
sarvamheaders = {
    "accept": "application/json",
    "content-type": "application/json",
    "api-subscription-key": subscription_key
}

# Initialize OpenAI client
LLMclient = OpenAI(
    api_key=xai_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
Nclient = OpenAI(
    api_key="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",
    base_url="https://integrate.api.nvidia.com/v1"
)

# Load LoRA fine-tuned model (Unsloth FastLanguageModel)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Meta-Llama-3-8B",  # Replace with your base model
    adapter_name="path/to/lora-adapter",      # Replace with your LoRA adapter path
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True
)
FastLanguageModel.for_inference(model)

# Initialize SQLite database for ingredient inventory
def init_sqlite_db():
    conn = sqlite3.connect("inventory.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ingredients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            quantity REAL NOT NULL,
            unit TEXT NOT NULL
        )
    """)
    # Insert sample data if table is empty
    cursor.execute("SELECT COUNT(*) FROM ingredients")
    if cursor.fetchone()[0] == 0:
        sample_ingredients = [
            ("Tomato", 5, "units"),
            ("Rice", 2, "kg"),
            ("Chicken", 1, "kg"),
            ("Onion", 3, "units"),
            ("Spices", 100, "g")
        ]
        cursor.executemany("INSERT INTO ingredients (name, quantity, unit) VALUES (?, ?, ?)", sample_ingredients)
    conn.commit()
    return conn

# Load CSS from file
with open("style.css", "r") as f:
    css = f.read()

# Add background image to CSS
css += """
body {
    background-image: url('https://images.unsplash.com/photo-1512621776951-a57141f2eefd'); /* Food-themed background */
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    height: 100vh;
}
"""

# Inject CSS into Streamlit
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

def create_connection():
    return s2.connect('admin:X8MBbWxI1NiuG6RGPhyIQcr7lz4oseOY@svc-8bd4e6d7-dd92-449e-b8af-56828e3aea12-dml.aws-mumbai-1.svc.singlestore.com:3306/miniDB')

# Function to send audio to Sarvam API for speech recognition
def transcribe_audio(audio_file, subscription_key, language_code):
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_file.read()))
        duration_ms = len(audio)
        if duration_ms > 30000:
            full_transcript = ""
            start = 0
            while start < duration_ms:
                end = min(start + 30000, duration_ms)
                chunk = audio[start:end]
                chunk_io = io.BytesIO()
                chunk.export(chunk_io, format="wav")
                chunk_io.seek(0)
                files = [('file', ('audio_chunk.wav', chunk_io, 'audio/wav'))]
                headers = {'api-subscription-key': subscription_key}
                payload = {
                    'model': 'saarika:v1',
                    'language_code': language_code,
                    'with_timesteps': 'false'
                }
                response = requests.post("https://api.sarvam.ai/speech-to-text", headers=headers, data=payload, files=files)
                if response.status_code == 200:
                    full_transcript += response.json().get('transcript', '') + " "
                else:
                    return f"Error: {response.status_code}, {response.text}"
                start += 30000
            return full_transcript.strip()
        else:
            files = [('file', ('audio.wav', audio_file, 'audio/wav'))]
            headers = {'api-subscription-key': subscription_key}
            payload = {
                'model': 'saarika:v1',
                'language_code': language_code,
                'with_timesteps': 'false'
            }
            response = requests.post("https://api.sarvam.ai/speech-to-text", headers=headers, data=payload, files=files)
            if response.status_code == 200:
                return response.json().get('transcript', 'No transcript available.')
            else:
                return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"An error occurred during transcription: {str(e)}"

# Function to get embeddings and nearest neighbors from SingleStoreDB
def get_embeddings_and_neighbors(sentence):
    response = Nclient.embeddings.create(
        input=[sentence],
        model="baai/bge-m3",
        encoding_format="float",
        extra_body={"truncate": "NONE"}
    )
    embeddings_json = json.dumps(response.data[0].embedding)
    with create_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, recipe_name, ingredients, instructions, embedding <-> %s AS score
                FROM webgen
                ORDER BY score
                LIMIT 5
            """, (embeddings_json,))
            results = cur.fetchall()
            with open("embeddings.txt", "w") as f:
                for row in results:
                    f.write(str(row) + "\n")
            return results

# Function to get food suggestions from LoRA fine-tuned model
def get_lora_food_suggestions(user_input):
    prompt = f"Suggest a food recipe based on the following user input: {user_input}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return suggestion

# Function to check ingredient inventory in SQLite
def check_inventory(ingredients_needed):
    conn = init_sqlite_db()
    cursor = conn.cursor()
    available = []
    unavailable = []
    for ingredient, quantity, unit in ingredients_needed:
        cursor.execute("SELECT quantity, unit FROM ingredients WHERE name = ?", (ingredient,))
        result = cursor.fetchone()
        if result and result[0] >= quantity:
            available.append((ingredient, quantity, unit))
        else:
            unavailable.append((ingredient, quantity, unit))
    conn.close()
    return available, unavailable

# Function to process text and generate recipe
def process_with_llm(text, recipe_data, lora_suggestion, conversation_history):
    ingredients = recipe_data[2]  # From SingleStoreDB
    instructions = recipe_data[3]
    history_context = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in conversation_history])
    prompt = f"""
    Based on the user input: '{text}', the following recipe data: 
    Ingredients: {ingredients}
    Instructions: {instructions}
    LoRA model suggestion: {lora_suggestion}
    Conversation history: {history_context}

    Generate a detailed recipe recommendation, including a list of ingredients, step-by-step instructions, and an HTML webpage code for the recipe. Ensure the recipe aligns with the user's language and preferences. Format the HTML code between ```html and ``` markers.
    """
    try:
        response = LLMclient.chat.completions.create(
            model=LLM,
            messages=[
                {"role": "system", "content": "You are a cooking and cuisine guide agent."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Language code mapping
language_mapping = {
    "Kannada": "kn-IN",
    "Hindi": "hi-IN",
    "Bengali": "bn-IN",
    "Malayalam": "ml-IN",
    "Marathi": "mr-IN",
    "Odia": "od-IN",
    "Punjabi": "pa-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Gujarati": "gu-IN"
}

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Streamlit UI
st.title("Cooking & Cuisine Guide Agent")

# Sidebar for language and cuisine selection
with st.sidebar:
    selected_language = st.selectbox("Select Language:", list(language_mapping.keys()), index=0)
    language_code = language_mapping[selected_language]
    cuisine_options = ["Indian", "Italian", "Chinese", "Mexican", "Any"]
    selected_cuisine = st.selectbox("Select Cuisine:", cuisine_options, index=4)
    st.subheader("Ingredient Inventory")
    conn = init_sqlite_db()
    cursor = conn.cursor()
    cursor.execute("SELECT name, quantity, unit FROM ingredients")
    inventory = cursor.fetchall()
    for name, quantity, unit in inventory:
        st.write(f"{name}: {quantity} {unit}")
    conn.close()

audio_value = st.audio_input("Record a voice message (e.g., 'I want a spicy Indian dish')")

if audio_value:
    if subscription_key:
        with st.spinner("Transcribing..."):
            transcript = transcribe_audio(audio_value, subscription_key, language_code)
            st.write(f"User 🙋 : {transcript}")

            # Fetch embeddings and nearest neighbors
            results = get_embeddings_and_neighbors(transcript)
            if results:
                recipe_data = results[0]  # Get the top recipe
                lora_suggestion = get_lora_food_suggestions(transcript)

                # Check ingredient availability
                ingredients_needed = [(ing.split(":")[0], float(ing.split(":")[1].split()[0]), ing.split(":")[1].split()[1])
                                     for ing in recipe_data[2].split(", ")]  # Parse ingredients
                available, unavailable = check_inventory(ingredients_needed)
                inventory_message = f"Available: {', '.join([f'{i[0]} ({i[1]} {i[2]})' for i in available])}\n" \
                                  f"Unavailable: {', '.join([f'{i[0]} ({i[1]} {i[2]})' for i in unavailable])}"
                
                # Process with LLM
                sarvam_message = process_with_llm(transcript, recipe_data, lora_suggestion, st.session_state.conversation_history)
                
                # Update conversation history
                st.session_state.conversation_history.append({
                    "user": transcript,
                    "assistant": sarvam_message.split('\n\n')[0]
                })

                # Display response and inventory status
                st.write(f"Model 🤖 : {sarvam_message.split('\n\n')[0]}")
                st.write(f"Inventory Status: {inventory_message}")

                # Extract and save HTML code
                html_code = ""
                start_marker = "```html"
                end_marker = "```"
                start_index = sarvam_message.find(start_marker)
                end_index = sarvam_message.find(end_marker, start_index + len(start_marker))
                if start_index != -1 and end_index != -1:
                    html_code = sarvam_message[start_index + len(start_marker):end_index].strip()
                    with open("recipe_page.html", "w", encoding='utf-8') as f:
                        f.write(html_code)
                    subprocess.run(["start", "recipe_page.html"], shell=True)

                # Sarvam AI text-to-speech
                payload = {
                    "inputs": [sarvam_message.split('\n\n')[0][:490]],
                    "target_language_code": language_code,
                    "speaker": "meera",
                    "pitch": 0.2,
                    "pace": 1.1,
                    "loudness": 0.8,
                    "enable_preprocessing": True,
                    "model": "bulbul:v1",
                    "speech_sample_rate": 8000
                }
                response = requests.request("POST", sarvamurl, json=payload, headers=sarvamheaders)
                audio_data = response.json()
                if "audios" in audio_data and audio_data["audios"]:
                    audio_bytes = base64.b64decode(audio_data["audios"][0])
                    st.markdown('<div class="st-ae">', unsafe_allow_html=True)
                    st.audio(audio_bytes, format="audio/wav", autoplay=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No recipes found.")
    else:
        st.error("API Subscription Key not found in environment variables.")