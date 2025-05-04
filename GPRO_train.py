

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import re
import pandas as pd
import json
import singlestoredb as s2
import ollama
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import BitsAndBytesConfig
from peft import PeftModel

max_seq_length = 1024
lora_rank = 64

# Define quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-4b-it",
    max_seq_length=max_seq_length,
    quantization_config=quant_config,
    fast_inference=False,  # Avoid vLLM pickling issue
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.5
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# System prompt for dish recommendations
SYSTEM_PROMPT = """
Given a description of food, provide a list of 20 food dishes that match the description. List each dish on a new line. Provide Indian food dishes unless the user specifically describes a foreign one. The dishes cant be regional ones which people havent heard much. The output format should be dish1 /n dish2 /n dish3 .... /n dish 20. If you give recipe, or other unnnessary text, you will be penalised
"""

# Function to generate embeddings using ollama
def get_embeddings(sentence: str) -> list[float]:
    response = ollama.embeddings(
        model="bge-m3:latest",
        prompt=sentence
    )
    return response["embedding"]

# Function to query vector database
def get_vector_db_dishes(description: str) -> list[str]:
    embeddings = get_embeddings(description)
    embeddings_json = json.dumps(embeddings)
    
    with s2.connect("string") as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT food
                FROM food
                ORDER BY embedding <-> %s
                LIMIT 20
            """, (embeddings_json,))
            results = cur.fetchall()
            return [row[0] for row in results]

# Load descriptions from XLSX
def load_food_descriptions(file_path: str = "merged_output.xlsx") -> Dataset:
    df = pd.read_excel(file_path)
    descriptions = df.iloc[:, 1].dropna().tolist()  # Second column
    data = {
        "prompt": [[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": desc}
        ] for desc in descriptions],
        "description": descriptions
    }
    return Dataset.from_dict(data)

dataset = load_food_descriptions()

# Reward function: Word-level matching with format penalty
def match_reward_func(prompts, completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    descriptions = [prompt[-1]["content"] for prompt in prompts]
    
    rewards = []
    for desc, response in zip(descriptions, responses):
        # Get expected dishes from vector database
        expected_dishes = get_vector_db_dishes(desc)
        # Extract model dishes (newline-separated)
        model_dishes = response.strip().split("\n")
        
        # Check for non-dish text (sentences, numbers, empty lines)
        has_extra_text = False
        cleaned_dishes = []
        for line in model_dishes:
            line = line.strip()
            if not line:
                has_extra_text = True
                continue
            # Detect numbered lines (e.g., "1. Mirchi ka Salan") or sentences
            if re.match(r"^\d+\.?\s+", line) or len(line.split()) > 5 or line.lower().startswith(("okay", "here", "list")):
                has_extra_text = True
            else:
                cleaned_dishes.append(line)
        
        # Get unique words from each set of dishes
        expected_words = set()
        for dish in expected_dishes:
            words = dish.lower().split()
            expected_words.update(words)
        
        model_words = set()
        for dish in cleaned_dishes:
            words = dish.lower().split()
            model_words.update(words)
        print("vectordb")
        print(expected_dishes)
        print("model op")
        print(model_words)
        # Count matching words
        matches = len(expected_words.intersection(model_words))
        # Base reward: 0.05 per matching word, capped at 2.0
        reward = matches * 0.05
        # Penalize 50% if extra text detected
        if has_extra_text:
            reward *= 0.5
        reward = min(reward, 2.0)
        print(f"Description: {desc}\nMatching Words: {matches}\nHas Extra Text: {has_extra_text}\nReward: {reward}")
        rewards.append(reward)
    return rewards

# Reward function: Strict format check (exactly 20 dish names)
def format_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response in responses:
        lines = response.strip().split("\n")
        valid = True
        if len(lines) != 20:
            valid = False
        else:
            for line in lines:
                line = line.strip()
                # Check for empty lines, numbered lines, or sentences
                if not line or re.match(r"^\d+\.?\s+", line) or len(line.split()) > 5 or line.lower().startswith(("okay", "here", "list")):
                    valid = False
                    break
        rewards.append(1.0 if valid else 0.0)
    return rewards

# Reward function: Check if output is non-empty
def non_empty_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if response.strip() else 0.0 for response in responses]

training_args = GRPOConfig(
    use_vllm=False,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    num_generations=2,
    max_prompt_length=512,
    max_completion_length=400,
    max_steps=200,  # Increased for better convergence
    save_steps=10,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        match_reward_func,
        format_reward_func,
        non_empty_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Example inference before saving LoRA
example_description = "Tamatar Pyaz Ki Sabzi Recipe - Tomato Onion Sabzi 4 Tomatoes - quartered,1 Onion - thinly sliced,2 Green Chillies - slit,4 cloves Garlic,1 teaspoon Turmeric powder (Haldi),1 teaspoon Red Chilli powder,Salt - to taste,1/2 teaspoon Mustard seeds,1 sprig Curry leaves,1/2 teaspoon Mustard oil,1 Bay leaves (tej patta) - torn into half,2 Cloves (Laung),1 inch Cinnamon stick 4 Tomatoes - quartered,1 Onion - thinly sliced,2 Green Chillies - slit,4 cloves Garlic,1 teaspoon Turmeric powder (Haldi),1 teaspoon Red Chilli powder,Salt - to taste,1/2 teaspoon Mustard seeds,1 sprig Curry leaves,1/2 teaspoon Mustard oil,1 Bay leaves (tej patta) - torn into half,2 Cloves (Laung),1 inch Cinnamon stick North Indian Recipes Side Dish Vegetarian To begin making the Tamatar Pyaz Ki Sabzi, first prep all the ingredients and keep them ready. Heat oil in a pressure cooker over medium heat; add the mustard seeds and allow it to crackle.Once it crackles, add the curry leaves, bay leaves, cloves, cinnamon stick and saute for a few seconds.Add the onion, green chillies and garlic. Saute the onions until they become slightly soft.Once soft, add the tomatoes, turmeric powder, red chilli powder and salt to taste.Add just a little water - about 2 to 3 tablespoons and cover the pressure cooker. Cook for 2 whistles and turn off the heat.Once the pressure released, check the salt and spices in the Tamatar Pyaz Ki Sabzi and adjust to suit your taste.Serve the Tamatar Pyaz Ki Sabzi along with Aloo Bhindi Sabzi, Phulka, Gujarati Kadhi for a delicious Sunday lunch."
text = tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": example_description},
], tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_length=1024,
    temperature=0.8,
    top_p=0.95,
    do_sample=True
)
output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Inference before saving LoRA:\n{output}")

# Save LoRA weights
model.save_pretrained("foodlora")

# Load LoRA weights
model = PeftModel.from_pretrained(model, "foodlora")
]
# Example inference after saving LoRA
text = tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": example_description},
], tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_length=1024,
    temperature=0.8,
    top_p=0.95,
    do_sample=True
)
output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Inference after saving LoRA:\n{output}")

