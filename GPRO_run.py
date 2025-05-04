

from unsloth import FastLanguageModel
import torch
from transformers import BitsAndBytesConfig
from peft import PeftModel

max_seq_length = 1024

# Define quantization config for 4-bit loading
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Load base model with quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-4b-it",
    max_seq_length=max_seq_length,
    quantization_config=quant_config,
    fast_inference=False,  # Avoid vLLM due to pickling issues
    gpu_memory_utilization=0.5
)

# Load saved LoRA weights
model = PeftModel.from_pretrained(model, "grpo_food_lora")

# Define system prompt for formatted output
SYSTEM_PROMPT = """
Given a description of food, provide a list of 20 food dishes that match the description. List each dish on a new line. Provide Indian food dishes unless the user specifically describes a foreign one. The dishes cant be regional ones which people havent heard much. The output format should be dish1 /n dish2 /n dish3 .... /n dish 20. If you give recipe, or other unnnessary text, you will be penalised
"""

# Prepare input for inference
text = tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Something spicy south indian sour soft?"},
], tokenize=False, add_generation_prompt=True)

# Tokenize input
inputs = tokenizer(text, return_tensors="pt").to("cuda")

# Generate output
outputs = model.generate(
    **inputs,
    max_length=1024,
    temperature=0.8,
    top_p=0.95,
    do_sample=True
)
output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output)


