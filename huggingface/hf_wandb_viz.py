import os
import time
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Initialize Weights & Biases anonymously
# This provides an instant visualization URL in the terminal output
run = wandb.init(
    project="oscar-llm-workshop", 
    name="phi3-inference-sweep",
    anonymous="allow"
)

# 2. Set up the model identifier and load tokenizer
# We point to a tiny, robust model
model_id = "microsoft/Phi-3-mini-4k-instruct"
print(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"Loading model layers onto GPU...")
# Load in 4-bit to demonstrate memory-efficient cluster operations
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    load_in_4bit=True
)

# 3. Create a simple test dataset (Prompts with different parameters)
eval_dataset = [
    {"prompt": "Explain Quantum Computing in one sentence.", "temperature": 0.1},
    {"prompt": "Write a short creative poem about Brown University.", "temperature": 0.7},
    {"prompt": "List three unique ways to optimize code on a supercomputer.", "temperature": 0.3},
    {"prompt": "What is the capital of Rhode Island?", "temperature": 0.1}
]

print("Starting inference loop tracking...")

# 4. Process data and stream metrics to WandB
for idx, item in enumerate(eval_dataset):
    prompt = item["prompt"]
    temp = item["temperature"]
    
    # Track execution time metrics
    start_time = time.time()
    
    # Format input with template architecture
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    # Generate tokens
    outputs = model.generate(
        inputs, 
        max_new_tokens=100, 
        temperature=temp, 
        do_sample=True if temp > 0.1 else False
    )
    
    # Decode outputs back to strings
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Isolate response generation from system template
    response_only = decoded_output.split("<|assistant|>")[-1].strip()
    
    generation_time = time.time() - start_time
    tokens_generated = len(outputs[0])
    tokens_per_sec = tokens_generated / generation_time

    print(f"Processed prompt {idx+1}/{len(eval_dataset)} | Speed: {tokens_per_sec:.2f} t/s")

    # Log operational metrics per execution step
    wandb.log({
        "prompt_index": idx,
        "temperature": temp,
        "generation_time_sec": generation_time,
        "tokens_per_second": tokens_per_sec,
        "prompt_text": prompt,
        "generated_response": response_only
    })

# Close out the logging stream safely
wandb.finish()
print("Run complete! Check the terminal output above for your unique visualization dashboard link.")
