import os
import time
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    wandb.init(
        project="oscar-llm-workshop",
        name=f"phi3-inference-{os.environ.get('USER', 'student')}",
        #mode="offline"
    )

    model_path = "/oscar/data/shared/bootcamp_2026/llm-workshop/phi3-mini"
    print(f"Loading tokenizer for {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    print(f"Loading model layers onto GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16,
        local_files_only=True
    )

    eval_dataset = [
        {"prompt": "Explain Quantum Computing in one sentence.", "temperature": 0.1},
        {"prompt": "Write a short creative poem about Brown University.", "temperature": 0.7},
        {"prompt": "List three unique ways to optimize code on a supercomputer.", "temperature": 0.3},
        {"prompt": "What is the capital of Rhode Island?", "temperature": 0.1}
    ]

    print("Starting inference loop tracking...")


    for idx, item in enumerate(eval_dataset):
        prompt = item["prompt"]
        temp = item["temperature"]

        start_time = time.time()

        messages = [{"role": "user", "content": prompt}]
        
        # We must tokenize it separately
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False  # Get the formatted string
        )
        
        # Now tokenize the string to get actual tensors
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False  # Already included by apply_chat_template
        ).to("cuda")

        outputs = model.generate(
            **inputs,  # Unpack the dict: input_ids, attention_mask
            max_new_tokens=100,
            do_sample=True,
            temperature=temp,
            top_p=0.9 if temp > 0.1 else 1.0
        )

        # Decode only the new tokens (skip the prompt)
        response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response_only = tokenizer.decode(response_tokens, skip_special_tokens=True)

        generation_time = time.time() - start_time
        tokens_generated = len(response_tokens)
        tokens_per_sec = tokens_generated / generation_time

        print(f"Processed prompt {idx+1}/{len(eval_dataset)} | Speed: {tokens_per_sec:.2f} t/s")

        # 1. Log metrics for CHARTS
        wandb.log({
            "prompt_index": idx,
            "temperature": temp,
            "generation_time_sec": generation_time,
            "tokens_per_second": tokens_per_sec,
            "step": idx,  # Explicit step counter
        })

        # 2. Log responses
        wandb.log({
            f"prompt_{idx}": prompt,
            f"temperature_{idx}": temp,
            f"generation_time_sec_{idx}": generation_time,
            f"tokens_per_second_{idx}": tokens_per_sec,
            f"generated_response_{idx}": response_only
        })

    wandb.finish()
    print("Run complete! Sync with: wandb sync wandb/offline-run-*")

if __name__ == "__main__":
    main()
