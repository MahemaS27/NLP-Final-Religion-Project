import os
import json
import torch
from transformers import pipeline, AutoTokenizer, logging

# 1. Configuration and Setup
logging.set_verbosity_error()

BASE_DIR = "/projectnb/cs505am/students/mscs2024"
INPUT_DIR = os.path.join(BASE_DIR, "data/final")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/instruction_sets")
BATCH_SIZE = 8 # Reduced for memory stability

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Model Initialization
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token

generator = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

def process_file(filepath):
    filename = os.path.basename(filepath)
    out_name = filename.replace(".txt", ".jsonl")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    
    # Skip if already processed
    if os.path.exists(out_path):
        print(f"Skipping {filename}: already exists in {OUTPUT_DIR}")
        return
    
    print(f"--- Processing {filename} ---")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        verses = [line.strip() for line in f if len(line.strip()) > 20]

    print(f"Generating {len(verses)} pairs for {out_name}...")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        # Batch loop
        for i in range(0, len(verses), BATCH_SIZE):
            batch = verses[i : i + BATCH_SIZE]
            
            # Format prompts
            prompts = [
                [
                    {"role": "system", "content": "You are a creative researcher. Generate questions that reflect a diverse set of intents (theology, morals, history). Return ONLY the question."},
                    {"role": "user", "content": f"Scripture: '{verse}'\n\nQuestion:"}
                ] for verse in batch
            ]
            
            # Run Inference
            with torch.inference_mode():
                results = generator(
                    prompts,
                    batch_size=len(batch),
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Save results
            for verse, res in zip(batch, results):
                instruction = res[0]['generated_text'][-1]['content'].strip()
                entry = {"instruction": instruction, "input": "", "output": verse}
                f.write(json.dumps(entry) + "\n")
            
            # Clear memory cache after every batch
            torch.cuda.empty_cache()
                
    print(f"Finished {filename}. Saved to {out_path}\n")

# 3. Main Loop
if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory {INPUT_DIR} not found.")
    else:
        # Sort files to ensure predictable processing order
        for filename in sorted(os.listdir(INPUT_DIR)):
            # Only process files that are clearly data splits
            if filename.endswith(".txt") and ("train" in filename or "val" in filename):
                process_file(os.path.join(INPUT_DIR, filename))
            else:
                continue