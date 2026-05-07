import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# Configuration
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
RELIGIONS = ["christianity", "islam", "hinduism", "judaism"]
DATA_DIR = Path("data/instruction_sets") 
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def formatting_prompts_func(example, religion_name):
    # This must match the exact format used during training
    text = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are practioner of religion well versed in {religion_name}. Answer in the voice and tone of the {religion_name} scripture.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{example['output']}<|eot_id|>"
    )
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--religion", choices=RELIGIONS, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.religion}] Loading model for Perplexity evaluation...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # Load the fine-tuned adapter
    model = PeftModel.from_pretrained(base_model, MODELS_DIR / args.religion / "final")
    model.eval()

    # Load validation data
    val_path = DATA_DIR / f"{args.religion}_val.jsonl"
    dataset = load_dataset("json", data_files={"val": str(val_path)})["val"]

    # Format the data exactly as the model expects
    formatted_texts = [formatting_prompts_func(ex, args.religion) for ex in dataset]
    full_text = "".join(formatted_texts)

    encodings = tokenizer(full_text, return_tensors="pt")
    max_length = 1024 
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    print(f"[{args.religion}] Calculating perplexity...")
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        
        # Mask tokens that are part of the context (history) to focus on the completion
        target_ids[:, :-trg_len] = -100 

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # Loss is the mean NLL for the specific window
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # Calculate final Perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    print(f"[{args.religion}] Final Perplexity: {ppl.item():.4f}")

    with open(RESULTS_DIR / f"{args.religion}_ppl.json", "w") as f:
        json.dump({"religion": args.religion, "perplexity": ppl.item()}, f)

if __name__ == "__main__":
    main()