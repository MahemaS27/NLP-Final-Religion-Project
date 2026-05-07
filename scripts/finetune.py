import os
import json
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,  # Ensure this is imported
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# --- NEW: Callback Class to handle JSON logging ---
class JsonLoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_data = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Triggered by the Trainer on every logging step.
        """
        if logs is None:
            return

        # Prepare entry formatted exactly how your plotter expects
        entry = {"step": state.global_step}
        
        # Capture training loss
        if "loss" in logs:
            entry["train_loss"] = logs["loss"]
        
        # Capture validation loss
        if "eval_loss" in logs:
            entry["eval_loss"] = logs["eval_loss"]

        # Only save if we captured relevant loss data
        if "train_loss" in entry or "eval_loss" in entry:
            self.log_data.append(entry)
            # Write/Overwrite the log file
            with open(self.log_path, "w") as f:
                json.dump(self.log_data, f, indent=4)
# ---------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

TRAIN_CONFIG = dict(
    max_steps=500,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    bf16=True,                        
    gradient_checkpointing=True,       
    logging_steps=10,
    save_strategy="steps",             
    save_steps=100,                    
    dataloader_num_workers=1,
    report_to="none", # This is fine; the callback will handle the logging
)

RELIGIONS = ["christianity", "islam", "hinduism", "judaism"]
DATA_DIR = Path("data/instruction_sets") 
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")

def formatting_prompts_func(example, religion_name):
    text = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are practioner of religion well versed in {religion_name}. Answer in the voice and tone of the {religion_name} scripture.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{example['output']}<|eot_id|>"
    )
    return text

def load_corpus(religion: str):
    train_path = DATA_DIR / f"{religion}_train.jsonl"
    val_path = DATA_DIR / f"{religion}_val.jsonl"
    dataset = load_dataset("json", data_files={"train": str(train_path), "test": str(val_path)})
    return dataset["train"], dataset["test"]

def load_model_and_tokenizer(hf_token: str | None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto", 
        token=hf_token,
        attn_implementation="sdpa"
    )
    
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--religion", required=True, choices=RELIGIONS)
    args = parser.parse_args()

    religion = args.religion
    hf_token = os.environ.get("HF_TOKEN")

    output_dir = MODELS_DIR / religion
    log_dir = LOGS_DIR / religion
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = log_dir / "train_log.json"
    json_callback = JsonLoggingCallback(log_file_path)

    train_ds, val_ds = load_corpus(religion)
    model, tokenizer = load_model_and_tokenizer(hf_token)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        eval_strategy="steps",
        eval_steps=100,
        do_eval=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,           
        **TRAIN_CONFIG,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=lambda x: formatting_prompts_func(x, religion),
        callbacks=[json_callback],
    )

    print(f"[{religion}] Starting Fine-Tuning...")
    trainer.train()
    
    trainer.save_model(str(output_dir / "final"))
    print(f"[{religion}] Process complete. Model saved.")

if __name__ == "__main__":
    main()