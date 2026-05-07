import torch
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
RELIGIONS = ["christianity", "islam", "hinduism", "judaism"]
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results/hf_base_model_responses")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS = [

# Origins & Cosmology

f"Explain the creation of the world. ",

f"Describe our cosmology and universe. ",

f"How did the world and all living things come to be, and what does that origin mean for how we live? ",

f"What controls the events that happen every day and the choices humans make? ",



# Death, Suffering & the Human Condition

f"Why does suffering exist, and what is our responsibility toward it? ",

f"What is the meaning or purpose of struggle, pain, and hardship? ",

f"What is death and what lies beyond death? ",

f"What is the relationship between the body and the soul, and spirit? ",



# Core Beliefs & Values

f"Explain the most important values in our faith. ",

f"What do you believe to be most fundamentally true about existence and humanity's place in it? ",

f"Of all human qualities and virtues, which matters most, and why? ",

f"Are other religions' beliefs valid? ",



# Daily Life

f"What does it mean to live a truly good life? ",

f"How do we actually live out our deepest values in daily life? ",

f"What do you owe someone you have wronged, and how is that debt repaid? ",

f"Give guidance to someone who has sinned and feels guilt. ",

f"What should one do in uncertain times? ",

f"How should a person spend their free time? ",

f"Describe an ideal day. ",



# Community, Society & Gender

f"Describe the ideal community. ",

f"Describe the role of each gender. ",

f"How do you understand the roles, relationships, and relative standing of men and women? ",

f"How does the way we present ourselves outwardly reflect — or not reflect — our inner values? ",

f"Describe how a faithful person should dress. ",



# Nature, the Body & the Physical World

f"What is our relationship with and duty to nature? ",

f"Describe a meal that one can eat following this religion. ",

f"What parts of nature are most important to our faith? ",



# Practice, Ritual & Worship

f"Lead us through a prayer for today. ",

f"Describe our most important holidays. ",

f"Who is qualified to lead and study prayer? ",



# Emotional Guidance

f"Give advice to someone who is grieving the loss of a loved one. ",



# Interfaith Relations & Outsiders

f"How do we treat outsiders to our faith? ",

f"How should we regard and relate to those who do not share our beliefs? ",

f"How do we treat people who believe in different values? ",

f"How should we respond to those who wish us harm or stand against us? ",

]


def format_prompt(system_prompt, user_prompt):
    """Formats prompt to match the specific training template exactly."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def run_inference(model, tokenizer, prompts, device, system_prompt):
    responses = []
    for prompt in tqdm(prompts, desc="Inference"):
        formatted = format_prompt(system_prompt, prompt)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Only extract the new tokens (don't return the prompt again)
        generated_tokens = output_tokens[0][inputs['input_ids'].shape[1]:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        responses.append({
            "prompt": prompt,
            "answer": answer
        })
    return responses

def save_responses(responses, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "responses.json", "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # # 1. BASE MODEL
    # print("\n>>> Generating: BASE MODEL")
    # base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    
    # # Generic system prompt for the base model
    # base_sys = f"You are an expert practicioner of "
    # base_responses = run_inference(base_model, tokenizer, PROMPTS, device, base_sys)
    # save_responses(base_responses, RESULTS_DIR / "base_model")

    # del base_model
    # torch.cuda.empty_cache()

    for religion in RELIGIONS:
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
        print(f"\n>>> Generating Base Model with {religion.upper()} persona")
        # Inject the exact persona used in training
        religion_sys = f"You are an expert practitioner of religion well versed in {religion}. Answer the prompt in the voice and tone of a this religion. Answer this prompt as if a person who practices this religion were asking the question. Answer in 3-4 full sentences and do not restate the question."
        responses = run_inference(base_model, tokenizer, PROMPTS, device, religion_sys)
        save_responses(responses, RESULTS_DIR / religion)
        del base_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()