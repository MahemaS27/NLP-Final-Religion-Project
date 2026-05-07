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
RESULTS_DIR = Path("results/prompt_responses")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_INSTRUCTION = "Answer in 3-4 full sentences. Do not ask further questions. Do not restate the question."

PROMPTS = [

# Origins & Cosmology

f"Explain the creation of the world. {PROMPT_INSTRUCTION}",

f"Describe our cosmology and universe. {PROMPT_INSTRUCTION}",

f"How did the world and all living things come to be, and what does that origin mean for how we live? {PROMPT_INSTRUCTION}",

f"What controls the events that happen every day and the choices humans make? {PROMPT_INSTRUCTION}",



# Death, Suffering & the Human Condition

f"Why does suffering exist, and what is our responsibility toward it? {PROMPT_INSTRUCTION}",

f"What is the meaning or purpose of struggle, pain, and hardship? {PROMPT_INSTRUCTION}",

f"What is death and what lies beyond death? {PROMPT_INSTRUCTION}",

f"What is the relationship between the body and the soul, and spirit? {PROMPT_INSTRUCTION}",



# Core Beliefs & Values

f"Explain the most important values in our faith. {PROMPT_INSTRUCTION}",

f"What do you believe to be most fundamentally true about existence and humanity's place in it? {PROMPT_INSTRUCTION}",

f"Of all human qualities and virtues, which matters most, and why? {PROMPT_INSTRUCTION}",

f"Are other religions' beliefs valid? {PROMPT_INSTRUCTION}",



# Daily Life

f"What does it mean to live a truly good life? {PROMPT_INSTRUCTION}",

f"How do we actually live out our deepest values in daily life? {PROMPT_INSTRUCTION}",

f"What do you owe someone you have wronged, and how is that debt repaid? {PROMPT_INSTRUCTION}",

f"Give guidance to someone who has sinned and feels guilt. {PROMPT_INSTRUCTION}",

f"What should one do in uncertain times? {PROMPT_INSTRUCTION}",

f"How should a person spend their free time? {PROMPT_INSTRUCTION}",

f"Describe an ideal day. {PROMPT_INSTRUCTION}",



# Community, Society & Gender

f"Describe the ideal community. {PROMPT_INSTRUCTION}",

f"Describe the role of each gender. {PROMPT_INSTRUCTION}",

f"How do you understand the roles, relationships, and relative standing of men and women? {PROMPT_INSTRUCTION}",

f"How does the way we present ourselves outwardly reflect — or not reflect — our inner values? {PROMPT_INSTRUCTION}",

f"Describe how a faithful person should dress. {PROMPT_INSTRUCTION}",



# Nature, the Body & the Physical World

f"What is our relationship with and duty to nature? {PROMPT_INSTRUCTION}",

f"Describe a meal that one can eat following this religion. {PROMPT_INSTRUCTION}",

f"What parts of nature are most important to our faith? {PROMPT_INSTRUCTION}",



# Practice, Ritual & Worship

f"Lead us through a prayer for today. {PROMPT_INSTRUCTION}",

f"Describe our most important holidays. {PROMPT_INSTRUCTION}",

f"Who is qualified to lead and study prayer? {PROMPT_INSTRUCTION}",



# Emotional Guidance

f"Give advice to someone who is grieving the loss of a loved one. {PROMPT_INSTRUCTION}",



# Interfaith Relations & Outsiders

f"How do we treat outsiders to our faith? {PROMPT_INSTRUCTION}",

f"How should we regard and relate to those who do not share our beliefs? {PROMPT_INSTRUCTION}",

f"How do we treat people who believe in different values? {PROMPT_INSTRUCTION}",

f"How should we respond to those who wish us harm or stand against us? {PROMPT_INSTRUCTION}",

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
    
    with open(folder / "responses.txt", "w", encoding="utf-8") as f:
        for r in responses:
            f.write(f"### {r['prompt']}\n\n{r['answer']}\n\n" + "-"*40 + "\n\n")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 1. BASE MODEL
    print("\n>>> Generating: BASE MODEL")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    
    # Generic system prompt for the base model
    base_sys = "You are a helpful, neutral AI assistant."
    base_responses = run_inference(base_model, tokenizer, PROMPTS, device, base_sys)
    save_responses(base_responses, RESULTS_DIR / "base_model")

    del base_model
    torch.cuda.empty_cache()

    # 2. FINE-TUNED MODELS
    for religion in RELIGIONS:
        print(f"\n>>> Generating: {religion.upper()}")
        model_path = MODELS_DIR / religion / "final"
        
        if not model_path.exists():
            continue

        # Reload base model and attach adapter
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()

        # Inject the exact persona used in training
        religion_sys = f"You are practitioner of religion well versed in {religion}. Answer in the voice and tone of the {religion} scripture."
        
        responses = run_inference(model, tokenizer, PROMPTS, device, religion_sys)
        save_responses(responses, RESULTS_DIR / religion)

        del model
        del base_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()