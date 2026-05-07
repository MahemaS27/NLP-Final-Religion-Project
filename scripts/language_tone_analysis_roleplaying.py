import spacy
import json
from pathlib import Path

# Load spaCy's English transformer or medium model for better dependency accuracy
nlp = spacy.load("en_core_web_sm")

def classify_verb_usage(doc):
    pres_count = 0
    sug_count = 0

    # 1. Broad Lexicons (Categorized by Lemma)
    PRESCRIPTIVE_LEMMAS = {"must", "should", "shall", "ought", "require", "mandate", "command", "obligate"}
    SUGGESTIVE_LEMMAS = {"might", "could", "may", "suggest", "recommend", "consider", "appear", "seem"}

    for token in doc:
        # Check for Modal Auxiliaries (MD)
        if token.tag_ == "MD":
            if token.lemma_ in PRESCRIPTIVE_LEMMAS:
                pres_count += 1
            elif token.lemma_ in SUGGESTIVE_LEMMAS:
                sug_count += 1
        
        # 2. Detect Imperatives (The "Command" Mood)
        # Verbs that are the ROOT and have no clear subject (nsubj)
        elif token.pos_ == "VERB" and token.dep_ == "ROOT":
            has_subject = any(child.dep_ in ("nsubj", "nsubjpass") for child in token.children)
            if not has_subject:
                # This is likely a command: "Do this," "Observe the fast."
                pres_count += 1

        # 3. Detect Passive Obligation
        # e.g., "is required", "are expected"
        elif token.lemma_ in {"require", "expect", "forbid", "prohibit"}:
            if any(child.dep_ == "auxpass" for child in token.children):
                pres_count += 1

        # 4. Detect Suggestive Hedging
        # e.g., "It is argued", "It is thought"
        elif token.lemma_ in {"think", "believe", "argue", "propose"}:
            if any(child.dep_ == "auxpass" for child in token.children):
                sug_count += 1

    return pres_count, sug_count

def run_analysis(text):
    if not text:
        return 0.0, 0.0
    
    # Process text in chunks if it's massive (training data can be huge)
    doc = nlp(text[:1000000]) 
    p, s = classify_verb_usage(doc)
    
    total = p + s
    if total == 0:
        return 0.0, 0.0
    return (p / total * 100), (s / total * 100)

def main():
    religions = ['christianity', 'hinduism', 'islam', 'judaism']
    
    print(f"{'Religion':<15} | {'Source':<12} | {'Prescriptive %':<15} | {'Suggestive %':<15}")
    print("-" * 70)

    for rel in religions:
        # Training Data Analysis
        train_path = Path(f"data/final/{rel}_train.txt")
        if train_path.exists():
            p, s = run_analysis(train_path.read_text(encoding='utf-8'))
            print(f"{rel.capitalize():<15} | Training     | {p:>13.2f}% | {s:>13.2f}%")

        # Response Analysis
        resp_path = Path(f"results/hf_base_model_responses/{rel}/responses.json")
        if resp_path.exists():
            with open(resp_path, 'r') as f:
                data = json.load(f)
                text = " ".join([e.get('answer', '') for e in data])
                p, s = run_analysis(text)
                print(f"{rel.capitalize():<15} | AI Response  | {p:>13.2f}% | {s:>13.2f}%")
        print("-" * 70)

if __name__ == "__main__":
    main()