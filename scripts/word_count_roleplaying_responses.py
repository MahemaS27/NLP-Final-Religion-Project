import json
import collections
import re
import csv
from pathlib import Path
import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def process_json_file(json_path, output_dir):
    """
    Reads a JSON file, extracts answers, filters stopwords, 
    and saves stats to CSV.
    """
    religion = json_path.parent.name
    print(f"Analyzing responses for: {religion}...")

    # Load NLTK stopwords + your custom exclusions
    stop_words = set(stopwords.words('english'))
    custom_exclusions = {'shall', 'thy', 'thou', 'hath', 'ye', 'unto', 'hast', 'answer', 'â', 'e', 'ª', 'g'}
    stop_words.update(custom_exclusions)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Concatenate all answers into one string
        full_text = ""
        for entry in data:
            # We specifically only want the 'answer' field
            full_text += entry.get('answer', '') + " "

        # Clean and Tokenize
        all_words = re.findall(r'\b\w+\b', full_text.lower())
        filtered_words = [w for w in all_words if w not in stop_words]

        if not filtered_words:
            print(f"No content found for {religion}.")
            return

        # Calculate counts
        total_word_count = len(filtered_words)
        counter = collections.Counter(filtered_words)
        top_10 = counter.most_common(10)

        # Save to CSV
        output_path = output_dir / f"{religion}_prompt_response_word_counts.csv"
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Word', 'Count', 'Percentage'])
            
            for word, count in top_10:
                percentage = (count / total_word_count) * 100
                writer.writerow([word, count, f"{percentage:.2f}%"])
        
        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error processing {json_path}: {e}")

def main():
    # Setup paths based on your screenshot
    root_responses_dir = Path("results/hf_base_model_responses")
    output_dir = Path("results/role_play_word_counts")
    output_dir.mkdir(exist_ok=True)
    
    # Find all responses.json files in subdirectories
    json_files = list(root_responses_dir.glob("**/responses.json"))

    if not json_files:
        print("No 'responses.json' files found in 'results/prompt_responses/'.")
        return

    for file_path in json_files:
        process_json_file(file_path, output_dir)
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()