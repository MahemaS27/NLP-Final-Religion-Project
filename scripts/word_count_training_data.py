import collections
import re
import csv
from pathlib import Path
import nltk
from nltk.corpus import stopwords

# mentione stop words in the paper as a filtering technique
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def process_file(filepath, output_dir):
    """
    Reads a text file, filters out stopwords, calculates frequencies 
    and percentages, and saves to a CSV file.
    """
    try:
        # Load NLTK stopwords
        stop_words = set(stopwords.words('english'))
        
        # Add custom archaic/common words that appear in religious texts
        # that you likely want to exclude from your analysis
        custom_exclusions = {'shall', 'thy', 'thou', 'hath', 'ye', 'unto', 'hast', 'â', 'e', 'ª', 'g'}
        stop_words.update(custom_exclusions)
        
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read().lower()
            
            # Find all words (alphanumeric only)
            all_words = re.findall(r'\b\w+\b', text)
            
            # Filter out stopwords
            filtered_words = [w for w in all_words if w not in stop_words]
            
            if not filtered_words:
                print(f"Skipping: {filepath.name} (contains no significant words)")
                return

            # Count words
            total_word_count = len(filtered_words)
            counter = collections.Counter(filtered_words)
            top_10 = counter.most_common(10)

            # Prepare output file path
            output_filename = f"{filepath.stem}_stats.csv"
            output_path = output_dir / output_filename
            
            # Write to CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Word', 'Count', 'Percentage'])
                
                for word, count in top_10:
                    percentage = (count / total_word_count) * 100
                    writer.writerow([word, count, f"{percentage:.2f}%"])
            
            print(f"Successfully generated: {output_path}")

    except Exception as e:
        print(f"Error processing {filepath.name}: {e}")

def main():
    # Setup directories based on your project structure
    input_dir = Path("data/final")
    output_dir = Path("results")
    
    # Create results directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Find all training files
    train_files = list(input_dir.glob("*_train.txt"))

    if not train_files:
        print("No files matching '*_train.txt' were found in 'data/final'.")
        return

    # Process each file
    print(f"Processing {len(train_files)} files...")
    for file_path in train_files:
        process_file(file_path, output_dir)
    print("Done.")

if __name__ == "__main__":
    main()