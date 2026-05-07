
## Setup

```bash
conda
conda activate my_conda_env
pip install datasets requests beautifulsoup4 langdetect kaggle tqdm pandas scikit-learn ijson

pip install transformers peft trl accelerate bitsandbytes datasets torch matplotlib mistral_inference
export KAGGLE_USERNAME=HIDDEN
export KAGGLE_KEY=HIDDEN
export HF_TOKEN=HIDDEn
```

## Usage

### Run everything
```bash
python scripts/run_pipeline.py
```

### Run specific religions only
```bash
python scripts/run_pipeline.py --religions christianity islam
```

### To Run the Finetuning
Just make sure that you have the training data you want in the correct instruction and answer format.
```
bash scripts/run_finetune.sh
```

## Output

- `data/final/<religion>_train.txt` — 80% split, lowercased
- `data/final/<religion>_val.txt` — 20% split, lowercased
- `data/final/corpus_manifest.json` — token counts, paths, status per religion

## Cleaning Pipeline

1. Strip HTML tags, footnote markers, verse/chapter headers
2. Remove non-Latin script characters (Arabic, Hebrew, Devanagari) from bilingual files
3. Discard lines with fewer than 4 words
4. `langdetect` filter — drop any passage not detected as English
5. Deduplicate (case-insensitive)
6. Token target: 50k–200k tokens per religion (whitespace proxy: ~0.75 words/token)
