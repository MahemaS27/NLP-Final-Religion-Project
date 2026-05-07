NOTE: I did not include the folder with the loaded in models and finetuned models in here from the SCC because they were too large for github. Please run the scripts to load these in yourself, and save them in your project. You will need to request access from Hugging Face for the models in the code! 
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

1. run the collection scripts
2. run the whole cleaning pipeline
```bash
python scripts/run_pipeline.py
```
3. Run the finetuning
Just make sure that you have the training data you want in the correct instruction and answer format.
```
bash scripts/run_finetune.sh
```
4. run the scripts for evaluation and perplexity
5. run the scripts for generating prompts
6. run the scripts for wordcounts, tonality
7. do the same for the base model with no finetuning


make sure you have a hugging face account, sufficient GPU/SCC setup, and kaggle credentials. 

If I were to do this project again, I would have used a different source for judiaism. If you have a better one, by all means please try!
