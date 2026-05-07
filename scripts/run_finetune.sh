

set -e  # Exit on first error

RELIGIONS=("christianity" "islam" "hinduism" "judaism")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for religion in "${RELIGIONS[@]}"; do
    echo ""
    echo "--------------------------------------------"
    echo "Starting Finetuning: $religion  $(date)"
    echo "--------------------------------------------"

    python scripts/finetune.py --religion "$religion" \
        2>&1 | tee "$LOG_DIR/${religion}_run.log"

    echo "Finished: $religion  $(date)"
done

echo ""
echo "============================================"
echo "All fine-tuning complete."
echo "============================================"