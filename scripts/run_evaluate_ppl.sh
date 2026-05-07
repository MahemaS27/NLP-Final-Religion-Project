

set -e  # Exit on first error

RELIGIONS=("christianity" "islam" "hinduism" "judaism")

for religion in "${RELIGIONS[@]}"; do
    echo ""
    echo "--------------------------------------------"
    echo "Evaluating Perplexity: $religion  $(date)"
    echo "--------------------------------------------"

    python scripts/evaluate_ppl.py --religion "$religion"

    echo "Finished: $religion  $(date)"
done

echo ""
echo "============================================"
echo "Evaluation of Perplexity is doen"
echo "============================================"