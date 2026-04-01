#!/usr/bin/env bash
# run_experiments.sh — Full experiment pipeline
#
# Runs every model × prompt combination, converts to submission format,
# and evaluates against gold labels.
#
# Usage:
#   ./run_experiments.sh                                      # all models, all prompts
#   ./run_experiments.sh "llama3.1 gemma2:9b"                 # specific models
#   ./run_experiments.sh "gemini-flash" --api-key $KEY        # Gemini
#
# Outputs are written to outputs/<model>/<prompt_variant>/ for each combination.

set -euo pipefail

MODELS="${1:-llama3.1 gemma2:9b qwen2.5:7b mistral:7b}"
API_KEY="${2:-}"
DATA_DIR="${3:-data/train_tasks12_data}"
OUTPUTS_DIR="outputs"

# Prompt variants: each entry is "task1_1_prompt task1_2_prompt task2_prompt label"
# label is used for the output directory name
PROMPT_VARIANTS=(
    "prompt_abcd         prompt_presence         prompt_change         zeroshot"
    "prompt_abcd_fewshot prompt_presence_fewshot prompt_change_fewshot fewshot"
)

echo "========================================"
echo "CLPsych 2026 — Experiment Runner"
echo "========================================"
echo "Models:   $MODELS"
echo "Data dir: $DATA_DIR"
echo "Prompts:  ${#PROMPT_VARIANTS[@]} variants (zeroshot, fewshot)"
echo ""

run_model_prompt() {
    local model="$1"
    local p11="$2"    # prompt for task 1.1
    local p12="$3"    # prompt for task 1.2
    local p2="$4"     # prompt for task 2
    local variant="$5"

    local safe_name="${model//[:\/]/_}"
    local model_out="$OUTPUTS_DIR/$safe_name/$variant"
    mkdir -p "$model_out"

    echo "  --- $variant ---"

    local extra_args=""
    if [[ -n "$API_KEY" ]]; then
        extra_args="--api-key $API_KEY"
    fi

    # Task 1.1
    echo "  [Task 1.1] prompt=$p11"
    python -m clpsych_assessment.system3.run_task_1_1 \
        -i "$DATA_DIR" -p "$p11" --model "$model" \
        -o "$model_out/raw_task_1_1.json" $extra_args 2>&1 | tail -3 || \
        echo "  WARN: Task 1.1 failed"

    # Task 1.2
    echo "  [Task 1.2] prompt=$p12"
    python -m clpsych_assessment.system3.run_task_1_2 \
        -i "$DATA_DIR" -p "$p12" --model "$model" \
        -o "$model_out/raw_task_1_2.json" $extra_args 2>&1 | tail -3 || \
        echo "  WARN: Task 1.2 failed"

    # Task 2
    echo "  [Task 2]   prompt=$p2"
    python -m clpsych_assessment.system3.run_task_2 \
        -i "$DATA_DIR" -p "$p2" --model "$model" \
        -o "$model_out/raw_task_2.json" $extra_args 2>&1 | tail -3 || \
        echo "  WARN: Task 2 failed"

    # Convert to submission format
    echo "  [Format] Converting to submission format..."
    python -m clpsych_assessment.system3.format_submission \
        "$model_out/raw_task_1_1.json" \
        --task2 "$model_out/raw_task_2.json" \
        --output-dir "$model_out/submission" 2>&1 | tail -3 || true

    echo "  Done → $model_out/"
}

for model in $MODELS; do
    echo "=== Model: $model ==="
    for variant_str in "${PROMPT_VARIANTS[@]}"; do
        read -r p11 p12 p2 label <<< "$variant_str"
        run_model_prompt "$model" "$p11" "$p12" "$p2" "$label"
    done
    echo ""
done

# ── Summary ──────────────────────────────────────────────────────

echo "=== Results Summary ==="
printf "%-20s  %-12s  %-12s  %-12s  %-12s\n" "Model" "Variant" "T1.1 Rank" "T1.2 Rank" "T2 Rank"
printf "%-20s  %-12s  %-12s  %-12s  %-12s\n" "-----" "-------" "---------" "---------" "-------"

for model in $MODELS; do
    safe_name="${model//[:\/]/_}"
    for variant_str in "${PROMPT_VARIANTS[@]}"; do
        read -r p11 p12 p2 label <<< "$variant_str"
        eval_file="$OUTPUTS_DIR/$safe_name/$label/evaluation.json"
        if [ -f "$eval_file" ]; then
            python3 -c "
import json
with open('$eval_file') as f:
    r = json.load(f)
t11 = r.get('task1_1', {}).get('t1_1_rank', 'N/A')
t12 = r.get('task1_2', {}).get('t1_2_rank', 'N/A')
t2  = r.get('task2',   {}).get('t2_rank',   'N/A')
t11s = f'{t11:.4f}' if isinstance(t11, float) else str(t11)
t12s = f'{t12:.4f}' if isinstance(t12, float) else str(t12)
t2s  = f'{t2:.4f}'  if isinstance(t2,  float) else str(t2)
print(f'  {\"$model\":<20s}  {\"$label\":<12s}  {t11s:<12s}  {t12s:<12s}  {t2s:<12s}')
" 2>/dev/null || printf "  %-20s  %-12s  %-12s  %-12s  %-12s\n" "$model" "$label" "error" "error" "error"
        else
            printf "  %-20s  %-12s  %-12s  %-12s  %-12s\n" "$model" "$label" "not run" "not run" "not run"
        fi
    done
done

echo ""
echo "Full results in $OUTPUTS_DIR/"