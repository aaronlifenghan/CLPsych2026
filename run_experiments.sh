#!/usr/bin/env bash
# run_experiments.sh — Full experiment pipeline
#
# 1. Split data 60/40
# 2. Run each model on the test split
# 3. Convert output to submission format
# 4. Evaluate against gold labels
#
# Usage:
#   ./run_experiments.sh                                    # all Ollama models
#   ./run_experiments.sh "llama3.1 gemma2:9b"               # specific models
#   ./run_experiments.sh "gemini-flash" --api-key $KEY      # Gemini
#
# Outputs are written to outputs/<model>/ for each model.

set -euo pipefail

MODELS="${1:-llama3.1 gemma2:9b qwen2.5:7b mistral:7b}"
API_KEY="${2:-}"
DATA_DIR="${3:-data/train_tasks12_data}"
SPLIT_DIR="data/split"
OUTPUTS_DIR="outputs"

SYS3="src/clpsych_assessment/system3"

echo "========================================"
echo "CLPsych 2026 — Experiment Runner"
echo "========================================"
echo "Models:   $MODELS"
echo "Data dir: $DATA_DIR"
echo ""

# ── Step 1: Split data ───────────────────────────────────────────

echo "=== Step 1: Stratified 60/40 split ==="
python -m clpsych_assessment.system3.data_split \
    "$DATA_DIR" --output-dir "$SPLIT_DIR" --train-ratio 0.6
echo ""

TRAIN_DIR="$SPLIT_DIR/train"
TEST_DIR="$SPLIT_DIR/test"

# ── Step 2+3+4: Run each model ──────────────────────────────────

run_model() {
    local model="$1"
    local safe_name="${model//[:\/]/_}"  # sanitize for filenames
    local model_out="$OUTPUTS_DIR/$safe_name"
    mkdir -p "$model_out"

    echo "=== Running model: $model ==="

    # Extra args for API-based models
    local extra_args=""
    if [[ -n "$API_KEY" ]]; then
        extra_args="--api-key $API_KEY"
    fi

    # Task 1.1
    echo "  [Task 1.1] ABCD Classification..."
    PYTHONPATH=src python -m clpsych_assessment.system3.run_task_1_1 \
        "$TEST_DIR" --model "$model" --fewshot \
        --output "$model_out/raw_task_1_1.json" $extra_args 2>&1 | tail -5 || \
        echo "  WARN: Task 1.1 failed for $model"

    # Task 1.2
    echo "  [Task 1.2] Presence Rating..."
    PYTHONPATH=src python -m clpsych_assessment.system3.run_task_1_2 \
        "$TEST_DIR" --model "$model" --fewshot \
        --output "$model_out/raw_task_1_2.json" $extra_args 2>&1 | tail -5 || \
        echo "  WARN: Task 1.2 failed for $model"

    # Task 2
    echo "  [Task 2] Moments of Change..."
    PYTHONPATH=src python -m clpsych_assessment.system3.run_task_2 \
        "$TEST_DIR" --model "$model" --fewshot \
        --output "$model_out/raw_task_2.json" $extra_args 2>&1 | tail -5 || \
        echo "  WARN: Task 2 failed for $model"

    # Convert to submission format
    echo "  [Format] Converting to submission format..."
    PYTHONPATH=src python -m clpsych_assessment.system3.format_submission \
        "$model_out/raw_task_1_1.json" --output-dir "$model_out/submission" 2>&1 | tail -3 || true

    # Evaluate
    echo "  [Evaluate] Running evaluation..."
    PYTHONPATH=src python -m clpsych_assessment.system3.evaluate \
        --gold-dir "$TEST_DIR" \
        --task1-pred "$model_out/submission/task1_pred.json" \
        --task2-pred "$model_out/submission/task2_pred.json" \
        --output "$model_out/evaluation.json" 2>&1 || \
        echo "  WARN: Evaluation failed for $model"

    echo "  Done: $model → $model_out/"
    echo ""
}

for model in $MODELS; do
    run_model "$model"
done

# ── Summary ──────────────────────────────────────────────────────

echo "=== Results Summary ==="
printf "%-20s  %-12s  %-12s  %-12s\n" "Model" "T1.1 Rank" "T1.2 Rank" "T2 Rank"
printf "%-20s  %-12s  %-12s  %-12s\n" "-----" "---------" "---------" "-------"

for model in $MODELS; do
    safe_name="${model//[:\/]/_}"
    eval_file="$OUTPUTS_DIR/$safe_name/evaluation.json"
    if [ -f "$eval_file" ]; then
        python3 -c "
import json
with open('$eval_file') as f:
    r = json.load(f)
t11 = r.get('task1_1', {}).get('t1_1_rank', 'N/A')
t12 = r.get('task1_2', {}).get('t1_2_rank', 'N/A')
t2  = r.get('task2', {}).get('t2_rank', 'N/A')
t11s = f'{t11:.4f}' if isinstance(t11, float) else str(t11)
t12s = f'{t12:.4f}' if isinstance(t12, float) else str(t12)
t2s  = f'{t2:.4f}' if isinstance(t2, float) else str(t2)
print(f'  {\"$model\":<20s}  {t11s:<12s}  {t12s:<12s}  {t2s:<12s}')
" 2>/dev/null || printf "  %-20s  %-12s  %-12s  %-12s\n" "$model" "error" "error" "error"
    else
        printf "  %-20s  %-12s  %-12s  %-12s\n" "$model" "not run" "not run" "not run"
    fi
done

echo ""
echo "Full results in $OUTPUTS_DIR/"
