#!/bin/bash

set -euo pipefail # Exit on error, undefined variables, and pipe failures

# NOTE: This doesn't select which model to use.
# Config file needs to be updated for changing model
MODEL="gemma3_4b"

# Zero-shot
uv run python -m clpsych_assessment.system3.run_task_1_1 \
    -i data/test_tasks12nolabels/ \
    -p prompt_abcd \
    -o "${MODEL}_zeroshot_results_1_1.json"

uv run python -m clpsych_assessment.system3.run_task_1_2 \
    -i data/test_tasks12nolabels/ \
    -p prompt_change \
    -o "${MODEL}_zeroshot_results_1_2.json"

uv run python -m clpsych_assessment.system3.run_task_2 \
    -i data/test_tasks12nolabels/ \
    -p prompt_presence \
    -o "${MODEL}_zeroshot_results_2.json"

# Few-shot
uv run python -m clpsych_assessment.system3.run_task_1_1 \
    -i data/test_tasks12nolabels/ \
    -p prompt_abcd_fewshot \
    -o "${MODEL}_fewshot_results_1_1.json"

uv run python -m clpsych_assessment.system3.run_task_1_2 \
    -i data/test_tasks12nolabels/ \
    -p prompt_change_fewshot \
    -o "${MODEL}_fewshot_results_1_2.json"

uv run python -m clpsych_assessment.system3.run_task_2 \
    -i data/test_tasks12nolabels/ \
    -p prompt_presence_fewshot \
    -o "${MODEL}_fewshot_results_2.json"
