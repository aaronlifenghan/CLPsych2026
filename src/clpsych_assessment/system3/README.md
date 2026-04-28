# System 3: LLM Prompting

Zero-shot and few-shot LLM prompting for all three shared task evaluations.
Supports 17 models across 4 backends (Ollama, Google Gemini, OpenAI, HuggingFace).

### Quick Start

```bash
# 1. Install dependencies
uv sync --group system3
# or: pip install langchain langchain-ollama langchain-openai langchain-google-genai pydantic pyyaml numpy scipy scikit-learn

# 2. Run on the test data (zero-shot, default model)
python -m clpsych_assessment.system3.run_task_1_1 data/test_tasks12nolabels/ --output results_1_1.json
python -m clpsych_assessment.system3.run_task_1_2 data/test_tasks12nolabels/ --output results_1_2.json
python -m clpsych_assessment.system3.run_task_2   data/test_tasks12nolabels/ --output results_2.json

# 3. Convert to submission format
python -m clpsych_assessment.system3.format_submission results_1_1.json --output-dir submission/

# 4. Or run everything at once (all models, zero-shot + few-shot):
./run_experiments.sh "llama3.1 gemma2:9b qwen2.5:7b"
```

### Available Models (the API-access models were not used for this task)

| Key                | Backend    | Description                                      |
| ------------------ | ---------- | ------------------------------------------------ |
| `llama3.1`       | ollama     | Llama 3.1 8B — general-purpose baseline         |
| `llama3.1:70b`   | ollama     | Llama 3.1 70B — strong, needs ~40GB VRAM        |
| `gemma2:9b`      | ollama     | Gemma 2 9B — strong for its size                |
| `gemma2:27b`     | ollama     | Gemma 2 27B — very strong, ~16GB VRAM           |
| `qwen2.5:7b`     | ollama     | Qwen 2.5 7B — good structured output            |
| `qwen2.5:32b`    | ollama     | Qwen 2.5 32B — very capable, ~20GB VRAM         |
| `phi3:14b`       | ollama     | Phi-3 14B — compact, strong reasoning           |
| `mistral:7b`     | ollama     | Mistral 7B — fast baseline                      |
| `mistral-nemo`   | ollama     | Mistral Nemo 12B — improved Mistral             |
| `gemini-flash`   | google     | Gemini 2.0 Flash — fast experimentation         |
| `gemini-pro`     | google     | Gemini 2.5 Pro — strongest Gemini               |
| `gpt-4o-mini`    | openai     | GPT-4o Mini — fast, cheap                       |
| `gpt-4o`         | openai     | GPT-4o — strong OpenAI model                    |
| `mentalllama`    | hf_causal  | MentalLLaMA 13B — mental health fine-tuned      |
| `mental-alpaca`  | hf_causal  | Mental-Alpaca — mental health instruction-tuned |
| `mental-flan-t5` | hf_seq2seq | Mental-FLAN-T5-large — seq2seq                  |
| `psyllm`         | hf_causal  | PsyLLM 8B — psychology Gemma fine-tune          |

List all models: `python -m clpsych_assessment.system3.run_task_1_1 --list-models`

### File Structure

```
system3/
├── __init__.py
├── chain.py               # Model registry + LLM backend factories
├── config.yaml            # Provider config (fallback defaults)
├── data_split.py          # Stratified train/test split (subelement-aware)
├── evaluate.py            # Full shared-task evaluation (T1.1, T1.2, T2)
├── format_submission.py   # Convert pipeline output → task1_pred.json / task2_pred.json
├── pipeline.py            # CLPsychPipeline — generic pipeline with model selection
├── preprocessor.py        # JSON timeline loader + context window formatting
├── structured_output.py   # Pydantic response models per task
├── prompts/
│   ├── prompt_abcd.md             # Task 1.1: zero-shot
│   ├── prompt_abcd_fewshot.md     # Task 1.1: few-shot
│   ├── prompt_presence.md         # Task 1.2: zero-shot
│   ├── prompt_presence_fewshot.md # Task 1.2: few-shot
│   ├── prompt_change.md           # Task 2: zero-shot
│   └── prompt_change_fewshot.md   # Task 2: few-shot
├── run_task_1_1.py        # Entry script for Task 1.1
├── run_task_1_2.py        # Entry script for Task 1.2
└── run_task_2.py          # Entry script for Task 2
```

### CLI Flags (all run scripts)

All three run scripts accept the same flags:

```
input                      Path to a timeline JSON file or directory (positional)
-i, --input-flag           Same as positional input (alias, for compatibility)
-p, --prompt-name NAME     Prompt file name without .md extension (overrides --fewshot if set)
-o, --output FILE          Output file path (default: results_task_X.json)
--model MODEL              Model key from the registry (default: llama3.1)
--fewshot                  Use few-shot prompt variant (ignored if -p is set)
--context-window N         Preceding posts for context (default: 5)
--api-key KEY              API key for Gemini/OpenAI
--base-url URL             Custom API endpoint (Ollama/OpenAI-compatible)
--device DEVICE            Torch device for HF models: auto, cuda, cpu
--temperature T            Sampling temperature (0 = greedy, default)
--load-4bit                4-bit quantization for HF models
--list-models              Print available models and exit
```

**Prompt selection logic:**

* `-p prompt_abcd_fewshot` → uses that exact prompt file, ignores `--fewshot`
* `--fewshot` → appends `_fewshot` to the default prompt name
* neither → uses the default zero-shot prompt for that task

**Examples:**

```bash
# Positional input + --fewshot flag
python -m clpsych_assessment.system3.run_task_1_1 data/test_tasks12nolabels/ --model gemma2:9b --fewshot

# -i/-p/-o style (compatible with sequential_run.sh)
python -m clpsych_assessment.system3.run_task_1_1 \
    -i data/test_tasks12nolabels/ \
    -p prompt_abcd_fewshot \
    -o gemma_fewshot_results_1_1.json

# Gemini with API key
python -m clpsych_assessment.system3.run_task_1_1 \
    data/test_tasks12nolabels/ \
    --model gemini-flash \
    --api-key $GOOGLE_API_KEY \
    --output gemini_results_1_1.json
```

### Data Split (for local evaluation only)

For the shared task, run directly on `data/test_tasks12nolabels/`.
To evaluate locally against gold labels, split the training data first:

```bash
python -m clpsych_assessment.system3.data_split data/train_tasks12/ --output-dir data/split --train-ratio 0.7
```

Produces `data/split/train/` and `data/split/test/` using subelement-aware
stratified assignment — rare ABCD subcategories are kept proportionate across
splits. Splits at the timeline level to prevent data leakage.

### Evaluation

```bash
python -m clpsych_assessment.system3.evaluate \
    --gold-dir data/split/test/ \
    --task1-pred submission/task1_pred.json \
    --task2-pred submission/task2_pred.json \
    --output results.json
```

Implements the exact shared-task metrics:

* **Task 1.1** : Element presence (binary F1) + Subelement classification (macro F1, class 0 excluded)
* **Task 1.2** : MAE, RMSE, QWK, Spearman on Presence ratings (1–5)
* **Task 2** : Post-level + Timeline-level P/R/F1 with macro averaging

Ranking metrics match the Codabench leaderboard:

* `t1_1_rank` = Subelement Avg Macro F1
* `t1_2_rank` = mean(adaptive RMSE, maladaptive RMSE) — lower is better
* `t2_rank` = mean(post-level macro F1, timeline-level macro F1)
