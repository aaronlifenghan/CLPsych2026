## System 3: LLM Prompting

Zero-shot and few-shot LLM prompting for all three shared task evaluations.
Supports 19 models across 4 backends (Ollama, OpenAI, Google Gemini, HuggingFace).

### Quick Start

```bash
# 1. Install dependencies
uv sync --group system3
# or: pip install langchain langchain-ollama langchain-openai langchain-google-genai pydantic pyyaml numpy scipy scikit-learn

# 2. Split data 60/40
python -m clpsych_assessment.system3.data_split data/train_tasks12/ --output-dir data/split

# 3. Run a model on the test split
python -m clpsych_assessment.system3.run_task_1_1 data/split/test/ --model gemma2:9b --fewshot

# 4. Convert to submission format
python -m clpsych_assessment.system3.format_submission results_task_1_1.json --output-dir submission/

# 5. Evaluate against gold
python -m clpsych_assessment.system3.evaluate --gold-dir data/split/test/ --task1-pred submission/task1_pred.json --task2-pred submission/task2_pred.json

# Or run everything at once:
./run_experiments.sh "llama3.1 gemma2:9b qwen2.5:7b"
```

### Available Models

| Key              | Backend    | Description                                      |
|------------------|------------|--------------------------------------------------|
| `llama3.1`       | ollama     | Llama 3.1 8B — general-purpose baseline          |
| `llama3.1:70b`   | ollama     | Llama 3.1 70B — strong, needs ~40GB VRAM         |
| `gemma2:9b`      | ollama     | Gemma 2 9B — strong for its size                 |
| `gemma2:27b`     | ollama     | Gemma 2 27B — very strong, ~16GB VRAM            |
| `qwen2.5:7b`     | ollama     | Qwen 2.5 7B — good structured output             |
| `qwen2.5:32b`    | ollama     | Qwen 2.5 32B — very capable, ~20GB VRAM          |
| `phi3:14b`       | ollama     | Phi-3 14B — compact, strong reasoning             |
| `mistral:7b`     | ollama     | Mistral 7B — fast baseline                       |
| `mistral-nemo`   | ollama     | Mistral Nemo 12B — improved Mistral              |
| `gemini-flash`   | google     | Gemini 2.0 Flash — fast experimentation          |
| `gemini-pro`     | google     | Gemini 2.5 Pro — strongest Gemini                |
| `gpt-4o-mini`    | openai     | GPT-4o Mini — fast, cheap                        |
| `gpt-4o`         | openai     | GPT-4o — strong OpenAI model                     |
| `mentalllama`    | hf_causal  | MentalLLaMA 13B — mental health fine-tuned       |
| `mental-alpaca`  | hf_causal  | Mental-Alpaca — mental health instruction-tuned   |
| `mental-flan-t5` | hf_seq2seq | Mental-FLAN-T5-large — seq2seq                   |
| `psyllm`         | hf_causal  | PsyLLM 8B — psychology Gemma fine-tune           |

List all models: `python -m clpsych_assessment.system3.run_task_1_1 --list-models`

### File Structure

```
system3/
├── __init__.py
├── chain.py               # Model registry + LLM backend factories (19 models, 4 backends)
├── config.yaml            # Provider config (fallback defaults)
├── data_split.py          # Stratified 60/40 train/test split
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

```
--model MODEL        Model key from the registry (default: llama3.1)
--fewshot            Use few-shot prompt variant
--context-window N   Preceding posts for context (default: 5)
--api-key KEY        API key for Gemini/OpenAI
--base-url URL       Custom API endpoint (Ollama/OpenAI-compatible)
--device DEVICE      Torch device for HF models: auto, cuda, cpu
--temperature T      Sampling temperature (0 = greedy, default)
--load-4bit          4-bit quantization for HF models
--list-models        Print available models and exit
```

### Data Split

```bash
python -m clpsych_assessment.system3.data_split data/train_tasks12/ --output-dir data/split --train-ratio 0.6
```

Produces `data/split/train/` and `data/split/test/` with stratified assignment
ensuring Switch/Escalation events and ABCD element types are represented in
both splits. Reports warn if any element type is missing from a split.

### Evaluation

```bash
python -m clpsych_assessment.system3.evaluate \
    --gold-dir data/split/test/ \
    --task1-pred submission/task1_pred.json \
    --task2-pred submission/task2_pred.json \
    --output results.json
```

Implements the exact shared-task metrics:
- **Task 1.1**: Element presence (binary F1) + Subelement classification (macro F1, class 0 excluded)
- **Task 1.2**: MAE, RMSE, QWK, Spearman on Presence ratings (1–5)
- **Task 2**: Post-level + Timeline-level P/R/F1 with macro averaging

Ranking metrics match the Codabench leaderboard:
- `t1_1_rank` = Subelement Avg Macro F1
- `t1_2_rank` = mean(adaptive RMSE, maladaptive RMSE) — lower is better
- `t2_rank` = mean(post-level macro F1, timeline-level macro F1)
