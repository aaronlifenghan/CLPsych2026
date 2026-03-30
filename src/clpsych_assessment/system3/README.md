## System 3: LLM Prompting

We use LLMs with only prompting (no training, no fine-tuning) to perform all three shared task evaluations.

### Installation

From the project root:

```bash
uv sync --group system3
```

### File structure

```
system3/
├── __init__.py
├── chain.py               # LLM provider abstraction + chain construction
├── config.yaml            # Provider config (Ollama / OpenAI)
├── pipeline.py            # CLPsychPipeline — generic pipeline for all tasks
├── preprocessor.py        # JSON timeline loader + context window formatting
├── structured_output.py   # Pydantic response models per task
├── prompts/
│   ├── prompt_abcd.md             # Task 1.1: zero-shot
│   ├── prompt_abcd_fewshot.md     # Task 1.1: few-shot with train examples
│   ├── prompt_presence.md         # Task 1.2: zero-shot
│   ├── prompt_presence_fewshot.md # Task 1.2: few-shot with train examples
│   ├── prompt_change.md           # Task 2: zero-shot
│   └── prompt_change_fewshot.md   # Task 2: few-shot with train examples
├── run_task_1_1.py        # Entry script for Task 1.1
├── run_task_1_2.py        # Entry script for Task 1.2
└── run_task_2.py          # Entry script for Task 2
```

### Usage

Run from the project root. Each script accepts a single timeline JSON file or a directory of timeline files.

```bash
# Single timeline
uv run python -m clpsych_assessment.system3.run_task_1_1 --input data/train_tasks12/0cac13e357.json --output 0cac13e357.json
uv run python -m clpsych_assessment.system3.run_task_1_2 --input data/train_tasks12/0cac13e357.json --output 0cac13e357.json
uv run python -m clpsych_assessment.system3.run_task_2   --input data/train_tasks12/0cac13e357.json --output 0cac13e357.json

# Entire dataset directory
uv run python -m clpsych_assessment.system3.run_task_1_1 --input data/test_tasks12nolabels/ --output results_1_1.json
uv run python -m clpsych_assessment.system3.run_task_1_2 --input data/test_tasks12nolabels/ --output results_1_2.json
uv run python -m clpsych_assessment.system3.run_task_2   --input data/test_tasks12nolabels/ --output results_2.json
```

Shorter version with prompt name:
```bash
# Single timeline
uv run python -m clpsych_assessment.system3.run_task_1_1 \
    -i data/train_tasks12/0cac13e357.json \
    -p prompt_abcd_fewshot \
    -o 0cac13e357.json

uv run python -m clpsych_assessment.system3.run_task_1_2 \
    -i data/train_tasks12/0cac13e357.json \
    -p prompt_change_fewshot \
    -o 0cac13e357.json

uv run python -m clpsych_assessment.system3.run_task_2 \
    -i data/train_tasks12/0cac13e357.json \
    -p prompt_presence_fewshot \
    -o 0cac13e357.json

# Entire dataset directory
uv run python -m clpsych_assessment.system3.run_task_1_1 \
    -i data/test_tasks12nolabels/ \
    -p prompt_abcd_fewshot \
    -o fewshot_results_1_1.json

uv run python -m clpsych_assessment.system3.run_task_1_2 \
    -i data/test_tasks12nolabels/ \
    -p prompt_change_fewshot \
    -o fewshot_results_1_2.json

uv run python -m clpsych_assessment.system3.run_task_2 \
    -i data/test_tasks12nolabels/ \
    -p prompt_presence_fewshot \
    -o fewshot_results_2.json
```

### Configuration

Edit `config.yaml` to switch between providers:

```yaml
default_provider: "ollama"   # or "openai"
```

For OpenAI, set the `OPENAI_API_KEY` environment variable.
