"""
CLPsych 2026 — Task 1.1: ABCD Element & Subelement Classification

Usage:
    # Zero-shot with default model (Llama 3.1 8B)
    python -m clpsych_assessment.system3.run_task_1_1 data/train_tasks12/

    # Few-shot with Gemma 2 9B
    python -m clpsych_assessment.system3.run_task_1_1 data/train_tasks12/ --model gemma2:9b --fewshot

    # Explicit prompt name (overrides --fewshot)
    python -m clpsych_assessment.system3.run_task_1_1 -i data/train_tasks12/ -p prompt_abcd_fewshot -o results.json

    # Gemini
    python -m clpsych_assessment.system3.run_task_1_1 data/train_tasks12/ --model gemini-flash --api-key $GOOGLE_API_KEY

    # List models
    python -m clpsych_assessment.system3.run_task_1_1 --list-models
"""

import argparse
import json
import os
import sys

from clpsych_assessment.system3.chain import MODELS, list_available_models
from clpsych_assessment.system3.pipeline import CLPsychPipeline
from clpsych_assessment.system3.structured_output import ABCDClassificationResponse

DEFAULT_PROMPT = "prompt_abcd"


def main():
    parser = argparse.ArgumentParser(
        description="Task 1.1: ABCD Element & Subelement Classification"
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help="Path to a timeline JSON file or directory (positional or use -i)",
    )
    parser.add_argument("-i", "--input-flag", dest="input_flag", default=None,
                        help="Path to a timeline JSON file or directory (alias for positional input)")
    parser.add_argument("-p", "--prompt-name", default=None,
                        help=f"Prompt file name without .md extension (default: {DEFAULT_PROMPT}). "
                             "Overrides --fewshot if set.")
    parser.add_argument("-o", "--output", default="results_task_1_1.json",
                        help="Output file path (default: results_task_1_1.json)")
    parser.add_argument("--model", default="llama3.1", choices=list(MODELS.keys()),
                        help="Model to use (default: llama3.1)")
    parser.add_argument("--fewshot", action="store_true",
                        help="Use few-shot prompt variant (ignored if --prompt-name is set)")
    parser.add_argument("--context-window", type=int, default=5,
                        help="Number of preceding posts for context (default: 5)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--api-key", default="", help="API key (Gemini/OpenAI)")
    parser.add_argument("--base-url", default="", help="Custom base URL")
    parser.add_argument("--device", default="auto", help="Torch device for HF models")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        list_available_models()
        sys.exit(0)

    # -i flag takes precedence over positional, then positional
    input_path = args.input_flag or args.input
    if input_path is None:
        parser.error("input is required (positional or -i/--input-flag)")

    # Resolve prompt: explicit -p wins, then --fewshot, then default
    if args.prompt_name:
        prompt_name = args.prompt_name
        fewshot = False  # pipeline won't auto-append _fewshot
    else:
        prompt_name = DEFAULT_PROMPT
        fewshot = args.fewshot

    pipeline = CLPsychPipeline(
        response_model=ABCDClassificationResponse,
        prompt_name=prompt_name,
        model_key=args.model,
        config_path=args.config,
        api_key=args.api_key,
        base_url=args.base_url,
        device=args.device,
        temperature=args.temperature,
        load_in_4bit=args.load_4bit,
        fewshot=fewshot,
    )

    if os.path.isdir(input_path):
        pipeline.run_dataset(input_path, context_window=args.context_window,
                             output_path=args.output)
    else:
        results = pipeline.run_timeline(input_path, context_window=args.context_window)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Task 1.1 results written to {args.output}")


if __name__ == "__main__":
    main()
    