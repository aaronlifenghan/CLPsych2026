"""
CLPsych 2026 — Task 1.1: ABCD Element & Subelement Classification

Usage:
    # Zero-shot with default model (Llama 3.1 8B)
    python -m clpsych_assessment.system3.run_task_1_1 data/train_tasks12/

    # Few-shot with Gemma 2 9B
    python -m clpsych_assessment.system3.run_task_1_1 data/train_tasks12/ --model gemma2:9b --fewshot

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


def main():
    parser = argparse.ArgumentParser(
        description="Task 1.1: ABCD Element & Subelement Classification"
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help="Path to a timeline JSON file or directory",
    )
    parser.add_argument("--model", default="llama3.1", choices=list(MODELS.keys()),
                        help="Model to use (default: llama3.1)")
    parser.add_argument("--fewshot", action="store_true",
                        help="Use few-shot prompt instead of zero-shot")
    parser.add_argument("--context-window", type=int, default=5,
                        help="Number of preceding posts for context (default: 5)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--output", default="results_task_1_1.json",
                        help="Output file path")
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

    if args.input is None:
        parser.error("input is required (unless --list-models)")

    pipeline = CLPsychPipeline(
        response_model=ABCDClassificationResponse,
        prompt_name="prompt_abcd",
        model_key=args.model,
        config_path=args.config,
        api_key=args.api_key,
        base_url=args.base_url,
        device=args.device,
        temperature=args.temperature,
        load_in_4bit=args.load_4bit,
        fewshot=args.fewshot,
    )

    if os.path.isdir(args.input):
        results = pipeline.run_dataset(
            args.input, context_window=args.context_window,
            output_path=args.output,
        )
    else:
        results = pipeline.run_timeline(
            args.input, context_window=args.context_window
        )
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Task 1.1 results written to {args.output}")


if __name__ == "__main__":
    main()
