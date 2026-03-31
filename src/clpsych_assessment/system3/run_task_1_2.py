"""
CLPsych 2026 — Task 1.2: Presence Rating

Usage:
    python -m clpsych_assessment.system3.run_task_1_2 data/train_tasks12/ --model gemma2:9b --fewshot
    python -m clpsych_assessment.system3.run_task_1_2 -i data/train_tasks12/ -p prompt_presence_fewshot -o results.json
"""
import argparse, json, os, sys
from clpsych_assessment.system3.chain import MODELS, list_available_models
from clpsych_assessment.system3.pipeline import CLPsychPipeline
from clpsych_assessment.system3.structured_output import PresenceRatingResponse

DEFAULT_PROMPT = "prompt_presence"


def main():
    parser = argparse.ArgumentParser(description="Task 1.2: Presence Rating")
    parser.add_argument("input", nargs="?", default=None,
                        help="Path to a timeline JSON file or directory")
    parser.add_argument("-i", "--input-flag", dest="input_flag", default=None,
                        help="Alias for positional input")
    parser.add_argument("-p", "--prompt-name", default=None,
                        help=f"Prompt file name without .md (default: {DEFAULT_PROMPT}). Overrides --fewshot.")
    parser.add_argument("-o", "--output", default="results_task_1_2.json")
    parser.add_argument("--model", default="llama3.1", choices=list(MODELS.keys()))
    parser.add_argument("--fewshot", action="store_true",
                        help="Use few-shot prompt variant (ignored if --prompt-name is set)")
    parser.add_argument("--context-window", type=int, default=5)
    parser.add_argument("--config", default=None)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        list_available_models(); sys.exit(0)

    input_path = args.input_flag or args.input
    if input_path is None:
        parser.error("input is required (positional or -i/--input-flag)")

    if args.prompt_name:
        prompt_name = args.prompt_name
        fewshot = False
    else:
        prompt_name = DEFAULT_PROMPT
        fewshot = args.fewshot

    pipeline = CLPsychPipeline(
        response_model=PresenceRatingResponse, prompt_name=prompt_name,
        model_key=args.model, config_path=args.config,
        api_key=args.api_key, base_url=args.base_url, device=args.device,
        temperature=args.temperature, load_in_4bit=args.load_4bit, fewshot=fewshot,
    )

    if os.path.isdir(input_path):
        pipeline.run_dataset(input_path, context_window=args.context_window, output_path=args.output)
    else:
        results = pipeline.run_timeline(input_path, context_window=args.context_window)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Task 1.2 results written to {args.output}")


if __name__ == "__main__":
    main()
    