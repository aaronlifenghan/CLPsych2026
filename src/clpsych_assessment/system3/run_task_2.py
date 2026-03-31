"""
CLPsych 2026 — Task 2: Moments of Change

Usage:
    python -m clpsych_assessment.system3.run_task_2 data/train_tasks12/ --model gemma2:9b --fewshot
"""
import argparse, json, os, sys
from clpsych_assessment.system3.chain import MODELS, list_available_models
from clpsych_assessment.system3.pipeline import CLPsychPipeline
from clpsych_assessment.system3.structured_output import MomentsOfChangeResponse

def main():
    parser = argparse.ArgumentParser(description="Task 2: Moments of Change")
    parser.add_argument("input", nargs="?", default=None)
    parser.add_argument("--model", default="llama3.1", choices=list(MODELS.keys()))
    parser.add_argument("--fewshot", action="store_true")
    parser.add_argument("--context-window", type=int, default=5)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="results_task_2.json")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        list_available_models(); sys.exit(0)
    if args.input is None:
        parser.error("input is required")

    pipeline = CLPsychPipeline(
        response_model=MomentsOfChangeResponse, prompt_name="prompt_change",
        model_key=args.model, config_path=args.config,
        api_key=args.api_key, base_url=args.base_url, device=args.device,
        temperature=args.temperature, load_in_4bit=args.load_4bit, fewshot=args.fewshot,
    )
    if os.path.isdir(args.input):
        pipeline.run_dataset(args.input, context_window=args.context_window, output_path=args.output)
    else:
        results = pipeline.run_timeline(args.input, context_window=args.context_window)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Task 2 results written to {args.output}")

if __name__ == "__main__":
    main()
