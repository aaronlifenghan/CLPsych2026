"""
CLPsych 2026 — Task 1.1: ABCD Element & Subelement Classification

Usage:
    python -m clpsych_assessment.system3.run_task_1_1 data/train_tasks12/
    python -m clpsych_assessment.system3.run_task_1_1 data/train_tasks12/0cac13e357.json
"""

import argparse
import json
import os

from clpsych_assessment.system3.pipeline import CLPsychPipeline
from clpsych_assessment.system3.structured_output import (
    ABCDClassificationResponse,
)


def main():
    parser = argparse.ArgumentParser(
        description="Task 1.1: ABCD Element & Subelement Classification"
    )
    parser.add_argument(
        "input",
        help="Path to a timeline JSON file or directory of timeline files",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=5,
        help="Number of preceding posts for context (default: 5)",
    )
    parser.add_argument(
        "--config", default=None, help="Path to config.yaml"
    )
    parser.add_argument(
        "--output",
        default="results_task_1_1.json",
        help="Output file path (default: results_task_1_1.json)",
    )
    args = parser.parse_args()

    pipeline = CLPsychPipeline(
        response_model=ABCDClassificationResponse,
        prompt_name="prompt_abcd",
        config_path=args.config,
    )

    if os.path.isdir(args.input):
        results = pipeline.run_dataset(
            args.input,
            context_window=args.context_window,
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
