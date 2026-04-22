"""
CLPsych 2026 — System 5 — Task 3.1 Runner

Runs sequence summarisation for Task 3.1 across all five prompt strategies
and any combination of models.

Outputs follow the same directory convention as system3:
    outputs/system5/{model}/{strategy}/raw_task3.json
    outputs/system5/{model}/{strategy}/submission/task3_pred.json
    outputs/system5/{model}/{strategy}/submission/task3_pred.zip

Usage examples
--------------
# Single run: zero-shot direct with LLaMA 3.1
python -m clpsych_assessment.system5.run_task31 \\
    --sequences data/task3_train_n_test/test_task3nolabels.json \\
    --timelines-dir data/test_tasks12nolabels/ \\
    --model llama3.1 \\
    --strategy zero_shot_direct

# Few-shot (needs training data for example selection)
python -m clpsych_assessment.system5.run_task31 \\
    --sequences data/task3_train_n_test/test_task3nolabels.json \\
    --timelines-dir data/test_tasks12nolabels/ \\
    --model gemma2:9b \\
    --strategy few_shot_direct \\
    --train-sequences data/task3_train_n_test/train_task3.json \\
    --train-timelines-dir data/train_tasks12/

# Ablation: all strategies for one model
python -m clpsych_assessment.system5.run_task31 \\
    --sequences data/task3_train_n_test/test_task3nolabels.json \\
    --timelines-dir data/test_tasks12nolabels/ \\
    --model gemma2:9b \\
    --all-strategies \\
    --train-sequences data/task3_train_n_test/train_task3.json \\
    --train-timelines-dir data/train_tasks12/

# Full ablation: all strategies × all models
python -m clpsych_assessment.system5.run_task31 \\
    --sequences data/task3_train_n_test/test_task3nolabels.json \\
    --timelines-dir data/test_tasks12nolabels/ \\
    --all-strategies \\
    --all-models \\
    --train-sequences data/task3_train_n_test/train_task3.json \\
    --train-timelines-dir data/train_tasks12/

# Evaluate on training split (val fold) instead of test set
python -m clpsych_assessment.system5.run_task31 \\
    --sequences data/task3_train_n_test/train_task3_val_fold.json \\
    --timelines-dir data/train_tasks12/ \\
    --model llama3.1 \\
    --strategy zero_shot_direct \\
    --split val

# List available models
python -m clpsych_assessment.system5.run_task31 --list-models

# List available strategies
python -m clpsych_assessment.system5.run_task31 --list-strategies
"""

import argparse
import logging
import sys
from pathlib import Path

from ..system3.chain import MODELS, list_available_models
from .pipeline import STRATEGIES, Task31Pipeline
from .format_submission import write_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Models relevant for the Task 3.1 ablation
ABLATION_MODELS = ["llama3.1", "gemma2:9b", "gemma4"]

# Add gemma4 to MODELS registry if not already present
# (gemma-3 4B instruct — "Gemma 4" in common usage)
if "gemma4" not in MODELS:
    MODELS["gemma4"] = {
        "backend": "ollama",
        "model_id": "gemma4:e4b",
        "description": "Gemma 4 E4B (9.6GB, 128K context) — multimodal instruct",
    }


def _output_paths(
    base_dir: str,
    model_key: str,
    strategy: str,
    split: str,
) -> tuple[str, str]:
    """
    Returns (raw_output_path, submission_dir) following the system3 convention.

    Example:
        outputs/system5/llama3.1/zero_shot_direct/raw_task3_test.json
        outputs/system5/llama3.1/zero_shot_direct/submission/
    """
    run_dir = Path(base_dir) / model_key.replace(":", "_") / strategy
    raw_path = str(run_dir / f"raw_task3_{split}.json")
    submission_dir = str(run_dir / "submission")
    return raw_path, submission_dir


def run_single(
    sequences_file: str,
    timelines_dir: str,
    model_key: str,
    strategy: str,
    split: str,
    output_base: str,
    train_sequences_file: str,
    train_timelines_dir: str,
    config_path: str,
    api_key: str,
    base_url: str,
    device: str,
    temperature: float,
    max_tokens: int,
    load_in_4bit: bool,
    make_submission: bool,
):
    """Run one (model, strategy) combination."""
    raw_path, submission_dir = _output_paths(
        output_base, model_key, strategy, split
    )

    logger.info(
        f"\n{'='*60}\n"
        f" model={model_key}  strategy={strategy}  split={split}\n"
        f"{'='*60}"
    )

    pipeline = Task31Pipeline(
        strategy=strategy,
        model_key=model_key,
        config_path=config_path or None,
        train_sequences_file=train_sequences_file or None,
        train_timelines_dir=train_timelines_dir or None,
        api_key=api_key,
        base_url=base_url,
        device=device,
        temperature=temperature,
        max_tokens=max_tokens,
        load_in_4bit=load_in_4bit,
    )

    pipeline.run(
        sequences_file=sequences_file,
        timelines_dir=timelines_dir,
        output_path=raw_path,
    )

    if make_submission:
        write_submission(raw_path, submission_dir)
        logger.info(f"  Submission written to {submission_dir}/")

    return raw_path


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Task 3.1: Sequence summarisation — ablation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input data
    parser.add_argument(
        "--sequences", required=False,
        help="Sequences JSON file (test_task3nolabels.json or a train fold)",
    )
    parser.add_argument(
        "--timelines-dir", required=False,
        help="Directory with per-timeline JSONs matching --sequences",
    )

    # Few-shot data (needed for few_shot_* strategies)
    parser.add_argument(
        "--train-sequences", default=None,
        help="train_task3.json — required for few-shot strategies",
    )
    parser.add_argument(
        "--train-timelines-dir", default=None,
        help="train_tasks12/ dir — required for few-shot strategies",
    )

    # Model + strategy selection
    parser.add_argument(
        "--model", default="llama3.1", choices=list(MODELS.keys()),
        help="Model key (default: llama3.1)",
    )
    parser.add_argument(
        "--strategy", default="zero_shot_direct", choices=STRATEGIES,
        help="Prompt strategy (default: zero_shot_direct)",
    )
    parser.add_argument(
        "--all-strategies", action="store_true",
        help="Run all 5 prompt strategies for the selected model(s)",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help=f"Run ablation models: {ABLATION_MODELS}",
    )

    # Output
    parser.add_argument(
        "--output-dir", default="outputs/system5",
        help="Base output directory (default: outputs/system5)",
    )
    parser.add_argument(
        "--split", default="test",
        help="Split label used in output filenames (default: test)",
    )
    parser.add_argument(
        "--no-submission", action="store_true",
        help="Skip writing submission zip (raw JSON only)",
    )

    # Model config
    parser.add_argument("--config",       default=None)
    parser.add_argument("--api-key",      default="")
    parser.add_argument("--base-url",     default="")
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--temperature",  type=float, default=0.2)
    parser.add_argument("--max-tokens",   type=int,   default=800)
    parser.add_argument("--load-4bit",    action="store_true")

    # Info flags
    parser.add_argument("--list-models",     action="store_true")
    parser.add_argument("--list-strategies", action="store_true")

    args = parser.parse_args()

    if args.list_models:
        list_available_models()
        sys.exit(0)

    if args.list_strategies:
        print("\nAvailable prompt strategies:\n")
        descs = {
            "zero_shot_direct":  "Zero-shot, direct output",
            "zero_shot_cot":     "Zero-shot, chain-of-thought then summary",
            "few_shot_direct":   "Few-shot (2 gold examples), direct output",
            "few_shot_cot":      "Few-shot (2 gold examples), CoT then summary",
            "baseline2_style":   "Enriched prompt with ABCD labels per post (Baseline 2 style)",
        }
        for s, d in descs.items():
            print(f"  {s:<22s}  {d}")
        print()
        sys.exit(0)

    # Validate required args
    if not args.sequences or not args.timelines_dir:
        parser.error("--sequences and --timelines-dir are required")

    # Build model and strategy lists
    models    = ABLATION_MODELS if args.all_models    else [args.model]
    strategies = STRATEGIES      if args.all_strategies else [args.strategy]

    make_submission = not args.no_submission

    total = len(models) * len(strategies)
    done  = 0

    for model_key in models:
        for strategy in strategies:
            done += 1
            logger.info(f"\n[{done}/{total}] model={model_key}, strategy={strategy}")
            try:
                run_single(
                    sequences_file     = args.sequences,
                    timelines_dir      = args.timelines_dir,
                    model_key          = model_key,
                    strategy           = strategy,
                    split              = args.split,
                    output_base        = args.output_dir,
                    train_sequences_file = args.train_sequences or "",
                    train_timelines_dir  = args.train_timelines_dir or "",
                    config_path        = args.config,
                    api_key            = args.api_key,
                    base_url           = args.base_url,
                    device             = args.device,
                    temperature        = args.temperature,
                    max_tokens         = args.max_tokens,
                    load_in_4bit       = args.load_4bit,
                    make_submission    = make_submission,
                )
            except Exception as e:
                logger.error(
                    f"  FAILED model={model_key}, strategy={strategy}: {e}"
                )
                continue

    logger.info(f"\nAll runs complete. Outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()