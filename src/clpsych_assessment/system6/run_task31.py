"""
CLPsych 2026 — System 6 — RAG-enhanced Task 3.1 Runner

Runs RAG-enhanced sequence summarisation for Task 3.1.

Outputs:
    outputs/system6/{model}/{strategy}/raw_task3_{split}.json
    outputs/system6/{model}/{strategy}/submission/task3_pred.json
    outputs/system6/{model}/{strategy}/submission/task3_pred.zip

Usage examples
--------------
# Single run: RAG few-shot direct with LLaMA 3.1
python -m clpsych_assessment.system6.run_task31 \
    --sequences data/task3_train_n_test/train_task3_val_fold.json \
    --timelines-dir data/train_tasks12/ \
    --model llama3.1 \
    --strategy rag_few_shot_direct \
    --train-sequences data/task3_train_n_test/train_task3_train_fold.json \
    --train-timelines-dir data/train_tasks12/ \
    --split val

# All RAG strategies for one model
python -m clpsych_assessment.system6.run_task31 \
    --sequences data/task3_train_n_test/train_task3_val_fold.json \
    --timelines-dir data/train_tasks12/ \
    --model llama3.1 \
    --all-strategies \
    --train-sequences data/task3_train_n_test/train_task3_train_fold.json \
    --train-timelines-dir data/train_tasks12/ \
    --split val

# Full ablation: all strategies × all models
python -m clpsych_assessment.system6.run_task31 \
    --sequences data/task3_train_n_test/train_task3_val_fold.json \
    --timelines-dir data/train_tasks12/ \
    --all-strategies --all-models \
    --train-sequences data/task3_train_n_test/train_task3_train_fold.json \
    --train-timelines-dir data/train_tasks12/ \
    --split val

# Test set run (after finding best config on val)
python -m clpsych_assessment.system6.run_task31 \
    --sequences data/task3_train_n_test/test_task3nolabels.json \
    --timelines-dir data/test_tasks12nolabels/ \
    --model llama3.1 \
    --strategy rag_few_shot_direct \
    --train-sequences data/task3_train_n_test/train_task3_train_fold.json \
    --train-timelines-dir data/train_tasks12/

# Override RAG parameters
python -m clpsych_assessment.system6.run_task31 \
    --sequences data/task3_train_n_test/train_task3_val_fold.json \
    --timelines-dir data/train_tasks12/ \
    --model gemma4 \
    --strategy rag_baseline2_style \
    --train-sequences data/task3_train_n_test/train_task3_train_fold.json \
    --train-timelines-dir data/train_tasks12/ \
    --rag-top-k 5 \
    --rag-strategy diverse \
    --embedding-backend tfidf \
    --split val
"""

import argparse
import logging
import sys
from pathlib import Path

from ..system3.chain import MODELS, list_available_models
from .pipeline import STRATEGIES, RAGTask31Pipeline
from .format_submission import write_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Models for ablation
ABLATION_MODELS = ["llama3.1", "gemma2:9b", "gemma4"]

# Register gemma4 if not already present
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
) -> tuple:
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
    rag_top_k: int = None,
    rag_strategy: str = None,
    embedding_backend: str = None,
):
    """Run one (model, strategy) combination with RAG."""
    raw_path, submission_dir = _output_paths(
        output_base, model_key, strategy, split
    )

    logger.info(
        f"\n{'='*60}\n"
        f" model={model_key}  strategy={strategy}  split={split}\n"
        f"{'='*60}"
    )

    pipeline = RAGTask31Pipeline(
        strategy=strategy,
        model_key=model_key,
        config_path=config_path or None,
        train_sequences_file=train_sequences_file,
        train_timelines_dir=train_timelines_dir,
        api_key=api_key,
        base_url=base_url,
        device=device,
        temperature=temperature,
        max_tokens=max_tokens,
        load_in_4bit=load_in_4bit,
        rag_top_k=rag_top_k,
        rag_strategy=rag_strategy,
        embedding_backend=embedding_backend,
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
        description="Task 3.1: RAG-enhanced sequence summarisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input data
    parser.add_argument(
        "--sequences", required=False,
        help="Sequences JSON file (test or val fold)",
    )
    parser.add_argument(
        "--timelines-dir", required=False,
        help="Directory with per-timeline JSONs",
    )

    # Training data (required for RAG)
    parser.add_argument(
        "--train-sequences", required=False,
        help="train_task3_train_fold.json — RAG library source",
    )
    parser.add_argument(
        "--train-timelines-dir", required=False,
        help="train_tasks12/ dir — RAG library source",
    )

    # Model + strategy selection
    parser.add_argument(
        "--model", default="llama3.1", choices=list(MODELS.keys()),
        help="Model key (default: llama3.1)",
    )
    parser.add_argument(
        "--strategy", default="rag_few_shot_direct", choices=STRATEGIES,
        help=f"Prompt strategy (default: rag_few_shot_direct)",
    )
    parser.add_argument(
        "--all-strategies", action="store_true",
        help=f"Run all RAG strategies: {STRATEGIES}",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help=f"Run ablation models: {ABLATION_MODELS}",
    )

    # RAG-specific parameters
    parser.add_argument(
        "--rag-top-k", type=int, default=None,
        help="Number of examples to retrieve (default: from config)",
    )
    parser.add_argument(
        "--rag-strategy", default=None,
        choices=["semantic", "filtered", "hybrid", "diverse"],
        help="Retrieval strategy (default: from config)",
    )
    parser.add_argument(
        "--embedding-backend", default=None,
        choices=["auto", "sentence_transformer", "tfidf"],
        help="Embedding backend (default: from config)",
    )

    # Output
    parser.add_argument(
        "--output-dir", default="outputs/system6",
        help="Base output directory (default: outputs/system6)",
    )
    parser.add_argument(
        "--split", default="test",
        help="Split label for output filenames (default: test)",
    )
    parser.add_argument(
        "--no-submission", action="store_true",
        help="Skip writing submission zip",
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
        print("\nAvailable RAG prompt strategies:\n")
        for s in STRATEGIES:
            print(f"  {s}")
        print()
        sys.exit(0)

    if not args.sequences or not args.timelines_dir:
        parser.error("--sequences and --timelines-dir are required")

    if not args.train_sequences or not args.train_timelines_dir:
        parser.error("--train-sequences and --train-timelines-dir are required for RAG")

    models = ABLATION_MODELS if args.all_models else [args.model]
    strategies = STRATEGIES if args.all_strategies else [args.strategy]

    make_submission = not args.no_submission
    total = len(models) * len(strategies)
    done = 0

    for model_key in models:
        for strategy in strategies:
            done += 1
            logger.info(f"\n[{done}/{total}] model={model_key}, strategy={strategy}")
            try:
                run_single(
                    sequences_file=args.sequences,
                    timelines_dir=args.timelines_dir,
                    model_key=model_key,
                    strategy=strategy,
                    split=args.split,
                    output_base=args.output_dir,
                    train_sequences_file=args.train_sequences,
                    train_timelines_dir=args.train_timelines_dir,
                    config_path=args.config,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    device=args.device,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    load_in_4bit=args.load_4bit,
                    make_submission=make_submission,
                    rag_top_k=args.rag_top_k,
                    rag_strategy=args.rag_strategy,
                    embedding_backend=args.embedding_backend,
                )
            except Exception as e:
                logger.error(
                    f"  FAILED model={model_key}, strategy={strategy}: {e}"
                )
                import traceback
                traceback.print_exc()
                continue

    logger.info(f"\nAll runs complete. Outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
