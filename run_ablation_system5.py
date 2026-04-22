"""
CLPsych 2026 — System 5 — Ablation Automation Script

Workflow
--------
Phase 1 — Val ablation (all 18 runs on the val fold)
    Runs all combinations of 3 models × 5 strategies on the validation fold.
    Scores each run with ROUGE-L Recall and BERTScore Recall against gold summaries.
    Logs all scores to outputs/system5/ablation_scores.csv.

Phase 2 — Top-4 test runs
    Selects the 4 highest-scoring (model, strategy) pairs by ROUGE-L Recall.
    Runs those on the test set and produces Codabench-ready zip files.

Usage
-----
# Full automation (val ablation → score → top-4 test runs)
python run_ablation.py

# Skip val ablation if already done, go straight to scoring + top-4
python run_ablation.py --skip-val

# Only run val ablation (no scoring, no test runs)
python run_ablation.py --val-only

# Override number of top runs to submit
python run_ablation.py --top-n 3

# Use a specific train fold for few-shot examples
python run_ablation.py --train-fold data/task3_train_n_test/train_task3_train_fold.json
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# CONFIG — edit these paths to match your project layout
# ─────────────────────────────────────────────────────────────────

VAL_SEQUENCES    = "data/task3_train_n_test/train_task3_val_fold.json"
VAL_TIMELINES    = "data/train_tasks12"
TEST_SEQUENCES   = "data/task3_train_n_test/test_task3nolabels.json"
TEST_TIMELINES   = "data/test_tasks12nolabels"

# Training data used for few-shot example selection
# After running data_split.py these will exist; fall back to full train set
TRAIN_SEQUENCES  = "data/task3_train_n_test/train_task3_train_fold.json"
TRAIN_SEQUENCES_FALLBACK = "data/task3_train_n_test/train_task3.json"
TRAIN_TIMELINES  = "data/train_tasks12"

# Gold summaries for scoring the val fold
GOLD_SUMMARIES   = "data/task3_train_n_test/train_task3.json"

OUTPUT_DIR       = "outputs/system5"
SCORES_CSV       = f"{OUTPUT_DIR}/ablation_scores.csv"

MODELS           = ["llama3.1", "gemma2:9b", "gemma4"]
STRATEGIES       = [
    "zero_shot_direct",
    "zero_shot_cot",
    "few_shot_direct",
    "few_shot_cot",
    "baseline2_style",
    "colleague_style",
]
FEW_SHOT_STRATEGIES = {"few_shot_direct", "few_shot_cot"}

TOP_N_DEFAULT    = 3


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _run_dir(model: str, strategy: str) -> Path:
    return Path(OUTPUT_DIR) / model.replace(":", "_") / strategy


def _raw_path(model: str, strategy: str, split: str) -> Path:
    return _run_dir(model, strategy) / f"raw_task3_{split}.json"


def _submission_dir(model: str, strategy: str) -> Path:
    return _run_dir(model, strategy) / "submission"


def _resolve_train_sequences(override: Optional[str]) -> str:
    """Return the train sequences path to use for few-shot examples."""
    if override and Path(override).exists():
        return override
    if Path(TRAIN_SEQUENCES).exists():
        return TRAIN_SEQUENCES
    if Path(TRAIN_SEQUENCES_FALLBACK).exists():
        logger.warning(
            f"Train fold not found, falling back to full train set: "
            f"{TRAIN_SEQUENCES_FALLBACK}"
        )
        return TRAIN_SEQUENCES_FALLBACK
    raise FileNotFoundError(
        f"No training sequences found. Run data_split.py first or pass "
        f"--train-fold explicitly."
    )


# ─────────────────────────────────────────────────────────────────
# PHASE 1 — VAL ABLATION
# ─────────────────────────────────────────────────────────────────

def build_run_command(
    model: str,
    strategy: str,
    split: str,
    sequences: str,
    timelines: str,
    train_sequences: str,
    extra_args: List[str],
) -> List[str]:
    """Build the python -m ... command for one run."""
    cmd = [
        sys.executable, "-m",
        "clpsych_assessment.system5.run_task31",
        "--sequences",    sequences,
        "--timelines-dir", timelines,
        "--model",        model,
        "--strategy",     strategy,
        "--split",        split,
        "--output-dir",   OUTPUT_DIR,
        "--no-submission",   # raw JSON only during val; submission comes later
    ]
    if strategy in FEW_SHOT_STRATEGIES:
        cmd += [
            "--train-sequences",    train_sequences,
            "--train-timelines-dir", TRAIN_TIMELINES,
        ]
    cmd += extra_args
    return cmd


def run_val_ablation(train_sequences: str, extra_args: List[str]) -> None:
    """Run all 15 (model, strategy) combinations on the validation fold."""
    total = len(MODELS) * len(STRATEGIES)
    done  = 0

    for model in MODELS:
        for strategy in STRATEGIES:
            done += 1
            raw = _raw_path(model, strategy, "val")
            if raw.exists():
                logger.info(
                    f"[{done}/{total}] SKIP (exists): {model} / {strategy}"
                )
                continue

            logger.info(
                f"[{done}/{total}] RUNNING val: model={model}  strategy={strategy}"
            )
            cmd = build_run_command(
                model, strategy, "val",
                VAL_SEQUENCES, VAL_TIMELINES,
                train_sequences, extra_args,
            )
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                logger.error(
                    f"  FAILED: model={model}, strategy={strategy} "
                    f"(exit code {result.returncode})"
                )


# ─────────────────────────────────────────────────────────────────
# PHASE 2 — SCORING
# ─────────────────────────────────────────────────────────────────

def load_gold_by_sequence(gold_file: str) -> Dict[str, str]:
    """Return {(timeline_id, sequence_id): summary} from gold file."""
    with open(gold_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        (item["timeline_id"], item["sequence_id"]): item["summary"]
        for item in data
        if item.get("summary")
    }


def load_predictions(pred_file: str) -> Dict[Tuple[str, str], str]:
    """Return {(timeline_id, sequence_id): summary} from a raw output file."""
    with open(pred_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        (item["timeline_id"], item["sequence_id"]): item.get("summary", "")
        for item in data
    }


def _truncate(text: str, word_limit: int = 350) -> str:
    words = text.split()
    return " ".join(words[:word_limit]) if len(words) > word_limit else text


def compute_rouge_l_recall(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L Recall (LCS-based) between prediction and reference.
    Pure Python implementation — no external library required.
    """
    pred_tokens = prediction.lower().split()
    ref_tokens  = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS length via DP
    m, n = len(ref_tokens), len(pred_tokens)
    # Use 1-D DP to save memory
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref_tokens[i-1] == pred_tokens[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr

    lcs_len = prev[n]
    # Recall = LCS / len(reference)
    return lcs_len / m


def compute_bertscore_recall(
    predictions: List[str],
    references: List[str],
) -> List[float]:
    """
    Compute BERTScore Recall. Uses bert_score library if available.
    Falls back to token overlap recall if not installed.
    Note: first call is slow (~30s) due to model/matplotlib loading.
    """
    try:
        from bert_score import score as bs_score
        _, _, F = bs_score(
            predictions, references,
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
            rescale_with_baseline=True,
            verbose=False,
        )
        return F.tolist()
    except ImportError:
        logger.warning(
            "bert_score not installed — using token overlap recall as proxy.\n"
            "  Install with: pip install bert-score\n"
            "  Note: this fallback is NOT the official metric."
        )
        scores = []
        for pred, ref in zip(predictions, references):
            pred_toks = set(pred.lower().split())
            ref_toks  = set(ref.lower().split())
            if not ref_toks:
                scores.append(0.0)
            else:
                scores.append(len(pred_toks & ref_toks) / len(ref_toks))
        return scores


def score_run(
    model: str,
    strategy: str,
    gold: Dict[Tuple[str, str], str],
) -> Optional[Dict]:
    """
    Score one (model, strategy) val run.
    Returns a dict of metrics, or None if the raw file doesn't exist.
    """
    raw = _raw_path(model, strategy, "val")
    if not raw.exists():
        return None

    preds = load_predictions(str(raw))

    # Diagnose empty summaries — common when model backend fails silently
    n_empty = sum(1 for v in preds.values() if not v.strip())
    if n_empty > 0:
        logger.warning(
            f"  {model}/{strategy}: {n_empty}/{len(preds)} summaries are empty. "
            f"Check that the model ran correctly and outputs were captured."
        )

    # Match predictions to gold — only score sequences present in both
    matched_keys = [k for k in preds if k in gold]
    if not matched_keys:
        logger.warning(
            f"  No matching sequences for {model}/{strategy}.\n"
            f"  Pred IDs (first 3): {list(preds.keys())[:3]}\n"
            f"  Gold IDs (first 3): {list(gold.keys())[:3]}"
        )
        return None

    pred_texts = [_truncate(preds[k]) for k in matched_keys]
    gold_texts = [gold[k] for k in matched_keys]

    # ROUGE-L Recall
    rouge_scores = [
        compute_rouge_l_recall(p, g)
        for p, g in zip(pred_texts, gold_texts)
    ]
    avg_rouge = sum(rouge_scores) / len(rouge_scores)

    # BERTScore Recall — computed independently from ROUGE-L
    logger.info(f"    BERTScore ({len(pred_texts)} sequences, first call loads model ~30s)...")
    bert_scores = compute_bertscore_recall(pred_texts, gold_texts)
    avg_bert = sum(bert_scores) / len(bert_scores)

    n_matched   = len(matched_keys)
    n_non_empty = sum(1 for t in pred_texts if t.strip())

    return {
        "model":            model,
        "strategy":         strategy,
        "n_sequences":      n_matched,
        "n_non_empty":      n_non_empty,
        "rouge_l_recall":   round(avg_rouge, 4),
        "bertscore_recall": round(avg_bert, 4),
    }


def score_all_runs(gold_file: str) -> List[Dict]:
    """Score all available val runs and return sorted results."""
    gold = load_gold_by_sequence(gold_file)
    logger.info(f"  Gold sequences loaded: {len(gold)}")

    results = []
    for model in MODELS:
        for strategy in STRATEGIES:
            logger.info(f"  Scoring: {model} / {strategy}")
            metrics = score_run(model, strategy, gold)
            if metrics:
                results.append(metrics)
                logger.info(
                    f"    ROUGE-L={metrics['rouge_l_recall']:.4f}  "
                    f"BERTScore={metrics['bertscore_recall']:.4f}"
                )
            else:
                logger.info("    (no output — skipped)")

    # Sort by ROUGE-L Recall descending (primary), BERTScore descending (tiebreak)
    results.sort(
        key=lambda x: (x["rouge_l_recall"], x["bertscore_recall"]),
        reverse=True,
    )
    return results


def save_scores_csv(results: List[Dict], path: str) -> None:
    """Save scores table to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "strategy", "n_sequences", "n_non_empty",
              "rouge_l_recall", "bertscore_recall"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"  Scores saved → {path}")


def print_scores_table(results: List[Dict]) -> None:
    """Pretty-print the scores table to stdout."""
    print(f"\n{'─'*80}")
    print(
        f"{'RANK':<5} {'MODEL':<14} {'STRATEGY':<22} "
        f"{'NON-EMPTY':>9} {'ROUGE-L':>8} {'BERTScore':>10}"
    )
    print(f"{'─'*80}")
    for i, r in enumerate(results, 1):
        marker = " ◄" if i <= 3 else ""
        non_empty = f"{r.get('n_non_empty', '?')}/{r['n_sequences']}"
        print(
            f"{i:<5} {r['model']:<14} {r['strategy']:<22} "
            f"{non_empty:>9} {r['rouge_l_recall']:>8.4f} "
            f"{r['bertscore_recall']:>10.4f}{marker}"
        )
    print(f"{'─'*80}\n")


# ─────────────────────────────────────────────────────────────────
# PHASE 3 — TOP-N TEST RUNS
# ─────────────────────────────────────────────────────────────────

def run_test_set(
    model: str,
    strategy: str,
    train_sequences: str,
    extra_args: List[str],
) -> None:
    """Run one (model, strategy) combination on the test set and format submission."""
    logger.info(f"  TEST RUN: model={model}  strategy={strategy}")

    cmd = build_run_command(
        model, strategy, "test",
        TEST_SEQUENCES, TEST_TIMELINES,
        train_sequences, extra_args,
    )
    # Remove --no-submission so the zip is produced
    cmd = [x for x in cmd if x != "--no-submission"]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error(
            f"  FAILED test run: model={model}, strategy={strategy} "
            f"(exit code {result.returncode})"
        )
        return

    # Write submission
    raw = _raw_path(model, strategy, "test")
    if raw.exists():
        from src.clpsych_assessment.system5.format_submission import write_submission
        sub_dir = str(_submission_dir(model, strategy))
        write_submission(str(raw), sub_dir)
        zip_path = Path(sub_dir) / "task3_pred.zip"
        logger.info(f"  Submission: {zip_path}")
    else:
        logger.error(f"  Raw output not found after test run: {raw}")


def run_top_n_test(
    results: List[Dict],
    top_n: int,
    train_sequences: str,
    extra_args: List[str],
) -> None:
    """Run the top_n (model, strategy) pairs on the test set."""
    top = results[:top_n]
    logger.info(
        f"\n{'='*60}\n"
        f"PHASE 3 — Top {top_n} test runs\n"
        f"{'='*60}"
    )
    for i, r in enumerate(top, 1):
        logger.info(
            f"  [{i}/{top_n}] {r['model']} / {r['strategy']}  "
            f"(ROUGE-L={r['rouge_l_recall']:.4f})"
        )
        run_test_set(r["model"], r["strategy"], train_sequences, extra_args)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="System 5 ablation: val runs → score → top-N test submissions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip-val", action="store_true",
        help="Skip the val ablation phase (use existing raw outputs)",
    )
    parser.add_argument(
        "--val-only", action="store_true",
        help="Run val ablation only — skip scoring and test runs",
    )
    parser.add_argument(
        "--top-n", type=int, default=TOP_N_DEFAULT,
        help=f"Number of top approaches to run on test set (default: {TOP_N_DEFAULT})",
    )
    parser.add_argument(
        "--train-fold", default=None,
        help="Path to train fold for few-shot examples (default: auto-detected)",
    )
    parser.add_argument(
        "--gold", default=GOLD_SUMMARIES,
        help=f"Gold summaries file for scoring (default: {GOLD_SUMMARIES})",
    )
    # Pass-through args for the run_task31 module
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens",  type=int,   default=800)
    parser.add_argument("--load-4bit",   action="store_true")
    args = parser.parse_args()

    train_sequences = _resolve_train_sequences(args.train_fold)
    logger.info(f"Train sequences for few-shot: {train_sequences}")

    # Extra args forwarded to run_task31
    extra_args = [
        "--device",      args.device,
        "--temperature", str(args.temperature),
        "--max-tokens",  str(args.max_tokens),
    ]
    if args.load_4bit:
        extra_args.append("--load-4bit")

    # ── Phase 1: Val ablation ─────────────────────────────────────
    if not args.skip_val:
        logger.info(
            f"\n{'='*60}\n"
            f"PHASE 1 — Val ablation ({len(MODELS)}×{len(STRATEGIES)}={len(MODELS)*len(STRATEGIES)} runs)\n"
            f"{'='*60}"
        )
        run_val_ablation(train_sequences, extra_args)
    else:
        logger.info("Phase 1 skipped (--skip-val).")

    if args.val_only:
        logger.info("Val-only mode — done.")
        return

    # ── Phase 2: Scoring ──────────────────────────────────────────
    logger.info(
        f"\n{'='*60}\n"
        f"PHASE 2 — Scoring val runs\n"
        f"{'='*60}"
    )

    if not Path(args.gold).exists():
        logger.error(
            f"Gold file not found: {args.gold}\n"
            f"Make sure train_task3.json is present before scoring."
        )
        sys.exit(1)

    results = score_all_runs(args.gold)
    print_scores_table(results)
    save_scores_csv(results, SCORES_CSV)

    if not results:
        logger.error("No val results found — cannot select top runs. Exiting.")
        sys.exit(1)

    # ── Phase 3: Top-N test runs ──────────────────────────────────
    run_top_n_test(results, args.top_n, train_sequences, extra_args)

    # ── Summary ───────────────────────────────────────────────────
    logger.info(
        f"\n{'='*60}\n"
        f"All done.\n"
        f"  Scores:      {SCORES_CSV}\n"
        f"  Submissions: {OUTPUT_DIR}/<model>/<strategy>/submission/task3_pred.zip\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()