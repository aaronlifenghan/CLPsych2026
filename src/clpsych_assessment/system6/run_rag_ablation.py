"""
CLPsych 2026 — System 6 — RAG Ablation Automation Script

Workflow
--------
Phase 1 — Val ablation (all model × strategy × RAG config combos on val fold)
Phase 2 — Score all runs against gold summaries
Phase 3 — Compare with System 5 baselines
Phase 4 — Top-N test runs with best configs

Usage
-----
# Full automation
python run_rag_ablation.py

# Skip val runs, just score existing outputs
python run_rag_ablation.py --skip-val

# Only score and compare (no test runs)
python run_rag_ablation.py --skip-val --score-only

# Run with specific RAG configs
python run_rag_ablation.py --rag-top-k 3 5 --rag-strategies hybrid diverse

# Override embedding backend
python run_rag_ablation.py --embedding-backend tfidf
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
# CONFIG
# ─────────────────────────────────────────────────────────────────

VAL_SEQUENCES      = "data/task3_train_n_test/train_task3_val_fold.json"
VAL_TIMELINES      = "data/train_tasks12"
TEST_SEQUENCES     = "data/task3_train_n_test/test_task3nolabels.json"
TEST_TIMELINES     = "data/test_tasks12nolabels"
TRAIN_SEQUENCES    = "data/task3_train_n_test/train_task3_train_fold.json"
TRAIN_TIMELINES    = "data/train_tasks12"
GOLD_SUMMARIES     = "data/task3_train_n_test/train_task3.json"

OUTPUT_DIR         = "outputs/system6"
SYSTEM5_OUTPUT_DIR = "outputs/system5"
SCORES_CSV         = f"{OUTPUT_DIR}/ablation_scores.csv"
COMPARISON_CSV     = f"{OUTPUT_DIR}/comparison_with_system5.csv"

MODELS = ["llama3.1", "gemma2:9b", "gemma4"]
PROMPT_STRATEGIES = [
    "rag_few_shot_direct",
    "rag_baseline2_style",
    "rag_colleague_style",
]

# Default ablation grid for RAG hyperparameters
DEFAULT_RAG_TOP_KS = [3]
DEFAULT_RAG_STRATEGIES = ["hybrid"]

TOP_N_DEFAULT = 3


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _run_dir(model: str, strategy: str, rag_topk: int = 3, rag_strat: str = "hybrid") -> Path:
    """Output directory for a specific run configuration."""
    suffix = f"_k{rag_topk}_{rag_strat}" if rag_topk != 3 or rag_strat != "hybrid" else ""
    return Path(OUTPUT_DIR) / model.replace(":", "_") / f"{strategy}{suffix}"


def _raw_path(model: str, strategy: str, split: str, rag_topk: int = 3, rag_strat: str = "hybrid") -> Path:
    return _run_dir(model, strategy, rag_topk, rag_strat) / f"raw_task3_{split}.json"


# ─────────────────────────────────────────────────────────────────
# PHASE 1 — VAL ABLATION
# ─────────────────────────────────────────────────────────────────

def build_run_command(
    model: str,
    strategy: str,
    split: str,
    sequences: str,
    timelines: str,
    rag_top_k: int,
    rag_strategy: str,
    embedding_backend: str,
    extra_args: List[str],
) -> List[str]:
    cmd = [
        sys.executable, "-m",
        "clpsych_assessment.system6.run_task31",
        "--sequences",        sequences,
        "--timelines-dir",    timelines,
        "--model",            model,
        "--strategy",         strategy,
        "--split",            split,
        "--output-dir",       OUTPUT_DIR,
        "--train-sequences",  TRAIN_SEQUENCES,
        "--train-timelines-dir", TRAIN_TIMELINES,
        "--no-submission",
        "--rag-top-k",        str(rag_top_k),
        "--rag-strategy",     rag_strategy,
        "--embedding-backend", embedding_backend,
    ]
    cmd += extra_args
    return cmd


def run_val_ablation(
    rag_top_ks: List[int],
    rag_strategies: List[str],
    embedding_backend: str,
    extra_args: List[str],
) -> None:
    """Run all combinations on the validation fold."""
    combos = [
        (m, ps, rk, rs)
        for m in MODELS
        for ps in PROMPT_STRATEGIES
        for rk in rag_top_ks
        for rs in rag_strategies
    ]
    total = len(combos)

    for i, (model, prompt_strat, rag_topk, rag_strat) in enumerate(combos, 1):
        raw = _raw_path(model, prompt_strat, "val", rag_topk, rag_strat)
        if raw.exists():
            logger.info(f"[{i}/{total}] SKIP (exists): {model} / {prompt_strat} / k={rag_topk} / {rag_strat}")
            continue

        logger.info(
            f"[{i}/{total}] RUNNING val: model={model}  strategy={prompt_strat}  "
            f"rag_top_k={rag_topk}  rag_strategy={rag_strat}"
        )

        # Override output dir for non-default configs
        cmd = build_run_command(
            model, prompt_strat, "val",
            VAL_SEQUENCES, VAL_TIMELINES,
            rag_topk, rag_strat, embedding_backend,
            extra_args,
        )

        # If non-default RAG params, adjust output-dir
        if rag_topk != 3 or rag_strat != "hybrid":
            # Replace output-dir to include suffix
            for j, arg in enumerate(cmd):
                if arg == "--output-dir":
                    cmd[j+1] = OUTPUT_DIR
                    break

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error(f"  FAILED: {model}/{prompt_strat} (exit code {result.returncode})")


# ─────────────────────────────────────────────────────────────────
# PHASE 2 — SCORING
# ─────────────────────────────────────────────────────────────────

def load_gold_by_sequence(gold_file: str) -> Dict[Tuple[str, str], str]:
    with open(gold_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        (item["timeline_id"], item["sequence_id"]): item["summary"]
        for item in data if item.get("summary")
    }


def load_predictions(pred_file: str) -> Dict[Tuple[str, str], str]:
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
    pred_tokens = prediction.lower().split()
    ref_tokens  = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(ref_tokens), len(pred_tokens)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref_tokens[i-1] == pred_tokens[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    return prev[n] / m


def compute_bertscore_recall(
    predictions: List[str],
    references: List[str],
) -> List[float]:
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
            "  Install with: pip install bert-score"
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
    raw_path: Path,
    gold: Dict[Tuple[str, str], str],
    label: str = "",
) -> Optional[Dict]:
    if not raw_path.exists():
        return None

    preds = load_predictions(str(raw_path))
    n_empty = sum(1 for v in preds.values() if not v.strip())
    if n_empty > 0:
        logger.warning(f"  {label}: {n_empty}/{len(preds)} summaries are empty")

    matched_keys = [k for k in preds if k in gold]
    if not matched_keys:
        logger.warning(f"  No matching sequences for {label}")
        return None

    pred_texts = [_truncate(preds[k]) for k in matched_keys]
    gold_texts = [gold[k] for k in matched_keys]

    rouge_scores = [compute_rouge_l_recall(p, g) for p, g in zip(pred_texts, gold_texts)]
    avg_rouge = sum(rouge_scores) / len(rouge_scores)

    logger.info(f"    Computing BERTScore ({len(pred_texts)} sequences)...")
    bert_scores = compute_bertscore_recall(pred_texts, gold_texts)
    avg_bert = sum(bert_scores) / len(bert_scores)

    return {
        "n_sequences":      len(matched_keys),
        "n_non_empty":      sum(1 for t in pred_texts if t.strip()),
        "rouge_l_recall":   round(avg_rouge, 4),
        "bertscore_recall": round(avg_bert, 4),
    }


def score_all_system6_runs(
    gold_file: str,
    rag_top_ks: List[int],
    rag_strategies: List[str],
) -> List[Dict]:
    gold = load_gold_by_sequence(gold_file)
    logger.info(f"  Gold sequences loaded: {len(gold)}")

    results = []
    for model in MODELS:
        for prompt_strat in PROMPT_STRATEGIES:
            for rag_topk in rag_top_ks:
                for rag_strat in rag_strategies:
                    raw = _raw_path(model, prompt_strat, "val", rag_topk, rag_strat)
                    label = f"{model}/{prompt_strat}/k={rag_topk}/{rag_strat}"
                    logger.info(f"  Scoring: {label}")

                    metrics = score_run(raw, gold, label)
                    if metrics:
                        metrics.update({
                            "system":          "system6",
                            "model":           model,
                            "prompt_strategy":  prompt_strat,
                            "rag_top_k":       rag_topk,
                            "rag_retrieval":   rag_strat,
                        })
                        results.append(metrics)
                        logger.info(
                            f"    ROUGE-L={metrics['rouge_l_recall']:.4f}  "
                            f"BERTScore={metrics['bertscore_recall']:.4f}"
                        )
                    else:
                        logger.info("    (no output — skipped)")

    results.sort(
        key=lambda x: (x["rouge_l_recall"], x["bertscore_recall"]),
        reverse=True,
    )
    return results


# ─────────────────────────────────────────────────────────────────
# PHASE 3 — COMPARISON WITH SYSTEM 5
# ─────────────────────────────────────────────────────────────────

def load_system5_scores() -> List[Dict]:
    """Load System 5 ablation scores for comparison."""
    csv_path = Path(SYSTEM5_OUTPUT_DIR) / "ablation_scores.csv"
    if not csv_path.exists():
        logger.warning(f"System 5 scores not found at {csv_path}")
        return []

    results = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "system":          "system5",
                "model":           row["model"],
                "prompt_strategy":  row["strategy"],
                "rag_top_k":       "-",
                "rag_retrieval":   "-",
                "n_sequences":     int(row["n_sequences"]),
                "n_non_empty":     int(row["n_non_empty"]),
                "rouge_l_recall":  float(row["rouge_l_recall"]),
                "bertscore_recall": float(row["bertscore_recall"]),
            })
    return results


def print_comparison_table(all_results: List[Dict]) -> None:
    """Pretty-print combined scores from both systems."""
    all_results.sort(
        key=lambda x: (x["rouge_l_recall"], x["bertscore_recall"]),
        reverse=True,
    )

    print(f"\n{'═'*110}")
    print("  SYSTEM 5 vs SYSTEM 6 (RAG) — Val Fold Comparison")
    print(f"{'═'*110}")
    print(
        f"{'RANK':<5} {'SYS':<8} {'MODEL':<14} {'PROMPT':<25} "
        f"{'RAG_K':>5} {'RAG_STRAT':>10} "
        f"{'ROUGE-L':>8} {'BERTScore':>10}"
    )
    print(f"{'─'*110}")

    for i, r in enumerate(all_results, 1):
        sys_tag = r["system"]
        marker = " ◄" if r["system"] == "system6" else ""
        rag_k = str(r.get("rag_top_k", "-"))
        rag_s = str(r.get("rag_retrieval", "-"))
        print(
            f"{i:<5} {sys_tag:<8} {r['model']:<14} {r['prompt_strategy']:<25} "
            f"{rag_k:>5} {rag_s:>10} "
            f"{r['rouge_l_recall']:>8.4f} {r['bertscore_recall']:>10.4f}{marker}"
        )

    print(f"{'═'*110}\n")


def save_comparison_csv(all_results: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "system", "model", "prompt_strategy",
        "rag_top_k", "rag_retrieval",
        "n_sequences", "n_non_empty",
        "rouge_l_recall", "bertscore_recall",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)
    logger.info(f"  Comparison saved → {path}")


# ─────────────────────────────────────────────────────────────────
# PHASE 4 — TOP-N TEST RUNS
# ─────────────────────────────────────────────────────────────────

def run_top_n_test(
    results: List[Dict],
    top_n: int,
    embedding_backend: str,
    extra_args: List[str],
) -> None:
    top = [r for r in results if r["system"] == "system6"][:top_n]
    logger.info(
        f"\n{'='*60}\n"
        f"PHASE 4 — Top {top_n} test runs\n"
        f"{'='*60}"
    )
    for i, r in enumerate(top, 1):
        logger.info(
            f"  [{i}/{top_n}] {r['model']} / {r['prompt_strategy']}  "
            f"(ROUGE-L={r['rouge_l_recall']:.4f})"
        )
        cmd = build_run_command(
            r["model"], r["prompt_strategy"], "test",
            TEST_SEQUENCES, TEST_TIMELINES,
            r["rag_top_k"], r["rag_retrieval"],
            embedding_backend, extra_args,
        )
        # Remove --no-submission for test runs
        cmd = [x for x in cmd if x != "--no-submission"]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error(
                f"  FAILED test run: {r['model']}/{r['prompt_strategy']} "
                f"(exit code {result.returncode})"
            )


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="System 6 RAG ablation: val runs → score → compare → top-N test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skip-val",    action="store_true", help="Skip val ablation")
    parser.add_argument("--score-only",  action="store_true", help="Score and compare only, no test runs")
    parser.add_argument("--top-n",       type=int, default=TOP_N_DEFAULT)
    parser.add_argument("--gold",        default=GOLD_SUMMARIES)

    # RAG ablation grid
    parser.add_argument(
        "--rag-top-k", type=int, nargs="+", default=DEFAULT_RAG_TOP_KS,
        help="RAG top_k values to ablate (default: [3])",
    )
    parser.add_argument(
        "--rag-strategies", nargs="+", default=DEFAULT_RAG_STRATEGIES,
        help="RAG retrieval strategies to ablate (default: [hybrid])",
    )
    parser.add_argument(
        "--embedding-backend", default="auto",
        choices=["auto", "sentence_transformer", "tfidf"],
    )

    # Model config pass-through
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens",  type=int,   default=800)
    parser.add_argument("--load-4bit",   action="store_true")

    args = parser.parse_args()

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
            f"PHASE 1 — Val ablation "
            f"({len(MODELS)}×{len(PROMPT_STRATEGIES)}×"
            f"{len(args.rag_top_k)}×{len(args.rag_strategies)} combos)\n"
            f"{'='*60}"
        )
        run_val_ablation(
            args.rag_top_k, args.rag_strategies,
            args.embedding_backend, extra_args,
        )

    # ── Phase 2: Score system6 runs ───────────────────────────────
    logger.info(
        f"\n{'='*60}\n"
        f"PHASE 2 — Scoring system6 val runs\n"
        f"{'='*60}"
    )

    if not Path(args.gold).exists():
        logger.error(f"Gold file not found: {args.gold}")
        sys.exit(1)

    s6_results = score_all_system6_runs(
        args.gold, args.rag_top_k, args.rag_strategies,
    )

    # Save system6-only scores
    if s6_results:
        save_s6_fields = [
            "model", "prompt_strategy", "rag_top_k", "rag_retrieval",
            "n_sequences", "n_non_empty", "rouge_l_recall", "bertscore_recall",
        ]
        Path(SCORES_CSV).parent.mkdir(parents=True, exist_ok=True)
        with open(SCORES_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=save_s6_fields)
            writer.writeheader()
            for r in s6_results:
                writer.writerow({k: r.get(k) for k in save_s6_fields})
        logger.info(f"  System 6 scores → {SCORES_CSV}")

    # ── Phase 3: Compare with System 5 ───────────────────────────
    logger.info(
        f"\n{'='*60}\n"
        f"PHASE 3 — Comparison with System 5\n"
        f"{'='*60}"
    )

    s5_results = load_system5_scores()
    all_results = s5_results + s6_results

    if all_results:
        print_comparison_table(all_results)
        save_comparison_csv(all_results, COMPARISON_CSV)

    if args.score_only:
        logger.info("Score-only mode — done.")
        return

    # ── Phase 4: Top-N test runs ──────────────────────────────────
    if s6_results:
        run_top_n_test(s6_results, args.top_n, args.embedding_backend, extra_args)

    logger.info(
        f"\n{'='*60}\n"
        f"All done.\n"
        f"  System 6 scores:  {SCORES_CSV}\n"
        f"  Comparison:       {COMPARISON_CSV}\n"
        f"  Submissions:      {OUTPUT_DIR}/<model>/<strategy>/submission/\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
