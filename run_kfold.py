"""
K-fold cross-validation runner for CLPsych 2026 System 4.

Usage (from the src/ directory):
    python ../run_kfold.py \
        --data-dir ../data/train_tasks12 \
        --folds 5 \
        --embed-backend combined \
        --embed-model all-mpnet-base-v2 \
        --unlabeled-dir ../data/test_tasks12nolabels \
        --output-dir outputs/system4_kfold
"""

import argparse
import glob
import json
import logging
import os
import random
import shutil
import sys
import tempfile

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def make_fold_dirs(all_files, fold_idx, folds, tmpdir):
    n = len(all_files)
    fold_size = n // folds
    start = fold_idx * fold_size
    end = start + fold_size if fold_idx < folds - 1 else n

    test_files  = all_files[start:end]
    train_files = all_files[:start] + all_files[end:]

    train_dir = os.path.join(tmpdir, f"fold{fold_idx}", "train")
    test_dir  = os.path.join(tmpdir, f"fold{fold_idx}", "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    for f in train_files:
        shutil.copy2(f, os.path.join(train_dir, os.path.basename(f)))
    for f in test_files:
        shutil.copy2(f, os.path.join(test_dir, os.path.basename(f)))

    return train_dir, test_dir


def safe_mean(fold_results, *keys):
    vals = []
    for r in fold_results:
        node = r
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                node = None
                break
        if node is not None and isinstance(node, (int, float)):
            vals.append(float(node))
    return float(np.mean(vals)) if vals else 0.0


def average_results(fold_results):
    avg = {}

    sc_keys = set()
    for r in fold_results:
        sc_keys |= set(r.get("task1_1", {}).get("subelement_classification", {}).keys())

    avg["task1_1"] = {
        "t1_1_rank": safe_mean(fold_results, "task1_1", "t1_1_rank"),
        "subelement_classification": {
            slot: {
                "macro_f1": safe_mean(fold_results, "task1_1", "subelement_classification", slot, "macro_f1"),
                "micro_f1": safe_mean(fold_results, "task1_1", "subelement_classification", slot, "micro_f1"),
            }
            for slot in sorted(sc_keys)
        },
    }

    t12_keys = set()
    for r in fold_results:
        t12_keys |= set(r.get("task1_2", {}).keys()) - {"t1_2_rank"}

    avg["task1_2"] = {"t1_2_rank": safe_mean(fold_results, "task1_2", "t1_2_rank")}
    for key in sorted(t12_keys):
        avg["task1_2"][key] = {
            metric: safe_mean(fold_results, "task1_2", key, metric)
            for metric in ["mae", "rmse", "qwk", "spearman"]
        }

    t2_keys = set()
    for r in fold_results:
        t2_keys |= set(r.get("task2", {}).keys()) - {"t2_rank"}

    avg["task2"] = {"t2_rank": safe_mean(fold_results, "task2", "t2_rank")}
    for key in sorted(t2_keys):
        sample = fold_results[0].get("task2", {}).get(key)
        if isinstance(sample, dict):
            avg["task2"][key] = {
                metric: safe_mean(fold_results, "task2", key, metric)
                for metric in sample.keys()
            }
        else:
            avg["task2"][key] = safe_mean(fold_results, "task2", key)

    return avg


def print_cv_results(avg, folds):
    print(f"\n{'='*65}")
    print(f"K-Fold CV Results ({folds} folds) — Averaged")
    print(f"{'='*65}")

    print(f"\n  Task 1.1 — Subelement Classification")
    print(f"  {'-'*45}")
    ada_scores, mal_scores = [], []
    sc = avg.get("task1_1", {}).get("subelement_classification", {})
    for slot, metrics in sorted(sc.items()):
        mf1  = metrics.get("macro_f1", 0)
        mif1 = metrics.get("micro_f1", 0)
        print(f"    {slot:<28} macroF1={mf1:.4f}  microF1={mif1:.4f}")
        (ada_scores if slot.startswith("adaptive") else mal_scores).append(mf1)
    if ada_scores:
        print(f"                               → adaptive avg: {np.mean(ada_scores):.4f}")
    if mal_scores:
        print(f"                               → maladaptive avg: {np.mean(mal_scores):.4f}")
    print(f"\n  ★ Task 1.1 Ranking = {avg['task1_1']['t1_1_rank']:.4f}")

    print(f"\n  Task 1.2 — Presence Rating")
    print(f"  {'-'*45}")
    for key, metrics in sorted(avg.get("task1_2", {}).items()):
        if key == "t1_2_rank":
            continue
        mae  = metrics.get("mae", 0)
        rmse = metrics.get("rmse", 0)
        qwk  = metrics.get("qwk", 0)
        print(f"    {key:<22} MAE={mae:.3f}  RMSE={rmse:.3f}  QWK={qwk:.3f}")
    print(f"\n  ★ Task 1.2 Ranking = {avg['task1_2']['t1_2_rank']:.4f} (lower is better)")

    print(f"\n  Task 2 — Moments of Change")
    print(f"  {'-'*45}")
    for key, val in sorted(avg.get("task2", {}).items()):
        if key == "t2_rank":
            continue
        if isinstance(val, dict):
            p = val.get("precision", 0)
            r = val.get("recall", 0)
            f = val.get("f1", 0)
            print(f"    {key:<40} P={p:.3f}  R={r:.3f}  F1={f:.3f}")
        else:
            print(f"    {key}: {val:.4f}")
    print(f"\n  ★ Task 2 Ranking = {avg['task2']['t2_rank']:.4f}")
    print(f"\n{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(description="K-fold CV for CLPsych System 4")
    parser.add_argument("--data-dir",       required=True)
    parser.add_argument("--folds",          type=int, default=5)
    parser.add_argument("--embed-backend",  default="combined",
                        choices=["tfidf", "sbert", "combined"])
    parser.add_argument("--embed-model",    default="all-MiniLM-L6-v2")
    parser.add_argument("--max-features",   type=int, default=2048)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--output-dir",     default="outputs/system4_kfold")
    parser.add_argument("--unlabeled-dir",  default=None,
                        help="Unlabeled test dir for TF-IDF vocab expansion")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    if not all_files:
        raise FileNotFoundError(f"No JSON files in {args.data_dir}")

    logger.info(f"Found {len(all_files)} timelines — running {args.folds}-fold CV")

    rng = random.Random(args.seed)
    shuffled = all_files[:]
    rng.shuffle(shuffled)

    src_dir = os.path.abspath(os.path.dirname(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from clpsych_assessment.system4.run import train_and_evaluate

    fold_results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for fold in range(args.folds):
            logger.info(f"\n{'─'*50}")
            logger.info(f"FOLD {fold+1}/{args.folds}")
            logger.info(f"{'─'*50}")

            train_dir, test_dir = make_fold_dirs(shuffled, fold, args.folds, tmpdir)
            fold_output = os.path.join(args.output_dir, f"fold{fold+1}")

            n_train = len(glob.glob(os.path.join(train_dir, "*.json")))
            n_test  = len(glob.glob(os.path.join(test_dir,  "*.json")))
            logger.info(f"  Train: {n_train} timelines  |  Test: {n_test} timelines")

            results = train_and_evaluate(
                train_dir=train_dir,
                test_dir=test_dir,
                output_dir=fold_output,
                embed_backend=args.embed_backend,
                embed_model=args.embed_model,
                max_features=args.max_features,
                unlabeled_dir=args.unlabeled_dir,
            )
            fold_results.append(results)

            t11 = results.get("task1_1", {}).get("t1_1_rank", 0)
            t12 = results.get("task1_2", {}).get("t1_2_rank", 0)
            t2  = results.get("task2",   {}).get("t2_rank",   0)
            logger.info(f"  Fold {fold+1} done → T1.1={t11:.4f}  T1.2={t12:.4f}  T2={t2:.4f}")

    avg = average_results(fold_results)
    print_cv_results(avg, folds=args.folds)

    out_path = os.path.join(args.output_dir, "cv_results.json")
    with open(out_path, "w") as f:
        json.dump({"folds": args.folds, "average": avg, "per_fold": fold_results}, f, indent=2)
    logger.info(f"CV results saved to {out_path}")


if __name__ == "__main__":
    main()
