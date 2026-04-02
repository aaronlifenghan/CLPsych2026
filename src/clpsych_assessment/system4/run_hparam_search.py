"""
Hyperparameter search for System 4 XGBoost models.
Tries combinations of n_estimators, max_depth, learning_rate, subsample.

Usage (from src/ directory):
    python ../run_hparam_search.py \
        --data-dir ../data/train_tasks12 \
        --embed-backend combined \
        --embed-model all-mpnet-base-v2 \
        --output-dir outputs/system4_hparam
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
from itertools import product

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Search grid — kept small to be practical on CPU
HPARAM_GRID = {
    "n_estimators":  [200, 400],
    "max_depth":     [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample":     [0.8],
    "colsample_bytree": [0.8],
}


def make_split(all_files, seed, train_ratio=0.7):
    rng = random.Random(seed)
    files = all_files[:]
    rng.shuffle(files)
    n_train = int(len(files) * train_ratio)
    return files[:n_train], files[n_train:]


def run_with_params(train_dir, test_dir, output_dir, embed_backend, embed_model,
                    max_features, xgb_params, unlabeled_dir=None):
    """Monkey-patch XGBoost params and run pipeline."""
    src_dir = os.path.abspath(os.path.dirname(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    import importlib
    import clpsych_assessment.system4.task1 as t1_mod
    import clpsych_assessment.system4.task2 as t2_mod

    # Patch default params
    orig_t1_1 = t1_mod.Task1_1Model.__init__
    orig_t1_2 = t1_mod.Task1_2Model.__init__
    orig_t2   = t2_mod.Task2Model.__init__

    def patched_t1_1(self, xgb_params_arg=None):
        orig_t1_1(self, xgb_params_arg or {**xgb_params,
            "eval_metric": "mlogloss", "random_state": 42, "n_jobs": -1})

    def patched_t1_2(self, xgb_params_arg=None):
        orig_t1_2(self, xgb_params_arg or {**xgb_params,
            "eval_metric": "mlogloss", "random_state": 42, "n_jobs": -1})

    def patched_t2(self, xgb_params_arg=None):
        orig_t2(self, xgb_params_arg or {**xgb_params,
            "eval_metric": "logloss", "random_state": 42, "n_jobs": -1})

    t1_mod.Task1_1Model.__init__ = patched_t1_1
    t1_mod.Task1_2Model.__init__ = patched_t1_2
    t2_mod.Task2Model.__init__   = patched_t2

    try:
        from clpsych_assessment.system4.run import train_and_evaluate
        results = train_and_evaluate(
            train_dir=train_dir, test_dir=test_dir, output_dir=output_dir,
            embed_backend=embed_backend, embed_model=embed_model,
            max_features=max_features, unlabeled_dir=unlabeled_dir,
        )
    finally:
        t1_mod.Task1_1Model.__init__ = orig_t1_1
        t1_mod.Task1_2Model.__init__ = orig_t1_2
        t2_mod.Task2Model.__init__   = orig_t2

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",      required=True)
    parser.add_argument("--embed-backend", default="combined",
                        choices=["tfidf", "sbert", "combined"])
    parser.add_argument("--embed-model",   default="all-mpnet-base-v2")
    parser.add_argument("--max-features",  type=int, default=2048)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--unlabeled-dir", default=None)
    parser.add_argument("--output-dir",    default="outputs/system4_hparam")
    parser.add_argument("--folds",         type=int, default=3,
                        help="CV folds per config (3 is fast enough)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    if not all_files:
        raise FileNotFoundError(f"No JSON files in {args.data_dir}")

    rng = random.Random(args.seed)
    shuffled = all_files[:]
    rng.shuffle(shuffled)

    # Build all param combos
    keys = list(HPARAM_GRID.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*HPARAM_GRID.values())]
    logger.info(f"Testing {len(combos)} hyperparameter combinations × {args.folds} folds")

    best_score = {"t1_1": 0, "t1_2": 999, "t2": 0, "combined": -999}
    best_params = None
    all_results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for ci, params in enumerate(combos):
            logger.info(f"\n[{ci+1}/{len(combos)}] params={params}")
            fold_t11, fold_t12, fold_t2 = [], [], []

            for fold in range(args.folds):
                fold_size = len(shuffled) // args.folds
                start = fold * fold_size
                end = start + fold_size if fold < args.folds - 1 else len(shuffled)
                test_f = shuffled[start:end]
                train_f = shuffled[:start] + shuffled[end:]

                train_dir = os.path.join(tmpdir, f"c{ci}_f{fold}_train")
                test_dir  = os.path.join(tmpdir, f"c{ci}_f{fold}_test")
                os.makedirs(train_dir, exist_ok=True)
                os.makedirs(test_dir,  exist_ok=True)
                for f in train_f:
                    shutil.copy2(f, os.path.join(train_dir, os.path.basename(f)))
                for f in test_f:
                    shutil.copy2(f, os.path.join(test_dir, os.path.basename(f)))

                fold_out = os.path.join(tmpdir, f"c{ci}_f{fold}_out")
                results = run_with_params(
                    train_dir, test_dir, fold_out,
                    args.embed_backend, args.embed_model, args.max_features,
                    params, args.unlabeled_dir,
                )
                fold_t11.append(results.get("task1_1", {}).get("t1_1_rank", 0))
                fold_t12.append(results.get("task1_2", {}).get("t1_2_rank", 999))
                fold_t2.append(results.get("task2",   {}).get("t2_rank",   0))

            avg_t11 = np.mean(fold_t11)
            avg_t12 = np.mean(fold_t12)
            avg_t2  = np.mean(fold_t2)
            # Combined score: normalize so higher = better
            combined = avg_t11 + (2.0 - avg_t12) + avg_t2

            logger.info(f"  → T1.1={avg_t11:.4f}  T1.2={avg_t12:.4f}  T2={avg_t2:.4f}  combined={combined:.4f}")

            all_results.append({
                "params": params, "t1_1": avg_t11, "t1_2": avg_t12,
                "t2": avg_t2, "combined": combined,
            })

            if combined > best_score["combined"]:
                best_score = {"t1_1": avg_t11, "t1_2": avg_t12, "t2": avg_t2, "combined": combined}
                best_params = params

    # Sort and print results
    all_results.sort(key=lambda x: -x["combined"])
    print(f"\n{'='*65}")
    print(f"Hyperparameter Search Results (top 5)")
    print(f"{'='*65}")
    for r in all_results[:5]:
        print(f"  T1.1={r['t1_1']:.4f}  T1.2={r['t1_2']:.4f}  T2={r['t2']:.4f}  combined={r['combined']:.4f}")
        print(f"    params: {r['params']}")

    print(f"\n★ Best params: {best_params}")
    print(f"  T1.1={best_score['t1_1']:.4f}  T1.2={best_score['t1_2']:.4f}  T2={best_score['t2']:.4f}")

    out_path = os.path.join(args.output_dir, "hparam_results.json")
    with open(out_path, "w") as f:
        json.dump({"best_params": best_params, "best_score": best_score,
                   "all_results": all_results}, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    print(f"\nTo retrain with best params, add to task1.py/task2.py xgb_params:")
    for k, v in best_params.items():
        print(f"  '{k}': {v},")


if __name__ == "__main__":
    main()
