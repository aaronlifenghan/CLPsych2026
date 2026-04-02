"""
CLPsych 2026 — System 4: Embedding + XGBoost Pipeline

Full pipeline: data split → embed → train → predict → evaluate.

Usage:
    # Train + evaluate on pre-split data
    python -m clpsych_assessment.system4.run \\
        --train-dir data/split/train \\
        --test-dir  data/split/test \\
        --output-dir outputs/system4

    # Split first, then train + evaluate
    python -m clpsych_assessment.system4.run \\
        --data-dir data/train_tasks12 \\
        --output-dir outputs/system4

    # Use sentence-transformer embeddings instead of TF-IDF
    python -m clpsych_assessment.system4.run \\
        --train-dir data/split/train \\
        --test-dir  data/split/test \\
        --embed-backend sbert \\
        --output-dir outputs/system4
"""

import argparse
import glob
import json
import logging
import os
import sys

import numpy as np

from .config import ELEMENTS, VALENCE_SHORT, VALENCES
from .embeddings import create_embedder
from .task1 import Task1_1Model, Task1_2Model, extract_task1_labels, format_task1_predictions
from .task2 import Task2Model, build_temporal_features, extract_task2_labels, format_task2_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ── Data loading ─────────────────────────────────────────────────

def load_timelines(data_dir: str) -> list[dict]:
    """Load all timeline JSON files from a directory."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON files in {data_dir}")

    timelines = []
    for f in files:
        with open(f) as fh:
            tl = json.load(fh)
        tid = tl.get("timeline_id", os.path.splitext(os.path.basename(f))[0])
        for p in tl.get("posts", []):
            p["timeline_id"] = tid
        timelines.append(tl)

    return timelines


def _has_evidence(post: dict) -> bool:
    """Check if a post has annotated evidence for Task 1."""
    ev = post.get("evidence", {})
    ada = ev.get("adaptive-state", {}).get("Presence")
    mal = ev.get("maladaptive-state", {}).get("Presence")
    return ada is not None or mal is not None


def extract_posts(timelines: list[dict], task1_only: bool = False) -> list[dict]:
    """Flatten timelines into posts, optionally filtering to Task-1-eligible ones."""
    posts = []
    for tl in timelines:
        tid = tl.get("timeline_id", "")
        for p in tl.get("posts", []):
            p["timeline_id"] = tid
            if task1_only and not _has_evidence(p):
                continue
            posts.append(p)
    return posts


# ── Pipeline steps ───────────────────────────────────────────────

def train_and_evaluate(
    train_dir: str,
    test_dir: str,
    output_dir: str,
    embed_backend: str = "tfidf",
    embed_model: str = "all-MiniLM-L6-v2",
    max_features: int = 2048,
    unlabeled_dir: str = None,
):
    """
    Full pipeline: embed → train → predict → evaluate.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # ── Load data ──
    logger.info(f"Loading training data from {train_dir}")
    train_timelines = load_timelines(train_dir)
    logger.info(f"Loading test data from {test_dir}")
    test_timelines = load_timelines(test_dir)

    train_posts_all = extract_posts(train_timelines)
    test_posts_all = extract_posts(test_timelines)
    train_posts_t1 = extract_posts(train_timelines, task1_only=True)
    test_posts_t1 = extract_posts(test_timelines, task1_only=True)

    # If test set has no annotated posts (unlabeled), predict on all posts
    if not test_posts_t1:
        logger.info("  No evidence in test set — predicting Task 1 on all test posts")
        test_posts_t1 = test_posts_all

    logger.info(
        f"Train: {len(train_timelines)} timelines, "
        f"{len(train_posts_all)} posts ({len(train_posts_t1)} with evidence)"
    )
    logger.info(
        f"Test:  {len(test_timelines)} timelines, "
        f"{len(test_posts_all)} posts ({len(test_posts_t1)} with evidence)"
    )

    # ── Extract texts ──
    train_texts_t1 = [p.get("post", "") for p in train_posts_t1]
    test_texts_t1 = [p.get("post", "") for p in test_posts_t1]
    all_texts_t2 = [p.get("post", "") for p in train_posts_all + test_posts_all]

    # Expand TF-IDF vocabulary with unlabeled test texts if provided
    vocab_texts_t1 = list(train_texts_t1)
    vocab_texts_t2 = list(all_texts_t2)
    if unlabeled_dir and os.path.isdir(unlabeled_dir):
        try:
            unlabeled_timelines = load_timelines(unlabeled_dir)
            unlabeled_texts = [p.get("post", "") for tl in unlabeled_timelines
                               for p in tl.get("posts", [])]
            vocab_texts_t1 = vocab_texts_t1 + unlabeled_texts
            vocab_texts_t2 = vocab_texts_t2 + unlabeled_texts
            logger.info(f"  Vocab expansion: +{len(unlabeled_texts)} unlabeled texts")
        except Exception as e:
            logger.warning(f"  Could not load unlabeled dir: {e}")

    # ── Embeddings ──
    logger.info("═" * 50)
    logger.info("Step 1: Computing embeddings")
    logger.info("═" * 50)

    # Task 1 embedder (fit on task1 train texts)
    embedder_t1 = create_embedder(
        backend=embed_backend,
        model_name=embed_model,
        max_features=max_features,
    )
    embedder_t1.fit(vocab_texts_t1)
    X_train_t1 = embedder_t1.transform(train_texts_t1)
    X_test_t1 = embedder_t1.transform(test_texts_t1)
    embedder_t1.save(os.path.join(model_dir, "embedder_t1.pkl"))
    logger.info(f"  Task 1 embeddings: train={X_train_t1.shape}, test={X_test_t1.shape}")

    # Task 2 embedder (fit on ALL texts for better vocabulary)
    embedder_t2 = create_embedder(
        backend=embed_backend,
        model_name=embed_model,
        max_features=max_features,
    )
    embedder_t2.fit(vocab_texts_t2)
    embedder_t2.save(os.path.join(model_dir, "embedder_t2.pkl"))

    # ── Task 1.1: Subelement classification ──
    logger.info("═" * 50)
    logger.info("Step 2: Training Task 1.1 (subelement classification)")
    logger.info("═" * 50)

    train_t1_labels = [extract_task1_labels(p) for p in train_posts_t1]
    train_sub_labels = [l["subelement_labels"] for l in train_t1_labels]
    train_pres_ratings = [l["presence_ratings"] for l in train_t1_labels]

    model_t1_1 = Task1_1Model()
    model_t1_1.fit(X_train_t1, train_sub_labels)
    model_t1_1.save(model_dir)

    # ── Task 1.2: Presence rating ──
    logger.info("═" * 50)
    logger.info("Step 3: Training Task 1.2 (presence rating)")
    logger.info("═" * 50)

    model_t1_2 = Task1_2Model()
    model_t1_2.fit(X_train_t1, train_pres_ratings)
    model_t1_2.save(model_dir)

    # ── Task 2: Moments of change ──
    logger.info("═" * 50)
    logger.info("Step 4: Training Task 2 (moments of change)")
    logger.info("═" * 50)

    # Build temporal features per timeline
    train_tl_embeddings = []
    train_tl_texts = []
    train_t2_labels = []
    for tl in train_timelines:
        posts = tl.get("posts", [])
        texts = [p.get("post", "") for p in posts]
        if not texts:
            continue
        emb = embedder_t2.transform(texts)
        train_tl_embeddings.append(emb)
        train_tl_texts.append(texts)
        for p in posts:
            train_t2_labels.append(extract_task2_labels(p))

    X_train_t2 = build_temporal_features(train_tl_embeddings, train_tl_texts)
    logger.info(f"  Temporal features: {X_train_t2.shape}")

    model_t2 = Task2Model()
    model_t2.fit(X_train_t2, train_t2_labels)
    model_t2.save(model_dir)

    # ── Predict on test set ──
    logger.info("═" * 50)
    logger.info("Step 5: Generating predictions on test set")
    logger.info("═" * 50)

    # Task 1 predictions
    sub_preds = model_t1_1.predict(X_test_t1)
    pres_preds = model_t1_2.predict(X_test_t1)

    test_pids_t1 = [p["post_id"] for p in test_posts_t1]
    test_tids_t1 = [p["timeline_id"] for p in test_posts_t1]

    t1_entries = format_task1_predictions(
        test_pids_t1, test_tids_t1, sub_preds, pres_preds
    )

    t1_path = os.path.join(output_dir, "task1_pred.json")
    with open(t1_path, "w") as f:
        json.dump(t1_entries, f, indent=2)
    logger.info(f"  Task 1: {len(t1_entries)} entries → {t1_path}")

    # Task 2 predictions
    test_tl_embeddings = []
    test_tl_texts = []
    test_t2_pids = []
    test_t2_tids = []
    for tl in test_timelines:
        posts = tl.get("posts", [])
        texts = [p.get("post", "") for p in posts]
        if not texts:
            continue
        emb = embedder_t2.transform(texts)
        test_tl_embeddings.append(emb)
        test_tl_texts.append(texts)
        for p in posts:
            test_t2_pids.append(p["post_id"])
            test_t2_tids.append(p.get("timeline_id", tl.get("timeline_id", "")))

    X_test_t2 = build_temporal_features(test_tl_embeddings, test_tl_texts)
    t2_preds = model_t2.predict(X_test_t2)

    t2_entries = format_task2_predictions(test_t2_pids, test_t2_tids, t2_preds)

    t2_path = os.path.join(output_dir, "task2_pred.json")
    with open(t2_path, "w") as f:
        json.dump(t2_entries, f, indent=2)
    logger.info(f"  Task 2: {len(t2_entries)} entries → {t2_path}")

    # ── Evaluate ──
    logger.info("═" * 50)
    logger.info("Step 6: Evaluating predictions")
    logger.info("═" * 50)

    # Import evaluation directly from the file to avoid system3.__init__.py
    # which imports pipeline.py → pydantic (not needed for evaluation).
    import importlib.util
    _eval_path = os.path.join(
        os.path.dirname(__file__), "..", "system3", "evaluate.py"
    )
    _spec = importlib.util.spec_from_file_location("_sys3_evaluate", _eval_path)
    _eval_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_eval_mod)
    run_full_evaluation = _eval_mod.run_full_evaluation
    print_evaluation = _eval_mod.print_evaluation

    results = run_full_evaluation(
        gold_dir=test_dir,
        task1_pred_path=t1_path,
        task2_pred_path=t2_path,
    )
    print_evaluation(results)

    # Save results
    eval_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {eval_path}")

    return results


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="System 4: Embedding + XGBoost pipeline for CLPsych 2026"
    )

    # Data arguments
    group = parser.add_argument_group("data")
    group.add_argument(
        "--data-dir", default=None,
        help="Raw data directory (will be split automatically)",
    )
    group.add_argument(
        "--train-dir", default=None,
        help="Pre-split training data directory",
    )
    group.add_argument(
        "--test-dir", default=None,
        help="Pre-split test data directory",
    )
    group.add_argument(
        "--train-ratio", type=float, default=0.7,
        help="Train ratio if splitting (default: 0.7)",
    )
    group.add_argument("--seed", type=int, default=42)

    # Embedding arguments
    group = parser.add_argument_group("embeddings")
    group.add_argument(
        "--embed-backend", default="tfidf", choices=["tfidf", "sbert", "combined"],
        help="Embedding backend (default: tfidf)",
    )
    group.add_argument(
        "--embed-model", default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (only for sbert)",
    )
    group.add_argument(
        "--max-features", type=int, default=2048,
        help="TF-IDF vocabulary size (default: 2048)",
    )
    group.add_argument(
        "--unlabeled-dir", default=None,
        help="Directory of unlabeled timelines for TF-IDF vocab expansion",
    )

    # Output
    group = parser.add_argument_group("output")
    group.add_argument(
        "--output-dir", default="outputs/system4",
        help="Output directory (default: outputs/system4)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.data_dir is None and (args.train_dir is None or args.test_dir is None):
        parser.error("Provide --data-dir (to auto-split) OR both --train-dir and --test-dir")

    # If raw data dir given, split first using system3's data_split
    if args.data_dir:
        logger.info("Splitting data using stratified split...")
        import importlib.util
        _ds_path = os.path.join(
            os.path.dirname(__file__), "..", "system3", "data_split.py"
        )
        _spec = importlib.util.spec_from_file_location("_sys3_ds", _ds_path)
        _ds_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_ds_mod)
        stratified_split = _ds_mod.stratified_split
        print_split_report = _ds_mod.print_split_report

        split_dir = os.path.join(args.output_dir, "split")
        result = stratified_split(
            args.data_dir, split_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
        print_split_report(result)
        train_dir = result["train_dir"]
        test_dir = result["test_dir"]
    else:
        train_dir = args.train_dir
        test_dir = args.test_dir

    # Run pipeline
    train_and_evaluate(
        train_dir=train_dir,
        test_dir=test_dir,
        output_dir=args.output_dir,
        embed_backend=args.embed_backend,
        embed_model=args.embed_model,
        max_features=args.max_features,
        unlabeled_dir=args.unlabeled_dir,
    )


if __name__ == "__main__":
    main()
