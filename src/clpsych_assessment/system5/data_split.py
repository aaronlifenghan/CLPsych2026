"""
CLPsych 2026 — System 5 — Training data splitter

Splits train_task3.json into a train fold and a validation fold,
stratified by change_type (Switch / Escalation) so both folds
contain examples of each type.

Usage:
    python -m clpsych_assessment.system5.data_split \\
        data/task3_train_n_test/train_task3.json \\
        --val-ratio 0.2 \\
        --seed 42 \\
        --output-dir data/task3_train_n_test/

Produces:
    data/task3_train_n_test/train_task3_train_fold.json
    data/task3_train_n_test/train_task3_val_fold.json
"""

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def stratified_split(
    sequences: List[Dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split sequences into train/val preserving change_type distribution.

    Sequences without a change_type field are treated as a single group.
    """
    rng = random.Random(seed)

    # Group by change_type
    groups: Dict[str, List] = defaultdict(list)
    for seq in sequences:
        key = seq.get("change_type", "unknown")
        groups[key].append(seq)

    train_fold: List[Dict] = []
    val_fold: List[Dict] = []

    for change_type, group in groups.items():
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_ratio))
        val_fold.extend(group[:n_val])
        train_fold.extend(group[n_val:])
        logger.info(
            f"  {change_type}: {len(group)} total → "
            f"{len(group) - n_val} train / {n_val} val"
        )

    # Shuffle both folds
    rng.shuffle(train_fold)
    rng.shuffle(val_fold)

    return train_fold, val_fold


def save_split(data: List[Dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved {len(data)} sequences → {path}")


# ── CLI ──────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Stratified train/val split for Task 3.1 training data"
    )
    parser.add_argument(
        "sequences_file",
        help="Path to train_task3.json",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: same directory as input file)",
    )
    args = parser.parse_args()

    with open(args.sequences_file, "r", encoding="utf-8") as f:
        sequences = json.load(f)

    logger.info(
        f"Splitting {len(sequences)} sequences "
        f"(val_ratio={args.val_ratio}, seed={args.seed})"
    )

    train_fold, val_fold = stratified_split(
        sequences, val_ratio=args.val_ratio, seed=args.seed
    )

    output_dir = args.output_dir or str(Path(args.sequences_file).parent)
    stem = Path(args.sequences_file).stem  # e.g. "train_task3"

    save_split(train_fold, os.path.join(output_dir, f"{stem}_train_fold.json"))
    save_split(val_fold,   os.path.join(output_dir, f"{stem}_val_fold.json"))

    print(
        f"\nSplit complete: {len(train_fold)} train / {len(val_fold)} val"
        f"\nOutput dir: {output_dir}"
    )


if __name__ == "__main__":
    main()
