"""
Stratified train/test split for CLPsych 2026.

Ensures both splits have representative coverage of:
  - ABCD subelement types, especially rare subcategories
  - Switch and Escalation events
  - Adaptive and maladaptive presence levels

Splits at the timeline level to prevent data leakage.

Stratification logic
--------------------
Each timeline is assigned a bucket label based on which specific
subelements appear in it (not just which elements). The 12 rarest
subelements (n<15 in the full dataset) drive the bucket — timelines
that share rare subcategories cluster together and are dealt
proportionately into each split. This is more informative than
binning on switch/escalation rates alone, because subelement
imbalance is the primary challenge for Task 1.

Timelines are then dealt one-by-one to whichever split is furthest
below its proportional target, which handles tiny buckets correctly
where floor(n * ratio) breaks down.
"""

import json
import glob
import logging
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────

ELEMENTS = ["A", "B-O", "B-S", "C-O", "C-S", "D"]
VALENCES = ["adaptive-state", "maladaptive-state"]

# Subelements with fewer than ~15 occurrences in the full training set.
# Stratification prioritises keeping these rare categories balanced.
_RARE_SUBELEMENTS = {
    "adaptive-state:A:1",     # Calm/laid back           n=1
    "adaptive-state:A:7",     # Vigor/energetic           n=1
    "adaptive-state:A:9",     # Justifiable anger         n=3
    "adaptive-state:A:13",    # Feel loved/belong         n=1
    "adaptive-state:B-O:3",   # Autonomous control        n=9
    "adaptive-state:C-O:3",   # Facilitating autonomy     n=2
    "maladaptive-state:A:6",  # Mania                     n=1
    "maladaptive-state:A:8",  # Apathic                   n=7
    "maladaptive-state:A:10", # Angry/aggression          n=11
    "maladaptive-state:B-O:4",# Over-controlling          n=6
    "maladaptive-state:C-O:4",# Blocking autonomy needs   n=12
    "maladaptive-state:D:4",  # Autonomy needs unmet      n=10
}


# ── Feature extraction per timeline ──────────────────────────────

def _parse_subelement_num(category_str: str) -> int:
    """Extract the leading integer from e.g. '(4) Depressed, despair, hopeless' → 4."""
    try:
        return int(category_str.split(")")[0].strip("(").strip())
    except (ValueError, IndexError):
        return 0


def _timeline_features(data: dict) -> dict:
    """Extract features used for stratified splitting and split statistics."""
    posts = data.get("posts", [])
    n_posts = len(posts)
    n_switch = sum(1 for p in posts if p.get("Switch") == "S")
    n_escalation = sum(1 for p in posts if p.get("Escalation") == "E")

    n_annotated = 0
    ada_elements = Counter()
    mal_elements = Counter()
    subelement_profile: set[str] = set()

    for p in posts:
        ev  = p.get("evidence", {})
        ada = ev.get("adaptive-state", {})
        mal = ev.get("maladaptive-state", {})

        ada_pres = ada.get("Presence")
        mal_pres = mal.get("Presence")

        if ada_pres is not None or mal_pres is not None:
            has_content = any(
                k in ada and isinstance(ada[k], dict) for k in ELEMENTS
            ) or any(
                k in mal and isinstance(mal[k], dict) for k in ELEMENTS
            )
            if has_content or (ada_pres and ada_pres > 1) or (mal_pres and mal_pres > 1):
                n_annotated += 1

        for valence, state in [("adaptive-state", ada), ("maladaptive-state", mal)]:
            if not state.get("Presence"):
                continue
            for elem in ELEMENTS:
                if elem in state and isinstance(state[elem], dict):
                    ada_elements[elem] += 1 if valence == "adaptive-state" else 0
                    mal_elements[elem] += 1 if valence == "maladaptive-state" else 0
                    sub = _parse_subelement_num(state[elem].get("Category", ""))
                    subelement_profile.add(f"{valence}:{elem}:{sub}")

    # Bucket label: rare subelements present + dominant valence
    ada_count = sum(1 for s in subelement_profile if s.startswith("adaptive"))
    mal_count = sum(1 for s in subelement_profile if s.startswith("maladaptive"))
    dominant  = "ada" if ada_count >= mal_count else "mal"
    rare_present = tuple(sorted(
        s.split(":")[-1] for s in _RARE_SUBELEMENTS if s in subelement_profile
    ))
    strat_key = f"rare_{rare_present}_{dominant}" if rare_present else f"common_{dominant}"

    return {
        "n_posts": n_posts,
        "n_annotated": n_annotated,
        "n_switch": n_switch,
        "n_escalation": n_escalation,
        "ada_elements": dict(ada_elements),
        "mal_elements": dict(mal_elements),
        "strat_key": strat_key,
    }


def stratified_split(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.6,
    seed: int = 42,
) -> dict:
    """
    Split timeline JSON files into train/test directories.

    Uses subelement-aware stratified assignment to ensure both splits
    have representative coverage of rare ABCD subcategories as well as
    Switch/Escalation events.

    Args:
        data_dir:    Directory containing timeline JSON files.
        output_dir:  Directory to write train/ and test/ subdirs.
        train_ratio: Fraction of timelines for training (default 0.6).
        seed:        Random seed for reproducibility.

    Returns:
        Dict with split statistics.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON files in {data_dir}")

    rng = random.Random(seed)

    # Load and featurize each timeline
    timelines = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        feat = _timeline_features(data)
        timelines.append({
            "file": f,
            "timeline_id": data.get("timeline_id", Path(f).stem),
            "data": data,
            **feat,
        })

    n = len(timelines)
    n_train = max(1, round(n * train_ratio))
    n_test  = n - n_train

    # Group timelines by stratification key, shuffle within each bucket
    groups: dict[str, list[int]] = defaultdict(list)
    for i, tl in enumerate(timelines):
        groups[tl["strat_key"]].append(i)

    # Flatten into one ordered list (sorted by bucket so similar timelines
    # are adjacent), then deal one-by-one to whichever split is furthest
    # below its proportional target. This handles tiny buckets correctly
    # where floor(n * ratio) would always round to 1 regardless of ratio.
    ordered: list[int] = []
    for key in sorted(groups.keys()):
        indices = groups[key][:]
        rng.shuffle(indices)
        ordered.extend(indices)

    train_idx: list[int] = []
    test_idx:  list[int] = []
    for idx in ordered:
        train_deficit = n_train - len(train_idx)
        test_deficit  = n_test  - len(test_idx)
        if train_deficit / n_train >= test_deficit / n_test:
            train_idx.append(idx)
        else:
            test_idx.append(idx)

    # Write files
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for idx in train_idx:
        src = timelines[idx]["file"]
        shutil.copy2(src, os.path.join(train_dir, os.path.basename(src)))
    for idx in test_idx:
        src = timelines[idx]["file"]
        shutil.copy2(src, os.path.join(test_dir, os.path.basename(src)))

    # Compute split statistics
    def _split_stats(indices):
        sw = sum(timelines[i]["n_switch"] for i in indices)
        esc = sum(timelines[i]["n_escalation"] for i in indices)
        posts = sum(timelines[i]["n_posts"] for i in indices)
        ann = sum(timelines[i]["n_annotated"] for i in indices)
        ada_elem = Counter()
        mal_elem = Counter()
        for i in indices:
            for k, v in timelines[i]["ada_elements"].items():
                ada_elem[k] += v
            for k, v in timelines[i]["mal_elements"].items():
                mal_elem[k] += v
        return {
            "n_timelines": len(indices),
            "n_posts": posts,
            "n_annotated": ann,
            "n_switches": sw,
            "n_escalations": esc,
            "ada_elements": dict(ada_elem),
            "mal_elements": dict(mal_elem),
        }

    train_stats = _split_stats(train_idx)
    test_stats = _split_stats(test_idx)

    result = {
        "train": train_stats,
        "test": test_stats,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "train_ids": [timelines[i]["timeline_id"] for i in sorted(train_idx)],
        "test_ids": [timelines[i]["timeline_id"] for i in sorted(test_idx)],
    }

    # Log coverage check
    logger.info(
        f"Split: {train_stats['n_timelines']} train / "
        f"{test_stats['n_timelines']} test timelines"
    )
    logger.info(
        f"  Train: {train_stats['n_posts']} posts, "
        f"{train_stats['n_switches']}S, {train_stats['n_escalations']}E"
    )
    logger.info(
        f"  Test:  {test_stats['n_posts']} posts, "
        f"{test_stats['n_switches']}S, {test_stats['n_escalations']}E"
    )

    # Warn if any element type is missing from a split
    for elem in ELEMENTS:
        for side, label in [("ada_elements", "adaptive"), ("mal_elements", "maladaptive")]:
            train_count = train_stats[side].get(elem, 0)
            test_count = test_stats[side].get(elem, 0)
            if train_count == 0:
                logger.warning(
                    f"  ⚠ {label} {elem} has 0 examples in train split"
                )
            if test_count == 0:
                logger.warning(
                    f"  ⚠ {label} {elem} has 0 examples in test split"
                )

    return result


def print_split_report(result: dict):
    """Pretty-print split statistics."""
    train_tl = result["train"]["n_timelines"]
    test_tl  = result["test"]["n_timelines"]
    total_tl = train_tl + test_tl
    ratio_pct = round(100 * train_tl / total_tl) if total_tl else 0
    print("\n" + "=" * 65)
    print(f"Data Split Report ({ratio_pct}/{100-ratio_pct} stratified)")
    print("=" * 65)

    for split_name in ["train", "test"]:
        s = result[split_name]
        print(f"\n  {split_name.upper()} ({s['n_timelines']} timelines):")
        print(f"    Posts: {s['n_posts']}  Annotated: {s['n_annotated']}")
        print(f"    Switches: {s['n_switches']}  Escalations: {s['n_escalations']}")
        print(f"    Adaptive elements:    {s['ada_elements']}")
        print(f"    Maladaptive elements: {s['mal_elements']}")

    print("\n" + "=" * 65)


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Stratified 60/40 split for CLPsych 2026 training data"
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing timeline JSON files",
    )
    parser.add_argument(
        "--output-dir", default="data/split",
        help="Output directory for train/ and test/ (default: data/split)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7,
        help="Fraction for training (default: 0.7)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = stratified_split(
        args.data_dir, args.output_dir,
        train_ratio=args.train_ratio, seed=args.seed,
    )
    print_split_report(result)

    # Save split metadata
    meta_path = os.path.join(args.output_dir, "split_meta.json")
    with open(meta_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSplit metadata saved to {meta_path}")