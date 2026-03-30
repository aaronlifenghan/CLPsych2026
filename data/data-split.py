"""
CLPsych 2026 – Task 1 Train/Val Splits
=======================================
Produces two things:
  1. A single stratified 70/30 train/val split
  2. K-fold cross-validation splits (default k=5)

Key design decisions
--------------------
* Unit of splitting = TIMELINE (not post).
  Posts from the same person must stay together — otherwise the model
  sees future posts from the same user at train time and you get leakage.

* Stratification label = subelement-aware profile per timeline.
  For each timeline we record which specific subelements (not just elements)
  appear. We then focus the bucket label on RARE subelements (those with
  fewer than ~15 occurrences in the full dataset), because these are the
  hardest to keep balanced. Timelines that share the same rare subcategories
  cluster together and are dealt proportionately into each split.

* K-fold uses the same bucket logic plus a round-robin deal that keeps
  fold sizes as equal as possible (fixes the old 13/13/5/2/2 imbalance).

* Encoder-ready output: each example has text, post_id, timeline_id,
  and a flat labels dict suitable for a Sentence-BERT multi-label head.

Usage
-----
  python split_task1.py --data-dir train_tasks12/ --output-dir splits/
  python split_task1.py --data-dir train_tasks12/ --output-dir splits/ --k 5 --seed 42
"""

import argparse
import json
import glob
import os
import random
from collections import defaultdict, Counter


# ── constants ──────────────────────────────────────────────────────────────

ELEMENTS = ["A", "B-O", "B-S", "C-O", "C-S", "D"]
VALENCES = ["adaptive-state", "maladaptive-state"]
ALL_SLOTS = [f"{v}:{e}" for v in VALENCES for e in ELEMENTS]

# Subelements with fewer than ~15 occurrences in the full training set.
# Stratification prioritises keeping these rare categories balanced.
RARE_SUBELEMENTS = {
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


# ── data loading ───────────────────────────────────────────────────────────

def load_data(data_dir):
    """
    Returns:
      timelines   – list of raw timeline dicts
      task1_posts – flat list of posts that have gold evidence (Task 1 only)
    """
    timelines = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        with open(path) as f:
            tl = json.load(f)
        for p in tl["posts"]:
            p["timeline_id"] = tl["timeline_id"]
        timelines.append(tl)

    def has_evidence(post):
        ev = post.get("evidence", {})
        return (ev.get("adaptive-state", {}).get("Presence") or
                ev.get("maladaptive-state", {}).get("Presence"))

    task1_posts = [p for tl in timelines for p in tl["posts"] if has_evidence(p)]
    return timelines, task1_posts


# ── stratification ─────────────────────────────────────────────────────────

def parse_subelement_num(category_str):
    """Extract the leading integer from e.g. '(4) Depressed, despair, hopeless' → 4."""
    try:
        return int(category_str.split(")")[0].strip("(").strip())
    except (ValueError, IndexError):
        return 0


def timeline_subelement_profile(tl_posts):
    """
    Return a frozenset of 'valence:element:subelement_num' strings that
    appear in *any* post of this timeline.

    Using specific subelement numbers (not just element presence) means rare
    subcategories like Mania or Feel-loved are represented in the signal.
    """
    active = set()
    for p in tl_posts:
        ev = p.get("evidence", {})
        for valence in VALENCES:
            state = ev.get(valence, {})
            if not state.get("Presence"):
                continue
            for elem in ELEMENTS:
                if elem in state:
                    sub = parse_subelement_num(state[elem].get("Category", ""))
                    active.add(f"{valence}:{elem}:{sub}")
    return frozenset(active)


def coarse_bucket(profile):
    """
    Map a subelement profile to a stratification bucket.

    Primary signal: which rare subelements appear (sorted tuple).
    Tiebreaker: dominant valence (adaptive vs maladaptive element count).

    Timelines with no rare subelements share a single 'common_ada' or
    'common_mal' bucket and are dealt evenly across splits from there.
    """
    ada_count = sum(1 for s in profile if s.startswith("adaptive"))
    mal_count = sum(1 for s in profile if s.startswith("maladaptive"))
    dominant  = "ada" if ada_count >= mal_count else "mal"

    rare_present = tuple(sorted(s.split(":")[-1] for s in RARE_SUBELEMENTS if s in profile))
    if rare_present:
        return f"rare_{rare_present}_{dominant}"
    return f"common_{dominant}"


def build_stratification(timelines, task1_posts):
    """Returns {timeline_id: bucket_label}."""
    by_tl = defaultdict(list)
    for p in task1_posts:
        by_tl[p["timeline_id"]].append(p)

    tl_bucket = {}
    for tl in timelines:
        tid   = tl["timeline_id"]
        posts = by_tl.get(tid, [])
        if not posts:
            tl_bucket[tid] = "no_evidence"
            continue
        profile = timeline_subelement_profile(posts)
        tl_bucket[tid] = coarse_bucket(profile)
    return tl_bucket


# ── splitting helpers ──────────────────────────────────────────────────────

def group_by_bucket(timeline_ids, tl_bucket):
    groups = defaultdict(list)
    for tid in timeline_ids:
        groups[tl_bucket[tid]].append(tid)
    return groups


def stratified_split(timeline_ids, tl_bucket, train_ratio=0.7, seed=42):
    """
    Proportionate stratified split over timelines.

    Timelines are sorted by bucket (so similar ones are adjacent), then dealt
    one-by-one to whichever split is furthest below its proportional target.
    This handles tiny buckets correctly where floor(n * ratio) breaks down.
    """
    rng = random.Random(seed)

    groups  = group_by_bucket(timeline_ids, tl_bucket)
    ordered = []
    for bucket in sorted(groups):
        tids = groups[bucket][:]
        rng.shuffle(tids)
        ordered.extend(tids)

    n_total = len(ordered)
    n_train = round(n_total * train_ratio)
    n_val   = n_total - n_train

    train_tids, val_tids = [], []
    for tid in ordered:
        train_deficit = n_train - len(train_tids)
        val_deficit   = n_val   - len(val_tids)
        if train_deficit / n_train >= val_deficit / n_val:
            train_tids.append(tid)
        else:
            val_tids.append(tid)

    return train_tids, val_tids


def kfold_split(timeline_ids, tl_bucket, k=5, seed=42):
    """
    Stratified k-fold over timelines with balanced fold sizes.

    Strategy: sort all timelines by bucket (so similar ones are adjacent),
    shuffle within each bucket, then flatten into one global list and assign
    folds round-robin (0,1,2,...,k-1,0,1,...) across the full list.

    This gives:
      - Fold sizes as equal as possible (differ by at most 1 timeline)
      - Each bucket's timelines spread across different folds rather than
        all landing in fold 0 (the bug when doing round-robin per-bucket
        with buckets smaller than k)

    Returns a list of k (train_tids, val_tids) tuples.
    """
    rng = random.Random(seed)

    # Build globally ordered list: shuffle within bucket, then concatenate
    groups  = group_by_bucket(timeline_ids, tl_bucket)
    ordered = []
    for bucket in sorted(groups):
        tids = groups[bucket][:]
        rng.shuffle(tids)
        ordered.extend(tids)

    # Assign folds round-robin across the full ordered list
    fold_assignments = {tid: i % k for i, tid in enumerate(ordered)}

    folds = []
    for fold_idx in range(k):
        val_tids   = [tid for tid, f in fold_assignments.items() if f == fold_idx]
        train_tids = [tid for tid, f in fold_assignments.items() if f != fold_idx]
        folds.append((train_tids, val_tids))
    return folds


# ── post flattening & encoder examples ────────────────────────────────────

def posts_for_timelines(task1_posts, timeline_ids):
    tid_set = set(timeline_ids)
    return [p for p in task1_posts if p["timeline_id"] in tid_set]


def make_encoder_examples(posts):
    """
    Convert raw posts into flat training examples for an encoder / SBERT model.

    Each example contains three parallel label structures so downstream code
    never has to re-derive them:

      presence_labels  – {valence:element → 0/1}  binary, for Task 1.1 element presence
      subelement_labels– {valence:element → int}   subelement ID (0 = absent), for Task 1.1 subelement classification
      presence_ratings – {valence → 1-5}           ordinal rating, for Task 1.2
      subelement_names – {valence:element → str}   human-readable label, for debugging / prompting

    Subelement IDs are the raw integers from the annotation scheme
    (e.g. 4 = "Depressed", 0 = absent). They are category IDs, not ordinal
    scores — treat each valence×element slot as its own classification head.
    """
    examples = []
    for p in posts:
        ev    = p.get("evidence", {})
        text  = p.get("post", "").strip()

        presence_labels   = {}   # valence:element → 0/1
        subelement_labels = {}   # valence:element → subelement int (0=absent)
        presence_ratings  = {}   # valence → 1-5
        subelement_names  = {}   # valence:element → readable string

        for valence in VALENCES:
            state    = ev.get(valence, {})
            presence = state.get("Presence")
            if not presence:
                continue

            presence_ratings[valence] = presence

            for elem in ELEMENTS:
                key = f"{valence}:{elem}"
                if elem in state:
                    cat_str = state[elem].get("Category", "")
                    sub_id  = parse_subelement_num(cat_str)
                    # strip the leading "(N) " to get just the name
                    name    = cat_str.split(")", 1)[-1].strip() if ")" in cat_str else cat_str

                    presence_labels[key]   = 1
                    subelement_labels[key] = sub_id
                    subelement_names[key]  = name
                else:
                    presence_labels[key]   = 0
                    subelement_labels[key] = 0
                    subelement_names[key]  = "absent"

        examples.append({
            "post_id":          p["post_id"],
            "timeline_id":      p["timeline_id"],
            "text":             text,
            "presence_labels":  presence_labels,    # Task 1.1 element presence
            "subelement_labels":subelement_labels,  # Task 1.1 subelement classification
            "presence_ratings": presence_ratings,   # Task 1.2
            "subelement_names": subelement_names,   # human-readable, for debugging/prompting
        })
    return examples


# ── reporting ──────────────────────────────────────────────────────────────

def label_distribution(examples):
    counts = Counter()
    for ex in examples:
        for slot in ALL_SLOTS:
            if ex["presence_labels"].get(slot, 0) != 0:
                counts[slot] += 1
    return counts


def print_split_summary(name, train_ex, val_ex):
    total_ex = len(train_ex) + len(val_ex)
    print(f"\n{'─'*62}")
    print(f"  {name}")
    print(f"{'─'*62}")
    print(f"  Train posts: {len(train_ex):>4}   Val posts: {len(val_ex):>4}   "
          f"(actual val%: {100*len(val_ex)/total_ex:.1f}%)")

    train_dist = label_distribution(train_ex)
    val_dist   = label_distribution(val_ex)

    # Also show subelement breakdown for rare categories
    print(f"\n  {'Slot':<30} {'Train':>6} {'Val':>6}  {'Val%':>6}")
    for slot in ALL_SLOTS:
        t = train_dist[slot]
        v = val_dist[slot]
        total = t + v
        pct = 100 * v / total if total else 0
        print(f"  {slot:<30} {t:>6} {v:>6}  {pct:>5.1f}%")

    # Rare subelement coverage
    print(f"\n  Rare subelement coverage:")
    print(f"  {'Subelement':<45} {'Train':>6} {'Val':>6}")
    for sub_slot in sorted(RARE_SUBELEMENTS):
        sub_num = int(sub_slot.split(":")[-1])
        elem    = sub_slot.split(":")[1]  # e.g. "A"
        valence = sub_slot.split(":")[0]  # e.g. "adaptive-state"
        t = sum(1 for ex in train_ex
                if ex["subelement_labels"].get(f"{valence}:{elem}", 0) == sub_num)
        v = sum(1 for ex in val_ex
                if ex["subelement_labels"].get(f"{valence}:{elem}", 0) == sub_num)
        print(f"  {sub_slot:<45} {t:>6} {v:>6}")


# ── output ─────────────────────────────────────────────────────────────────

def save_split(train_examples, val_examples, path_train, path_val):
    os.makedirs(os.path.dirname(path_train), exist_ok=True)
    with open(path_train, "w") as f:
        json.dump(train_examples, f, indent=2)
    with open(path_val, "w") as f:
        json.dump(val_examples, f, indent=2)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CLPsych Task 1 – train/val splits")
    parser.add_argument("--data-dir",    default="train_tasks12/", help="Directory with timeline JSON files")
    parser.add_argument("--output-dir",  default="splits/",        help="Where to write split files")
    parser.add_argument("--train-ratio", type=float, default=0.7,  help="Train fraction for stratified split (default 0.7)")
    parser.add_argument("--k",           type=int,   default=5,    help="Number of folds for k-fold (default 5)")
    parser.add_argument("--seed",        type=int,   default=42,   help="Random seed")
    args = parser.parse_args()

    # ── load ──
    timelines, task1_posts = load_data(args.data_dir)
    print(f"Loaded {len(timelines)} timelines, {len(task1_posts)} Task-1 posts")

    tids_with_evidence = sorted(set(p["timeline_id"] for p in task1_posts))
    print(f"Timelines with evidence: {len(tids_with_evidence)}")

    # ── stratification ──
    tl_bucket     = build_stratification(timelines, task1_posts)
    bucket_counts = Counter(tl_bucket[tid] for tid in tids_with_evidence)
    print(f"\nStratification buckets:")
    for bucket, n in sorted(bucket_counts.items()):
        print(f"  {bucket:<40} n={n}")

    # ══════════════════════════════════════════════════════════════
    # 1.  STRATIFIED 70/30 SPLIT
    # ══════════════════════════════════════════════════════════════
    train_tids, val_tids = stratified_split(
        tids_with_evidence, tl_bucket,
        train_ratio=args.train_ratio, seed=args.seed
    )

    train_ex = make_encoder_examples(posts_for_timelines(task1_posts, train_tids))
    val_ex   = make_encoder_examples(posts_for_timelines(task1_posts, val_tids))

    print_split_summary(
        f"Stratified split ({args.train_ratio:.0%}/{1-args.train_ratio:.0%})  "
        f"— {len(train_tids)} train timelines / {len(val_tids)} val timelines",
        train_ex, val_ex
    )

    save_split(
        train_ex, val_ex,
        os.path.join(args.output_dir, "stratified", "train.json"),
        os.path.join(args.output_dir, "stratified", "val.json"),
    )
    print(f"\n  Saved → {args.output_dir}/stratified/{{train,val}}.json")

    # ══════════════════════════════════════════════════════════════
    # 2.  K-FOLD SPLITS
    # ══════════════════════════════════════════════════════════════
    folds = kfold_split(tids_with_evidence, tl_bucket, k=args.k, seed=args.seed)

    print(f"\n\n{'═'*62}")
    print(f"  {args.k}-FOLD CROSS-VALIDATION")
    print(f"{'═'*62}")

    for fold_idx, (train_tids_f, val_tids_f) in enumerate(folds):
        train_ex_f = make_encoder_examples(posts_for_timelines(task1_posts, train_tids_f))
        val_ex_f   = make_encoder_examples(posts_for_timelines(task1_posts, val_tids_f))

        print_split_summary(
            f"Fold {fold_idx+1}/{args.k}  "
            f"— {len(train_tids_f)} train timelines / {len(val_tids_f)} val timelines",
            train_ex_f, val_ex_f
        )

        save_split(
            train_ex_f, val_ex_f,
            os.path.join(args.output_dir, f"fold_{fold_idx+1}", "train.json"),
            os.path.join(args.output_dir, f"fold_{fold_idx+1}", "val.json"),
        )

    print(f"\n  Saved → {args.output_dir}/fold_*/{{train,val}}.json")
    print(f"\nDone.")


if __name__ == "__main__":
    main()