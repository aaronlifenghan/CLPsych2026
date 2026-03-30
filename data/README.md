# split_task1.py — CLPsych 2026 Task 1 Data Splits

Generates train/val splits from the CLPsych 2026 training data for Task 1 (ABCD element and subelement classification + presence rating).

## Usage

```bash
# Default: 70/30 stratified split + 5-fold CV
python split_task1.py --data-dir train_tasks12/ --output-dir splits/

# Custom options
python split_task1.py --data-dir train_tasks12/ --output-dir splits/ --k 3 --seed 99
```

**Arguments**

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `train_tasks12/` | Folder containing the timeline JSON files |
| `--output-dir` | `splits/` | Where to write output files |
| `--train-ratio` | `0.7` | Train fraction for the stratified split |
| `--k` | `5` | Number of folds for k-fold CV |
| `--seed` | `42` | Random seed |

## Output files

```
splits/
  stratified/
    train.json       ← 70% of timelines (~164 posts)
    val.json         ← 30% of timelines (~72 posts)
  fold_1/
    train.json
    val.json
  fold_2/ ... fold_5/
```

Each file is a JSON array of post examples (see format below).

## Example format

Each example in the JSON looks like this:

```json
{
  "post_id": "13a844f48c",
  "timeline_id": "0cac13e357",
  "text": "I wish someone in this world actually liked me...",

  "presence_labels": {
    "adaptive-state:A": 0,
    "adaptive-state:C-O": 1,
    "adaptive-state:D": 1,
    "maladaptive-state:C-O": 1,
    ...
  },

  "subelement_labels": {
    "adaptive-state:C-O": 1,
    "adaptive-state:D": 5,
    "maladaptive-state:C-O": 2,
    ...
  },

  "presence_ratings": {
    "adaptive-state": 3,
    "maladaptive-state": 4
  },

  "subelement_names": {
    "adaptive-state:C-O": "Perception of the other as related",
    "adaptive-state:D": "Competence, self esteem, self-care",
    "maladaptive-state:C-O": "Perception of the other as detached or over attached",
    ...
  }
}
```

**The four label fields map directly to the three tasks:**

- `presence_labels` → Task 1.1 element presence (binary per slot, 0/1)
- `subelement_labels` → Task 1.1 subelement classification (integer ID per slot, 0 = absent)
- `presence_ratings` → Task 1.2 presence rating (ordinal 1–5 per valence)
- `subelement_names` → human-readable labels, useful for prompting or debugging

The subelement IDs are the numbers from the annotation scheme (e.g. 4 = "Depressed, despair, hopeless"). They are **category IDs, not ordinal scores** — adaptive and maladaptive use parallel numbering (1↔2, 3↔4, etc.).

## Key design decisions

### 1. Split timelines, not posts
The unit of splitting is the **timeline** (one user's post history), not individual posts. Posts from the same user must stay together to avoid leakage — if the same user appears in both train and val, the model can pick up on writing style rather than learning the labels.

### 2. Stratification uses subelement categories
Each timeline gets a stratification label based on which **specific subelements** appear in it (not just which elements). The label prioritises the 12 rarest subelements (those with fewer than ~15 occurrences in the full dataset), because these are the hardest to keep balanced. Timelines that share the same rare categories are grouped together and dealt proportionately into each split.

### 3. Proportionate dealing (not floor-rounding)
With only 30 timelines, the naive `floor(n * ratio)` approach breaks down for small groups — a bucket with 1 timeline always puts it in train regardless of ratio. Instead, timelines are dealt one-by-one to whichever split is furthest below its proportional target, keeping the overall ratio accurate.

### 4. K-fold assigns globally, not per-bucket
Fold indices are assigned round-robin across all timelines sorted by bucket (not within each bucket separately). This prevents the case where small buckets dump all their timelines into fold 0, and ensures every fold gets the same number of timelines (6 val timelines per fold for k=5).

## Honest caveats

With only 30 timelines the splits are inherently noisy. A few things to keep in mind:

- The post counts per fold vary (42–58) even though timeline counts are equal, because timelines differ in length. This is real data variance, not a sampling bug.
- Subelements with n=1 (Mania, Calm/laid back, Vigor, Feel loved/belong) will inevitably land entirely in train or val for some splits. Cross-validation gives a fairer picture for these than the single stratified split.
- Val% per slot in the stratified split ranges from ~19–37% rather than a clean 30% — this is the best achievable given the constraint of keeping timelines intact.