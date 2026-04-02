# System 4: Embedding + XGBoost Pipeline

System 4 is a classical ML pipeline using combined TF-IDF + Sentence-BERT embeddings
fed into XGBoost classifiers. It covers all three shared tasks:

- **Task 1.1** — ABCD subelement classification (12 XGBoost classifiers)
- **Task 1.2** — Presence rating 1–5 (2 XGBoost classifiers)
- **Task 2** — Moments of change: Switch & Escalation detection (2 XGBoost classifiers)

---

## Directory Structure

```
clp-project-updated/
├── data/
│   ├── train_tasks12/          ← gold training timelines (30 JSON files)
│   ├── train_silver_voted/     ← silver-augmented training data (generated)
│   └── test_tasks12nolabels/   ← unlabeled competition test timelines
├── src/
│   └── clpsych_assessment/
│       ├── system3/            ← LLM pipeline (System 3)
│       └── system4/
│           ├── run.py          ← main pipeline entry point
│           ├── task1.py        ← Task 1.1 and 1.2 models
│           ├── task2.py        ← Task 2 model + temporal/linguistic features
│           ├── embeddings.py   ← TF-IDF, SBERT, Combined embedders
│           └── config.py       ← subelement schema constants
├── run_kfold.py                ← 5-fold cross-validation runner
├── convert_s3_to_gold.py       ← converts LLM outputs to silver training labels
└── run_hparam_search.py        ← XGBoost hyperparameter search
```

---

## Installation

```
pip install -e .
pip install sentence-transformers xgboost scikit-learn numpy scipy
```

---

## Step 0 — Generate Silver Training Labels (optional but recommended)

Uses majority-vote across System 3 LLM outputs to label unannotated posts,
giving System 4 more training signal for rare classes.

```bash
cd ~/Documents/CLPsych/CLPsych2026

python convert_s3_to_gold.py \
    --s3-files src/outputs/system3/gemma2_9b/fewshot/raw_task_1_1.json \
               src/outputs/system3/llama3.1/fewshot/raw_task_1_1.json \
               src/outputs/system3/qwen3_8b/fewshot/raw_task_1_1.json \
    --train-dir data/train_tasks12 \
    --output-dir data/train_silver_voted
```

Output: `data/train_silver_voted/` — 30 timeline JSONs with gold labels preserved
and silver labels injected into previously unannotated posts.

---

## Step 1 — Evaluate with Cross-Validation

Run 5-fold CV to get reliable performance estimates. Uses combined mpnet embeddings
(best configuration found during experiments).

```bash
cd src

TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ../run_kfold.py \
    --data-dir ../data/train_silver_voted \
    --folds 5 \
    --embed-backend combined \
    --embed-model all-mpnet-base-v2 \
    --unlabeled-dir ../data/test_tasks12nolabels \
    --output-dir outputs/system4_cv
```

**Key flags:**

| Flag                | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `--data-dir`      | Training data directory                                  |
| `--folds`         | Number of CV folds (5 recommended)                       |
| `--embed-backend` | `tfidf`, `sbert`, or `combined` (use `combined`) |
| `--embed-model`   | SBERT model name (use `all-mpnet-base-v2`)             |
| `--unlabeled-dir` | Unlabeled test texts for TF-IDF vocab expansion          |
| `--output-dir`    | Where to save fold results and averaged scores           |

**Expected scores (5-fold CV):**

- Task 1.1 macro F1: ~0.27
- Task 1.2 RMSE: ~0.99
- Task 2 macro F1: ~0.44

---

## Step 2 — Hyperparameter Search (optional)

Tests 8 XGBoost configurations × 3 folds. Takes ~30–40 min on CPU.

```bash
cd src

TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ../run_hparam_search.py \
    --data-dir ../data/train_silver_voted \
    --embed-backend combined \
    --embed-model all-mpnet-base-v2 \
    --unlabeled-dir ../data/test_tasks12nolabels \
    --output-dir outputs/system4_hparam
```

**Best params found:** `n_estimators=400, max_depth=6, learning_rate=0.05`

To apply them, update the `xgb_params` dicts in `task1.py` and `task2.py`:

```python
self.xgb_params = xgb_params or {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    ...
}
```

---

## Step 3 — Generate Final Submission

Train on ALL 30 training timelines, predict on the unlabeled test set.

```bash
cd src

TRANSFORMERS_OFFLINE=1 python -m clpsych_assessment.system4.run \
    --train-dir ../data/train_silver_voted \
    --test-dir  ../data/test_tasks12nolabels \
    --embed-backend combined \
    --embed-model all-mpnet-base-v2 \
    --unlabeled-dir ../data/test_tasks12nolabels \
    --output-dir outputs/system4_submission
```

Output files:

- `outputs/system4_submission/task1_pred.json` — Task 1 predictions
- `outputs/system4_submission/task2_pred.json` — Task 2 predictions

Zip for submission:

```bash
cd outputs/system4_submission
zip task1_submission.zip task1_pred.json
zip task2_submission.zip task2_pred.json
```

---

## Quick Single-Split Run (for testing)

If you just want a fast sanity check without CV:

```bash
cd src

TRANSFORMERS_OFFLINE=1 python -m clpsych_assessment.system4.run \
    --data-dir ../data/train_tasks12 \
    --embed-backend combined \
    --embed-model all-mpnet-base-v2 \
    --output-dir outputs/system4_test
```

This auto-splits 70/30 and reports evaluation scores immediately.

---

## Embedding Backends

| Backend      | Description                 | Speed  | Quality           |
| ------------ | --------------------------- | ------ | ----------------- |
| `tfidf`    | TF-IDF bag-of-words         | Fast   | Good for Task 2   |
| `sbert`    | Sentence-BERT only          | Medium | Good for Task 1.2 |
| `combined` | TF-IDF + SBERT concatenated | Slower | Best overall ✓   |

SBERT models (set via `--embed-model`):

- `all-MiniLM-L6-v2` — fast, smaller (default)
- `all-mpnet-base-v2` — slower, better quality ✓ (recommended)

---

## Architecture Summary

### Task 1.1 — Subelement Classification

- 12 independent XGBoost classifiers (one per valence × element)
- Input: combined TF-IDF + SBERT post embedding
- Output: subelement class (0 = absent, or valid subelement ID)
- Class imbalance handled via sub-linear inverse frequency sample weights

### Task 1.2 — Presence Rating

- 2 XGBoost classifiers (adaptive, maladaptive)
- 5-class ordinal classification (ratings 1–5 mapped to 0–4)
- Prediction via expected value: sum(k × p(k)) for k=1..5

### Task 2 — Moments of Change

- 2 independent binary XGBoost classifiers (Switch, Escalation)
- Features: embedding + temporal diff + element-wise product + position + 14 linguistic features
- Linguistic features include: sentiment words, punctuation density, post length, caps ratio
- Class imbalance handled via `scale_pos_weight`

---

## Experiment Results Summary

| Config                                         | T1.1 F1 ↑       | T1.2 RMSE ↓     | T2 F1 ↑         |
| ---------------------------------------------- | ---------------- | ---------------- | ---------------- |
| TF-IDF only                                    | 0.2732           | 1.0701           | 0.4197           |
| SBERT only                                     | 0.2477           | 1.0387           | 0.3283           |
| Combined MiniLM                                | 0.2579           | 0.9930           | 0.4078           |
| Combined mpnet                                 | 0.2696           | 0.9883           | 0.4365           |
| + silver labels                                | 0.2701           | 0.9952           | 0.4399           |
| **+ linguistic features + best hparams** | **0.2713** | **0.9890** | **0.4206** |
