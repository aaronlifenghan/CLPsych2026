"""
Task 2: Moments of Change (Switch & Escalation) using XGBoost
with temporal difference features.

Key insight: "moments of change" are about what's DIFFERENT between
consecutive posts. So we augment each post's embedding with:
  - h_t - h_{t-1}     (what changed)
  - h_t * h_{t-1}     (element-wise interaction)
  - position features  (normalized position in timeline, is_first, is_last)

Two independent binary classifiers: one for Switch, one for Escalation.
Focal-loss-style class weighting handles the severe class imbalance
(most posts are neither Switch nor Escalation).
"""

import logging
import os
import pickle
from collections import Counter

import numpy as np
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ── Temporal feature construction ────────────────────────────────

def _linguistic_features(text: str) -> np.ndarray:
    """
    Extract linguistic features from post text that signal emotional change.
    Returns a fixed-length float32 array of 14 features.
    """
    import re
    t = text or ""
    words = t.split()
    n_words = max(len(words), 1)
    n_chars = max(len(t), 1)

    # Lexical features
    n_sentences = max(len(re.split(r'[.!?]+', t)), 1)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    frac_upper = sum(1 for c in t if c.isupper()) / n_chars
    frac_punct = sum(1 for c in t if c in '!?.,;:') / n_chars
    n_exclaim = t.count('!')
    n_question = t.count('?')
    n_ellipsis = t.count('...')

    # Negative sentiment words (simple lexicon)
    neg_words = {'not','never','no','cant','cant','wont','dont','doesnt',
                 'hate','terrible','awful','horrible','bad','worst','fail',
                 'sad','depressed','angry','hurt','pain','lost','alone'}
    pos_words = {'good','great','happy','love','hope','better','improve',
                 'proud','excited','grateful','thankful','wonderful','joy'}
    words_lower = {w.lower().strip('.,!?') for w in words}
    frac_neg = len(words_lower & neg_words) / n_words
    frac_pos = len(words_lower & pos_words) / n_words

    # Structural
    log_len = np.log1p(n_words)
    has_removed = 1.0 if '[removed]' in t.lower() or '[deleted]' in t.lower() else 0.0

    return np.array([
        log_len, n_sentences, avg_word_len,
        frac_upper, frac_punct,
        n_exclaim, n_question, n_ellipsis,
        frac_neg, frac_pos,
        has_removed,
        n_words / n_sentences,  # words per sentence
        frac_neg - frac_pos,    # sentiment balance
        min(n_exclaim + n_question, 10),  # emotional punctuation count
    ], dtype=np.float32)


def build_temporal_features(
    embeddings_by_timeline: list[np.ndarray],
    texts_by_timeline: list[list[str]] | None = None,
) -> np.ndarray:
    """
    Build temporal difference features for all posts across timelines.

    Args:
        embeddings_by_timeline: list of (n_posts_in_tl, embed_dim) arrays,
            one per timeline, posts in chronological order.
        texts_by_timeline: optional list of post text lists, one per timeline.
            If provided, linguistic features are appended.

    Returns:
        (total_posts, feature_dim) array where feature_dim =
            embed_dim*3 + 3 (position) + 14 (linguistic, if texts provided)
    """
    all_features = []

    for tl_idx, tl_emb in enumerate(embeddings_by_timeline):
        n_posts, dim = tl_emb.shape
        texts = texts_by_timeline[tl_idx] if texts_by_timeline else None

        for t in range(n_posts):
            h_t = tl_emb[t]

            if t > 0:
                h_prev = tl_emb[t - 1]
                diff = h_t - h_prev
                prod = h_t * h_prev
            else:
                diff = np.zeros(dim, dtype=np.float32)
                prod = np.zeros(dim, dtype=np.float32)

            pos_norm = t / max(n_posts - 1, 1)
            is_first = 1.0 if t == 0 else 0.0
            is_last = 1.0 if t == n_posts - 1 else 0.0

            parts = [h_t, diff, prod, np.array([pos_norm, is_first, is_last], dtype=np.float32)]

            if texts is not None and t < len(texts):
                ling = _linguistic_features(texts[t])
                # Also add diff of linguistic features from previous post
                if t > 0 and (t - 1) < len(texts):
                    ling_prev = _linguistic_features(texts[t - 1])
                    ling_diff = ling - ling_prev
                    parts.append(ling)
                    parts.append(ling_diff)
                else:
                    parts.append(ling)
                    parts.append(np.zeros_like(ling))

            all_features.append(np.concatenate(parts))

    return np.array(all_features, dtype=np.float32)


def build_temporal_features_single(
    tl_emb: np.ndarray,
) -> np.ndarray:
    """Build temporal features for a single timeline."""
    return build_temporal_features([tl_emb])


# ── Label extraction ─────────────────────────────────────────────

def extract_task2_labels(post: dict) -> dict[str, int]:
    """
    Extract Task 2 binary labels from a post.

    Returns:
        {"Switch": 0 or 1, "Escalation": 0 or 1}
    """
    sw = post.get("Switch", "0")
    esc = post.get("Escalation", "0")
    return {
        "Switch": 1 if sw == "S" else 0,
        "Escalation": 1 if esc == "E" else 0,
    }


# ── Task 2 Model ────────────────────────────────────────────────

class Task2Model:
    """
    Two independent binary XGBoost classifiers for Switch and Escalation.

    Uses temporal difference features to capture "moments of change".
    """

    def __init__(self, xgb_params: dict | None = None):
        self.xgb_params = xgb_params or {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        self.models: dict[str, XGBClassifier] = {}

    def fit(
        self,
        X: np.ndarray,
        labels: list[dict[str, int]],
    ) -> "Task2Model":
        """
        Train Switch and Escalation classifiers.

        Args:
            X: (n_posts, feature_dim) temporal features.
            labels: list of {"Switch": 0/1, "Escalation": 0/1} per post.
        """
        for label_name in ["Switch", "Escalation"]:
            y = np.array([lab[label_name] for lab in labels], dtype=int)
            pos = int(y.sum())
            neg = len(y) - pos

            # Heavy positive-class weighting for imbalanced data
            scale = neg / max(pos, 1)
            scale = min(scale, 20.0)  # cap to avoid instability

            clf = XGBClassifier(
                objective="binary:logistic",
                scale_pos_weight=scale,
                use_label_encoder=False,
                verbosity=0,
                **self.xgb_params,
            )
            clf.fit(X, y)
            self.models[label_name] = clf

            logger.info(
                f"  Task 2 {label_name}: {len(y)} posts, "
                f"{pos} positive ({100*pos/len(y):.1f}%), "
                f"scale_pos_weight={scale:.1f}"
            )

        return self

    def predict(self, X: np.ndarray) -> list[dict[str, str]]:
        """
        Predict Switch and Escalation labels.

        Returns:
            list of {"Switch": "S"/"0", "Escalation": "E"/"0"} per post.
        """
        n = X.shape[0]
        results = [{"Switch": "0", "Escalation": "0"} for _ in range(n)]

        for label_name, pos_val in [("Switch", "S"), ("Escalation", "E")]:
            if label_name not in self.models:
                continue
            clf = self.models[label_name]
            preds = clf.predict(X)
            for i, p in enumerate(preds):
                if p == 1:
                    results[i][label_name] = pos_val

        return results

    def predict_proba(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return probabilities for threshold tuning."""
        probs = {}
        for label_name, clf in self.models.items():
            probs[label_name] = clf.predict_proba(X)[:, 1]
        return probs

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        for name, clf in self.models.items():
            clf.save_model(os.path.join(model_dir, f"t2_{name.lower()}.xgb"))

    def load(self, model_dir: str) -> "Task2Model":
        for name in ["Switch", "Escalation"]:
            path = os.path.join(model_dir, f"t2_{name.lower()}.xgb")
            if os.path.exists(path):
                clf = XGBClassifier(
                    objective="binary:logistic",
                    use_label_encoder=False,
                    verbosity=0,
                )
                clf.load_model(path)
                self.models[name] = clf
        return self


# ── Prediction formatting ────────────────────────────────────────

def format_task2_predictions(
    post_ids: list[str],
    timeline_ids: list[str],
    predictions: list[dict[str, str]],
) -> list[dict]:
    """Format predictions into task2_pred.json schema (§9)."""
    return [
        {
            "timeline_id": tid,
            "post_id": pid,
            "Switch": pred["Switch"],
            "Escalation": pred["Escalation"],
        }
        for pid, tid, pred in zip(post_ids, timeline_ids, predictions)
    ]
