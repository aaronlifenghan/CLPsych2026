"""
Task 1.1 (Element/Subelement Classification) + Task 1.2 (Presence Rating)
using XGBoost on frozen embeddings.

Task 1.1 approach:
  One XGBoost classifier per (valence × element) = 12 models.
  Each does (K+1)-class classification: class 0 = absent, classes 1..K = subelements.
  This jointly handles element presence AND subelement selection.

Task 1.2 approach:
  One XGBoost classifier per valence = 2 models.
  5-class classification (ratings 1–5).
  Predicted rating = argmax (or expected value for softer prediction).
"""

import logging
import os
import pickle
import re
from collections import Counter

import numpy as np
from xgboost import XGBClassifier

from .config import (
    ELEMENTS,
    VALID_SUBELEMENTS,
    VALENCE_SHORT,
    VALENCES,
    classes_for_slot,
)

logger = logging.getLogger(__name__)


# ── Label extraction from timeline JSON ──────────────────────────

def _parse_subelement_num(category_str: str) -> int:
    """Extract the leading integer from '(N) description' format."""
    m = re.match(r"^\((\d+)\)", category_str.strip())
    return int(m.group(1)) if m else 0


def extract_task1_labels(post: dict) -> dict:
    """
    Extract Task 1 labels from a single post's evidence.

    Returns:
        {
            "subelement_labels": {"adaptive:A": 5, "adaptive:B-O": 0, ...},
            "presence_ratings":  {"adaptive": 3, "maladaptive": None},
        }
    """
    ev = post.get("evidence", {})
    sub_labels = {}
    pres_ratings = {}

    for valence_key in VALENCES:
        vs = VALENCE_SHORT[valence_key]
        state = ev.get(valence_key, {})
        presence = state.get("Presence")

        if presence is None:
            pres_ratings[vs] = None
            for elem in ELEMENTS:
                sub_labels[f"{vs}:{elem}"] = None  # not evaluated
            continue

        pres_ratings[vs] = int(presence)

        for elem in ELEMENTS:
            key = f"{vs}:{elem}"
            elem_data = state.get(elem)
            if elem_data and isinstance(elem_data, dict):
                cat = elem_data.get("Category", "")
                sub_labels[key] = _parse_subelement_num(cat)
            else:
                sub_labels[key] = 0  # absent

    return {
        "subelement_labels": sub_labels,
        "presence_ratings": pres_ratings,
    }


# ── Task 1.1: Subelement classifiers ────────────────────────────

class Task1_1Model:
    """
    12 independent XGBoost classifiers, one per (valence, element).

    Each classifier maps a post embedding → class in {0, sub1, sub2, ...}
    where 0 = element absent.
    """

    def __init__(self, xgb_params: dict | None = None):
        self.xgb_params = xgb_params or {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 2,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        self.models: dict[str, XGBClassifier] = {}
        self._class_maps: dict[str, dict[int, int]] = {}  # slot → {orig_class: xgb_idx}
        self._inv_maps: dict[str, dict[int, int]] = {}    # slot → {xgb_idx: orig_class}

    def fit(
        self,
        X: np.ndarray,
        labels: list[dict[str, int | None]],
    ) -> "Task1_1Model":
        """
        Train one classifier per (valence, element) slot.

        Args:
            X: (n_posts, embed_dim) array of post embeddings.
            labels: list of dicts from extract_task1_labels()["subelement_labels"],
                    one per post. Keys like "adaptive:A" → int (0=absent).
        """
        for valence_key in VALENCES:
            vs = VALENCE_SHORT[valence_key]
            for elem in ELEMENTS:
                slot = f"{vs}:{elem}"
                valid_classes = classes_for_slot(vs, elem)

                # Collect (X, y) pairs where this slot was evaluated
                indices = []
                y_raw = []
                for i, lab in enumerate(labels):
                    val = lab.get(slot)
                    if val is not None:
                        indices.append(i)
                        y_raw.append(val)

                if not indices:
                    logger.warning(f"  {slot}: no training samples, skipping")
                    continue

                X_slot = X[indices]
                y_arr = np.array(y_raw, dtype=int)

                # Map original class IDs to contiguous 0..K for XGBoost.
                # Only include classes actually present in training data —
                # XGBoost requires contiguous labels [0, 1, ..., n_classes-1].
                present_classes = sorted(set(y_arr.tolist()) | {0})
                class_map = {c: i for i, c in enumerate(present_classes)}
                inv_map = {i: c for c, i in class_map.items()}
                self._class_maps[slot] = class_map
                self._inv_maps[slot] = inv_map

                y_mapped = np.array([class_map[c] for c in y_arr])
                n_classes = len(present_classes)

                # Class weights: sub-linear inverse frequency to strongly
                # up-weight rare subelements without going to extremes
                counts = Counter(y_mapped.tolist())
                total = len(y_mapped)
                weights = np.ones(total, dtype=float)
                for i in range(total):
                    c = y_mapped[i]
                    freq = counts[c] / total
                    weights[i] = (1.0 / max(freq, 1e-6)) ** 0.75

                # For binary slots also set scale_pos_weight for the minority class
                extra_params = {}
                if n_classes == 2:
                    n_neg = counts.get(0, 1)
                    n_pos = counts.get(1, 1)
                    extra_params["scale_pos_weight"] = max(n_neg / max(n_pos, 1), 1.0)

                clf = XGBClassifier(
                    num_class=n_classes if n_classes > 2 else None,
                    objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                    use_label_encoder=False,
                    verbosity=0,
                    **{**self.xgb_params, **extra_params},
                )
                clf.fit(X_slot, y_mapped, sample_weight=weights)
                self.models[slot] = clf

                dist = Counter(y_arr.tolist())
                logger.info(
                    f"  {slot}: {len(indices)} samples, "
                    f"{n_classes} classes, dist={dict(sorted(dist.items()))}"
                )

        return self

    def predict(self, X: np.ndarray) -> list[dict[str, int]]:
        """
        Predict subelement labels for each post.

        Returns:
            list of dicts: {slot → predicted_class} for each post.
            Predicted class is the original subelement ID (0 = absent).
        """
        n = X.shape[0]
        results = [{} for _ in range(n)]

        for slot, clf in self.models.items():
            inv_map = self._inv_maps[slot]
            preds = clf.predict(X)
            for i, p in enumerate(preds):
                results[i][slot] = inv_map[int(p)]

        return results

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        for slot, clf in self.models.items():
            safe = slot.replace(":", "_")
            clf.save_model(os.path.join(model_dir, f"t1_1_{safe}.xgb"))
        with open(os.path.join(model_dir, "t1_1_maps.pkl"), "wb") as f:
            pickle.dump({"class_maps": self._class_maps, "inv_maps": self._inv_maps}, f)

    def load(self, model_dir: str) -> "Task1_1Model":
        with open(os.path.join(model_dir, "t1_1_maps.pkl"), "rb") as f:
            maps = pickle.load(f)
        self._class_maps = maps["class_maps"]
        self._inv_maps = maps["inv_maps"]
        for slot in self._class_maps:
            safe = slot.replace(":", "_")
            path = os.path.join(model_dir, f"t1_1_{safe}.xgb")
            if os.path.exists(path):
                n_classes = len(self._class_maps[slot])
                clf = XGBClassifier(
                    num_class=n_classes if n_classes > 2 else None,
                    use_label_encoder=False,
                    verbosity=0,
                )
                clf.load_model(path)
                self.models[slot] = clf
        return self


# ── Task 1.2: Presence rating ───────────────────────────────────

class Task1_2Model:
    """
    Two XGBoost classifiers (adaptive, maladaptive).
    5-class classification for Presence rating 1–5.
    """

    def __init__(self, xgb_params: dict | None = None):
        self.xgb_params = xgb_params or {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        self.models: dict[str, XGBClassifier] = {}

    def fit(
        self,
        X: np.ndarray,
        ratings: list[dict[str, int | None]],
    ) -> "Task1_2Model":
        """
        Train one classifier per valence.

        Args:
            X: (n_posts, embed_dim) post embeddings.
            ratings: list of dicts from extract_task1_labels()["presence_ratings"].
        """
        for vs in ["adaptive", "maladaptive"]:
            indices = []
            y = []
            for i, r in enumerate(ratings):
                val = r.get(vs)
                if val is not None and 1 <= val <= 5:
                    indices.append(i)
                    y.append(val - 1)  # map 1-5 → 0-4

            if not indices:
                logger.warning(f"  Task 1.2 {vs}: no samples, skipping")
                continue

            X_slot = X[indices]
            y_arr = np.array(y, dtype=int)

            clf = XGBClassifier(
                num_class=5,
                objective="multi:softprob",
                use_label_encoder=False,
                verbosity=0,
                **self.xgb_params,
            )
            clf.fit(X_slot, y_arr)
            self.models[vs] = clf
            logger.info(
                f"  Task 1.2 {vs}: {len(indices)} samples, "
                f"dist={dict(Counter(y_arr.tolist()))}"
            )

        return self

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """
        Predict presence ratings.

        Returns:
            {"adaptive": array of ints (1-5), "maladaptive": array of ints (1-5)}
        """
        results = {}
        for vs, clf in self.models.items():
            probs = clf.predict_proba(X)  # (n, 5)
            # Expected value: sum(k * p(k)) for k=1..5
            ratings_float = probs @ np.arange(1, 6)
            ratings_int = np.clip(np.round(ratings_float), 1, 5).astype(int)
            results[vs] = ratings_int
        return results

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        for vs, clf in self.models.items():
            clf.save_model(os.path.join(model_dir, f"t1_2_{vs}.xgb"))

    def load(self, model_dir: str) -> "Task1_2Model":
        for vs in ["adaptive", "maladaptive"]:
            path = os.path.join(model_dir, f"t1_2_{vs}.xgb")
            if os.path.exists(path):
                clf = XGBClassifier(
                    num_class=5,
                    use_label_encoder=False,
                    verbosity=0,
                )
                clf.load_model(path)
                self.models[vs] = clf
        return self


# ── Prediction formatting ────────────────────────────────────────

def format_task1_predictions(
    post_ids: list[str],
    timeline_ids: list[str],
    subelement_preds: list[dict[str, int]],
    presence_preds: dict[str, np.ndarray],
) -> list[dict]:
    """
    Format predictions into task1_pred.json schema.

    Returns a list of submission entries matching §9.
    """
    entries = []
    for i, (pid, tid) in enumerate(zip(post_ids, timeline_ids)):
        entry = {"timeline_id": tid, "post_id": pid}

        for valence_key in VALENCES:
            vs = VALENCE_SHORT[valence_key]
            state_out = {}

            # Presence rating
            if vs in presence_preds:
                state_out["Presence"] = int(presence_preds[vs][i])

            # Subelement predictions
            sub_pred = subelement_preds[i]
            for elem in ELEMENTS:
                slot = f"{vs}:{elem}"
                pred_class = sub_pred.get(slot, 0)
                if pred_class != 0:
                    state_out[elem] = {"subelement": pred_class}

            # Only include valence if it has content
            if state_out.get("Presence") or any(
                k != "Presence" for k in state_out
            ):
                if "Presence" not in state_out:
                    state_out["Presence"] = 1
                entry[valence_key] = state_out

        if "adaptive-state" in entry or "maladaptive-state" in entry:
            entries.append(entry)

    return entries
