"""
CLPsych 2026 Shared Task — Evaluation Metrics

Implements the exact evaluation logic from the shared task specification:
  - Task 1.1: Element presence (binary) + Subelement classification (multi-class)
  - Task 1.2: Presence rating (MAE, RMSE, QWK, Spearman)
  - Task 2:   Moments of Change (post-level + timeline-level P/R/F1)

All metrics follow the filtering rules, class-0 exclusion, and edge-case
handling described in the shared task README.
"""

import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

ELEMENTS = ["A", "B-O", "B-S", "C-O", "C-S", "D"]

# Valid subelement IDs per (valence, element) from §8
VALID_SUBELEMENTS = {
    ("adaptive", "A"):   [1, 3, 5, 7, 9, 11, 13],
    ("adaptive", "B-O"): [1, 3],
    ("adaptive", "B-S"): [1],
    ("adaptive", "C-O"): [1, 3],
    ("adaptive", "C-S"): [1],
    ("adaptive", "D"):   [1, 3, 5],
    ("maladaptive", "A"):   [2, 4, 6, 8, 10, 12, 14],
    ("maladaptive", "B-O"): [2, 4],
    ("maladaptive", "B-S"): [2],
    ("maladaptive", "C-O"): [2, 4],
    ("maladaptive", "C-S"): [2],
    ("maladaptive", "D"):   [2, 4, 6],
}


# ── Helpers ──────────────────────────────────────────────────────

def _prf1(tp, fp, fn):
    """Compute precision, recall, F1 from counts."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _qwk(y_true, y_pred, min_rating=1, max_rating=5):
    """Quadratic Weighted Kappa for ordinal ratings."""
    n = max_rating - min_rating + 1
    hist_true = np.zeros(n, dtype=float)
    hist_pred = np.zeros(n, dtype=float)
    confusion = np.zeros((n, n), dtype=float)

    for t, p in zip(y_true, y_pred):
        ti = t - min_rating
        pi = p - min_rating
        ti = max(0, min(n - 1, ti))
        pi = max(0, min(n - 1, pi))
        hist_true[ti] += 1
        hist_pred[pi] += 1
        confusion[ti][pi] += 1

    total = len(y_true)
    if total == 0:
        return 0.0

    weight = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            weight[i][j] = ((i - j) ** 2) / ((n - 1) ** 2)

    expected = np.outer(hist_true, hist_pred) / total
    observed = confusion

    num = np.sum(weight * observed)
    den = np.sum(weight * expected)

    if den == 0:
        return 1.0
    return 1.0 - num / den


def _spearman(x, y):
    """Spearman rank correlation."""
    if len(x) < 2:
        return 0.0
    from scipy.stats import spearmanr
    corr, _ = spearmanr(x, y)
    if np.isnan(corr):
        return 0.0
    return float(corr)


# ── Gold label loading ───────────────────────────────────────────

def load_gold_from_dir(gold_dir: str) -> dict:
    """
    Load gold labels from a directory of timeline JSON files.

    Returns:
        {post_id: {timeline_id, evidence, Switch, Escalation, ...}}
    """
    gold = {}
    timeline_posts = defaultdict(list)

    for f in sorted(Path(gold_dir).glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        tid = data.get("timeline_id", f.stem)
        for p in data.get("posts", []):
            pid = p.get("post_id", "")
            gold[pid] = {
                "timeline_id": tid,
                "post_id": pid,
                "evidence": p.get("evidence", {}),
                "Switch": p.get("Switch", "0"),
                "Escalation": p.get("Escalation", "0"),
            }
            timeline_posts[tid].append(pid)

    return gold, dict(timeline_posts)


def _get_gold_subelement(evidence_state: dict, element: str, valence: str) -> int:
    """Extract gold subelement ID from evidence, or 0 if absent."""
    elem_data = evidence_state.get(element)
    if elem_data is None or not isinstance(elem_data, dict):
        return 0
    cat = elem_data.get("Category", "")
    if not cat:
        return 0
    # Extract number from "(N) ..." format
    import re
    m = re.match(r"^\((\d+)\)", cat.strip())
    if m:
        return int(m.group(1))
    return 0


def _get_pred_subelement(pred_state: dict, element: str) -> int:
    """Extract predicted subelement ID, or 0 if absent."""
    elem_data = pred_state.get(element)
    if elem_data is None or not isinstance(elem_data, dict):
        return 0
    return elem_data.get("subelement", 0)


# ── Task 1.1: Element Presence + Subelement Classification ──────

def evaluate_task1_1(
    gold: dict,
    predictions: list[dict],
) -> dict:
    """
    Evaluate Task 1.1 following the exact shared task pipeline.

    Steps:
      1. Filter to posts with gold evidence
      2. Per-valence filtering
      3. Element presence (binary)
      4. Subelement classification (multi-class, class 0 excluded)
    """
    # Index predictions by post_id
    pred_by_id = {p["post_id"]: p for p in predictions}

    # Element presence: collect binary labels per (valence, element)
    ep_gold = defaultdict(list)  # key: (valence, element) -> list of 0/1
    ep_pred = defaultdict(list)

    # Subelement classification: collect class labels per (valence, element)
    sc_gold = defaultdict(list)  # key: (valence, element) -> list of int
    sc_pred = defaultdict(list)

    for pid, g in gold.items():
        ev = g.get("evidence", {})

        for valence_key in ["adaptive-state", "maladaptive-state"]:
            valence = "adaptive" if "adaptive-state" == valence_key else "maladaptive"
            gold_state = ev.get(valence_key, {})

            # Step 2: Per-valence filtering — only evaluate if gold has valid Presence
            gold_pres = gold_state.get("Presence")
            if gold_pres is None:
                continue
            if not isinstance(gold_pres, (int, float)):
                continue

            # Check this valence has actual content (Presence alone with no
            # elements still means we evaluate — elements will be 0/absent)
            pred_entry = pred_by_id.get(pid, {})
            pred_state = pred_entry.get(valence_key, {})

            for elem in ELEMENTS:
                g_sub = _get_gold_subelement(gold_state, elem, valence)
                p_sub = _get_pred_subelement(pred_state, elem)

                # Element presence: binary
                g_present = 1 if g_sub != 0 else 0
                p_present = 1 if p_sub != 0 else 0
                ep_gold[(valence, elem)].append(g_present)
                ep_pred[(valence, elem)].append(p_present)

                # Subelement classification: full class labels
                sc_gold[(valence, elem)].append(g_sub)
                sc_pred[(valence, elem)].append(p_sub)

    # ── Element Presence metrics ──
    ep_results = {}
    for valence in ["adaptive", "maladaptive"]:
        for elem in ELEMENTS:
            key = (valence, elem)
            if key not in ep_gold:
                continue
            g = np.array(ep_gold[key])
            p = np.array(ep_pred[key])
            tp = int(np.sum((g == 1) & (p == 1)))
            fp = int(np.sum((g == 0) & (p == 1)))
            fn = int(np.sum((g == 1) & (p == 0)))
            prec, rec, f1 = _prf1(tp, fp, fn)
            support = int(np.sum(g == 1))
            ep_results[f"{valence}:{elem}"] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": support,
            }

    # Per-valence aggregation
    for valence in ["adaptive", "maladaptive"]:
        f1s = [
            ep_results[f"{valence}:{e}"]["f1"]
            for e in ELEMENTS if f"{valence}:{e}" in ep_results
        ]
        if f1s:
            ep_results[f"{valence}_macro_f1"] = round(float(np.mean(f1s)), 4)

    # Overall
    all_f1s = [v["f1"] for k, v in ep_results.items() if isinstance(v, dict)]
    if all_f1s:
        ep_results["overall_macro_f1"] = round(float(np.mean(all_f1s)), 4)

    # ── Subelement Classification metrics ──
    sc_results = {}
    for valence in ["adaptive", "maladaptive"]:
        for elem in ELEMENTS:
            key = (valence, elem)
            if key not in sc_gold:
                continue

            valid_classes = VALID_SUBELEMENTS.get((valence, elem), [])
            g = sc_gold[key]
            p = sc_pred[key]

            # Per-class P/R/F1, excluding class 0
            class_f1s = []
            total_tp, total_fp, total_fn = 0, 0, 0

            for cls in valid_classes:
                tp = sum(1 for gi, pi in zip(g, p) if gi == cls and pi == cls)
                fp = sum(1 for gi, pi in zip(g, p) if gi != cls and pi == cls)
                fn = sum(1 for gi, pi in zip(g, p) if gi == cls and pi != cls)
                _, _, f1 = _prf1(tp, fp, fn)
                class_f1s.append(f1)
                total_tp += tp
                total_fp += fp
                total_fn += fn

            macro_f1 = float(np.mean(class_f1s)) if class_f1s else 0.0
            _, _, micro_f1 = _prf1(total_tp, total_fp, total_fn)

            sc_results[f"{valence}:{elem}"] = {
                "macro_f1": round(macro_f1, 4),
                "micro_f1": round(micro_f1, 4),
            }

    # Per-valence and overall aggregation
    for valence in ["adaptive", "maladaptive"]:
        macro_f1s = [
            sc_results[f"{valence}:{e}"]["macro_f1"]
            for e in ELEMENTS if f"{valence}:{e}" in sc_results
        ]
        if macro_f1s:
            sc_results[f"{valence}_macro_f1"] = round(float(np.mean(macro_f1s)), 4)

    ada_mf1 = sc_results.get("adaptive_macro_f1", 0.0)
    mal_mf1 = sc_results.get("maladaptive_macro_f1", 0.0)
    sc_results["avg_macro_f1"] = round((ada_mf1 + mal_mf1) / 2, 4)

    # Ranking metric
    ranking = sc_results["avg_macro_f1"]

    return {
        "element_presence": ep_results,
        "subelement_classification": sc_results,
        "t1_1_rank": ranking,
    }


# ── Task 1.2: Presence Rating ───────────────────────────────────

def evaluate_task1_2(
    gold: dict,
    predictions: list[dict],
) -> dict:
    """Evaluate Task 1.2: Presence rating (1-5 ordinal)."""
    pred_by_id = {p["post_id"]: p for p in predictions}

    pairs = {"adaptive": [], "maladaptive": []}

    for pid, g in gold.items():
        ev = g.get("evidence", {})

        for valence_key, valence in [
            ("adaptive-state", "adaptive"),
            ("maladaptive-state", "maladaptive"),
        ]:
            gold_state = ev.get(valence_key, {})
            gold_pres = gold_state.get("Presence")

            if gold_pres is None or not isinstance(gold_pres, (int, float)):
                continue
            gold_pres = int(gold_pres)
            if not (1 <= gold_pres <= 5):
                continue

            pred_entry = pred_by_id.get(pid, {})
            pred_state = pred_entry.get(valence_key, {})
            pred_pres = pred_state.get("Presence", 1)  # default 1 if missing
            pred_pres = int(pred_pres) if pred_pres is not None else 1
            pred_pres = max(1, min(5, pred_pres))

            pairs[valence].append((gold_pres, pred_pres))

    results = {}
    for valence in ["adaptive", "maladaptive"]:
        if not pairs[valence]:
            results[valence] = {"mae": 0.0, "rmse": 0.0, "qwk": 0.0, "spearman": 0.0, "n": 0}
            continue

        golds = [p[0] for p in pairs[valence]]
        preds = [p[1] for p in pairs[valence]]

        errors = [abs(g - p) for g, p in zip(golds, preds)]
        mae = sum(errors) / len(errors)
        rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
        qwk = _qwk(golds, preds)
        spear = _spearman(golds, preds)

        results[valence] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "qwk": round(qwk, 4),
            "spearman": round(spear, 4),
            "n": len(golds),
        }

    # Combined
    all_golds = [p[0] for v in pairs.values() for p in v]
    all_preds = [p[1] for v in pairs.values() for p in v]
    if all_golds:
        errors = [abs(g - p) for g, p in zip(all_golds, all_preds)]
        results["combined"] = {
            "mae": round(sum(errors) / len(errors), 4),
            "rmse": round(math.sqrt(sum(e ** 2 for e in errors) / len(errors)), 4),
            "qwk": round(_qwk(all_golds, all_preds), 4),
            "spearman": round(_spearman(all_golds, all_preds), 4),
            "n": len(all_golds),
        }

    # Ranking metric: mean(adaptive RMSE, maladaptive RMSE) — lower is better
    ada_rmse = results.get("adaptive", {}).get("rmse", 0.0)
    mal_rmse = results.get("maladaptive", {}).get("rmse", 0.0)
    results["t1_2_rank"] = round((ada_rmse + mal_rmse) / 2, 4)

    return results


# ── Task 2: Moments of Change ───────────────────────────────────

def evaluate_task2(
    gold: dict,
    timeline_posts: dict,
    predictions: list[dict],
) -> dict:
    """
    Evaluate Task 2: Switch and Escalation detection.

    Computes both post-level (pooled) and timeline-level (macro-averaged).
    """
    pred_by_id = {p["post_id"]: p for p in predictions}

    results = {}

    for label_name, gold_key, pred_key, pos_val in [
        ("switch", "Switch", "Switch", "S"),
        ("escalation", "Escalation", "Escalation", "E"),
    ]:
        # ── Post-level ──
        tp, fp, fn = 0, 0, 0
        for pid, g in gold.items():
            is_gold = g.get(gold_key, "0") == pos_val
            pred_entry = pred_by_id.get(pid, {})
            is_pred = pred_entry.get(pred_key, "0") == pos_val

            if is_gold and is_pred:
                tp += 1
            elif is_pred and not is_gold:
                fp += 1
            elif is_gold and not is_pred:
                fn += 1

        pl_prec, pl_rec, pl_f1 = _prf1(tp, fp, fn)
        results[f"post_level_{label_name}"] = {
            "precision": round(pl_prec, 4),
            "recall": round(pl_rec, 4),
            "f1": round(pl_f1, 4),
            "support_positive": tp + fn,
            "support_total": len(gold),
        }

        # ── Timeline-level ──
        tl_precs, tl_recs, tl_f1s = [], [], []
        for tid, pids in timeline_posts.items():
            tl_tp, tl_fp, tl_fn = 0, 0, 0
            for pid in pids:
                g = gold.get(pid, {})
                is_gold = g.get(gold_key, "0") == pos_val
                pred_entry = pred_by_id.get(pid, {})
                is_pred = pred_entry.get(pred_key, "0") == pos_val

                if is_gold and is_pred:
                    tl_tp += 1
                elif is_pred and not is_gold:
                    tl_fp += 1
                elif is_gold and not is_pred:
                    tl_fn += 1

            # Edge case: no events in gold or pred
            if tl_tp == 0 and tl_fp == 0 and tl_fn == 0:
                tl_precs.append(1.0)
                tl_recs.append(1.0)
                tl_f1s.append(1.0)
            else:
                p, r, f = _prf1(tl_tp, tl_fp, tl_fn)
                tl_precs.append(p)
                tl_recs.append(r)
                tl_f1s.append(f)

        results[f"timeline_level_{label_name}"] = {
            "precision": round(float(np.mean(tl_precs)), 4) if tl_precs else 0.0,
            "recall": round(float(np.mean(tl_recs)), 4) if tl_recs else 0.0,
            "f1": round(float(np.mean(tl_f1s)), 4) if tl_f1s else 0.0,
            "num_timelines": len(timeline_posts),
        }

    # Macro F1s
    pl_macro_f1 = (
        results["post_level_switch"]["f1"] +
        results["post_level_escalation"]["f1"]
    ) / 2
    tl_macro_f1 = (
        results["timeline_level_switch"]["f1"] +
        results["timeline_level_escalation"]["f1"]
    ) / 2

    results["post_level_macro_f1"] = round(pl_macro_f1, 4)
    results["timeline_level_macro_f1"] = round(tl_macro_f1, 4)
    results["t2_rank"] = round((pl_macro_f1 + tl_macro_f1) / 2, 4)

    return results


# ── Full evaluation ──────────────────────────────────────────────

def run_full_evaluation(
    gold_dir: str,
    task1_pred_path: Optional[str] = None,
    task2_pred_path: Optional[str] = None,
) -> dict:
    """
    Run the full evaluation pipeline.

    Args:
        gold_dir: Directory with gold timeline JSON files.
        task1_pred_path: Path to task1_pred.json.
        task2_pred_path: Path to task2_pred.json.

    Returns:
        Dict with all metrics and ranking scores.
    """
    gold, timeline_posts = load_gold_from_dir(gold_dir)
    results = {}

    if task1_pred_path and os.path.exists(task1_pred_path):
        with open(task1_pred_path) as f:
            t1_preds = json.load(f)
        results["task1_1"] = evaluate_task1_1(gold, t1_preds)
        results["task1_2"] = evaluate_task1_2(gold, t1_preds)
        logger.info(
            f"Task 1.1 Rank (Subelement Avg Macro F1): "
            f"{results['task1_1']['t1_1_rank']}"
        )
        logger.info(
            f"Task 1.2 Rank (Avg RMSE): {results['task1_2']['t1_2_rank']}"
        )

    if task2_pred_path and os.path.exists(task2_pred_path):
        with open(task2_pred_path) as f:
            t2_preds = json.load(f)
        results["task2"] = evaluate_task2(gold, timeline_posts, t2_preds)
        logger.info(
            f"Task 2 Rank (Avg Macro F1): {results['task2']['t2_rank']}"
        )

    return results


def print_evaluation(results: dict):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 65)
    print("CLPsych 2026 — Evaluation Results")
    print("=" * 65)

    if "task1_1" in results:
        r = results["task1_1"]
        print("\n  Task 1.1 — Subelement Classification")
        print("  " + "-" * 45)
        sc = r["subelement_classification"]
        for valence in ["adaptive", "maladaptive"]:
            for elem in ELEMENTS:
                key = f"{valence}:{elem}"
                if key in sc:
                    m = sc[key]
                    print(f"    {key:25s}  macroF1={m['macro_f1']:.4f}  microF1={m['micro_f1']:.4f}")
            v_key = f"{valence}_macro_f1"
            if v_key in sc:
                print(f"    {'':25s}  → {valence} avg: {sc[v_key]:.4f}")
        print(f"\n  ★ Task 1.1 Ranking = {r['t1_1_rank']:.4f}")

    if "task1_2" in results:
        r = results["task1_2"]
        print("\n  Task 1.2 — Presence Rating")
        print("  " + "-" * 45)
        for valence in ["adaptive", "maladaptive", "combined"]:
            if valence in r:
                m = r[valence]
                print(
                    f"    {valence:20s}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
                    f"QWK={m['qwk']:.3f}  Spearman={m['spearman']:.3f}  n={m['n']}"
                )
        print(f"\n  ★ Task 1.2 Ranking = {r['t1_2_rank']:.4f} (lower is better)")

    if "task2" in results:
        r = results["task2"]
        print("\n  Task 2 — Moments of Change")
        print("  " + "-" * 45)
        for level in ["post_level", "timeline_level"]:
            for label in ["switch", "escalation"]:
                key = f"{level}_{label}"
                if key in r:
                    m = r[key]
                    print(
                        f"    {key:35s}  P={m['precision']:.3f}  "
                        f"R={m['recall']:.3f}  F1={m['f1']:.3f}"
                    )
        print(f"    Post-level macro F1:     {r.get('post_level_macro_f1', 0):.4f}")
        print(f"    Timeline-level macro F1: {r.get('timeline_level_macro_f1', 0):.4f}")
        print(f"\n  ★ Task 2 Ranking = {r['t2_rank']:.4f}")

    print("\n" + "=" * 65)


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="CLPsych 2026 — Evaluate predictions against gold labels"
    )
    parser.add_argument(
        "--gold-dir", required=True,
        help="Directory containing gold timeline JSON files",
    )
    parser.add_argument("--task1-pred", default=None, help="Path to task1_pred.json")
    parser.add_argument("--task2-pred", default=None, help="Path to task2_pred.json")
    parser.add_argument("--output", default=None, help="Save results to JSON")
    args = parser.parse_args()

    results = run_full_evaluation(
        args.gold_dir,
        task1_pred_path=args.task1_pred,
        task2_pred_path=args.task2_pred,
    )
    print_evaluation(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
