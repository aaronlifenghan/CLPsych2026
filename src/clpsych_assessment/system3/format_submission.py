"""
Convert pipeline outputs to CLPsych 2026 shared task submission format.

Produces:
  - task1_pred.json  (ABCD elements + subelements + presence rating)
  - task2_pred.json  (Switch / Escalation labels)

See the shared task README §9 for the exact schema.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Subelement schema (matches shared task §8) ──────────────────

# Maps (valence, element) -> {category_substring: subelement_int}
# We use substring matching because LLM outputs may not exactly match.

ADAPTIVE_SUBELEMENTS = {
    "A": {
        "calm": 1, "laid back": 1,
        "sad": 3, "emotional pain": 3, "grieving": 3,
        "content": 5, "happy": 5, "joy": 5, "hopeful": 5,
        "vigor": 7, "energetic": 7,
        "justifiable anger": 9, "assertive anger": 9, "outrage": 9,
        "proud": 11,
        "feel loved": 13, "belong": 13,
    },
    "B-O": {
        "relating": 1, "relating behavior": 1,
        "autonomous": 3, "adaptive control behavior": 3,
    },
    "B-S": {
        "self care": 1, "self-care": 1, "improvement": 1,
    },
    "C-O": {
        "perception of the other as related": 1, "as related": 1,
        "facilitating autonomy": 3,
    },
    "C-S": {
        "self-acceptance": 1, "self acceptance": 1, "compassion": 1,
    },
    "D": {
        "relatedness": 1,
        "autonomy": 3, "adaptive control": 3,
        "competence": 5, "self esteem": 5, "self-esteem": 5, "self-care": 5,
    },
}

MALADAPTIVE_SUBELEMENTS = {
    "A": {
        "anxious": 2, "fearful": 2, "tense": 2,
        "depressed": 4, "despair": 4, "hopeless": 4,
        "mania": 6,
        "apathic": 8, "apathetic": 8, "don't care": 8, "blunted": 8,
        "angry": 10, "aggression": 10, "disgust": 10, "contempt": 10,
        "ashamed": 12, "guilty": 12,
        "lonely": 14, "feel lonely": 14,
    },
    "B-O": {
        "fight": 2, "flight": 2, "fight or flight": 2,
        "over controlled": 4, "overcontrolled": 4, "controlling": 4,
    },
    "B-S": {
        "self harm": 2, "self-harm": 2, "neglect": 2, "avoidance": 2,
    },
    "C-O": {
        "detached": 2, "over attached": 2, "overattached": 2,
        "blocking autonomy": 4,
    },
    "C-S": {
        "self criticism": 2, "self-criticism": 2,
    },
    "D": {
        "relatedness": 2, "relatedness needs will not": 2,
        "autonomy": 4, "autonomy needs will not": 4,
        "competence": 6, "competence needs will not": 6,
    },
}

# Number-based fallback: extract subelement from "(N) ..." category strings
_NUM_PATTERN = re.compile(r"^\((\d+)\)")


def _resolve_subelement(
    category_str: str,
    element: str,
    valence: str,
) -> Optional[int]:
    """
    Resolve a category string to a subelement integer.

    Tries:
      1. Extract number from "(N) ..." format
      2. Substring match against the schema
      3. Return the first valid subelement as fallback
    """
    if not category_str:
        return None

    cat_lower = category_str.lower().strip()

    # Method 1: number prefix like "(4) Depressed, despair, hopeless"
    m = _NUM_PATTERN.match(category_str.strip())
    if m:
        return int(m.group(1))

    # Method 2: substring match
    schema = (
        ADAPTIVE_SUBELEMENTS if valence == "adaptive" else MALADAPTIVE_SUBELEMENTS
    )
    elem_map = schema.get(element, {})
    for substring, sub_id in elem_map.items():
        if substring in cat_lower:
            return sub_id

    # Method 3: fallback to first valid subelement
    if elem_map:
        first = list(set(elem_map.values()))[0]
        logger.warning(
            f"Could not resolve '{category_str}' for {valence}/{element}, "
            f"defaulting to subelement {first}"
        )
        return first

    return None


# ── Conversion functions ─────────────────────────────────────────

def convert_to_task1_submission(
    pipeline_output: list[dict],
) -> list[dict]:
    """
    Convert pipeline predictions to task1_pred.json format.

    Input: list of timeline dicts from the pipeline, each with:
        {"timeline_id": ..., "assessments": [{post_id, adaptive_state, ...}, ...]}
      OR the format from run_pipeline.py (my-version):
        [{"timeline_id": ..., "posts": [{"post_id": ..., "evidence": {...}}, ...]}]

    Output: flat list matching §9 Task 1 submission format.
    """
    entries = []

    for tl in pipeline_output:
        timeline_id = tl["timeline_id"]

        # Support both pipeline output formats
        posts = tl.get("assessments") or tl.get("posts", [])

        for post in posts:
            post_id = post.get("post_id", "")
            entry = {
                "timeline_id": timeline_id,
                "post_id": post_id,
            }

            # Handle system3 format (from CLPsychPipeline)
            if "adaptive_state" in post or "maladaptive_state" in post:
                entry.update(
                    _convert_system3_post_task1(post)
                )
            # Handle my-version format (evidence-based)
            elif "evidence" in post:
                entry.update(
                    _convert_evidence_post_task1(post)
                )

            # Only include if at least one state has content
            has_adaptive = "adaptive-state" in entry and entry["adaptive-state"]
            has_maladaptive = "maladaptive-state" in entry and entry["maladaptive-state"]
            if has_adaptive or has_maladaptive:
                entries.append(entry)

    return entries


def _convert_system3_post_task1(post: dict) -> dict:
    """Convert a system3 pipeline post (Pydantic-dumped) to task1 format."""
    result = {}

    for valence_key, state_key in [
        ("adaptive-state", "adaptive_state"),
        ("maladaptive-state", "maladaptive_state"),
    ]:
        state = post.get(state_key)
        if not state:
            continue

        valence = "adaptive" if "adaptive" in valence_key else "maladaptive"
        elements = state.get("elements", [])

        # Filter to present elements (subelement != 0)
        present = [e for e in elements if e.get("subelement", 0) != 0]
        if not present:
            continue

        state_out = {}
        # Presence: count present elements -> estimate, or use from data
        # For now estimate from element count
        state_out["Presence"] = min(5, max(1, len(present) + 1))

        for elem in present:
            element_name = elem["element"]
            sub = elem["subelement"]
            state_out[element_name] = {"subelement": sub}

        result[valence_key] = state_out

    return result


def _convert_evidence_post_task1(post: dict) -> dict:
    """Convert a my-version pipeline post (evidence format) to task1 format."""
    result = {}
    evidence = post.get("evidence", {})

    for valence_key in ["adaptive-state", "maladaptive-state"]:
        state = evidence.get(valence_key, {})
        if not state:
            continue

        valence = "adaptive" if "adaptive" in valence_key else "maladaptive"
        presence = state.get("Presence")

        # Check if there are any elements
        elements = {
            k: v for k, v in state.items()
            if k != "Presence" and isinstance(v, dict)
        }

        if not elements and (presence is None or presence <= 1):
            continue

        state_out = {}
        if presence is not None:
            state_out["Presence"] = int(presence)

        for elem_key, elem_val in elements.items():
            category = elem_val.get("Category", "")
            sub_id = _resolve_subelement(category, elem_key, valence)
            if sub_id is not None:
                state_out[elem_key] = {"subelement": sub_id}

        if state_out.get("Presence") or any(
            k != "Presence" for k in state_out
        ):
            result[valence_key] = state_out

    return result


def convert_to_task2_submission(
    pipeline_output: list[dict],
) -> list[dict]:
    """
    Convert pipeline predictions to task2_pred.json format.

    Output: flat list of {timeline_id, post_id, Switch, Escalation}.
    """
    entries = []

    for tl in pipeline_output:
        timeline_id = tl["timeline_id"]
        posts = tl.get("assessments") or tl.get("posts", [])

        for post in posts:
            post_id = post.get("post_id", "")

            # System3 format uses bool; my-version uses string
            switch = post.get("Switch", post.get("switch", "0"))
            escalation = post.get("Escalation", post.get("escalation", "0"))

            # Normalize booleans to "S"/"0" and "E"/"0"
            if isinstance(switch, bool):
                switch = "S" if switch else "0"
            elif switch not in ("S", "0"):
                switch = "0"

            if isinstance(escalation, bool):
                escalation = "E" if escalation else "0"
            elif escalation not in ("E", "0"):
                escalation = "0"

            entries.append({
                "timeline_id": timeline_id,
                "post_id": post_id,
                "Switch": switch,
                "Escalation": escalation,
            })

    return entries


# ── Write submission files ───────────────────────────────────────

def write_submission(
    pipeline_output: list[dict],
    output_dir: str,
    task1: bool = True,
    task2: bool = True,
):
    """
    Write task1_pred.json and task2_pred.json to output_dir.

    Args:
        pipeline_output: Pipeline results (list of timeline dicts).
        output_dir: Directory to write submission files.
        task1: Whether to write task1_pred.json.
        task2: Whether to write task2_pred.json.
    """
    os.makedirs(output_dir, exist_ok=True)

    if task1:
        t1 = convert_to_task1_submission(pipeline_output)
        path = os.path.join(output_dir, "task1_pred.json")
        with open(path, "w") as f:
            json.dump(t1, f, indent=2)
        logger.info(f"Task 1 submission: {len(t1)} entries → {path}")

    if task2:
        t2 = convert_to_task2_submission(pipeline_output)
        path = os.path.join(output_dir, "task2_pred.json")
        with open(path, "w") as f:
            json.dump(t2, f, indent=2)
        logger.info(f"Task 2 submission: {len(t2)} entries → {path}")


import os  # noqa: E402 (needed at module level for write_submission)


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Convert pipeline output to shared task submission format"
    )
    parser.add_argument("input", help="Pipeline output JSON file")
    parser.add_argument(
        "--output-dir", default="submission",
        help="Directory for submission files (default: submission/)",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    write_submission(data, args.output_dir)
    print(f"\nSubmission files written to {args.output_dir}/")
