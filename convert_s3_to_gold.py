"""
Convert System 3 (LLM) outputs to gold-label format for System 4 training.
Supports majority voting across multiple model outputs.

Usage:
    python convert_s3_to_gold.py \
        --s3-files outputs/system3/gemma2_9b/fewshot/raw_task_1_1.json \
                   outputs/system3/llama3.1/fewshot/raw_task_1_1.json \
                   outputs/system3/qwen3_8b/fewshot/raw_task_1_1.json \
        --train-dir data/train_tasks12 \
        --output-dir data/train_silver
"""

import argparse
import json
import os
import glob
from collections import Counter
from pathlib import Path

ADAPTIVE_LABELS = {
    "A":   {1: "Calm/ laid back", 3: "Sad, Emotional pain, grieving",
            5: "Content, happy, joy, hopeful", 7: "Vigor / energetic",
            9: "Justifiable anger", 11: "Proud", 13: "Feel loved, belong"},
    "B-O": {1: "Relating behavior", 3: "Autonomous or adaptive control behavior"},
    "B-S": {1: "Self care and improvement"},
    "C-O": {1: "Perception of the other as related",
            3: "Perception of the other as facilitating autonomy needs"},
    "C-S": {1: "Self-acceptance and compassion"},
    "D":   {1: "Relatedness", 3: "Autonomy and adaptive control",
            5: "Competence, self esteem, self-care"},
}

MALADAPTIVE_LABELS = {
    "A":   {2: "Anxious/ fearful/ tense", 4: "Depressed, despair, hopeless",
            6: "Mania", 8: "Apathic, don't care, blunted",
            10: "Angry (aggression), disgust, contempt",
            12: "Ashamed, guilty", 14: "Feel lonely"},
    "B-O": {2: "Fight or flight behavior", 4: "Over controlled or controlling behavior"},
    "B-S": {2: "Self harm, neglect and avoidance"},
    "C-O": {2: "Perception of the other as detached or over attached",
            4: "Perception of the other as blocking autonomy needs"},
    "C-S": {2: "Self criticism"},
    "D":   {2: "Expectation that relatedness needs will not be met",
            4: "Expectation that autonomy needs will not be met",
            6: "Expectation that competence needs will not be met"},
}

ELEMENTS = ["A", "B-O", "B-S", "C-O", "C-S", "D"]
VALID_SUBS = {
    "adaptive": {e: set(ADAPTIVE_LABELS[e].keys()) for e in ELEMENTS},
    "maladaptive": {e: set(MALADAPTIVE_LABELS[e].keys()) for e in ELEMENTS},
}


def sub_to_category(element, subelement, valence_labels):
    label = valence_labels.get(element, {}).get(subelement, "")
    return f"({subelement}) {label}" if label else None


def extract_state_votes(state_dict):
    """Extract element->subelement mapping from one assessment's state."""
    if not state_dict:
        return {}
    votes = {}
    for e in state_dict.get("elements", []):
        elem = e.get("element")
        sub = e.get("subelement", 0)
        if elem in ELEMENTS:
            votes[elem] = sub
    return votes


def majority_vote_state(all_votes, valence, valence_labels):
    """
    all_votes: list of dicts {elem -> subelement} from each model
    Returns gold-format state dict or None if all absent.
    """
    if not all_votes:
        return None

    result = {}
    any_present = False

    for elem in ELEMENTS:
        # Collect votes for this element
        elem_votes = [v.get(elem, 0) for v in all_votes]
        # Majority vote — pick most common, break ties toward 0 (absent)
        counter = Counter(elem_votes)
        winner = counter.most_common(1)[0][0]

        # Validate subelement is legal
        valid = VALID_SUBS[valence][elem]
        if winner != 0 and winner not in valid:
            winner = 0

        if winner != 0:
            category = sub_to_category(elem, winner, valence_labels)
            if category:
                result[elem] = {"Category": category}
                any_present = True

    if not any_present and all(v.get(e, 0) == 0 for v in all_votes for e in ELEMENTS):
        return {"Presence": 1}

    # Estimate presence from active element count
    n_present = len(result)
    result["Presence"] = min(1 + n_present, 5)
    return result


def collect_silver_labels(s3_files):
    """
    Collect all assessments from multiple files and majority-vote per post.
    Returns {post_id: {"adaptive-state": ..., "maladaptive-state": ...}}
    """
    # Group by post_id
    from collections import defaultdict
    ada_votes = defaultdict(list)
    mal_votes = defaultdict(list)

    for path in s3_files:
        data = json.load(open(path))
        for tl in data:
            for a in tl.get("assessments", []):
                pid = a.get("post_id")
                if not pid:
                    continue
                ada = extract_state_votes(a.get("adaptive_state"))
                mal = extract_state_votes(a.get("maladaptive_state"))
                if ada is not None:
                    ada_votes[pid].append(ada)
                if mal is not None:
                    mal_votes[pid].append(mal)

    silver = {}
    all_pids = set(ada_votes.keys()) | set(mal_votes.keys())
    for pid in all_pids:
        evidence = {}
        if ada_votes[pid]:
            state = majority_vote_state(ada_votes[pid], "adaptive", ADAPTIVE_LABELS)
            if state:
                evidence["adaptive-state"] = state
        if mal_votes[pid]:
            state = majority_vote_state(mal_votes[pid], "maladaptive", MALADAPTIVE_LABELS)
            if state:
                evidence["maladaptive-state"] = state
        if evidence:
            silver[pid] = evidence

    return silver


def build_post_index(train_dir):
    post_index = {}
    timeline_index = {}
    files = sorted(glob.glob(os.path.join(train_dir, "*.json")))
    for f in files:
        tl = json.load(open(f))
        if not isinstance(tl, dict):
            continue
        tid = tl.get("timeline_id", Path(f).stem)
        timeline_index[tid] = tl
        for p in tl.get("posts", []):
            pid = p.get("post_id")
            if pid:
                post_index[pid] = {"timeline_id": tid, "post": p}
    return post_index, timeline_index


def merge_and_write(timeline_index, silver_evidence, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    n_gold = n_silver = n_empty = 0

    for tid, tl in timeline_index.items():
        new_posts = []
        for p in tl.get("posts", []):
            pid = p.get("post_id")
            new_p = dict(p)
            gold_ev = p.get("evidence", {})
            has_gold = (
                gold_ev.get("adaptive-state", {}).get("Presence") is not None or
                gold_ev.get("maladaptive-state", {}).get("Presence") is not None
            )
            if has_gold:
                n_gold += 1
            elif pid in silver_evidence:
                new_p["evidence"] = silver_evidence[pid]
                n_silver += 1
            else:
                n_empty += 1
            new_posts.append(new_p)

        new_tl = dict(tl)
        new_tl["posts"] = new_posts
        with open(os.path.join(output_dir, f"{tid}.json"), "w") as f:
            json.dump(new_tl, f, indent=2)

    print(f"Written {len(timeline_index)} timelines to {output_dir}")
    print(f"  Gold labels kept:    {n_gold}")
    print(f"  Silver labels added: {n_silver}")
    print(f"  No labels:           {n_empty}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-files", nargs="+", required=True,
                        help="One or more System 3 raw_task_1_1.json files")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    print(f"Loading {len(args.s3_files)} System 3 file(s)...")
    silver = collect_silver_labels(args.s3_files)
    print(f"  {len(silver)} posts have silver labels (majority vote)")

    print(f"Loading training data from {args.train_dir}...")
    _, timeline_index = build_post_index(args.train_dir)
    print(f"  {len(timeline_index)} timelines loaded")

    print("Merging gold + silver labels...")
    merge_and_write(timeline_index, silver, args.output_dir)


if __name__ == "__main__":
    main()
