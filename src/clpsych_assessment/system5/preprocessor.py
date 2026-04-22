"""
CLPsych 2026 — System 5 — Preprocessor for Task 3.1

Handles:
  - Loading per-timeline JSON files (train_tasks12 / test_tasks12nolabels)
  - Loading sequence files (train_task3.json / test_task3nolabels.json)
  - Formatting post context for each prompt strategy
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Data loading ─────────────────────────────────────────────────

def load_timelines(timelines_dir: str) -> Dict[str, Dict[str, Any]]:
    """Return {timeline_id: {post_id: post_dict}}"""
    timelines: Dict[str, Dict[str, Any]] = {}
    for fpath in sorted(Path(timelines_dir).glob("*.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tid = data.get("timeline_id") or fpath.stem
        timelines[tid] = {p["post_id"]: p for p in data.get("posts", [])}
    return timelines


def load_sequences(sequences_file: str) -> List[Dict[str, Any]]:
    """Load the task3 sequences file (train or test)."""
    with open(sequences_file, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_sequence_posts(
    seq: Dict[str, Any],
    timelines: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return ordered list of post dicts for a sequence. Missing posts are skipped."""
    tid = seq["timeline_id"]
    timeline = timelines.get(tid, {})
    posts = []
    for pid in seq["postids"]:
        post = timeline.get(pid)
        if post:
            posts.append(post)
        else:
            print(f"  WARNING: post {pid} not found in timeline {tid}")
    return posts


# ── Presence label helper ─────────────────────────────────────────

def _presence_label(score: Optional[int]) -> str:
    """Convert numeric presence score to a descriptive label."""
    if score is None or score <= 1:
        return "absent"
    if score == 2:
        return "subtle"
    if score == 3:
        return "moderate"
    if score == 4:
        return "dominant"
    return "highly dominant"


# ── Post formatters ───────────────────────────────────────────────

def format_post_zero_shot(post: Dict[str, Any]) -> str:
    """
    Format for zero-shot / few-shot / colleague prompts.
    Includes post text, Switch/Escalation labels, wellbeing.
    Does NOT include ABCD evidence — model infers from text.
    """
    wb = post.get("Well-being")
    wb_str = str(wb) if wb is not None else "N/A"
    lines = [
        f"[Post {post['post_index']} | {post.get('date', '')}]",
        f"Switch={post.get('Switch', '0')}  "
        f"Escalation={post.get('Escalation', '0')}  "
        f"Well-being={wb_str}",
        f"Text: {post.get('post', '')}",
    ]
    return "\n".join(lines)


def format_post_enriched_compact(post: Dict[str, Any]) -> str:
    """
    Compact enriched format for baseline2_style.
    Includes ABCD category labels but omits full evidence spans
    to prevent context window overflow on long sequences.
    """
    wb = post.get("Well-being")
    wb_str = str(wb) if wb is not None else "N/A"
    ev = post.get("evidence", {})

    def fmt_state(state: dict, label: str) -> str:
        presence = state.get("Presence", 1)
        parts = []
        for k, v in state.items():
            if k == "Presence" or not isinstance(v, dict):
                continue
            cat = v.get("Category", "")
            parts.append(f"{k}: {cat}")
        body = ", ".join(parts) if parts else "(none)"
        return f"  {label} [{_presence_label(presence)}]: {body}"

    lines = [
        f"[Post {post['post_index']} | Switch={post.get('Switch','0')} "
        f"Escalation={post.get('Escalation','0')} Well-being={wb_str}]",
        fmt_state(ev.get("adaptive-state",   {}), "Adaptive"),
        fmt_state(ev.get("maladaptive-state", {}), "Maladaptive"),
        f"  Text: {post.get('post', '')}",
    ]
    return "\n".join(lines)


def format_post_enriched(post: Dict[str, Any]) -> str:
    """
    Full enriched format with ABCD labels + evidence spans.
    Used by few-shot example display (not sequence body).
    """
    wb = post.get("Well-being")
    wb_str = str(wb) if wb is not None else "N/A"
    ev = post.get("evidence", {})

    def fmt_state(state: dict, label: str) -> str:
        presence = state.get("Presence", 1)
        lines = [f"  {label} self-state [{_presence_label(presence)}]:"]
        for k, v in state.items():
            if k == "Presence" or not isinstance(v, dict):
                continue
            cat  = v.get("Category", "")
            evid = v.get("highlighted_evidence", "").strip()
            line = f"    {k}: {cat}"
            if evid:
                line += f'  → "{evid}"'
            lines.append(line)
        if len(lines) == 1:
            lines.append("    (no elements)")
        return "\n".join(lines)

    lines = [
        f"[Post {post['post_index']} | {post.get('date', '')}]",
        f"Switch={post.get('Switch', '0')}  "
        f"Escalation={post.get('Escalation', '0')}  "
        f"Well-being={wb_str}",
        fmt_state(ev.get("adaptive-state",   {}), "Adaptive"),
        fmt_state(ev.get("maladaptive-state", {}), "Maladaptive"),
        f"Text: {post.get('post', '')}",
    ]
    return "\n".join(lines)


# ── Sequence formatter ────────────────────────────────────────────

def format_sequence_for_prompt(
    posts: List[Dict[str, Any]],
    strategy: str,
) -> str:
    """
    Format all posts in a sequence for the LLM prompt.

    Formatter used per strategy:
      zero_shot_direct / zero_shot_cot / few_shot_direct /
      few_shot_cot / colleague_style  → format_post_zero_shot
      baseline2_style                 → format_post_enriched_compact
    """
    formatter = (
        format_post_enriched_compact
        if strategy == "baseline2_style"
        else format_post_zero_shot
    )
    blocks = [formatter(p) for p in posts]
    return "\n\n---\n\n".join(blocks)