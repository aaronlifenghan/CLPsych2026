"""
CLPsych 2026 — System 5 — Few-shot example selector

Selects gold training examples from train_task3.json to prepend to prompts.

Selection strategy: "diverse"
  - Pick one Switch sequence and one Escalation sequence
  - Prefer shorter sequences (fewer posts) to save context space
  - Always use the same pair for reproducibility (deterministic by sort order)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .preprocessor import (
    format_post_zero_shot,
    format_post_enriched,
    load_timelines,
)


def load_gold_sequences(train_sequences_file: str) -> List[Dict[str, Any]]:
    """Load gold training sequences (train_task3.json)."""
    with open(train_sequences_file, "r", encoding="utf-8") as f:
        return json.load(f)


def select_examples(
    gold_sequences: List[Dict[str, Any]],
    n: int = 2,
    strategy: str = "diverse",
) -> List[Dict[str, Any]]:
    """
    Select n gold sequences to use as few-shot examples.

    strategy='diverse': pick one Switch and one Escalation (shortest first).
    strategy='random':  pick the first n sequences.
    """
    if strategy == "random" or n <= 1:
        return gold_sequences[:n]

    # Diverse: one Switch + one Escalation, shortest sequences preferred
    switches = sorted(
        [s for s in gold_sequences if s.get("change_type") == "Switch"],
        key=lambda s: len(s["postids"]),
    )
    escalations = sorted(
        [s for s in gold_sequences if s.get("change_type") == "Escalation"],
        key=lambda s: len(s["postids"]),
    )

    selected = []
    if switches:
        selected.append(switches[0])
    if escalations and len(selected) < n:
        selected.append(escalations[0])
    # Fill remaining with whatever is available
    remaining = [s for s in gold_sequences if s not in selected]
    while len(selected) < n and remaining:
        selected.append(remaining.pop(0))

    return selected[:n]


def format_example(
    seq: Dict[str, Any],
    timelines: Dict[str, Dict[str, Any]],
    strategy: str,
) -> str:
    """
    Format one gold example (sequence + gold summary) for inclusion in a prompt.
    The format depends on the prompt strategy.
    """
    tid = seq["timeline_id"]
    seq_id = seq["sequence_id"]
    change_type = seq.get("change_type", "unknown")
    gold_summary = seq.get("summary", "")

    timeline = timelines.get(tid, {})
    posts = [timeline[pid] for pid in seq["postids"] if pid in timeline]

    use_enriched = strategy == "baseline2_style"
    formatter = format_post_enriched if use_enriched else format_post_zero_shot
    post_blocks = "\n\n---\n\n".join(formatter(p) for p in posts)

    lines = [
        f"### Example (timeline={tid}, sequence={seq_id}, change_type={change_type})",
        "",
        "**Posts:**",
        post_blocks,
        "",
        "**Gold summary:**",
        gold_summary,
    ]
    return "\n".join(lines)


def build_few_shot_block(
    gold_sequences: List[Dict[str, Any]],
    timelines: Dict[str, Dict[str, Any]],
    strategy: str,
    n: int = 2,
    selection: str = "diverse",
) -> str:
    """
    Build the full few-shot examples block to inject into a prompt template.
    Returns an empty string if n=0 (zero-shot strategies).
    """
    if n == 0:
        return ""

    examples = select_examples(gold_sequences, n=n, strategy=selection)
    blocks = [
        format_example(ex, timelines, strategy)
        for ex in examples
    ]
    header = (
        f"## Few-shot Examples\n\n"
        f"The following {len(blocks)} example(s) show the expected output format.\n\n"
    )
    return header + "\n\n".join(blocks)
