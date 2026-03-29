"""
CLPsych 2026 Data Preprocessor
Handles loading and formatting social media timeline data for LLM processing.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def load_timeline_data(file_path: str) -> Dict[str, Any]:
    """Load timeline data from a JSON file"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_post_for_llm(post: Dict[str, Any], post_id: str) -> str:
    """
    Format a single post for LLM consumption.

    Args:
        post: Post data dictionary
        post_id: Identifier for the post

    Returns:
        Formatted string representation of the post
    """
    text = post.get("post", "")
    date = post.get("date", "")

    formatted = f"Post ID: {post_id}\n"
    if date:
        formatted += f"Date: {date}\n"
    formatted += f"Content: {text}\n"

    return formatted


def format_timeline_for_llm(
    posts: List[Dict[str, Any]],
    post_ids: List[str],
    context_window: int = 5,
    target_idx: int = 0,
) -> str:
    """
    Format a window of posts around a target post for context.

    Args:
        posts: List of post data dictionaries
        post_ids: List of post identifiers
        context_window: Number of preceding posts to include for context
        target_idx: Index of the target post to assess

    Returns:
        Formatted string with context and target post
    """
    start_idx = max(0, target_idx - context_window)
    context_posts = []

    for i in range(start_idx, target_idx):
        context_posts.append(format_post_for_llm(posts[i], post_ids[i]))

    target_post = format_post_for_llm(posts[target_idx], post_ids[target_idx])

    formatted = ""
    if context_posts:
        formatted += "--- PRECEDING POSTS (for context) ---\n\n"
        formatted += "\n".join(context_posts)
        formatted += "\n--- TARGET POST (assess this post) ---\n\n"

    formatted += target_post

    return formatted
