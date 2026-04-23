"""
CLPsych 2026 — System 6 — RAG Index Builder

Builds a searchable vector index from:
  1. Task 3 train fold sequences (59 sequences with gold summaries)
  2. Task 1+2 per-post ABCD annotations (30 timelines × all posts)

Two embedding backends:
  - sentence-transformers (preferred): all-MiniLM-L6-v2, dense cosine similarity
  - sklearn TF-IDF (fallback): sparse vectors, cosine similarity

The index is serialized to disk so it only needs to be built once.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Entry types ──────────────────────────────────────────────────

def _posts_to_text(posts: List[Dict[str, Any]]) -> str:
    """Concatenate post texts for embedding."""
    parts = []
    for p in posts:
        text = p.get("post", "").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _posts_to_clinical_text(posts: List[Dict[str, Any]]) -> str:
    """
    Build a richer text representation including ABCD labels for embedding.
    This captures clinical semantics better than raw post text alone.
    """
    parts = []
    for p in posts:
        text = p.get("post", "").strip()
        ev = p.get("evidence", {})

        # Extract ABCD categories
        abcd_parts = []
        for state_key in ["adaptive-state", "maladaptive-state"]:
            state = ev.get(state_key, {})
            for k, v in state.items():
                if k == "Presence" or not isinstance(v, dict):
                    continue
                cat = v.get("Category", "")
                if cat:
                    abcd_parts.append(cat)

        clinical = " ".join(abcd_parts)
        switch = "Switch" if str(p.get("Switch", "0")) == "1" else ""
        escalation = "Escalation" if str(p.get("Escalation", "0")) == "1" else ""
        change = f" {switch} {escalation}".strip()

        combined = f"{text} {clinical} {change}".strip()
        if combined:
            parts.append(combined)

    return " ".join(parts)


# ── Index entries ────────────────────────────────────────────────

class IndexEntry:
    """One entry in the RAG library."""

    __slots__ = [
        "entry_type",       # "sequence" or "post_group"
        "timeline_id",
        "sequence_id",      # None for post_group entries
        "change_type",      # "Switch" / "Escalation" / None
        "post_indices",     # list of post indices
        "text_for_embed",   # text used for embedding
        "formatted_example",  # pre-formatted few-shot block (for sequence entries)
        "gold_summary",     # gold summary (for sequence entries)
        "posts",            # raw post dicts (for formatting at retrieval time)
    ]

    def __init__(self, **kwargs):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))


# ── Embedding backends ───────────────────────────────────────────

class EmbeddingBackend:
    """Abstract embedding backend."""
    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    @staticmethod
    def load(path: str) -> "EmbeddingBackend":
        raise NotImplementedError


class SentenceTransformerBackend(EmbeddingBackend):
    """Dense embeddings via sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def save(self, path: str):
        data = {"backend": "sentence_transformer", "model_name": self.model_name}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path: str) -> "SentenceTransformerBackend":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return SentenceTransformerBackend(data["model_name"])


class TfidfBackend(EmbeddingBackend):
    """Sparse TF-IDF embeddings via sklearn (fallback)."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )
        self._fitted = False

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        sparse = self.vectorizer.transform(texts)
        # Normalize rows for cosine similarity via dot product
        from sklearn.preprocessing import normalize
        return normalize(sparse, norm="l2").toarray()

    def save(self, path: str):
        data = {"backend": "tfidf", "vectorizer": self.vectorizer}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path: str) -> "TfidfBackend":
        with open(path, "rb") as f:
            data = pickle.load(f)
        backend = TfidfBackend()
        backend.vectorizer = data["vectorizer"]
        backend._fitted = True
        return backend


def get_embedding_backend(name: str = "auto") -> EmbeddingBackend:
    """
    Get embedding backend.
    'auto': try sentence-transformers first, fall back to TF-IDF.
    'sentence_transformer': require sentence-transformers.
    'tfidf': always use TF-IDF.
    """
    if name == "tfidf":
        logger.info("Using TF-IDF embedding backend")
        return TfidfBackend()

    if name in ("auto", "sentence_transformer"):
        try:
            backend = SentenceTransformerBackend()
            logger.info("Using sentence-transformers embedding backend (all-MiniLM-L6-v2)")
            return backend
        except ImportError:
            if name == "sentence_transformer":
                raise
            logger.warning(
                "sentence-transformers not installed. Falling back to TF-IDF.\n"
                "  Install with: pip install sentence-transformers"
            )
            return TfidfBackend()

    raise ValueError(f"Unknown embedding backend: {name}")


# ── RAG Index ────────────────────────────────────────────────────

class RAGIndex:
    """
    The RAG vector index.

    Stores entries + their embeddings for fast retrieval.
    """

    def __init__(self, backend: EmbeddingBackend):
        self.backend = backend
        self.entries: List[IndexEntry] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_sequence_entries(
        self,
        sequences: List[Dict[str, Any]],
        timelines: Dict[str, Dict[str, Any]],
        use_clinical_text: bool = True,
    ):
        """Add Task 3 sequence entries (with gold summaries) to the index."""
        for seq in sequences:
            tid = seq["timeline_id"]
            timeline = timelines.get(tid, {})
            posts = [timeline[pid] for pid in seq["postids"] if pid in timeline]

            if not posts:
                logger.warning(f"No posts found for {tid}/{seq['sequence_id']}")
                continue

            if use_clinical_text:
                text = _posts_to_clinical_text(posts)
            else:
                text = _posts_to_text(posts)

            entry = IndexEntry(
                entry_type="sequence",
                timeline_id=tid,
                sequence_id=seq["sequence_id"],
                change_type=seq.get("change_type"),
                post_indices=seq.get("postindices", []),
                text_for_embed=text,
                gold_summary=seq.get("summary", ""),
                posts=posts,
            )
            self.entries.append(entry)

        logger.info(f"Added {len(sequences)} sequence entries to index")

    def add_post_group_entries(
        self,
        timelines: Dict[str, Dict[str, Any]],
        window_size: int = 3,
        stride: int = 2,
    ):
        """
        Add sliding-window post groups from Task 1+2 timelines.
        These provide additional context from the full timeline history.
        """
        count = 0
        for tid, posts_dict in timelines.items():
            posts_sorted = sorted(posts_dict.values(), key=lambda p: p.get("post_index", 0))

            for i in range(0, max(1, len(posts_sorted) - window_size + 1), stride):
                window = posts_sorted[i:i + window_size]
                if not window:
                    continue

                text = _posts_to_clinical_text(window)

                # Detect if this window has any change events
                has_switch = any(str(p.get("Switch", "0")) == "1" for p in window)
                has_escalation = any(str(p.get("Escalation", "0")) == "1" for p in window)
                if has_switch:
                    change_type = "Switch"
                elif has_escalation:
                    change_type = "Escalation"
                else:
                    change_type = None

                entry = IndexEntry(
                    entry_type="post_group",
                    timeline_id=tid,
                    sequence_id=None,
                    change_type=change_type,
                    post_indices=[p.get("post_index") for p in window],
                    text_for_embed=text,
                    gold_summary=None,
                    posts=window,
                )
                self.entries.append(entry)
                count += 1

        logger.info(f"Added {count} post-group entries from {len(timelines)} timelines")

    def build_embeddings(self):
        """Compute embeddings for all entries."""
        texts = [e.text_for_embed for e in self.entries]
        if not texts:
            logger.warning("No entries to embed")
            return

        # For TF-IDF, fit on all texts first
        if isinstance(self.backend, TfidfBackend) and not self.backend._fitted:
            self.backend.fit(texts)

        self.embeddings = self.backend.encode(texts)
        logger.info(f"Built embeddings: shape={self.embeddings.shape}")

    def save(self, path: str):
        """Save index to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": self.entries,
            "embeddings": self.embeddings,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        # Save backend separately
        backend_path = str(Path(path).with_suffix(".backend.pkl"))
        self.backend.save(backend_path)
        logger.info(f"Index saved to {path}")

    @staticmethod
    def load(path: str) -> "RAGIndex":
        """Load index from disk."""
        backend_path = str(Path(path).with_suffix(".backend.pkl"))
        with open(backend_path, "rb") as f:
            backend_data = pickle.load(f)

        backend_type = backend_data.get("backend", "tfidf")
        if backend_type == "sentence_transformer":
            backend = SentenceTransformerBackend.load(backend_path)
        else:
            backend = TfidfBackend.load(backend_path)

        index = RAGIndex(backend)
        with open(path, "rb") as f:
            data = pickle.load(f)
        index.entries = data["entries"]
        index.embeddings = data["embeddings"]

        logger.info(
            f"Index loaded: {len(index.entries)} entries, "
            f"embeddings shape={index.embeddings.shape if index.embeddings is not None else 'None'}"
        )
        return index


# ── Build function ───────────────────────────────────────────────

def build_rag_index(
    train_sequences_file: str,
    timelines_dir: str,
    embedding_backend: str = "auto",
    include_post_groups: bool = True,
    use_clinical_text: bool = True,
    save_path: Optional[str] = None,
    exclude_sequence_ids: Optional[set] = None,
) -> RAGIndex:
    """
    Build the full RAG index from training data.

    Parameters
    ----------
    train_sequences_file : str
        Path to train_task3_train_fold.json (sequences with gold summaries)
    timelines_dir : str
        Path to train_tasks12/ directory
    embedding_backend : str
        'auto', 'sentence_transformer', or 'tfidf'
    include_post_groups : bool
        Whether to include sliding-window post groups from timelines
    use_clinical_text : bool
        Whether to include ABCD labels in the text used for embedding
    save_path : str, optional
        If provided, save the index to this path
    exclude_sequence_ids : set, optional
        Sequence IDs to exclude (e.g., val fold sequences to prevent leakage)
    """
    backend = get_embedding_backend(embedding_backend)
    index = RAGIndex(backend)

    # Load data
    with open(train_sequences_file, "r", encoding="utf-8") as f:
        sequences = json.load(f)

    # Filter out excluded sequences
    if exclude_sequence_ids:
        original_len = len(sequences)
        sequences = [s for s in sequences if s["sequence_id"] not in exclude_sequence_ids]
        logger.info(f"Excluded {original_len - len(sequences)} sequences from index")

    # Load timelines
    timelines: Dict[str, Dict[str, Any]] = {}
    for fpath in sorted(Path(timelines_dir).glob("*.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tid = data.get("timeline_id") or fpath.stem
        timelines[tid] = {p["post_id"]: p for p in data.get("posts", [])}

    logger.info(f"Loaded {len(sequences)} train sequences, {len(timelines)} timelines")

    # Add entries
    index.add_sequence_entries(sequences, timelines, use_clinical_text=use_clinical_text)

    if include_post_groups:
        index.add_post_group_entries(timelines)

    # Build embeddings
    index.build_embeddings()

    # Save if requested
    if save_path:
        index.save(save_path)

    return index
