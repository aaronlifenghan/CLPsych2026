"""
CLPsych 2026 — System 6 — RAG Retriever

Given a query sequence, retrieves the most similar entries from the RAG index.

Retrieval strategies:
  - semantic: pure cosine similarity (default)
  - filtered: cosine similarity filtered by change_type
  - hybrid: combines semantic similarity with change_type boost
  - diverse: ensures diversity in retrieved examples (different timelines)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .rag_index import (
    IndexEntry,
    RAGIndex,
    _posts_to_clinical_text,
    _posts_to_text,
)

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Retrieves relevant examples from the RAG index for a query sequence.
    """

    def __init__(
        self,
        index: RAGIndex,
        top_k: int = 3,
        strategy: str = "hybrid",
        change_type_boost: float = 0.15,
        sequence_only: bool = True,
        diversity_penalty: float = 0.3,
        use_clinical_text: bool = True,
    ):
        """
        Parameters
        ----------
        index : RAGIndex
            The pre-built RAG index.
        top_k : int
            Number of examples to retrieve.
        strategy : str
            'semantic', 'filtered', 'hybrid', or 'diverse'
        change_type_boost : float
            Score boost for same change_type (used in 'hybrid' strategy).
        sequence_only : bool
            If True, only retrieve sequence entries (with gold summaries).
            If False, also retrieve post_group entries for context.
        diversity_penalty : float
            Penalty for retrieving multiple entries from the same timeline.
        use_clinical_text : bool
            Whether to include ABCD labels in the query text.
        """
        self.index = index
        self.top_k = top_k
        self.strategy = strategy
        self.change_type_boost = change_type_boost
        self.sequence_only = sequence_only
        self.diversity_penalty = diversity_penalty
        self.use_clinical_text = use_clinical_text

    def _embed_query(self, posts: List[Dict[str, Any]]) -> np.ndarray:
        """Embed a query sequence."""
        if self.use_clinical_text:
            text = _posts_to_clinical_text(posts)
        else:
            text = _posts_to_text(posts)
        return self.index.backend.encode([text])[0]

    def _compute_scores(
        self,
        query_embedding: np.ndarray,
        query_change_type: Optional[str] = None,
        query_timeline_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute retrieval scores for all entries.
        Returns array of scores (higher = more relevant).
        """
        if self.index.embeddings is None:
            raise ValueError("Index has no embeddings. Call build_embeddings() first.")

        # Cosine similarity (embeddings are already L2-normalized)
        scores = self.index.embeddings @ query_embedding

        # Apply strategy-specific adjustments
        for i, entry in enumerate(self.index.entries):
            # Filter: only sequence entries if requested
            if self.sequence_only and entry.entry_type != "sequence":
                scores[i] = -np.inf
                continue

            # Don't retrieve entries from the same timeline as the query
            # (prevents data leakage when query is from train timelines)
            if query_timeline_id and entry.timeline_id == query_timeline_id:
                scores[i] = -np.inf
                continue

            # Strategy-specific scoring
            if self.strategy == "filtered":
                # Hard filter: must match change_type
                if query_change_type and entry.change_type != query_change_type:
                    scores[i] = -np.inf

            elif self.strategy == "hybrid":
                # Soft boost for matching change_type
                if query_change_type and entry.change_type == query_change_type:
                    scores[i] += self.change_type_boost

        return scores

    def _apply_diversity(
        self,
        scores: np.ndarray,
        top_indices: np.ndarray,
    ) -> List[int]:
        """
        Re-rank top candidates to ensure diversity (different timelines).
        Uses Maximal Marginal Relevance (MMR)-inspired approach.
        """
        if len(top_indices) <= 1:
            return list(top_indices)

        selected = [top_indices[0]]
        candidates = list(top_indices[1:])

        while len(selected) < self.top_k and candidates:
            best_idx = None
            best_score = -np.inf

            for idx in candidates:
                entry = self.index.entries[idx]
                score = scores[idx]

                # Penalize if same timeline as already selected
                for sel_idx in selected:
                    if self.index.entries[sel_idx].timeline_id == entry.timeline_id:
                        score -= self.diversity_penalty

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                candidates.remove(best_idx)
            else:
                break

        return selected

    def retrieve(
        self,
        posts: List[Dict[str, Any]],
        query_change_type: Optional[str] = None,
        query_timeline_id: Optional[str] = None,
    ) -> List[Tuple[IndexEntry, float]]:
        """
        Retrieve top-k similar entries for a query sequence.

        Parameters
        ----------
        posts : list
            The query sequence's posts (list of post dicts).
        query_change_type : str, optional
            Change type of the query sequence (if known).
        query_timeline_id : str, optional
            Timeline ID of the query (to exclude from results).

        Returns
        -------
        List of (entry, score) tuples, sorted by relevance.
        """
        query_embedding = self._embed_query(posts)
        scores = self._compute_scores(
            query_embedding,
            query_change_type=query_change_type,
            query_timeline_id=query_timeline_id,
        )

        # Get top candidates (fetch more than needed for diversity re-ranking)
        n_candidates = min(self.top_k * 3, len(scores))
        top_indices = np.argsort(scores)[::-1][:n_candidates]

        # Filter out -inf scores
        top_indices = np.array([i for i in top_indices if scores[i] > -np.inf])

        if len(top_indices) == 0:
            logger.warning("No valid retrieval candidates found")
            return []

        # Apply diversity if requested
        if self.strategy == "diverse" or self.diversity_penalty > 0:
            selected_indices = self._apply_diversity(scores, top_indices)
        else:
            selected_indices = list(top_indices[:self.top_k])

        results = [
            (self.index.entries[i], float(scores[i]))
            for i in selected_indices[:self.top_k]
        ]

        if results:
            logger.debug(
                f"Retrieved {len(results)} examples. "
                f"Scores: {[f'{s:.3f}' for _, s in results]}. "
                f"Types: {[e.change_type for e, _ in results]}"
            )

        return results


def format_retrieved_examples(
    retrieved: List[Tuple[IndexEntry, float]],
    strategy: str,
    include_posts: bool = True,
    include_scores: bool = False,
) -> str:
    """
    Format retrieved examples into a few-shot block for the prompt.

    Parameters
    ----------
    retrieved : list
        Output from RAGRetriever.retrieve().
    strategy : str
        Prompt strategy name (affects post formatting).
    include_posts : bool
        Whether to include the full post text (True) or just the summary (False).
    include_scores : bool
        Whether to include similarity scores (for debugging).
    """
    from ..system5.preprocessor import (
        format_post_enriched_compact,
        format_post_zero_shot,
    )

    if not retrieved:
        return ""

    blocks = []
    for i, (entry, score) in enumerate(retrieved, 1):
        parts = [f"### Retrieved Example {i}"]
        if entry.change_type:
            parts.append(f"Change type: {entry.change_type}")
        if include_scores:
            parts.append(f"Relevance: {score:.3f}")
        parts.append("")

        if include_posts and entry.posts:
            # Choose formatter based on strategy
            use_enriched = "baseline2" in strategy
            formatter = format_post_enriched_compact if use_enriched else format_post_zero_shot

            parts.append("**Posts:**")
            post_blocks = "\n\n---\n\n".join(formatter(p) for p in entry.posts)
            parts.append(post_blocks)
            parts.append("")

        if entry.gold_summary:
            parts.append("**Gold summary:**")
            parts.append(entry.gold_summary)

        blocks.append("\n".join(parts))

    header = (
        f"## Retrieved Examples\n\n"
        f"The following {len(blocks)} example(s) are the most similar to your "
        f"input sequence. Use them as reference for format and analytical depth.\n\n"
    )
    return header + "\n\n---\n\n".join(blocks)
