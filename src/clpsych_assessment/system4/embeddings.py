"""
Feature extraction for System 4.

Supports two embedding backends:
  1. TF-IDF (default, no GPU needed)
  2. Sentence-Transformers (optional, needs `pip install sentence-transformers`)

Both produce a fixed-size vector per post that feeds into XGBoost.
"""

import logging
import os
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


# ── Text cleaning ────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Minimal cleaning: lowercase, collapse whitespace, strip URLs."""
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text


# ── Embedding backends ───────────────────────────────────────────

class TfidfEmbedder:
    """TF-IDF bag-of-words embeddings — always available."""

    def __init__(self, max_features: int = 2048, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            min_df=2,
            max_df=0.95,
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        cleaned = [clean_text(t) for t in texts]
        self.vectorizer.fit(cleaned)
        self._fitted = True
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        cleaned = [clean_text(t) for t in texts]
        return self.vectorizer.transform(cleaned).toarray().astype(np.float32)

    @property
    def dim(self) -> int:
        return len(self.vectorizer.vocabulary_)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: str) -> "TfidfEmbedder":
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self._fitted = True
        return self


class SentenceTransformerEmbedder:
    """Sentence-Transformer embeddings — optional, higher quality."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        # No fitting needed for pretrained models
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        cleaned = [clean_text(t) for t in texts]
        return self.model.encode(
            cleaned,
            show_progress_bar=False,
            batch_size=32,
            normalize_embeddings=True,
        ).astype(np.float32)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def save(self, path: str):
        # Just save the model name — the model itself is from HF hub
        with open(path, "w") as f:
            f.write(self.model_name)

    def load(self, path: str) -> "SentenceTransformerEmbedder":
        with open(path) as f:
            name = f.read().strip()
        self.__init__(name)
        return self


# ── Combined embedder ────────────────────────────────────────────

class CombinedEmbedder:
    """
    Concatenates TF-IDF and SBERT embeddings.
    Gives XGBoost both keyword signals (TF-IDF) and semantic signals (SBERT).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_features: int = 2048):
        self.tfidf = TfidfEmbedder(max_features=max_features)
        self.sbert = SentenceTransformerEmbedder(model_name)
        self.model_name = model_name
        self.max_features = max_features

    def fit(self, texts: list[str]) -> "CombinedEmbedder":
        self.tfidf.fit(texts)
        self.sbert.fit(texts)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        tfidf_vecs = self.tfidf.transform(texts)
        sbert_vecs = self.sbert.transform(texts)
        return np.concatenate([tfidf_vecs, sbert_vecs], axis=1).astype(np.float32)

    @property
    def dim(self) -> int:
        return self.tfidf.dim + self.sbert.dim

    def save(self, path: str):
        tfidf_path = path + ".tfidf.pkl"
        sbert_path = path + ".sbert.txt"
        self.tfidf.save(tfidf_path)
        self.sbert.save(sbert_path)
        with open(path, "w") as f:
            f.write(f"{self.model_name}\n{self.max_features}\n")

    def load(self, path: str) -> "CombinedEmbedder":
        with open(path) as f:
            lines = f.read().strip().splitlines()
        self.model_name = lines[0]
        self.max_features = int(lines[1])
        self.tfidf = TfidfEmbedder(max_features=self.max_features)
        self.tfidf.load(path + ".tfidf.pkl")
        self.sbert = SentenceTransformerEmbedder(self.model_name)
        return self


# ── Factory ──────────────────────────────────────────────────────

def create_embedder(
    backend: str = "tfidf",
    model_name: str = "all-MiniLM-L6-v2",
    max_features: int = 2048,
) -> TfidfEmbedder | SentenceTransformerEmbedder | CombinedEmbedder:
    """
    Create an embedder.

    Args:
        backend: "tfidf", "sbert", or "combined" (TF-IDF + SBERT concatenated)
        model_name: Sentence-transformer model name (only for sbert/combined)
        max_features: TF-IDF vocabulary size (only for tfidf/combined)
    """
    if backend == "sbert":
        logger.info(f"Using Sentence-Transformer: {model_name}")
        return SentenceTransformerEmbedder(model_name)
    elif backend == "combined":
        logger.info(f"Using Combined TF-IDF (max_features={max_features}) + SBERT ({model_name})")
        return CombinedEmbedder(model_name=model_name, max_features=max_features)
    else:
        logger.info(f"Using TF-IDF (max_features={max_features})")
        return TfidfEmbedder(max_features=max_features)
