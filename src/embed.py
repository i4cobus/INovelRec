"""Embedding helpers for compact novel profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"


class SupportsEncode(Protocol):
    """Minimal protocol for SentenceTransformer-like models used in tests."""

    def encode(self, texts: list[str], **kwargs: object) -> object:
        """Encode text strings into dense vectors."""


def load_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str | None = None,
) -> SentenceTransformer:
    """Load a SentenceTransformer embedding model once per process."""

    from sentence_transformers import SentenceTransformer

    kwargs = {"trust_remote_code": True}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)


def ensure_float32_2d(embeddings: object) -> np.ndarray:
    """Convert model output to a 2D float32 numpy array."""

    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding array, got shape {array.shape}")
    return array


def encode_texts(
    model: SupportsEncode,
    texts: list[str],
    batch_size: int = 32,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """Encode texts into float32 embeddings with optional L2 normalization."""

    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return ensure_float32_2d(embeddings)
