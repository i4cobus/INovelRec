"""Embedding helpers for compact novel profiles."""

from __future__ import annotations

import math
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

# Tuned for A100-80GB. The previous value of 32 was an RTX 4080 (16 GB) constraint:
# a 4B encoder in fp16 left only ~8 GB for activations. Lower this for smaller cards.
DEFAULT_BATCH_SIZE = 256

# Qwen3-Embedding is instruction-aware: queries carry a task instruction prefix and
# documents do not. sentence-transformers exposes this as encode_query/encode_document,
# which read the prompt templates the model ships with. Encoding both sides identically
# (the previous behaviour) silently runs the model outside its intended mode.
Role = Literal["query", "document", "plain"]


class SupportsEncode(Protocol):
    """Minimal protocol for SentenceTransformer-like models used in tests.

    Only ``encode`` is required. When a model also exposes ``encode_query`` /
    ``encode_document`` those are preferred for the matching role.
    """

    def encode(self, texts: list[str], **kwargs: object) -> object:
        """Encode text strings into dense vectors."""


def load_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str | None = None,
) -> SentenceTransformer:
    """Load a SentenceTransformer embedding model once per process."""

    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)


def open_encode_pool(model: SentenceTransformer, target_devices: list[str] | None = None) -> Any:
    """Start a multi-GPU worker pool for large document encoding jobs.

    Pass the returned pool to ``encode_documents``. Remember to close it with
    ``close_encode_pool``. Single-query encoding should not use a pool: process
    startup dominates the work.
    """

    if target_devices is None:
        import torch

        count = torch.cuda.device_count()
        target_devices = [f"cuda:{index}" for index in range(count)] if count else None
    return model.start_multi_process_pool(target_devices=target_devices)


def close_encode_pool(model: SentenceTransformer, pool: Any) -> None:
    """Shut down a pool created by ``open_encode_pool``."""

    model.stop_multi_process_pool(pool)


def ensure_float32_2d(embeddings: object) -> np.ndarray:
    """Convert model output to a 2D float32 numpy array."""

    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding array, got shape {array.shape}")
    return array


def resolve_encoder(model: SupportsEncode, role: Role) -> Any:
    """Return the role-specific encode callable, falling back to plain ``encode``."""

    if role == "query":
        return getattr(model, "encode_query", model.encode)
    if role == "document":
        return getattr(model, "encode_document", model.encode)
    return model.encode


def encode_texts(
    model: SupportsEncode,
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize_embeddings: bool = True,
    role: Role = "plain",
    pool: Any | None = None,
    chunk_size: int | None = None,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Encode texts into float32 embeddings with optional L2 normalization."""

    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    encoder = resolve_encoder(model, role)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "normalize_embeddings": normalize_embeddings,
        "show_progress_bar": show_progress_bar,
        "convert_to_numpy": True,
    }
    if pool is not None:
        kwargs["pool"] = pool
        if chunk_size is not None:
            kwargs["chunk_size"] = chunk_size

    return ensure_float32_2d(encoder(texts, **kwargs))


def encode_queries(
    model: SupportsEncode,
    texts: list[str],
    batch_size: int = 1,
    normalize_embeddings: bool = True,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Encode search queries using the model's query-side instruction prompt."""

    return encode_texts(
        model,
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        role="query",
        show_progress_bar=show_progress_bar,
    )


def encode_documents(
    model: SupportsEncode,
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize_embeddings: bool = True,
    pool: Any | None = None,
    chunk_size: int | None = None,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Encode indexed documents (novel profiles) without a query instruction prefix."""

    return encode_texts(
        model,
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        role="document",
        pool=pool,
        chunk_size=chunk_size,
        show_progress_bar=show_progress_bar,
    )


def encode_shard(
    model_name: str,
    device: str,
    texts: list[str],
    batch_size: int,
    normalize_embeddings: bool,
    out_path: str,
) -> str:
    """Encode one shard on one GPU and persist it. Module-level so spawn can pickle it."""

    model = load_embedding_model(model_name, device=device)
    vectors = encode_documents(
        model,
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )
    np.save(out_path, vectors)
    return out_path


def encode_documents_multi_gpu(
    model_name: str,
    texts: list[str],
    devices: list[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize_embeddings: bool = True,
    work_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Shard documents across GPUs, one independent worker process per device.

    Replaces ``sentence_transformers.start_multi_process_pool``, which hung here:
    it created its semaphores but never started a worker, leaving the parent
    blocked on an empty queue for an hour with seven idle GPUs. Owning the
    sharding makes failures visible per shard and keeps this step — rerun on every
    profile change — debuggable.

    Row order is preserved: shard *i* covers a contiguous slice and results are
    concatenated in device order.
    """

    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    if devices is None:
        import torch

        count = torch.cuda.device_count()
        devices = [f"cuda:{index}" for index in range(count)] or ["cpu"]
    devices = devices[: max(1, min(len(devices), len(texts)))]

    if len(devices) == 1:
        model = load_embedding_model(model_name, device=devices[0])
        return encode_documents(model, texts, batch_size=batch_size, normalize_embeddings=normalize_embeddings)

    with tempfile.TemporaryDirectory(dir=work_dir) as scratch:
        size = math.ceil(len(texts) / len(devices))
        jobs = []
        for index, device in enumerate(devices):
            shard = texts[index * size : (index + 1) * size]
            if not shard:
                continue
            jobs.append((model_name, device, shard, batch_size, normalize_embeddings, str(Path(scratch) / f"shard{index}.npy")))

        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=get_context("spawn")) as executor:
            futures = {executor.submit(encode_shard, *job): job[1] for job in jobs}
            for future in as_completed(futures):
                device = futures[future]
                future.result()  # surface worker exceptions instead of hanging
                if progress:
                    progress(device)

        return ensure_float32_2d(np.concatenate([np.load(job[5]) for job in jobs], axis=0))
