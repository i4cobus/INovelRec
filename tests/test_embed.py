import numpy as np
import pytest

from src.embed import encode_documents, encode_queries, encode_texts, ensure_float32_2d, resolve_encoder


class FakeModel:
    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        return np.array([[float(len(text)), 1.0] for text in texts], dtype=np.float64)


class RoleAwareModel:
    """Stands in for a Qwen3-Embedding model exposing role-specific encoders."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def _record(self, role: str, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append((role, kwargs))
        return np.zeros((len(texts), 2), dtype=np.float32)

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        return self._record("plain", texts, **kwargs)

    def encode_query(self, texts: list[str], **kwargs: object) -> np.ndarray:
        return self._record("query", texts, **kwargs)

    def encode_document(self, texts: list[str], **kwargs: object) -> np.ndarray:
        return self._record("document", texts, **kwargs)


def test_encode_texts_empty_input() -> None:
    embeddings = encode_texts(FakeModel(), [])
    assert embeddings.shape == (0, 0)
    assert embeddings.dtype == np.float32


def test_encode_texts_converts_to_float32() -> None:
    embeddings = encode_texts(FakeModel(), ["abc", "hello"])
    assert embeddings.dtype == np.float32
    assert embeddings.shape == (2, 2)


def test_encode_texts_rejects_bad_batch_size() -> None:
    with pytest.raises(ValueError):
        encode_texts(FakeModel(), ["abc"], batch_size=0)


def test_ensure_float32_2d_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        ensure_float32_2d(np.zeros((1, 2, 3)))


def test_queries_and_documents_use_role_specific_encoders() -> None:
    model = RoleAwareModel()
    encode_queries(model, ["凡人流 仙侠"])
    encode_documents(model, ["标题：某某\n开篇样本：..."])
    assert [role for role, _ in model.calls] == ["query", "document"]


def test_role_encoders_fall_back_to_plain_encode() -> None:
    """Stub models implementing only ``encode`` must keep working."""

    model = FakeModel()
    assert resolve_encoder(model, "query").__name__ == "encode"
    assert resolve_encoder(model, "document").__name__ == "encode"
    assert encode_queries(model, ["abc"]).shape == (1, 2)


def test_pool_is_only_forwarded_when_provided() -> None:
    model = RoleAwareModel()
    encode_documents(model, ["a"])
    encode_documents(model, ["a"], pool={"sentinel": True}, chunk_size=64)
    without_pool, with_pool = (kwargs for _, kwargs in model.calls)
    assert "pool" not in without_pool
    assert with_pool["pool"] == {"sentinel": True}
    assert with_pool["chunk_size"] == 64
