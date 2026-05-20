import numpy as np
import pytest

from src.embed import encode_texts, ensure_float32_2d


class FakeModel:
    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        return np.array([[float(len(text)), 1.0] for text in texts], dtype=np.float64)


def test_encode_texts_empty_input() -> None:
    embeddings = encode_texts(FakeModel(), [])
    assert embeddings.shape == (0, 0)
    assert embeddings.dtype == np.float32


def test_encode_texts_converts_to_float32() -> None:
    embeddings = encode_texts(FakeModel(), ["abc", "hello"])
    assert embeddings.dtype == np.float32
    assert embeddings.shape == (2, 2)


def test_ensure_float32_2d_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        ensure_float32_2d(np.zeros((1, 2, 3)))

