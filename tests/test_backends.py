import pytest

from src.backends import BACKENDS, backend_uses_gpu, create_matcher, normalize_backend
from src.http_matcher import OpenAICompatibleMatcher


def test_normalize_backend_accepts_known_names() -> None:
    assert normalize_backend("HTTP") == "http"
    assert normalize_backend(" transformers ") == "transformers"


def test_normalize_backend_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        normalize_backend("ollama")


def test_all_declared_backends_normalize() -> None:
    assert [normalize_backend(name) for name in BACKENDS] == list(BACKENDS)


def test_create_http_matcher_without_touching_the_network() -> None:
    matcher = create_matcher(
        backend="http",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        base_url="http://127.0.0.1:9999/v1/",
        api_key="k",
        max_workers=4,
    )
    assert isinstance(matcher, OpenAICompatibleMatcher)
    assert matcher.provider == "openai_compatible"
    assert matcher.max_workers == 4
    assert matcher.transport.model == "Qwen/Qwen3-4B-Instruct-2507"
    assert matcher.transport.base_url == "http://127.0.0.1:9999/v1"


def test_http_matcher_satisfies_the_three_protocols() -> None:
    matcher = create_matcher(backend="http", model_name="m", base_url="http://x/v1")
    for method in ("score", "expand_queries", "generate"):
        assert callable(getattr(matcher, method))
    assert callable(getattr(matcher, "score_many"))


def test_backend_uses_gpu_only_for_in_process_weights() -> None:
    assert backend_uses_gpu("transformers") is True
    assert backend_uses_gpu("http") is False


def test_create_matcher_rejects_unknown_backend_before_importing_anything() -> None:
    with pytest.raises(ValueError):
        create_matcher(backend="nope", model_name="m")


def test_http_matcher_is_its_own_explanation_generator() -> None:
    from src.backends import as_explanation_generator

    matcher = create_matcher(backend="http", model_name="m", base_url="http://x/v1")
    assert as_explanation_generator(matcher) is matcher


def test_non_generating_matcher_gets_wrapped() -> None:
    from src.backends import as_explanation_generator

    class Bare:
        provider = "bare"

        def generate_response(self, prompt: str, max_new_tokens: int | None = None) -> str:
            return "wrapped:" + prompt

    generator = as_explanation_generator(Bare())
    assert generator.generate("hi", max_new_tokens=8) == "wrapped:hi"


def test_thinking_is_disabled_by_default() -> None:
    """A reasoning trace eats a small token budget and the JSON never arrives —
    which surfaces as a plausible 0.0 score, not as an error."""

    matcher = create_matcher(backend="http", model_name="Qwen/Qwen3-32B", base_url="http://x/v1")
    assert matcher.transport.extra_body["chat_template_kwargs"] == {"enable_thinking": False}


def test_thinking_can_be_re_enabled() -> None:
    matcher = create_matcher(
        backend="http", model_name="m", base_url="http://x/v1", enable_thinking=True
    )
    assert "chat_template_kwargs" not in matcher.transport.extra_body


def test_explicit_extra_body_wins() -> None:
    matcher = create_matcher(
        backend="http",
        model_name="m",
        base_url="http://x/v1",
        extra_body={"chat_template_kwargs": {"enable_thinking": True}, "top_p": 0.8},
    )
    assert matcher.transport.extra_body["chat_template_kwargs"] == {"enable_thinking": True}
    assert matcher.transport.extra_body["top_p"] == 0.8
