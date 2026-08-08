import json
import threading
from typing import Any

import pytest

from src.http_matcher import (
    HTTPChatTransport,
    OpenAICompatibleMatcher,
    extract_message_content,
    parse_match_response,
)
from src.llm_matcher import LLMMatchResult


MATCH_JSON = json.dumps(
    {
        "llm_match_score": 0.8,
        "confidence": "high",
        "matched_preferences": ["仙侠"],
        "violated_preferences": [],
        "risk_flags": [],
        "reason": "题材吻合。",
    },
    ensure_ascii=False,
)


class FakeTransport:
    """Records prompts and replays canned responses without touching the network."""

    def __init__(self, responses: list[str] | str = MATCH_JSON, fail_on: set[int] | None = None) -> None:
        self.responses = responses
        self.fail_on = fail_on or set()
        self.prompts: list[str] = []
        self.max_tokens: list[int] = []
        self._lock = threading.Lock()

    def complete(self, prompt: str, max_tokens: int) -> str:
        with self._lock:
            index = len(self.prompts)
            self.prompts.append(prompt)
            self.max_tokens.append(max_tokens)
        if index in self.fail_on:
            raise RuntimeError("simulated transport failure")
        if isinstance(self.responses, str):
            return self.responses
        return self.responses[index % len(self.responses)]


def candidate(novel_id: str = "n0") -> dict[str, Any]:
    return {"novel_id": novel_id, "title_guess": f"Title {novel_id}", "score": 0.5}


def test_score_parses_json_response() -> None:
    matcher = OpenAICompatibleMatcher(FakeTransport())
    result = matcher.score("凡人流 仙侠", candidate(), "profile text")
    assert result.llm_match_score == 0.8
    assert result.confidence == "high"
    assert result.matched_preferences == ["仙侠"]


def test_score_falls_back_on_unparseable_response() -> None:
    matcher = OpenAICompatibleMatcher(FakeTransport(responses="not json at all"))
    result = matcher.score("query", candidate(), "profile")
    assert result.llm_match_score == 0.0
    assert result.risk_flags == ["llm_parse_failed"]


def test_score_respects_profile_truncation() -> None:
    transport = FakeTransport()
    matcher = OpenAICompatibleMatcher(transport)
    matcher.score("query", candidate(), "x" * 5000, max_profile_chars=50)
    assert ("x" * 60) not in transport.prompts[0]


def test_score_many_preserves_order_and_scores_all() -> None:
    responses = [
        json.dumps({"llm_match_score": score, "confidence": "medium"})
        for score in (0.1, 0.2, 0.3, 0.4, 0.5)
    ]
    transport = FakeTransport(responses=responses)
    matcher = OpenAICompatibleMatcher(transport, max_workers=1)
    items = [(candidate(f"n{i}"), f"profile {i}") for i in range(5)]

    results = matcher.score_many("query", items)

    assert len(results) == 5
    assert [result.llm_match_score for result in results] == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_score_many_runs_concurrently_without_losing_slots() -> None:
    transport = FakeTransport()
    matcher = OpenAICompatibleMatcher(transport, max_workers=8)
    items = [(candidate(f"n{i}"), f"profile {i}") for i in range(24)]

    results = matcher.score_many("query", items)

    assert len(results) == 24
    assert all(isinstance(result, LLMMatchResult) for result in results)
    assert len(transport.prompts) == 24


def test_score_many_marks_failed_requests_as_none() -> None:
    """A dead request must be distinguishable from a genuine low score."""

    transport = FakeTransport(fail_on={1})
    matcher = OpenAICompatibleMatcher(transport, max_workers=1)
    items = [(candidate(f"n{i}"), "profile") for i in range(3)]

    results = matcher.score_many("query", items)

    assert results[1] is None
    assert results[0] is not None and results[2] is not None


def test_score_many_reports_progress() -> None:
    seen: list[int] = []
    matcher = OpenAICompatibleMatcher(FakeTransport(), max_workers=1)
    items = [(candidate(f"n{i}"), "profile") for i in range(3)]

    matcher.score_many("query", items, on_result=lambda index, _result: seen.append(index))

    assert sorted(seen) == [0, 1, 2]


def test_score_many_empty_input() -> None:
    assert OpenAICompatibleMatcher(FakeTransport()).score_many("query", []) == []


def test_matcher_satisfies_expansion_and_explanation_protocols() -> None:
    transport = FakeTransport(responses='{"expanded_queries":[{"text":"修仙","source":"llm","weight":0.9}]}')
    matcher = OpenAICompatibleMatcher(transport)

    expanded = matcher.expand_queries("凡人流", max_queries=3)
    explanation = matcher.generate("explain this", max_new_tokens=512)

    assert "expanded_queries" in expanded
    assert explanation
    assert transport.max_tokens == [256, 512]


def test_extract_message_content_requires_choices() -> None:
    with pytest.raises(ValueError):
        extract_message_content({"choices": []})


def test_parse_match_response_accepts_wrapped_json() -> None:
    result = parse_match_response(f"prefix text {MATCH_JSON} suffix")
    assert result.llm_match_score == 0.8


def test_transport_builds_openai_payload_and_auth_header() -> None:
    transport = HTTPChatTransport(model="qwen", base_url="http://host:8000/v1/", api_key="secret")
    payload = transport.build_payload("hello", max_tokens=64)

    assert transport.base_url == "http://host:8000/v1"
    assert payload["model"] == "qwen"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["max_tokens"] == 64
    assert payload["temperature"] == 0.0


def test_transport_rejects_zero_retries() -> None:
    with pytest.raises(ValueError):
        HTTPChatTransport(model="qwen", max_retries=0)


def test_transport_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INOVELREC_LLM_API_KEY", "from-env")
    assert HTTPChatTransport(model="qwen").api_key == "from-env"


def test_local_urls_bypass_the_proxy_but_gateways_do_not() -> None:
    """This host exports http_proxy; the proxy closes connections to 127.0.0.1."""

    from src.http_matcher import is_local_url

    assert is_local_url("http://127.0.0.1:8000/v1")
    assert is_local_url("http://localhost:8000/v1")
    assert not is_local_url("https://llm.echo.tech/v1")

    local = HTTPChatTransport(model="m", base_url="http://127.0.0.1:8000/v1")
    remote = HTTPChatTransport(model="m", base_url="https://llm.echo.tech/v1")
    assert local.bypass_proxy is True
    assert remote.bypass_proxy is False


def test_proxy_bypass_can_be_forced() -> None:
    transport = HTTPChatTransport(model="m", base_url="https://llm.echo.tech/v1", bypass_proxy=True)
    assert transport.bypass_proxy is True
