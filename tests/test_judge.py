import json
from pathlib import Path

import pytest

from src.evidence import PROFILE_FRACTIONS, judge_fractions, sample_judge_evidence, window_at
from src.http_matcher import TokenUsage
from src.judge import (
    BudgetExceeded,
    BudgetGuard,
    JudgeTask,
    JudgeVerdict,
    PricePerMillion,
    build_judge_prompt,
    judge_cache_key,
    parse_judge_verdict,
    run_judgements,
)

CHEAP = PricePerMillion(input_usd=1.0, output_usd=5.0)


def task(novel_id: str = "n0", evidence: str = "正文摘录内容") -> JudgeTask:
    return JudgeTask(
        query_id="q001",
        query="凡人流 仙侠 慢热 理性主角 不系统",
        novel_id=novel_id,
        title=f"小说{novel_id}",
        evidence=evidence,
        wanted=["凡人流", "仙侠"],
        unwanted=["系统"],
    )


class FakeJudgeTransport:
    def __init__(self, verdict: dict | None = None, usage: TokenUsage | None = None, fail_ids: set[str] | None = None) -> None:
        self.verdict = verdict or {"relevance_label": 2, "constraint_violation": False, "judge_confidence": "high"}
        self.usage = usage or TokenUsage(prompt_tokens=2500, completion_tokens=150)
        self.fail_ids = fail_ids or set()
        self.calls = 0

    def complete_with_usage(self, prompt: str, max_tokens: int) -> tuple[str, TokenUsage]:
        self.calls += 1
        for bad in self.fail_ids:
            if bad in prompt:
                raise RuntimeError("simulated failure")
        return json.dumps(self.verdict, ensure_ascii=False), self.usage


# --------------------------- evidence independence ---------------------------


def test_judge_fractions_avoid_profile_windows() -> None:
    for fraction in judge_fractions("n0", windows=4):
        assert all(abs(fraction - avoided) >= 0.05 for avoided in PROFILE_FRACTIONS)


def test_judge_fractions_are_deterministic_per_novel() -> None:
    assert judge_fractions("abc") == judge_fractions("abc")
    assert judge_fractions("abc") != judge_fractions("xyz")


def test_sample_judge_evidence_excludes_profile_text() -> None:
    """The judge must not be handed the same opening the profile embeds."""

    text = "".join(f"第{i}段内容。" for i in range(4000))
    opening = text[:650]
    evidence = sample_judge_evidence(text, "n0", windows=4, window_chars=300)
    assert evidence
    assert opening not in evidence


def test_sample_judge_evidence_handles_empty_and_short_text() -> None:
    assert sample_judge_evidence("", "n0") == ""
    assert sample_judge_evidence("短", "n0")
    assert window_at("abc", 0.9, 10) == "abc"


# ------------------------------- verdict parsing ------------------------------


def test_parse_judge_verdict_plain_json() -> None:
    verdict = parse_judge_verdict('{"relevance_label":2,"constraint_violation":true,"judge_confidence":"high"}')
    assert verdict.relevance_label == 2
    assert verdict.constraint_violation is True


def test_parse_judge_verdict_wrapped_json() -> None:
    verdict = parse_judge_verdict('说明文字 {"relevance_label":1,"constraint_violation":false} 结尾')
    assert verdict.relevance_label == 1


def test_parse_judge_verdict_abstains_on_garbage() -> None:
    verdict = parse_judge_verdict("完全不是 JSON")
    assert verdict.relevance_label == 0
    assert verdict.reason == "judge_parse_failed"


def test_labels_are_clamped() -> None:
    assert JudgeVerdict.from_dict({"relevance_label": 9}).relevance_label == 2
    assert JudgeVerdict.from_dict({"relevance_label": -3}).relevance_label == 0
    assert JudgeVerdict.from_dict({"relevance_label": "oops"}).relevance_label == 0


def test_prompt_carries_constraints_and_evidence() -> None:
    prompt = build_judge_prompt(task(evidence="独立摘录"))
    assert "独立摘录" in prompt
    assert "系统" in prompt
    assert "relevance_label" in prompt


# ---------------------------------- caching -----------------------------------


def test_cache_key_changes_when_evidence_changes() -> None:
    """A verdict is only valid for the text the judge actually read."""

    assert judge_cache_key(task(evidence="A"), "m") != judge_cache_key(task(evidence="B"), "m")
    assert judge_cache_key(task(), "m1") != judge_cache_key(task(), "m2")
    assert judge_cache_key(task(), "m") == judge_cache_key(task(), "m")


def test_cached_verdicts_avoid_new_requests(tmp_path: Path) -> None:
    transport = FakeJudgeTransport()
    cache_path = tmp_path / "judge.jsonl"
    tasks = [task("n0"), task("n1")]
    budget = BudgetGuard(limit_usd=200.0, prices=CHEAP)

    run_judgements(tasks, transport, "judge-model", budget, cache_path=cache_path, max_workers=1)
    first_calls = transport.calls
    _, summary = run_judgements(tasks, transport, "judge-model", budget, cache_path=cache_path, max_workers=1)

    assert first_calls == 2
    assert transport.calls == 2
    assert summary.cache_hits == 2
    assert summary.judged == 0


# ---------------------------------- budgeting ---------------------------------


def test_estimate_matches_hand_computed_cost() -> None:
    budget = BudgetGuard(limit_usd=200.0, prices=CHEAP)
    # 100 items x (2500 in, 150 out) = 250k in, 15k out -> 0.25*1 + 0.015*5
    assert round(budget.estimate_usd(100), 6) == round(0.25 * 1.0 + 0.015 * 5.0, 6)


def test_run_refuses_to_start_when_projection_exceeds_budget(tmp_path: Path) -> None:
    budget = BudgetGuard(limit_usd=0.001, prices=CHEAP)
    with pytest.raises(BudgetExceeded, match="exceeds the remaining"):
        run_judgements(
            [task(f"n{i}") for i in range(50)],
            FakeJudgeTransport(),
            "judge-model",
            budget,
            cache_path=tmp_path / "c.jsonl",
            max_workers=1,
        )


def test_spend_is_tracked_from_reported_usage(tmp_path: Path) -> None:
    transport = FakeJudgeTransport(usage=TokenUsage(prompt_tokens=1_000_000, completion_tokens=0))
    budget = BudgetGuard(limit_usd=200.0, prices=CHEAP)

    _, summary = run_judgements(
        [task("n0")], transport, "judge-model", budget, cache_path=tmp_path / "c.jsonl", max_workers=1
    )

    assert round(summary.spent_usd, 4) == 1.0
    assert summary.usage.prompt_tokens == 1_000_000


def test_run_stops_once_the_cap_is_reached(tmp_path: Path) -> None:
    """Each call burns the whole budget, so later tasks must be skipped, not billed."""

    transport = FakeJudgeTransport(usage=TokenUsage(prompt_tokens=1_000_000, completion_tokens=0))
    budget = BudgetGuard(limit_usd=1.0, prices=CHEAP)

    _, summary = run_judgements(
        [task(f"n{i}") for i in range(5)],
        transport,
        "judge-model",
        budget,
        cache_path=tmp_path / "c.jsonl",
        max_workers=1,
    )

    assert summary.judged == 1
    assert summary.skipped_over_budget == 4
    assert summary.stopped_early is True
    assert transport.calls == 1


def test_completed_work_is_persisted_when_stopping_early(tmp_path: Path) -> None:
    cache_path = tmp_path / "c.jsonl"
    transport = FakeJudgeTransport(usage=TokenUsage(prompt_tokens=1_000_000, completion_tokens=0))
    budget = BudgetGuard(limit_usd=1.0, prices=CHEAP)

    run_judgements(
        [task(f"n{i}") for i in range(5)], transport, "judge-model", budget, cache_path=cache_path, max_workers=1
    )

    assert len(cache_path.read_text(encoding="utf-8").strip().splitlines()) == 1


# ----------------------------------- failures ---------------------------------


def test_failed_request_is_absent_rather_than_scored_zero(tmp_path: Path) -> None:
    transport = FakeJudgeTransport(fail_ids={"小说n1"})
    budget = BudgetGuard(limit_usd=200.0, prices=CHEAP)

    verdicts, summary = run_judgements(
        [task("n0"), task("n1"), task("n2")],
        transport,
        "judge-model",
        budget,
        cache_path=tmp_path / "c.jsonl",
        max_workers=1,
    )

    assert summary.failed == 1
    assert summary.judged == 2
    assert judge_cache_key(task("n1"), "judge-model") not in verdicts


def test_empty_task_list_is_free() -> None:
    budget = BudgetGuard(limit_usd=0.0, prices=CHEAP)
    verdicts, summary = run_judgements([], FakeJudgeTransport(), "m", budget)
    assert verdicts == {}
    assert summary.requested == 0
