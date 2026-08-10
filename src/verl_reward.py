"""verl entry point for the GRPO reward.

verl loads this file inside its own interpreter (``/data/huangyanyu/.venv-verl``),
which has torch 2.10 + vLLM 0.19 and none of this project's dependencies. That works
because everything reachable from here imports only the standard library at module
level — ``torch`` and ``transformers`` are deferred inside functions in
``llm_matcher``. Reaching it needs nothing but ``PYTHONPATH=/data/huangyanyu/INovelRec``.

The contract verl calls is::

    compute_score(data_source, solution_str, ground_truth, extra_info) -> float | dict

Returning a dict makes ``score`` the reward and files every other key into
``reward_extra_info``, which verl logs per step. That is how the reward-hacking
monitors get their curves: a single scalar cannot separate "learned the constraint"
from "learned to emit shorter JSON".
"""

from __future__ import annotations

from typing import Any

from src.grpo_reward import RewardWeights, compute_reward


def _as_list(value: Any) -> list[str]:
    """Coerce whatever the parquet round-trip produced back into a list of terms."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in value]


def compute_score(
    data_source: str | None = None,
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    *,
    weight_constraint: float = RewardWeights().constraint,
    weight_score: float = RewardWeights().score,
    weight_terminate: float = RewardWeights().terminate,
) -> dict[str, Any]:
    """Score one rollout for verl.

    ``ground_truth`` carries the rule's verdict on the full novel, computed offline
    from the density table by ``20_build_grpo_data.py`` — the reward never reads a
    novel here, so it stays a microsecond-scale lookup inside the rollout loop.
    """

    info = extra_info or {}
    terms = _as_list(info.get("constraint_terms"))

    # The verdict travels as a string because parquet -> verl -> here is not a
    # faithful round trip for numpy bools, and "None" must stay distinguishable from
    # False: None means no reward signal, False means the rule read the book and
    # found it clean.
    raw = ground_truth if ground_truth is not None else info.get("rule_verdict")
    if isinstance(raw, str):
        rule_verdict: bool | None = {"true": True, "false": False}.get(raw.strip().lower())
    elif raw is None:
        rule_verdict = None
    else:
        rule_verdict = bool(raw)

    breakdown = compute_reward(
        solution_str,
        terms=terms,
        rule_verdict=rule_verdict,
        weights=RewardWeights(
            constraint=weight_constraint,
            score=weight_score,
            terminate=weight_terminate,
        ),
    )

    # None would break verl's aggregation of reward_extra_info into tensors, so the
    # inactive terms are reported as their neutral value and the *_active flags say
    # whether they contributed. Averaging r_constraint over a step is only meaningful
    # alongside how often it was defined.
    return {
        "score": breakdown.total,
        "format_ok": float(breakdown.format_ok),
        "r_constraint": float(breakdown.constraint) if breakdown.constraint is not None else 0.0,
        "r_constraint_active": float(breakdown.constraint is not None),
        "r_score": float(breakdown.score) if breakdown.score is not None else 0.0,
        "r_score_active": float(breakdown.score is not None),
        "r_terminate": breakdown.terminate,
        # The reward-hacking monitor. The pool is balanced 50/50, so a policy that
        # learns to discriminate holds this near 0.5; a drift toward 1.0 is the
        # "claim everything violates" collapse, and toward 0.0 is silence.
        "claimed_violation": float(breakdown.claimed_violation),
        "llm_match_score": float(breakdown.verdict.get("llm_match_score", 0.0)) if breakdown.verdict else 0.0,
    }
