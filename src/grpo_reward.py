"""Verifiable reward for GRPO over the pointwise reranker.

One episode is one ``(query, candidate)`` pair — the same unit the model sees at
inference. A listwise episode would give K generations a single scalar and destroy
credit assignment; the listwise objective stays in evaluation, which also keeps the
reward rule and the evaluation source separate (see ``docs/post_training_plan.md``).

The reward is

    r = r_format x (w_constraint * r_constraint + w_score * r_score) + w_terminate * r_terminate

``r_format`` gates rather than adds: the other terms read fields, so their values are
meaningless when the fields are absent. ``r_terminate`` sits *outside* the gate on
purpose — measured at temperature 1.0, 94.8% of rollouts emit a valid verdict object
but only 91% stop cleanly, so a model that answered correctly and then rambled must
lose the rambling points, not the answer points.

Nothing here loads a model or reads a novel: constraint truth arrives as a
precomputed verdict from ``preferences.constraint_violation_from_densities``, so the
whole thing is a pure function and tests run on CPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.llm_matcher import split_first_json_object

REQUIRED_FIELDS = ("llm_match_score", "confidence", "matched_preferences", "violated_preferences", "risk_flags", "reason")
VALID_CONFIDENCE = ("high", "medium", "low")

DEFAULT_WEIGHT_CONSTRAINT = 0.58
DEFAULT_WEIGHT_SCORE = 0.4
# Format and termination both sit at 0.999 after one GRPO round — saturated, so most
# of this weight was paying for behaviour already learned. A token amount keeps the
# gradient from disappearing; the rest goes to the constraint term, which has not
# saturated.
DEFAULT_WEIGHT_TERMINATE = 0.02

# A missed violation and a false alarm are not equally bad for the product. Missing
# one recommends a book the reader explicitly excluded; a false alarm demotes a book
# that was fine. Symmetric accuracy let the policy settle wherever the two balanced,
# and it settled on near-silence in the region that matters: on the evaluation top
# ten, GRPO-v2 flagged only 0.4% of scored candidates against the 32B teacher's
# 19.6%, so the 27.1% of rule-arm violations still surviving there are all misses.
# Penalty size cannot fix a miss; only recall can.
FALSE_ALARM_CREDIT = 0.4
# The teacher-agreement term is an ablation arm, off for the first run: the teacher's
# rerank was not shown to beat baseline on relevance, so optimising toward it would
# be fitting noise. Turned on later only to answer "does removing it collapse?".
DEFAULT_WEIGHT_CONSISTENCY = 0.0


@dataclass(frozen=True)
class RewardWeights:
    """Weights for the reward terms."""

    constraint: float = DEFAULT_WEIGHT_CONSTRAINT
    score: float = DEFAULT_WEIGHT_SCORE
    terminate: float = DEFAULT_WEIGHT_TERMINATE
    consistency: float = DEFAULT_WEIGHT_CONSISTENCY


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-term reward, kept so training logs can show *why* a reward moved.

    A single scalar cannot distinguish "learned the constraint" from "learned to
    emit shorter JSON", and those need separate curves to detect reward hacking.
    """

    total: float
    format_ok: bool
    constraint: float | None
    score: float | None
    terminate: float
    claimed_violation: bool
    verdict: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.total,
            "format_ok": self.format_ok,
            "r_constraint": self.constraint,
            "r_score": self.score,
            "r_terminate": self.terminate,
            "claimed_violation": self.claimed_violation,
        }


def trailing_text(text: str) -> str | None:
    """Whatever the model emitted after its first complete verdict object.

    ``None`` when there is no complete object at all. This is how termination is
    scored: verl's reward interface passes only the decoded string, not
    ``finish_reason``, and the string is the better signal anyway. Hitting the token
    cap and rambling under it are the same defect, and 9% of rollouts at temperature
    1.0 emit a correct verdict followed by garbage — sometimes a second copy of the
    verdict, sometimes unrelated text.
    """

    try:
        return split_first_json_object(text)[1]
    except (ValueError, TypeError):
        return None


def parse_verdict(text: str) -> dict[str, Any] | None:
    """Return the first balanced JSON object if it is a well-formed verdict.

    Depth-matched, never a greedy ``{.*}``. Rollouts that fail to stop emit a valid
    verdict *followed by* garbage — sometimes a second copy of the verdict — so a
    greedy match spans both and fails on text that actually contains a correct
    answer. Measured on 1,024 rollouts, greedy scored 83.6% valid where depth-matched
    scored 94.8%; the difference is entirely answers that were already right.
    """

    try:
        payload = json.loads(split_first_json_object(text)[0])
    except (ValueError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or not all(field_name in payload for field_name in REQUIRED_FIELDS):
        return None
    try:
        score = float(payload["llm_match_score"])
    except (TypeError, ValueError):
        return None
    if not 0.0 <= score <= 1.0:
        return None
    if str(payload.get("confidence", "")).strip().lower() not in VALID_CONFIDENCE:
        return None
    for list_field in ("matched_preferences", "violated_preferences", "risk_flags"):
        if not isinstance(payload[list_field], list):
            return None
    return payload


def claims_violation(verdict: dict[str, Any], terms: list[str] | tuple[str, ...]) -> bool:
    """Whether the model named any of this query's exclusions as violated.

    Substring rather than equality, matching ``normalize_violated_terms``: models
    write 「不系统」 or bury the term in a phrase, and a claim is a claim.
    """

    claimed = [str(item) for item in verdict.get("violated_preferences", [])]
    return any(term and any(term in item for item in claimed) for term in terms)


def constraint_reward(verdict: dict[str, Any], terms: list[str], rule_verdict: bool) -> float:
    """Score the claim against the rule's reading of the full novel, asymmetrically.

    Correct either way earns 1. A false alarm still earns ``FALSE_ALARM_CREDIT``
    because demoting a clean book is a mild error; a missed violation earns nothing,
    because it puts an excluded book in front of the reader. See FALSE_ALARM_CREDIT
    for why the symmetric version was not enough.
    """

    claimed = claims_violation(verdict, terms)
    if claimed == rule_verdict:
        return 1.0
    return FALSE_ALARM_CREDIT if claimed else 0.0


MARGIN_SCALE = 0.3


def pairwise_score_reward(
    verdict: dict[str, Any],
    rule_verdict: bool,
    partner_anchor: float | None,
    margin_scale: float = MARGIN_SCALE,
) -> float | None:
    """Score this candidate *relative to a contrasting candidate of the same query*.

    The previous absolute form — ``1 - llm_match_score`` on violations, nothing on
    clean candidates — had ``score = 0`` everywhere as its trivial optimum, and the
    policy found it: validation ``llm_match_score`` went 0.058 -> 0.000 within 25
    steps and stayed there. Since ``llm_match_score`` carries weight 0.50 in the
    final ranking, a constant collapses the reranker's relevance contribution
    entirely, and judge-scored relevance fell (paired sign test against the SFT
    student, p=0.012).

    The fix is to reward *separation* rather than magnitude. ``partner_anchor`` is the
    reference policy's score on a candidate of the same query with the opposite rule
    verdict, precomputed offline. A violating candidate should land below it; a clean
    one above it. Both halves are scored, so any constant output earns exactly 0.5 on
    each — the degenerate solution is no longer optimal, it is average.

    Returns ``None`` when no partner exists, which contributes no signal rather than
    a guess.
    """

    if partner_anchor is None:
        return None
    score = float(verdict["llm_match_score"])
    margin = (partner_anchor - score) if rule_verdict else (score - partner_anchor)
    return max(0.0, min(0.5 + margin / (2.0 * margin_scale), 1.0))


def compute_reward(
    text: str,
    *,
    terms: list[str],
    rule_verdict: bool | None,
    partner_anchor: float | None = None,
    finish_reason: str | None = None,
    weights: RewardWeights | None = None,
) -> RewardBreakdown:
    """Score one rollout.

    ``rule_verdict`` is ``None`` when the density rule abstained or the exclusion is
    not rule-checkable. Such a pair carries no verifiable signal and should not be in
    the batch at all — ``20_build_grpo_data.py`` filters them out — but if one arrives
    the constraint terms are simply skipped rather than guessed at. Scoring an
    abstention as 0 would teach the model that ambiguity is punished.
    """

    active = weights or RewardWeights()
    verdict = parse_verdict(text)
    # Default to reading termination off the text, because that is all verl's reward
    # interface receives. ``finish_reason`` overrides it when a caller has one.
    remainder = trailing_text(text)
    clean_stop = remainder is not None and not remainder.strip()
    terminate = float(finish_reason == "stop") if finish_reason is not None else float(clean_stop)

    if verdict is None:
        return RewardBreakdown(
            total=active.terminate * terminate,
            format_ok=False,
            constraint=None,
            score=None,
            terminate=terminate,
            claimed_violation=False,
        )

    gated = 0.0
    r_constraint: float | None = None
    r_score: float | None = None
    if rule_verdict is not None and terms:
        r_constraint = constraint_reward(verdict, terms, rule_verdict)
        gated += active.constraint * r_constraint
        r_score = pairwise_score_reward(verdict, rule_verdict, partner_anchor)
        if r_score is not None:
            gated += active.score * r_score

    return RewardBreakdown(
        total=gated + active.terminate * terminate,
        format_ok=True,
        constraint=r_constraint,
        score=r_score,
        terminate=terminate,
        claimed_violation=claims_violation(verdict, terms),
        verdict=verdict,
    )


def group_advantages(rewards: list[float]) -> list[float]:
    """Normalize rewards within a GRPO group, treating a degenerate group as no signal.

    When every sample in a group earns the same reward the standard deviation is
    zero and the advantage is undefined. Dividing by ``std + eps`` there would turn
    floating-point dust into enormous advantages pointing in arbitrary directions, so
    the group contributes nothing instead.

    The share of groups that land here is worth logging on its own: it rises as the
    policy converges, and a sudden jump means the reward has stopped discriminating.
    """

    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    if variance <= 1e-12:
        return [0.0] * len(rewards)
    deviation = variance**0.5
    return [(value - mean) / deviation for value in rewards]
