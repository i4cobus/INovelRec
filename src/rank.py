"""Hybrid recommendation ranking using FAISS scores plus local LLM features."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import pandas as pd

from src.config import DATA_DIR
from src.llm_matcher import LLMMatchResult, PROMPT_VERSION, confidence_to_score
from src.vector_index import DEFAULT_PROFILES_PATH

CACHE_PATH = DATA_DIR / "cache" / "llm_rerank_cache.jsonl"
BOILERPLATE_PATTERNS = ("知轩藏书", "zxcs", "www.zxcs", "精校小说下载", "更多精校小说")

# Legacy LLM budget from the single-GPU era, where one candidate cost ~8.8s. Pass
# LLM_CANDIDATE_K_ALL to score every candidate once a batching backend is available.
DEFAULT_LLM_CANDIDATE_K = 10
LLM_CANDIDATE_K_ALL = 0

# How to score candidates the LLM never saw.
#
# "impute" keeps every candidate on ONE scale: unscored rows reuse the normal formula
# with the LLM features filled in from the mean of the scored rows, i.e. treated as
# average rather than bad.
#
# "legacy_semantic" reproduces the original two-formula behaviour, kept as an A/B
# control. It is biased: its ceiling is 0.40 while the scored path's ceiling is 1.00,
# so a candidate the LLM actively rated 0.1 can still outrank the best unscored
# candidate. Being *selected* was worth up to 0.6 points independent of content.
FallbackPolicy = Literal["impute", "legacy_semantic"]
DEFAULT_FALLBACK_POLICY: FallbackPolicy = "impute"

# Used only when nothing at all was scored, so there is no mean to impute from.
PRIOR_LLM_MATCH_SCORE = 0.5
PRIOR_CONFIDENCE_SCORE = confidence_to_score("low")


class CandidateMatcher(Protocol):
    provider: str

    def score(self, query: str, candidate: dict[str, Any], profile_text: str, max_profile_chars: int = 1200) -> LLMMatchResult:
        """Return local LLM candidate analysis."""


@dataclass(frozen=True)
class RankingWeights:
    semantic_weight: float = 0.40
    llm_match_weight: float = 0.50
    confidence_weight: float = 0.10


@dataclass
class TimingSummary:
    preference_parsing: float = 0.0
    faiss_retrieval: float = 0.0
    load_profiles: float = 0.0
    llm_scoring: float = 0.0
    average_llm_scoring_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    final_reranking: float = 0.0
    total_runtime: float = 0.0
    candidate_k: int = 0
    llm_candidate_k: int = 0
    top_k: int = 0
    provider: str = ""
    llm_model: str = ""
    llm_profile_max_chars: int = 1200
    fallback_policy: str = ""


@dataclass(frozen=True)
class LLMProgressEvent:
    index: int
    total: int
    title: str
    faiss_rank: int
    cache_status: str
    elapsed_seconds: float = 0.0
    average_seconds: float = 0.0
    estimated_remaining_seconds: float | None = None
    llm_match_score: float | None = None
    confidence: str | None = None
    phase: str = "start"


ProgressCallback = Callable[[LLMProgressEvent], None]


def candidate_key(candidate: dict[str, Any], fallback: int = 0) -> str:
    """Return a stable candidate key for selection and scoring maps."""

    return str(candidate.get("novel_id") or candidate.get("title_guess") or fallback)


def normalize_semantic_scores(candidates: list[dict[str, Any]]) -> dict[str, float]:
    """Min-max normalize semantic scores to 0..1 across the candidate pool.

    Always min-max, unconditionally. The previous version passed scores through
    unchanged whenever every value already fell inside [0, 1] and only min-maxed
    otherwise — so the function did two different things depending on whether the
    pool happened to contain a negative cosine.

    That mattered because it usually did nothing: cosine similarities over a
    retrieved pool cluster in a narrow band (~0.64-0.72 is typical), so the
    "already in range" branch fired almost every time and ``semantic_weight``'s
    nominal 0.40 contributed only ~0.03 of actual score spread, while
    ``llm_match_weight``'s 0.50 contributed its full range. Weights on paper were
    not the weights doing the ranking. Use ``score_component_contributions`` to
    check the effective split rather than trusting the constants.
    """

    if not candidates:
        return {}
    scores = [float(candidate.get("score", 0.0)) for candidate in candidates]
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        # One distinct value: every candidate is equally (un)supported by semantics,
        # so give them all the same score and let the other components decide.
        return {str(candidate.get("novel_id", idx)): 1.0 for idx, candidate in enumerate(candidates)}
    span = max_score - min_score
    return {
        str(candidate.get("novel_id", idx)): (score - min_score) / span
        for idx, (candidate, score) in enumerate(zip(candidates, scores, strict=False))
    }


def score_component_contributions(rows: list[dict[str, Any]], weights: RankingWeights | None = None) -> dict[str, float]:
    """Report how much each weighted component actually moves the final score.

    Nominal weights describe intent; this describes behaviour. A component whose
    inputs barely vary across the pool contributes almost nothing no matter how
    large its weight, which is exactly how a reranker ends up decorative.
    """

    weights = weights or RankingWeights()
    if not rows:
        return {}

    def spread(values: list[float]) -> float:
        usable = [value for value in values if value is not None]
        return (max(usable) - min(usable)) if usable else 0.0

    semantic = spread([float(row.get("normalized_semantic_score", 0.0)) for row in rows])
    llm = spread([row.get("llm_match_score") for row in rows if row.get("llm_match_score") is not None])
    confidence = spread([row.get("confidence_score") for row in rows if row.get("confidence_score") is not None])
    risk = spread([float(row.get("risk_penalty", 0.0)) for row in rows])

    contributions = {
        "semantic": weights.semantic_weight * semantic,
        "llm_match": weights.llm_match_weight * llm,
        "confidence": weights.confidence_weight * confidence,
        "risk_penalty": risk,
    }
    total = sum(contributions.values())
    if total <= 0:
        return {**{key: round(value, 6) for key, value in contributions.items()}, "total_spread": 0.0}
    shares = {f"{key}_share": round(value / total, 4) for key, value in contributions.items()}
    return {
        **{key: round(value, 6) for key, value in contributions.items()},
        **shares,
        "total_spread": round(total, 6),
    }


def load_profile_text_lookup(profiles_path: Path = DEFAULT_PROFILES_PATH) -> dict[str, str]:
    """Load full profile text by novel_id if the profile parquet is available."""

    if not profiles_path.exists():
        return {}
    profiles = pd.read_parquet(profiles_path, columns=["novel_id", "profile_text"])
    profiles = profiles.dropna(subset=["novel_id", "profile_text"])
    return dict(zip(profiles["novel_id"].astype(str), profiles["profile_text"].astype(str), strict=False))


def resolve_llm_candidate_k(candidate_k: int, llm_candidate_k: int | None) -> tuple[int, str | None]:
    """Resolve and clamp the number of candidates sent to the local LLM.

    ``LLM_CANDIDATE_K_ALL`` (0) means "score every candidate", which removes the
    unscored path entirely. That is only affordable with a batching backend.
    """

    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if llm_candidate_k is None:
        resolved = min(DEFAULT_LLM_CANDIDATE_K, candidate_k)
    elif llm_candidate_k == LLM_CANDIDATE_K_ALL:
        return candidate_k, None
    else:
        resolved = llm_candidate_k
    if resolved <= 0:
        raise ValueError("llm_candidate_k must be positive, or LLM_CANDIDATE_K_ALL to score everything")
    if resolved > candidate_k:
        return candidate_k, f"llm-candidate-k {resolved} exceeds candidate-k {candidate_k}; clamped to {candidate_k}."
    return resolved, None


@dataclass(frozen=True)
class PendingScore:
    """One cache-missed candidate awaiting an LLM call."""

    selection_key: str
    selected_index: int
    candidate: dict[str, Any]
    truncated_profile: str
    cache_key: str
    title: str
    faiss_rank: int


def score_pending(
    *,
    query: str,
    pending: list[PendingScore],
    matcher: CandidateMatcher,
    llm_profile_max_chars: int,
) -> list[LLMMatchResult | None]:
    """Score cache-missed candidates, using a batching backend when one is offered.

    ``None`` marks a request that never produced an answer, which the caller must
    not cache. Matchers exposing ``score_many`` (see ``src/http_matcher.py``) run
    the whole batch concurrently; everything else falls back to one call at a time.
    """

    batch_score = getattr(matcher, "score_many", None)
    if callable(batch_score) and len(pending) > 1:
        items = [(item.candidate, item.truncated_profile) for item in pending]
        return list(batch_score(query, items, llm_profile_max_chars))

    results: list[LLMMatchResult | None] = []
    for item in pending:
        try:
            results.append(
                matcher.score(
                    query=query,
                    candidate=item.candidate,
                    profile_text=item.truncated_profile,
                    max_profile_chars=llm_profile_max_chars,
                )
            )
        except Exception:  # noqa: BLE001 - one dead candidate must not kill the run
            results.append(None)
    return results


def impute_unscored_features(scored: list[LLMMatchResult]) -> tuple[float, float]:
    """Estimate LLM features for candidates that never reached the model.

    Returns the mean match score and confidence score of the candidates that were
    scored, so an unscored candidate ranks as *average* rather than as bad. Falls
    back to a neutral prior when nothing was scored.
    """

    if not scored:
        return PRIOR_LLM_MATCH_SCORE, PRIOR_CONFIDENCE_SCORE
    return (
        sum(match.llm_match_score for match in scored) / len(scored),
        sum(match.confidence_score for match in scored) / len(scored),
    )


def imputed_final_score(
    normalized_semantic_score: float,
    imputed_llm_match_score: float,
    imputed_confidence_score: float,
    weights: RankingWeights,
) -> float:
    """Score an unscored candidate on the same scale as scored candidates."""

    return (
        weights.semantic_weight * normalized_semantic_score
        + weights.llm_match_weight * imputed_llm_match_score
        + weights.confidence_weight * imputed_confidence_score
    )


def llm_selection_quotas(llm_candidate_k: int) -> tuple[int, int, int]:
    """Allocate LLM scoring slots across retrieval, semantic, and FAISS-rank views."""

    if llm_candidate_k <= 0:
        raise ValueError("llm_candidate_k must be positive")
    if llm_candidate_k == 1:
        return 1, 0, 0
    if llm_candidate_k == 2:
        return 1, 1, 0

    retrieval_slots = max(1, math.ceil(0.5 * llm_candidate_k))
    semantic_slots = max(1, math.ceil(0.3 * llm_candidate_k))
    faiss_slots = max(1, llm_candidate_k - retrieval_slots - semantic_slots)

    while retrieval_slots + semantic_slots + faiss_slots > llm_candidate_k:
        if retrieval_slots >= semantic_slots and retrieval_slots > 1:
            retrieval_slots -= 1
        elif semantic_slots > 1:
            semantic_slots -= 1
        else:
            faiss_slots -= 1
    return retrieval_slots, semantic_slots, faiss_slots


def semantic_score_value(candidate: dict[str, Any]) -> float:
    """Return the best available semantic score for a candidate."""

    return float(candidate.get("best_semantic_score", candidate.get("semantic_score", candidate.get("score", 0.0))))


def best_faiss_rank_value(candidate: dict[str, Any]) -> int:
    """Return best FAISS rank, using a large value when missing."""

    try:
        return int(candidate.get("best_faiss_rank", candidate.get("rank", 1_000_000)))
    except (TypeError, ValueError):
        return 1_000_000


def add_selection_candidate(
    selected: dict[str, dict[str, Any]],
    candidate: dict[str, Any],
    reason: str,
    *,
    fallback_index: int,
) -> None:
    """Add a selected candidate or append an additional selection reason."""

    key = candidate_key(candidate, fallback=fallback_index)
    if key in selected:
        reasons = selected[key].setdefault("llm_selection_reasons", [])
        if reason not in reasons:
            reasons.append(reason)
        if len(reasons) > 1 and "already_selected_multiple_reasons" not in reasons:
            reasons.append("already_selected_multiple_reasons")
        return

    selected[key] = {
        **candidate,
        "llm_selection_reasons": [reason],
        "llm_selection_reason": reason,
    }


def select_llm_candidates(
    candidates: list[dict[str, Any]],
    llm_candidate_k: int,
    debug_target_title: str | None = None,
) -> list[dict[str, Any]]:
    """Select a diversified set of candidates for expensive local LLM scoring."""

    if llm_candidate_k <= 0:
        raise ValueError("llm_candidate_k must be positive")
    if not candidates:
        return []

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, candidate in enumerate(candidates):
        key = candidate_key(candidate, fallback=idx)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    selected: dict[str, dict[str, Any]] = {}
    retrieval_slots, semantic_slots, faiss_slots = llm_selection_quotas(min(llm_candidate_k, len(deduped)))
    views = [
        (
            sorted(deduped, key=lambda item: float(item.get("retrieval_score", item.get("score", 0.0))), reverse=True)[:retrieval_slots],
            "retrieval_score_top",
        ),
        (
            sorted(deduped, key=semantic_score_value, reverse=True)[:semantic_slots],
            "semantic_score_top",
        ),
        (
            sorted(deduped, key=best_faiss_rank_value)[:faiss_slots],
            "best_faiss_rank_top",
        ),
    ]

    for view, reason in views:
        for candidate in view:
            if len(selected) >= llm_candidate_k and candidate_key(candidate) not in selected:
                continue
            add_selection_candidate(selected, candidate, reason, fallback_index=deduped.index(candidate))

    if len(selected) < llm_candidate_k:
        fill_view = sorted(deduped, key=lambda item: float(item.get("retrieval_score", item.get("score", 0.0))), reverse=True)
        for candidate in fill_view:
            if len(selected) >= llm_candidate_k:
                break
            if candidate_key(candidate) not in selected:
                add_selection_candidate(selected, candidate, "retrieval_score_top", fallback_index=deduped.index(candidate))

    if debug_target_title:
        target = next((candidate for candidate in deduped if debug_target_title in str(candidate.get("title_guess", ""))), None)
        if target is not None:
            key = candidate_key(target)
            if key in selected:
                pass
            elif len(selected) < llm_candidate_k:
                add_selection_candidate(selected, target, "debug_target_forced", fallback_index=deduped.index(target))
            else:
                # Replace the lowest-priority selected candidate so debug can verify the expected title.
                last_key = next(reversed(selected))
                selected.pop(last_key)
                forced = {**target, "llm_selection_forced_replacement": True}
                add_selection_candidate(selected, forced, "debug_target_forced", fallback_index=deduped.index(target))

    return list(selected.values())[:llm_candidate_k]


def truncate_profile(profile_text: str, max_chars: int) -> str:
    """Limit profile text sent to the local LLM."""

    if max_chars <= 0:
        raise ValueError("llm_profile_max_chars must be positive")
    return profile_text[:max_chars]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_cache_key(
    *,
    query: str,
    novel_id: str,
    profile_text: str,
    llm_model: str,
    provider: str,
    llm_profile_max_chars: int,
) -> str:
    """Build a stable cache key for one candidate analysis."""

    payload = {
        "query_hash": sha256_text(query),
        "novel_id": novel_id,
        "profile_hash": sha256_text(profile_text),
        "llm_model": llm_model,
        "provider": provider,
        "prompt_version": PROMPT_VERSION,
        "llm_profile_max_chars": llm_profile_max_chars,
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def load_llm_cache(cache_path: Path = CACHE_PATH) -> dict[str, dict[str, Any]]:
    """Load JSONL cache entries keyed by cache_key."""

    if not cache_path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = str(item.get("cache_key", ""))
        if key:
            cache[key] = item
    return cache


def append_llm_cache(cache_path: Path, cache_key: str, result: LLMMatchResult) -> None:
    """Append one candidate analysis result to the JSONL cache."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"cache_key": cache_key, "result": result.to_dict()}
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_boilerplate_penalty(text: str) -> float:
    lowered = text.lower()
    hits = sum(1 for pattern in BOILERPLATE_PATTERNS if pattern.lower() in lowered)
    return 0.05 if hits else 0.0


def compute_risk_penalty(match: LLMMatchResult, profile_text: str) -> float:
    """Compute rule-based risk penalty from LLM fields and visible profile risks."""

    penalty = 0.0
    if match.violated_preferences:
        penalty += 0.15
    if compute_boilerplate_penalty(profile_text) or any("boilerplate" in flag.lower() or "source" in flag.lower() for flag in match.risk_flags):
        penalty += 0.05
    if match.confidence == "low":
        penalty += 0.05
    return min(penalty, 1.0)


def llm_final_score(normalized_semantic_score: float, match: LLMMatchResult, risk_penalty: float, weights: RankingWeights) -> float:
    """Apply the Stage 4 final scoring formula."""

    return (
        weights.semantic_weight * normalized_semantic_score
        + weights.llm_match_weight * match.llm_match_score
        + weights.confidence_weight * match.confidence_score
        - risk_penalty
    )


def semantic_fallback_score(normalized_semantic_score: float, matched_query_count: int = 1) -> float:
    """Keep fallback candidates comparable while below strong LLM-scored candidates."""

    matched_query_bonus = min(max(matched_query_count, 0) / 5.0, 1.0)
    return (0.35 * normalized_semantic_score) + (0.05 * matched_query_bonus)


def build_output_row(
    *,
    candidate: dict[str, Any],
    normalized_semantic_score: float,
    selected_for_llm: bool,
    analysis_provider: str,
    cache_hit: bool,
    final_score: float,
    match: LLMMatchResult | None,
    risk_penalty: float,
    profile_text: str,
    imputed_llm_match_score: float | None = None,
    imputed_confidence_score: float | None = None,
) -> dict[str, Any]:
    return {
        "imputed_llm_match_score": None if imputed_llm_match_score is None else round(imputed_llm_match_score, 6),
        "imputed_confidence_score": None if imputed_confidence_score is None else round(imputed_confidence_score, 6),
        "final_rank": 0,
        "faiss_rank": int(candidate.get("rank", 0)),
        "best_faiss_rank": int(candidate.get("best_faiss_rank", candidate.get("rank", 0))),
        "matched_query_count": int(candidate.get("matched_query_count", 1)),
        "retrieval_sources": list(candidate.get("retrieval_sources", ["raw"])),
        "retrieval_score": round(float(candidate.get("retrieval_score", candidate.get("score", 0.0))), 6),
        "llm_selection_reasons": list(candidate.get("llm_selection_reasons", [])),
        "llm_selection_reason": str(candidate.get("llm_selection_reason", "")),
        "llm_selection_forced_replacement": bool(candidate.get("llm_selection_forced_replacement", False)),
        "selected_for_llm": selected_for_llm,
        "novel_id": str(candidate.get("novel_id", "")),
        "title_guess": str(candidate.get("title_guess", "")),
        "semantic_score": float(candidate.get("score", 0.0)),
        "normalized_semantic_score": round(normalized_semantic_score, 6),
        "llm_match_score": None if match is None else round(match.llm_match_score, 6),
        "confidence": None if match is None else match.confidence,
        "confidence_score": None if match is None else round(match.confidence_score, 6),
        "risk_penalty": round(risk_penalty, 6),
        "final_score": round(final_score, 6),
        "cache_hit": cache_hit,
        "analysis_provider": analysis_provider,
        "matched_preferences": [] if match is None else match.matched_preferences,
        "violated_preferences": [] if match is None else match.violated_preferences,
        "risk_flags": [] if match is None else match.risk_flags,
        "reason": "" if match is None else match.reason,
        "profile_text_preview": str(candidate.get("profile_text_preview") or profile_text[:300]),
    }


def rerank_candidates_with_llm(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    matcher: CandidateMatcher,
    llm_candidate_k: int,
    llm_profile_max_chars: int = 1200,
    profile_lookup: dict[str, str] | None = None,
    weights: RankingWeights | None = None,
    use_cache: bool = True,
    cache_path: Path = CACHE_PATH,
    llm_model: str = "",
    debug_target_title: str | None = None,
    progress_callback: ProgressCallback | None = None,
    fallback_policy: FallbackPolicy = DEFAULT_FALLBACK_POLICY,
) -> tuple[list[dict[str, Any]], TimingSummary]:
    """Rerank semantic candidates with budgeted local LLM analysis, cache, and timing.

    Pass ``llm_candidate_k=LLM_CANDIDATE_K_ALL`` to score every candidate.
    ``fallback_policy`` controls how unscored candidates are placed on the score
    scale; see the constant definitions for why the legacy policy is biased.
    """

    started = time.perf_counter()
    weights = weights or RankingWeights()
    profile_lookup = profile_lookup or {}
    selected_count, _ = resolve_llm_candidate_k(len(candidates), llm_candidate_k) if candidates else (0, None)
    selected_candidates = select_llm_candidates(candidates, selected_count, debug_target_title=debug_target_title) if selected_count else []
    selected_by_id = {candidate_key(candidate, fallback=idx): candidate for idx, candidate in enumerate(selected_candidates)}
    normalized = normalize_semantic_scores(candidates)
    cache = load_llm_cache(cache_path) if use_cache else {}
    rows: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    llm_elapsed_total = 0.0

    # Pass 1 scores the selected candidates. Imputing features for the rest needs the
    # scored distribution, so no output row can be finalised until scoring completes.
    scored: dict[str, tuple[LLMMatchResult, str, str]] = {}
    pending: list[PendingScore] = []
    selected_index = 0

    for idx, candidate in enumerate(candidates):
        novel_id = str(candidate.get("novel_id", idx))
        selection_key = candidate_key(candidate, fallback=idx)
        if selection_key not in selected_by_id:
            continue

        candidate_for_output = selected_by_id[selection_key]
        profile_text = profile_lookup.get(novel_id) or str(candidate.get("profile_text_preview", ""))
        selected_index += 1
        truncated_profile = truncate_profile(profile_text, llm_profile_max_chars)
        cache_key = make_cache_key(
            query=query,
            novel_id=novel_id,
            profile_text=truncated_profile,
            llm_model=llm_model,
            provider=getattr(matcher, "provider", "unknown"),
            llm_profile_max_chars=llm_profile_max_chars,
        )
        title = str(candidate_for_output.get("title_guess", ""))
        cached = cache.get(cache_key)

        if cached:
            match = LLMMatchResult.from_dict(cached.get("result", {}))
            cache_hits += 1
            if progress_callback:
                progress_callback(
                    LLMProgressEvent(
                        index=selected_index,
                        total=selected_count,
                        title=title,
                        faiss_rank=int(candidate.get("rank", 0)),
                        cache_status="cache hit",
                        llm_match_score=match.llm_match_score,
                        confidence=match.confidence,
                        phase="done",
                    )
                )
            provider = "cache"
            elapsed = 0.0
        else:
            cache_misses += 1
            if progress_callback:
                progress_callback(
                    LLMProgressEvent(
                        index=selected_index,
                        total=selected_count,
                        title=title,
                        faiss_rank=int(candidate.get("rank", 0)),
                        cache_status="cache miss",
                        phase="start",
                    )
                )
            pending.append(
                PendingScore(
                    selection_key=selection_key,
                    selected_index=selected_index,
                    candidate=candidate_for_output,
                    truncated_profile=truncated_profile,
                    cache_key=cache_key,
                    title=title,
                    faiss_rank=int(candidate.get("rank", 0)),
                )
            )
            continue

        scored[selection_key] = (match, provider, truncated_profile)

    if pending:
        batch_started = time.perf_counter()
        matches = score_pending(
            query=query,
            pending=pending,
            matcher=matcher,
            llm_profile_max_chars=llm_profile_max_chars,
        )
        llm_elapsed_total += time.perf_counter() - batch_started
        average_elapsed = llm_elapsed_total / max(len(pending), 1)

        for item, match in zip(pending, matches, strict=True):
            provider = "rule_fallback" if match is None else getattr(matcher, "provider", "transformers")
            if match is None:
                match = LLMMatchResult(
                    llm_match_score=0.0,
                    confidence="low",
                    risk_flags=["llm_request_failed"],
                    reason="Scoring request failed; candidate kept with a neutral-low result.",
                )
            elif use_cache:
                append_llm_cache(cache_path, item.cache_key, match)
            scored[item.selection_key] = (match, provider, item.truncated_profile)
            if progress_callback:
                progress_callback(
                    LLMProgressEvent(
                        index=item.selected_index,
                        total=selected_count,
                        title=item.title,
                        faiss_rank=item.faiss_rank,
                        cache_status="cache miss",
                        elapsed_seconds=average_elapsed,
                        average_seconds=average_elapsed,
                        estimated_remaining_seconds=0.0,
                        llm_match_score=match.llm_match_score,
                        confidence=match.confidence,
                        phase="done",
                    )
                )

    imputed_llm_score, imputed_confidence = impute_unscored_features([item[0] for item in scored.values()])

    # Pass 2 builds every row. Under the default "impute" policy both branches use the
    # same formula and therefore the same 0..1 scale, so selection into the LLM pool
    # carries no score advantage of its own.
    for idx, candidate in enumerate(candidates):
        novel_id = str(candidate.get("novel_id", idx))
        selection_key = candidate_key(candidate, fallback=idx)
        candidate_for_output = selected_by_id.get(selection_key, candidate)
        profile_text = profile_lookup.get(novel_id) or str(candidate.get("profile_text_preview", ""))
        normalized_score = normalized.get(novel_id, 0.0)

        if selection_key in scored:
            match, provider, truncated_profile = scored[selection_key]
            risk_penalty = compute_risk_penalty(match, truncated_profile)
            rows.append(
                build_output_row(
                    candidate=candidate_for_output,
                    normalized_semantic_score=normalized_score,
                    selected_for_llm=True,
                    analysis_provider=provider,
                    cache_hit=provider == "cache",
                    final_score=llm_final_score(normalized_score, match, risk_penalty, weights),
                    match=match,
                    risk_penalty=risk_penalty,
                    profile_text=profile_text,
                )
            )
            continue

        if fallback_policy == "legacy_semantic":
            final_score = semantic_fallback_score(normalized_score, int(candidate.get("matched_query_count", 1)))
            row_imputed: tuple[float | None, float | None] = (None, None)
        else:
            final_score = imputed_final_score(normalized_score, imputed_llm_score, imputed_confidence, weights)
            row_imputed = (imputed_llm_score, imputed_confidence)

        rows.append(
            build_output_row(
                candidate=candidate_for_output,
                normalized_semantic_score=normalized_score,
                selected_for_llm=False,
                analysis_provider="semantic_fallback",
                cache_hit=False,
                final_score=final_score,
                match=None,
                risk_penalty=0.0,
                profile_text=profile_text,
                imputed_llm_match_score=row_imputed[0],
                imputed_confidence_score=row_imputed[1],
            )
        )

    rerank_started = time.perf_counter()
    rows.sort(key=lambda item: item["final_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["final_rank"] = rank
    final_reranking = time.perf_counter() - rerank_started
    total = time.perf_counter() - started
    timing = TimingSummary(
        llm_scoring=llm_elapsed_total,
        average_llm_scoring_time=llm_elapsed_total / max(cache_misses, 1) if cache_misses else 0.0,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        final_reranking=final_reranking,
        total_runtime=total,
        llm_candidate_k=selected_count,
        provider=getattr(matcher, "provider", ""),
        llm_model=llm_model,
        llm_profile_max_chars=llm_profile_max_chars,
        fallback_policy=fallback_policy,
    )
    return rows, timing
