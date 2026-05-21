"""Hybrid recommendation ranking using FAISS scores plus local LLM features."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd

from src.config import DATA_DIR
from src.llm_matcher import LLMMatchResult, PROMPT_VERSION, confidence_to_score
from src.vector_index import DEFAULT_PROFILES_PATH

CACHE_PATH = DATA_DIR / "cache" / "llm_rerank_cache.jsonl"
BOILERPLATE_PATTERNS = ("知轩藏书", "zxcs", "www.zxcs", "精校小说下载", "更多精校小说")


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
    """Normalize semantic scores to 0..1 across candidates."""

    if not candidates:
        return {}
    scores = [float(candidate.get("score", 0.0)) for candidate in candidates]
    if all(0.0 <= score <= 1.0 for score in scores):
        return {
            str(candidate.get("novel_id", idx)): score
            for idx, (candidate, score) in enumerate(zip(candidates, scores, strict=False))
        }
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return {str(candidate.get("novel_id", idx)): 1.0 for idx, candidate in enumerate(candidates)}
    return {
        str(candidate.get("novel_id", idx)): (float(candidate.get("score", 0.0)) - min_score) / (max_score - min_score)
        for idx, candidate in enumerate(candidates)
    }


def load_profile_text_lookup(profiles_path: Path = DEFAULT_PROFILES_PATH) -> dict[str, str]:
    """Load full profile text by novel_id if the profile parquet is available."""

    if not profiles_path.exists():
        return {}
    profiles = pd.read_parquet(profiles_path, columns=["novel_id", "profile_text"])
    profiles = profiles.dropna(subset=["novel_id", "profile_text"])
    return dict(zip(profiles["novel_id"].astype(str), profiles["profile_text"].astype(str), strict=False))


def resolve_llm_candidate_k(candidate_k: int, llm_candidate_k: int | None) -> tuple[int, str | None]:
    """Resolve and clamp the number of candidates sent to the local LLM."""

    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    resolved = min(10, candidate_k) if llm_candidate_k is None else llm_candidate_k
    if resolved <= 0:
        raise ValueError("llm_candidate_k must be positive")
    if resolved > candidate_k:
        return candidate_k, f"llm-candidate-k {resolved} exceeds candidate-k {candidate_k}; clamped to {candidate_k}."
    return resolved, None


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
) -> dict[str, Any]:
    return {
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
) -> tuple[list[dict[str, Any]], TimingSummary]:
    """Rerank semantic candidates with budgeted local LLM analysis, cache, and timing."""

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

    for idx, candidate in enumerate(candidates):
        novel_id = str(candidate.get("novel_id", idx))
        selection_key = candidate_key(candidate, fallback=idx)
        candidate_for_output = selected_by_id.get(selection_key, candidate)
        profile_text = profile_lookup.get(novel_id) or str(candidate.get("profile_text_preview", ""))
        normalized_score = normalized.get(novel_id, 0.0)

        if selection_key not in selected_by_id:
            rows.append(
                build_output_row(
                    candidate=candidate_for_output,
                    normalized_semantic_score=normalized_score,
                    selected_for_llm=False,
                    analysis_provider="semantic_fallback",
                    cache_hit=False,
                    final_score=semantic_fallback_score(normalized_score, int(candidate.get("matched_query_count", 1))),
                    match=None,
                    risk_penalty=0.0,
                    profile_text=profile_text,
                )
            )
            continue

        selected_index = len([row for row in rows if row["selected_for_llm"]]) + 1
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
            item_started = time.perf_counter()
            try:
                match = matcher.score(query=query, candidate=candidate_for_output, profile_text=truncated_profile, max_profile_chars=llm_profile_max_chars)
                provider = getattr(matcher, "provider", "transformers")
            except Exception as exc:  # pragma: no cover - defensive runtime fallback
                match = LLMMatchResult(
                    llm_match_score=0.0,
                    confidence="low",
                    risk_flags=["llm_exception"],
                    reason=str(exc)[:180],
                )
                provider = "rule_fallback"
            elapsed = time.perf_counter() - item_started
            llm_elapsed_total += elapsed
            if use_cache and provider != "rule_fallback":
                append_llm_cache(cache_path, cache_key, match)
            avg = llm_elapsed_total / max(cache_misses, 1)
            remaining = max(selected_count - selected_index, 0) * avg
            if progress_callback:
                progress_callback(
                    LLMProgressEvent(
                        index=selected_index,
                        total=selected_count,
                        title=title,
                        faiss_rank=int(candidate.get("rank", 0)),
                        cache_status="cache miss",
                        elapsed_seconds=elapsed,
                        average_seconds=avg,
                        estimated_remaining_seconds=remaining,
                        llm_match_score=match.llm_match_score,
                        confidence=match.confidence,
                        phase="done",
                    )
                )

        risk_penalty = compute_risk_penalty(match, truncated_profile)
        final_score = llm_final_score(normalized_score, match, risk_penalty, weights)
        rows.append(
            build_output_row(
                candidate=candidate_for_output,
                normalized_semantic_score=normalized_score,
                selected_for_llm=True,
                analysis_provider=provider,
                cache_hit=provider == "cache",
                final_score=final_score,
                match=match,
                risk_penalty=risk_penalty,
                profile_text=profile_text,
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
    )
    return rows, timing
