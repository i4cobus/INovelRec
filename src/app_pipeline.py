"""Reusable Stage 4 -> Stage 5 pipeline helpers for CLI and Streamlit."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.embed import DEFAULT_EMBEDDING_MODEL, SupportsEncode
from src.explain import ExplanationProgressEvent, RecommendationExplanation, explain_recommendations
from src.llm_explain import TransformersExplanationGenerator
from src.llm_matcher import DEFAULT_LLM_MODEL
from src.preferences import parse_preference_query
from src.query_expansion import ExpandedQuery, build_expanded_queries, expansion_summary_by_source
from src.rank import CACHE_PATH, LLMProgressEvent, load_profile_text_lookup, rerank_candidates_with_llm, resolve_llm_candidate_k
from src.report import format_report
from src.search import multi_query_semantic_search
from src.vector_index import DEFAULT_ID_MAP_PATH, DEFAULT_INDEX_PATH, DEFAULT_PROFILES_PATH


@dataclass(frozen=True)
class PipelineSettings:
    """Runtime settings shared by the Streamlit app and tests."""

    candidate_k: int = 200
    top_k_per_query: int = 100
    llm_candidate_k: int | None = None
    top_k: int = 5
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    device: str = "auto"
    explanation_profile_max_chars: int = 1200
    llm_profile_max_chars: int = 1200
    use_query_expansion: bool = True
    use_domain_hints: bool = True
    max_expanded_queries: int = 5
    use_cache: bool = True
    cache_path: Path = CACHE_PATH
    index_path: Path = DEFAULT_INDEX_PATH
    id_map_path: Path = DEFAULT_ID_MAP_PATH
    profiles_path: Path = DEFAULT_PROFILES_PATH


@dataclass(frozen=True)
class PipelineResult:
    """Complete app-facing result from recommendation plus explanation."""

    query: str
    settings: PipelineSettings
    resolved_device: str | None
    resolved_llm_candidate_k: int
    expanded_queries: list[ExpandedQuery]
    candidates: list[dict[str, Any]]
    ranked_candidates: list[dict[str, Any]]
    explanations: list[RecommendationExplanation]
    report_markdown: str
    report_json: str
    retrieval_summary: dict[str, Any]
    timing: dict[str, float]

    def raw_dict(self) -> dict[str, Any]:
        """Return JSON-serializable debug data."""

        return {
            "query": self.query,
            "settings": {
                **self.settings.__dict__,
                "cache_path": str(self.settings.cache_path),
                "index_path": str(self.settings.index_path),
                "id_map_path": str(self.settings.id_map_path),
                "profiles_path": str(self.settings.profiles_path),
            },
            "resolved_device": self.resolved_device,
            "resolved_llm_candidate_k": self.resolved_llm_candidate_k,
            "expanded_queries": [query.__dict__ for query in self.expanded_queries],
            "retrieval_summary": self.retrieval_summary,
            "ranked_candidates": self.ranked_candidates,
            "explanations": [explanation.to_dict() for explanation in self.explanations],
            "timing": self.timing,
        }


ProgressCallback = Callable[[str, str, float | None], None]


def resolve_device(device: str) -> str | None:
    """Resolve auto/cuda/cpu into the device value used by model loaders."""

    normalized = device.strip().lower()
    if normalized in {"", "auto"}:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else None
        except ImportError:
            return None
    return normalized


def validate_runtime_files(settings: PipelineSettings) -> list[str]:
    """Return missing artifact messages instead of raising cryptic file errors."""

    missing: list[str] = []
    if not settings.index_path.exists():
        missing.append(f"FAISS index not found: {settings.index_path}. Run Stage 3 first.")
    if not settings.id_map_path.exists():
        missing.append(f"Novel id map not found: {settings.id_map_path}. Run Stage 3 first.")
    if not settings.profiles_path.exists():
        missing.append(f"Novel profiles not found: {settings.profiles_path}. Run Stage 2 first.")
    return missing


def retrieval_summary(expanded_queries: list[ExpandedQuery], candidates: list[dict[str, Any]], settings: PipelineSettings) -> dict[str, Any]:
    """Build a compact retrieval summary for UI display."""

    source_counts: dict[str, int] = {}
    for candidate in candidates:
        for source in candidate.get("retrieval_sources", []):
            source_counts[str(source)] = source_counts.get(str(source), 0) + 1
    return {
        "expanded_query_count": len(expanded_queries),
        "expanded_query_sources": expansion_summary_by_source(expanded_queries),
        "top_k_per_query": settings.top_k_per_query,
        "unique_candidates_after_merge": len(candidates),
        "final_candidate_k": settings.candidate_k,
        "candidate_sources": source_counts,
    }


def load_profiles_for_app(profiles_path: Path = DEFAULT_PROFILES_PATH) -> dict[str, str]:
    """Load profile lookup for app/runtime use."""

    return load_profile_text_lookup(profiles_path)


def run_discovery_pipeline(
    *,
    query: str,
    settings: PipelineSettings,
    embedder: SupportsEncode,
    matcher: Any,
    index: Any,
    id_map: dict[int, dict[str, str]],
    profile_lookup: dict[str, str],
    progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
    """Run Stage 4 ranking followed by Stage 5 grounded explanations."""

    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")
    missing = validate_runtime_files(settings)
    if missing:
        raise FileNotFoundError("\n".join(missing))

    started = time.perf_counter()
    timing: dict[str, float] = {}
    resolved_device = resolve_device(settings.device)
    resolved_llm_k, _ = resolve_llm_candidate_k(settings.candidate_k, settings.llm_candidate_k)

    if progress_callback:
        progress_callback("parse", "Parsing query and building expanded retrieval queries.", 0.05)
    stage_started = time.perf_counter()
    expanded_queries = build_expanded_queries(
        raw_query=query,
        structured_preference=parse_preference_query(query),
        llm_provider=matcher if settings.use_query_expansion else None,
        use_llm_expansion=settings.use_query_expansion,
        use_domain_hints=settings.use_domain_hints,
        max_expanded_queries=settings.max_expanded_queries,
    )
    timing["query_expansion"] = time.perf_counter() - stage_started

    if progress_callback:
        progress_callback("retrieve", "Running multi-query FAISS retrieval.", 0.25)
    stage_started = time.perf_counter()
    candidates = multi_query_semantic_search(
        expanded_queries=expanded_queries,
        model=embedder,
        index=index,
        id_map=id_map,
        top_k_per_query=settings.top_k_per_query,
        final_candidate_k=settings.candidate_k,
    )
    timing["faiss_retrieval"] = time.perf_counter() - stage_started
    if not candidates:
        raise ValueError("No recommendation candidates were retrieved.")

    def rerank_progress(event: LLMProgressEvent) -> None:
        if progress_callback:
            message = f"LLM reranking {event.index}/{event.total}: {event.title}"
            if event.phase == "done":
                message += f" score={event.llm_match_score} confidence={event.confidence}"
            progress_callback("rerank", message, 0.35 + (0.30 * event.index / max(event.total, 1)))

    if progress_callback:
        progress_callback("rerank", "Running local LLM candidate scoring.", 0.35)
    stage_started = time.perf_counter()
    ranked, rerank_timing = rerank_candidates_with_llm(
        query=query,
        candidates=candidates,
        matcher=matcher,
        llm_candidate_k=resolved_llm_k,
        llm_profile_max_chars=settings.llm_profile_max_chars,
        profile_lookup=profile_lookup,
        use_cache=settings.use_cache,
        cache_path=settings.cache_path,
        llm_model=settings.llm_model,
        progress_callback=rerank_progress,
    )
    timing["llm_reranking"] = time.perf_counter() - stage_started
    timing["rerank_cache_hits"] = float(rerank_timing.cache_hits)
    timing["rerank_cache_misses"] = float(rerank_timing.cache_misses)

    generator = TransformersExplanationGenerator(matcher)
    final_candidates = ranked[: settings.top_k]

    def explanation_progress(event: ExplanationProgressEvent) -> None:
        if progress_callback:
            message = f"Explanation {event.index}/{event.total}: {event.title}"
            if event.phase == "done":
                message += f" confidence={event.confidence}"
            progress_callback("explain", message, 0.70 + (0.25 * event.index / max(event.total, 1)))

    if progress_callback:
        progress_callback("explain", "Generating grounded explanations.", 0.70)
    stage_started = time.perf_counter()
    explanations, explanation_summary = explain_recommendations(
        query=query,
        candidates=final_candidates,
        generator=generator,
        profile_lookup=profile_lookup,
        max_profile_chars=settings.explanation_profile_max_chars,
        progress_callback=explanation_progress,
    )
    timing["explanation_generation"] = time.perf_counter() - stage_started
    timing["explanation_fallbacks"] = float(explanation_summary.fallback_explanations)

    if progress_callback:
        progress_callback("render", "Formatting recommendation report.", 0.98)
    report_markdown = format_report(query, explanations, output_format="markdown")
    report_json = format_report(query, explanations, output_format="json")
    timing["total_runtime"] = time.perf_counter() - started

    if progress_callback:
        progress_callback("done", "Recommendation report complete.", 1.0)
    return PipelineResult(
        query=query,
        settings=settings,
        resolved_device=resolved_device,
        resolved_llm_candidate_k=resolved_llm_k,
        expanded_queries=expanded_queries,
        candidates=candidates,
        ranked_candidates=final_candidates,
        explanations=explanations,
        report_markdown=report_markdown,
        report_json=report_json,
        retrieval_summary=retrieval_summary(expanded_queries, candidates, settings),
        timing=timing,
    )
