"""CLI demo for Stage 4 transformers-only local LLM recommendation ranking."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.embed import DEFAULT_EMBEDDING_MODEL, load_embedding_model
from src.llm_matcher import DEFAULT_LLM_MODEL, create_transformers_matcher
from src.preferences import parse_preference_query
from src.query_expansion import build_expanded_queries, expansion_summary_by_source
from src.rank import (
    CACHE_PATH,
    LLMProgressEvent,
    TimingSummary,
    load_profile_text_lookup,
    rerank_candidates_with_llm,
    resolve_llm_candidate_k,
)
from src.search import debug_target_status, load_faiss_index, load_id_map, multi_query_semantic_search
from src.vector_index import DEFAULT_ID_MAP_PATH, DEFAULT_INDEX_PATH, DEFAULT_PROFILES_PATH

app = typer.Typer(add_completion=False)
console = Console(width=200)


def resolve_device(device: str) -> str | None:
    normalized = device.strip().lower()
    if normalized in {"", "auto"}:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else None
        except ImportError:
            return None
    return normalized


def clear_cuda_if_needed(device: str | None) -> None:
    gc.collect()
    if device == "cuda":
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass


def progress_printer(event: LLMProgressEvent) -> None:
    if event.phase == "start":
        console.print(
            f"[LLM scoring] {event.index}/{event.total} | FAISS rank {event.faiss_rank} | "
            f"{event.title} | {event.cache_status} | scoring..."
        )
        return
    if event.cache_status == "cache hit":
        console.print(
            f"[LLM scoring] {event.index}/{event.total} | FAISS rank {event.faiss_rank} | "
            f"{event.title} | cache hit | llm_match_score={event.llm_match_score:.2f} | confidence={event.confidence}"
        )
        return
    eta = f" | ETA {event.estimated_remaining_seconds:.1f}s" if event.estimated_remaining_seconds is not None else ""
    console.print(
        f"[LLM scoring] {event.index}/{event.total} | done in {event.elapsed_seconds:.1f}s | "
        f"avg {event.average_seconds:.1f}s | llm_match_score={event.llm_match_score:.2f} | "
        f"confidence={event.confidence}{eta}"
    )


def print_expanded_queries(expanded_queries: list) -> None:
    console.print("Expanded retrieval queries:")
    for idx, query in enumerate(expanded_queries, start=1):
        console.print(f"{idx}. [{query.source}, weight={query.weight:.2f}] {query.text}", markup=False)


def print_retrieval_summary(expanded_queries: list, candidates: list[dict], top_k_per_query: int, candidate_k: int) -> None:
    source_counts = expansion_summary_by_source(expanded_queries)
    candidate_source_counts: dict[str, int] = {}
    for candidate in candidates:
        for source in candidate.get("retrieval_sources", []):
            candidate_source_counts[source] = candidate_source_counts.get(source, 0) + 1
    table = Table(title="Retrieval Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Expanded queries", str(len(expanded_queries)))
    table.add_row("Query sources", ", ".join(f"{key}={value}" for key, value in source_counts.items()))
    table.add_row("top-k-per-query", str(top_k_per_query))
    table.add_row("Unique candidates after merge", str(len(candidates)))
    table.add_row("final candidate-k", str(candidate_k))
    table.add_row("Candidate source hits", ", ".join(f"{key}={value}" for key, value in candidate_source_counts.items()) or "-")
    console.print(table)


def print_debug_target(title: str, candidates: list[dict], recommendations: list[dict] | None = None, top_k: int = 10) -> None:
    status = debug_target_status(title, candidates, recommendations, top_k=top_k)
    console.print(f"Debug target title: {title}")
    console.print(f"- merged candidate pool: {'yes' if status['in_merged_pool'] else 'no'}")
    console.print(f"- LLM-selected pool: {'yes' if status['in_llm_selected_pool'] else 'no'}")
    console.print(f"- final top-{top_k}: {'yes' if status['in_final_top_k'] else 'no'}")
    found = status.get("final") or status.get("llm_selected") or status.get("merged")
    if found:
        console.print(
            "- details: "
            f"retrieval_rank={found.get('retrieval_rank', found.get('final_rank', '-'))}, "
            f"best_faiss_rank={found.get('best_faiss_rank', '-')}, "
            f"matched_query_count={found.get('matched_query_count', '-')}, "
            f"retrieval_sources={found.get('retrieval_sources', '-')}, "
            f"semantic_score={found.get('semantic_score', found.get('best_semantic_score', '-'))}, "
            f"final_score={found.get('final_score', '-')}"
        )


def print_timing_summary(timing: TimingSummary) -> None:
    table = Table(title="Timing Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Preference parsing", f"{timing.preference_parsing:.2f}s")
    table.add_row("FAISS retrieval", f"{timing.faiss_retrieval:.2f}s")
    table.add_row("Load profiles", f"{timing.load_profiles:.2f}s")
    table.add_row("LLM scoring", f"{timing.llm_scoring:.2f}s")
    table.add_row("Average LLM scoring time", f"{timing.average_llm_scoring_time:.2f}s / candidate")
    table.add_row("Cache hits", str(timing.cache_hits))
    table.add_row("Cache misses", str(timing.cache_misses))
    table.add_row("Final reranking", f"{timing.final_reranking:.2f}s")
    table.add_row("Total runtime", f"{timing.total_runtime:.2f}s")
    table.add_row("candidate-k", str(timing.candidate_k))
    table.add_row("llm-candidate-k", str(timing.llm_candidate_k))
    table.add_row("top-k", str(timing.top_k))
    table.add_row("provider", "transformers")
    table.add_row("llm model", timing.llm_model)
    table.add_row("llm profile max chars", str(timing.llm_profile_max_chars))
    console.print(table)


@app.command()
def main(
    query: str = typer.Argument(..., help="Chinese preference query."),
    index: Path = typer.Option(DEFAULT_INDEX_PATH, help="FAISS index path."),
    id_map: Path = typer.Option(DEFAULT_ID_MAP_PATH, help="Row metadata JSON path."),
    profiles: Path = typer.Option(DEFAULT_PROFILES_PATH, help="Novel profiles parquet path."),
    model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer model name."),
    llm_model: str = typer.Option(DEFAULT_LLM_MODEL, help="Transformers local LLM model name."),
    llm_max_new_tokens: int = typer.Option(256, help="Maximum tokens for each local LLM JSON scoring response."),
    candidate_k: int = typer.Option(200, help="Number of merged candidates kept after multi-query retrieval."),
    top_k_per_query: int = typer.Option(100, help="FAISS results fetched per expanded query."),
    llm_candidate_k: int | None = typer.Option(None, help="Number of candidates sent to the local LLM."),
    llm_profile_max_chars: int = typer.Option(1200, help="Maximum profile characters sent to the local LLM."),
    top_k: int = typer.Option(10, help="Number of final recommendations to show."),
    use_query_expansion: bool = typer.Option(True, "--use-query-expansion/--no-query-expansion", help="Use local LLM query expansion."),
    use_domain_hints: bool = typer.Option(True, "--use-domain-hints/--no-domain-hints", help="Use small domain hints for retrieval expansion."),
    max_expanded_queries: int = typer.Option(5, help="Maximum number of retrieval query variants."),
    debug_target_title: str | None = typer.Option(None, help="Print retrieval/rerank status for an expected title."),
    use_cache: bool = typer.Option(True, "--use-cache/--no-cache", help="Reuse cached local LLM candidate analysis."),
    cache_path: Path = typer.Option(CACHE_PATH, help="LLM rerank cache JSONL path."),
    device: str = typer.Option("auto", help="Torch device: auto, cuda, or cpu."),
) -> None:
    """Retrieve semantic candidates with expansion, then rerank them with a transformers local LLM."""

    total_started = time.perf_counter()
    if candidate_k <= 0:
        raise typer.BadParameter("candidate-k must be positive")
    if top_k <= 0:
        raise typer.BadParameter("top-k must be positive")
    resolved_llm_k, warning = resolve_llm_candidate_k(candidate_k, llm_candidate_k)
    if warning:
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    resolved_device = resolve_device(device)
    preference_started = time.perf_counter()
    structured_preference = parse_preference_query(query)
    preference_parsing = time.perf_counter() - preference_started

    expansion_provider = None
    if use_query_expansion:
        expansion_provider = create_transformers_matcher(
            model_name=llm_model,
            device=resolved_device,
            max_new_tokens=llm_max_new_tokens,
        )
    expanded_queries = build_expanded_queries(
        raw_query=query,
        structured_preference=structured_preference,
        llm_provider=expansion_provider,
        use_llm_expansion=use_query_expansion,
        use_domain_hints=use_domain_hints,
        max_expanded_queries=max_expanded_queries,
    )
    print_expanded_queries(expanded_queries)
    if expansion_provider is not None:
        del expansion_provider
        clear_cuda_if_needed(resolved_device)

    retrieval_started = time.perf_counter()
    embedding_model = load_embedding_model(model, device=resolved_device)
    faiss_index = load_faiss_index(index)
    row_map = load_id_map(id_map)
    candidates = multi_query_semantic_search(
        expanded_queries=expanded_queries,
        model=embedding_model,
        index=faiss_index,
        id_map=row_map,
        top_k_per_query=top_k_per_query,
        final_candidate_k=candidate_k,
    )
    faiss_retrieval = time.perf_counter() - retrieval_started
    print_retrieval_summary(expanded_queries, candidates, top_k_per_query, candidate_k)
    if debug_target_title:
        print_debug_target(debug_target_title, candidates, None, top_k=top_k)

    del embedding_model
    clear_cuda_if_needed(resolved_device)

    profile_started = time.perf_counter()
    profile_lookup = load_profile_text_lookup(profiles)
    load_profiles = time.perf_counter() - profile_started

    matcher = create_transformers_matcher(
        model_name=llm_model,
        device=resolved_device,
        max_new_tokens=llm_max_new_tokens,
    )
    recommendations, timing = rerank_candidates_with_llm(
        query=query,
        candidates=candidates,
        matcher=matcher,
        llm_candidate_k=resolved_llm_k,
        llm_profile_max_chars=llm_profile_max_chars,
        profile_lookup=profile_lookup,
        use_cache=use_cache,
        cache_path=cache_path,
        llm_model=llm_model,
        progress_callback=progress_printer,
    )
    if debug_target_title:
        print_debug_target(debug_target_title, candidates, recommendations, top_k=top_k)

    timing.preference_parsing = preference_parsing
    timing.faiss_retrieval = faiss_retrieval
    timing.load_profiles = load_profiles
    timing.total_runtime = time.perf_counter() - total_started
    timing.candidate_k = candidate_k
    timing.top_k = top_k
    timing.provider = "transformers"

    table = Table(title="Hybrid Recommendations")
    table.add_column("Final", justify="right")
    table.add_column("BestFAISS", justify="right")
    table.add_column("Queries", justify="right")
    table.add_column("Sources")
    table.add_column("LLM?", justify="center")
    table.add_column("Title", overflow="fold")
    table.add_column("Semantic", justify="right")
    table.add_column("LLM", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Cache", justify="center")
    table.add_column("Reason", overflow="fold")

    for item in recommendations[:top_k]:
        table.add_row(
            str(item["final_rank"]),
            str(item["best_faiss_rank"]),
            str(item["matched_query_count"]),
            ",".join(item["retrieval_sources"]),
            "Y" if item["selected_for_llm"] else "N",
            str(item["title_guess"]),
            f"{item['semantic_score']:.4f}",
            "-" if item["llm_match_score"] is None else f"{item['llm_match_score']:.2f}",
            f"{item['final_score']:.4f}",
            "Y" if item["cache_hit"] else "N",
            str(item["reason"] or "-")[:120],
        )
    console.print(table)
    print_timing_summary(timing)


if __name__ == "__main__":
    app()
