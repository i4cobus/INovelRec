"""Stage 5 demo: run Stage 4 ranking and generate a recommendation report."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import typer
from rich.console import Console

from src.embed import DEFAULT_EMBEDDING_MODEL, load_embedding_model
from src.explain import ExplanationProgressEvent, explain_recommendations, save_report
from src.llm_explain import TransformersExplanationGenerator
from src.llm_matcher import DEFAULT_LLM_MODEL, create_transformers_matcher
from src.preferences import parse_preference_query
from src.query_expansion import build_expanded_queries
from src.rank import CACHE_PATH, load_profile_text_lookup, rerank_candidates_with_llm, resolve_llm_candidate_k
from src.report import format_report
from src.search import load_faiss_index, load_id_map, multi_query_semantic_search
from src.vector_index import DEFAULT_ID_MAP_PATH, DEFAULT_INDEX_PATH, DEFAULT_PROFILES_PATH

app = typer.Typer(add_completion=False)
console = Console(width=180)


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


def explanation_progress(event: ExplanationProgressEvent) -> None:
    if event.phase == "start":
        console.print(f"[Explanation] {event.index}/{event.total} | {event.title} | generating...")
    else:
        console.print(
            f"[Explanation] {event.index}/{event.total} | done in {event.elapsed_seconds:.1f}s | "
            f"confidence={event.confidence}"
        )


@app.command()
def main(
    query: str = typer.Argument(..., help="Chinese preference query."),
    candidate_k: int = typer.Option(200, help="Number of merged candidates kept after multi-query retrieval."),
    top_k_per_query: int = typer.Option(100, help="FAISS results fetched per expanded query."),
    llm_candidate_k: int | None = typer.Option(None, help="Number of candidates sent to the local LLM reranker."),
    top_k: int = typer.Option(5, help="Number of final recommendations to explain."),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer embedding model."),
    llm_model: str = typer.Option(DEFAULT_LLM_MODEL, help="Transformers local Qwen model."),
    device: str = typer.Option("auto", help="Torch device: auto, cuda, or cpu."),
    explanation_profile_max_chars: int = typer.Option(1200, help="Max profile chars included in explanation prompt."),
    llm_profile_max_chars: int = typer.Option(1200, help="Max profile chars included in Stage 4 reranking prompt."),
    output_format: str = typer.Option("text", help="Output format: text, markdown, or json."),
    save_report_path: Path | None = typer.Option(None, "--save-report", help="Optional report output path."),
    index: Path = typer.Option(DEFAULT_INDEX_PATH, help="FAISS index path."),
    id_map: Path = typer.Option(DEFAULT_ID_MAP_PATH, help="Row metadata JSON path."),
    profiles: Path = typer.Option(DEFAULT_PROFILES_PATH, help="Novel profiles parquet path."),
    use_query_expansion: bool = typer.Option(True, "--use-query-expansion/--no-query-expansion", help="Use local LLM query expansion."),
    use_domain_hints: bool = typer.Option(True, "--use-domain-hints/--no-domain-hints", help="Use retrieval-only domain hints."),
    max_expanded_queries: int = typer.Option(5, help="Maximum expanded retrieval queries."),
    use_cache: bool = typer.Option(True, "--use-cache/--no-cache", help="Reuse cached Stage 4 LLM candidate analysis."),
    cache_path: Path = typer.Option(CACHE_PATH, help="Stage 4 rerank cache JSONL path."),
) -> None:
    """Run Stage 4, then generate grounded explanations for final top-k candidates."""

    if output_format not in {"text", "markdown", "json"}:
        raise typer.BadParameter("output-format must be text, markdown, or json")
    resolved_llm_k, warning = resolve_llm_candidate_k(candidate_k, llm_candidate_k)
    if warning:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    resolved_device = resolve_device(device)
    started = time.perf_counter()

    expansion_provider = None
    if use_query_expansion:
        expansion_provider = create_transformers_matcher(model_name=llm_model, device=resolved_device, max_new_tokens=256)
    expanded_queries = build_expanded_queries(
        raw_query=query,
        structured_preference=parse_preference_query(query),
        llm_provider=expansion_provider,
        use_llm_expansion=use_query_expansion,
        use_domain_hints=use_domain_hints,
        max_expanded_queries=max_expanded_queries,
    )
    if expansion_provider is not None:
        del expansion_provider
        clear_cuda_if_needed(resolved_device)

    embedder = load_embedding_model(embedding_model, device=resolved_device)
    candidates = multi_query_semantic_search(
        expanded_queries=expanded_queries,
        model=embedder,
        index=load_faiss_index(index),
        id_map=load_id_map(id_map),
        top_k_per_query=top_k_per_query,
        final_candidate_k=candidate_k,
    )
    del embedder
    clear_cuda_if_needed(resolved_device)

    profile_lookup = load_profile_text_lookup(profiles)
    matcher = create_transformers_matcher(model_name=llm_model, device=resolved_device, max_new_tokens=256)
    ranked, _ = rerank_candidates_with_llm(
        query=query,
        candidates=candidates,
        matcher=matcher,
        llm_candidate_k=resolved_llm_k,
        llm_profile_max_chars=llm_profile_max_chars,
        profile_lookup=profile_lookup,
        use_cache=use_cache,
        cache_path=cache_path,
        llm_model=llm_model,
    )

    generator = TransformersExplanationGenerator(matcher)
    final_candidates = ranked[:top_k]
    explanations, summary = explain_recommendations(
        query=query,
        candidates=final_candidates,
        generator=generator,
        profile_lookup=profile_lookup,
        max_profile_chars=explanation_profile_max_chars,
        max_new_tokens=512,
        progress_callback=explanation_progress,
    )
    report = format_report(query, explanations, output_format=output_format)  # type: ignore[arg-type]
    console.print(report, markup=False)

    console.print("Explanation summary:")
    console.print(f"- explained candidates: {summary.explained_candidates}")
    console.print(f"- successful LLM JSON outputs: {summary.successful_llm_json_outputs}")
    console.print(f"- fallback explanations: {summary.fallback_explanations}")
    console.print(f"- total explanation time: {summary.total_explanation_time:.1f}s")
    console.print(f"- average time per candidate: {summary.average_time_per_candidate:.2f}s")
    console.print(f"- total runtime: {time.perf_counter() - started:.1f}s")

    if save_report_path is not None:
        save_report(save_report_path, report)
        console.print(f"Saved report: {save_report_path}")


if __name__ == "__main__":
    app()

