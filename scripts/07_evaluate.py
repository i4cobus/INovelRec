"""Run lightweight baseline/full recommendation evaluation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from src.app_pipeline import resolve_device
from src.embed import DEFAULT_EMBEDDING_MODEL, load_embedding_model
from src.evaluation import EvalQuery, compute_anchor_metrics, load_eval_queries, write_eval_outputs
from src.llm_matcher import DEFAULT_LLM_MODEL, create_transformers_matcher
from src.preferences import parse_preference_query
from src.query_expansion import ExpandedQuery, build_expanded_queries
from src.rank import load_profile_text_lookup, rerank_candidates_with_llm, resolve_llm_candidate_k
from src.search import load_faiss_index, load_id_map, multi_query_semantic_search, semantic_search
from src.vector_index import DEFAULT_ID_MAP_PATH, DEFAULT_INDEX_PATH, DEFAULT_PROFILES_PATH

app = typer.Typer(add_completion=False)
console = Console(width=180)


def result_row(query: EvalQuery, variant: str, rank: int, item: dict[str, Any]) -> dict[str, Any]:
    """Build a flat evaluation result row."""

    return {
        "query_id": query.query_id,
        "query": query.query,
        "system_variant": variant,
        "rank": rank,
        "title_guess": str(item.get("title_guess", "")),
        "novel_id": str(item.get("novel_id", "")),
        "score": item.get("final_score", item.get("score", "")),
        "semantic_score": item.get("semantic_score", item.get("score", "")),
        "llm_match_score": item.get("llm_match_score", ""),
        "confidence": item.get("confidence", ""),
        "selected_for_llm": item.get("selected_for_llm", False),
        "best_faiss_rank": item.get("best_faiss_rank", item.get("rank", "")),
        "matched_query_count": item.get("matched_query_count", ""),
        "retrieval_sources": ",".join(item.get("retrieval_sources", [])) if isinstance(item.get("retrieval_sources"), list) else item.get("retrieval_sources", ""),
        "llm_selection_reasons": ",".join(item.get("llm_selection_reasons", [])) if isinstance(item.get("llm_selection_reasons"), list) else item.get("llm_selection_reasons", ""),
        "reason": item.get("reason", ""),
        "anchor_titles": "|".join(query.anchor_titles),
        "wanted": "|".join(query.wanted),
        "unwanted": "|".join(query.unwanted),
    }


def run_baseline(query: EvalQuery, model: Any, index: Any, id_map: dict[int, dict[str, str]], top_k: int) -> list[dict[str, Any]]:
    """Run FAISS-only semantic retrieval using the raw query."""

    return semantic_search(query.query, model, index, id_map, top_k=top_k)


def run_full(
    query: EvalQuery,
    *,
    model: Any,
    index: Any,
    id_map: dict[int, dict[str, str]],
    matcher: Any,
    profile_lookup: dict[str, str],
    candidate_k: int,
    top_k_per_query: int,
    llm_candidate_k: int,
    llm_model: str,
) -> list[dict[str, Any]]:
    """Run query expansion, multi-query retrieval, and local LLM reranking."""

    expanded_queries = build_expanded_queries(
        raw_query=query.query,
        structured_preference=parse_preference_query(query.query),
        llm_provider=matcher,
        use_llm_expansion=True,
        use_domain_hints=True,
        max_expanded_queries=5,
    )
    candidates = multi_query_semantic_search(
        expanded_queries=expanded_queries,
        model=model,
        index=index,
        id_map=id_map,
        top_k_per_query=top_k_per_query,
        final_candidate_k=candidate_k,
    )
    resolved_llm_k, _ = resolve_llm_candidate_k(len(candidates), llm_candidate_k) if candidates else (0, None)
    ranked, _ = rerank_candidates_with_llm(
        query=query.query,
        candidates=candidates,
        matcher=matcher,
        llm_candidate_k=resolved_llm_k,
        profile_lookup=profile_lookup,
        llm_model=llm_model,
    )
    return ranked


def print_anchor_summary(summary: dict[str, Any]) -> None:
    """Print automatic anchor metrics."""

    console.print(f"Queries: {summary['num_queries']}")
    console.print(f"Queries with anchors: {summary['num_queries_with_anchors']}")
    table = Table(title="Anchor Metrics")
    table.add_column("Variant")
    table.add_column("Hit@1", justify="right")
    table.add_column("Hit@5", justify="right")
    table.add_column("Hit@10", justify="right")
    table.add_column("Avg first anchor rank", justify="right")
    for variant, values in summary["variants"].items():
        table.add_row(
            variant,
            f"{values.get('Anchor Hit@1', 0.0):.3f}",
            f"{values.get('Anchor Hit@5', 0.0):.3f}",
            f"{values.get('Anchor Hit@10', 0.0):.3f}",
            "-" if values.get("average_first_anchor_rank") is None else f"{values['average_first_anchor_rank']:.2f}",
        )
    console.print(table)


@app.command()
def main(
    eval_file: Path = typer.Option(Path("eval/eval_queries.jsonl"), help="Evaluation query JSONL file."),
    out_dir: Path = typer.Option(Path("eval/results"), help="Directory for CSV/JSONL result outputs."),
    top_k: int = typer.Option(10, help="Top-k results saved per query and variant."),
    candidate_k: int = typer.Option(100, help="Full-system candidate pool size."),
    llm_candidate_k: int = typer.Option(10, help="Number of candidates sent to the local LLM in full mode."),
    top_k_per_query: int = typer.Option(100, help="FAISS results per expanded query in full mode."),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer embedding model."),
    llm_model: str = typer.Option(DEFAULT_LLM_MODEL, help="Local Qwen LLM model."),
    index: Path = typer.Option(DEFAULT_INDEX_PATH, help="FAISS index path."),
    id_map: Path = typer.Option(DEFAULT_ID_MAP_PATH, help="Novel id map path."),
    profiles: Path = typer.Option(DEFAULT_PROFILES_PATH, help="Novel profiles parquet path."),
    device: str = typer.Option("auto", help="Torch device: auto, cuda, or cpu."),
    mode: str = typer.Option("baseline", help="Evaluation mode: baseline, full, or both."),
    skip_llm: bool = typer.Option(False, help="Skip full LLM mode and run baseline only."),
) -> None:
    """Run lightweight evaluation and write result files."""

    if top_k <= 0 or candidate_k <= 0:
        raise typer.BadParameter("top-k and candidate-k must be positive")
    if mode not in {"baseline", "full", "both"}:
        raise typer.BadParameter("mode must be baseline, full, or both")
    if skip_llm:
        mode = "baseline"

    started = time.perf_counter()
    queries = load_eval_queries(eval_file)
    resolved_device = resolve_device(device)
    embedder = load_embedding_model(embedding_model, device=resolved_device)
    faiss_index = load_faiss_index(index)
    row_map = load_id_map(id_map)

    matcher = None
    profile_lookup: dict[str, str] = {}
    if mode in {"full", "both"}:
        matcher = create_transformers_matcher(model_name=llm_model, device=resolved_device, max_new_tokens=256)
        profile_lookup = load_profile_text_lookup(profiles)

    rows: list[dict[str, Any]] = []
    for query in queries:
        console.print(f"Evaluating {query.query_id}: {query.query}")
        if mode in {"baseline", "both"}:
            baseline = run_baseline(query, embedder, faiss_index, row_map, top_k=top_k)
            rows.extend(result_row(query, "baseline_faiss", rank, item) for rank, item in enumerate(baseline[:top_k], start=1))

        if mode in {"full", "both"} and matcher is not None:
            full = run_full(
                query,
                model=embedder,
                index=faiss_index,
                id_map=row_map,
                matcher=matcher,
                profile_lookup=profile_lookup,
                candidate_k=candidate_k,
                top_k_per_query=top_k_per_query,
                llm_candidate_k=llm_candidate_k,
                llm_model=llm_model,
            )
            rows.extend(result_row(query, "full_llm_rerank", rank, item) for rank, item in enumerate(full[:top_k], start=1))

    csv_path, jsonl_path = write_eval_outputs(rows, out_dir)
    console.print(f"Wrote CSV: {csv_path}")
    console.print(f"Wrote JSONL: {jsonl_path}")
    print_anchor_summary(compute_anchor_metrics(rows, queries, ks=(1, 5, 10)))
    console.print(f"Runtime: {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    app()
