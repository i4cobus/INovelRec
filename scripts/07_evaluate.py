"""Run lightweight baseline/full recommendation evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

import pandas as pd

from src.app_pipeline import resolve_device
from src.embed import DEFAULT_EMBEDDING_MODEL, load_embedding_model
from src.evaluation import (
    EvalQuery,
    compute_anchor_metrics,
    load_eval_queries,
    resolve_anchor_folds,
    write_eval_outputs,
)
from src.backends import create_matcher
from src.config import DEFAULT_OUTPUT_PATH
from src.llm_matcher import DEFAULT_LLM_MODEL, PROMPT_VERSION
from src.preferences import parse_preference_query
from src.query_expansion import (
    EXPANSION_PROMPT_VERSION,
    ExpandedQuery,
    build_expanded_queries,
    load_expansion_cache,
)
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
        # Exported because their absence was once read as the teacher never filling
        # them. It fills violated_preferences on 78% of scored candidates; the column
        # simply was not written out, and a missing column looks exactly like an empty
        # one. Any field the analysis reasons about has to survive to the results file.
        "violated_preferences": "|".join(item.get("violated_preferences", [])) if isinstance(item.get("violated_preferences"), list) else item.get("violated_preferences", ""),
        "matched_preferences": "|".join(item.get("matched_preferences", [])) if isinstance(item.get("matched_preferences"), list) else item.get("matched_preferences", ""),
        "risk_flags": "|".join(item.get("risk_flags", [])) if isinstance(item.get("risk_flags"), list) else item.get("risk_flags", ""),
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
    fallback_policy: str,
    expansion_cache: dict[str, str],
) -> list[dict[str, Any]]:
    """Run query expansion, multi-query retrieval, and local LLM reranking."""

    # Expansion is cached so a re-run reproduces its candidate pool. vLLM's greedy
    # decoding is not bitwise stable across runs, and one flipped token rewrites an
    # expanded query, changing retrieval and therefore the whole comparison.
    expanded_queries = build_expanded_queries(
        raw_query=query.query,
        structured_preference=parse_preference_query(query.query),
        llm_provider=matcher,
        use_llm_expansion=True,
        use_domain_hints=True,
        max_expanded_queries=5,
        expansion_cache=expansion_cache,
        llm_model=llm_model,
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
        fallback_policy=fallback_policy,
    )
    return ranked


def load_anchor_folds(queries: list[EvalQuery], inventory: Path, splits: Path) -> dict[str, str]:
    """Resolve anchor -> fold, returning empty when the artifacts are missing."""

    if not inventory.exists() or not splits.exists():
        return {}
    novels = pd.read_parquet(inventory, columns=["novel_id", "title_guess"])
    titles = dict(zip(novels["novel_id"].astype(str), novels["title_guess"].astype(str), strict=False))
    folds = pd.read_parquet(splits, columns=["novel_id", "fold"])
    fold_lookup = dict(zip(folds["novel_id"].astype(str), folds["fold"].astype(str), strict=False))
    return resolve_anchor_folds(queries, titles, fold_lookup)


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

    for variant, values in summary["variants"].items():
        breakdown = values.get("Anchor Recall@10 by fold")
        if not breakdown:
            continue
        parts = ", ".join(
            f"{fold} {stats['found']}/{stats['total']} ({stats['recall']:.3f})"
            for fold, stats in breakdown.items()
        )
        console.print(f"{variant} Recall@10 by fold: {parts}")
    if any("Anchor Recall@10 by fold" in values for values in summary["variants"].values()):
        console.print(
            "[dim]Anchors in the train fold will be seen during teacher labelling; a train/eval "
            "gap after training measures memorisation, not retrieval quality.[/dim]"
        )


@app.command()
def main(
    eval_file: Path = typer.Option(Path("eval/eval_queries.jsonl"), help="Evaluation query JSONL file."),
    out_dir: Path = typer.Option(Path("eval/results"), help="Directory for CSV/JSONL result outputs."),
    top_k: int = typer.Option(10, help="Top-k results saved per query and variant."),
    candidate_k: int = typer.Option(100, help="Full-system candidate pool size."),
    llm_candidate_k: int = typer.Option(10, help="Number of candidates sent to the local LLM in full mode. Use 0 to score every candidate."),
    fallback_policy: str = typer.Option("impute", help="Scoring for candidates the LLM never saw: impute or legacy_semantic."),
    top_k_per_query: int = typer.Option(100, help="FAISS results per expanded query in full mode."),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer embedding model."),
    llm_model: str = typer.Option(DEFAULT_LLM_MODEL, help="Local Qwen LLM model."),
    index: Path = typer.Option(DEFAULT_INDEX_PATH, help="FAISS index path."),
    id_map: Path = typer.Option(DEFAULT_ID_MAP_PATH, help="Novel id map path."),
    profiles: Path = typer.Option(DEFAULT_PROFILES_PATH, help="Novel profiles parquet path."),
    splits: Path = typer.Option(Path("data/processed/book_splits.parquet"), help="Book fold assignment, for the train/eval anchor breakdown."),
    device: str = typer.Option("auto", help="Torch device: auto, cuda, or cpu."),
    backend: str = typer.Option("transformers", help="LLM backend: transformers (in-process) or http (OpenAI-compatible endpoint)."),
    llm_base_url: str | None = typer.Option(None, help="Base URL for the http backend, e.g. http://127.0.0.1:8000/v1. Env: INOVELREC_LLM_BASE_URL."),
    llm_max_workers: int | None = typer.Option(None, help="Concurrent requests for the http backend."),
    mode: str = typer.Option("baseline", help="Evaluation mode: baseline, full, or both."),
    skip_llm: bool = typer.Option(False, help="Skip full LLM mode and run baseline only."),
) -> None:
    """Run lightweight evaluation and write result files."""

    if top_k <= 0 or candidate_k <= 0:
        raise typer.BadParameter("top-k and candidate-k must be positive")
    if mode not in {"baseline", "full", "both"}:
        raise typer.BadParameter("mode must be baseline, full, or both")
    if fallback_policy not in {"impute", "legacy_semantic"}:
        raise typer.BadParameter("fallback-policy must be impute or legacy_semantic")
    if skip_llm:
        mode = "baseline"

    started = time.perf_counter()
    queries = load_eval_queries(eval_file)
    expansion_cache = load_expansion_cache()
    resolved_device = resolve_device(device)
    embedder = load_embedding_model(embedding_model, device=resolved_device)
    faiss_index = load_faiss_index(index)
    row_map = load_id_map(id_map)

    matcher = None
    profile_lookup: dict[str, str] = {}
    if mode in {"full", "both"}:
        matcher = create_matcher(backend=backend, model_name=llm_model, device=resolved_device, max_new_tokens=256, base_url=llm_base_url, max_workers=llm_max_workers)
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
                fallback_policy=fallback_policy,
                expansion_cache=expansion_cache,
            )
            rows.extend(result_row(query, "full_llm_rerank", rank, item) for rank, item in enumerate(full[:top_k], start=1))

    # The configuration goes next to the results. Reading llm_candidate_k back out
    # of a cache line count once cost a full re-run and a wrong conclusion about
    # nondeterminism; a results file that cannot state how it was produced cannot be
    # compared to another one.
    run_config = {
        "top_k": top_k,
        "candidate_k": candidate_k,
        "llm_candidate_k": llm_candidate_k,
        "top_k_per_query": top_k_per_query,
        "fallback_policy": fallback_policy,
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "backend": backend,
        "mode": mode,
        "queries": len(queries),
        "expansion_prompt_version": EXPANSION_PROMPT_VERSION,
        "llm_prompt_version": PROMPT_VERSION,
    }
    config_path = out_dir / "eval_run_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path, jsonl_path = write_eval_outputs(rows, out_dir)
    console.print(f"Wrote config: {config_path}")
    console.print(f"Wrote CSV: {csv_path}")
    console.print(f"Wrote JSONL: {jsonl_path}")
    anchor_folds = load_anchor_folds(queries, DEFAULT_OUTPUT_PATH, splits)
    print_anchor_summary(compute_anchor_metrics(rows, queries, ks=(1, 5, 10), anchor_folds=anchor_folds))
    console.print(f"Runtime: {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    app()
