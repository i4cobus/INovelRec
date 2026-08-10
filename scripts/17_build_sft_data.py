"""Assemble SFT samples: one (query, candidate) pair per line.

The input side is exactly ``llm_matcher.build_match_prompt`` — the prompt the
reranker already sees at inference — so SFT teaches the deployed interface rather
than a training-only one. The target is the multi-field JSON verdict.

Three decisions are baked in here; each was measured, and each is reversible only
by regenerating (roughly 6 GPU-hours of teacher time).

**Candidates are sampled across the retrieval pool, not taken from the top.**
``confidence`` has a real distribution over the full pool (medium 850 / low 471 /
high 12) but collapses to a single value inside the top-10 for 50 of 59 evaluation
queries. Training only on the top would teach the field as a constant and leave a
Brier-style calibration reward with no gradient. Low-scoring candidates are also
the ones a reranker most needs to learn to reject; showing it only survivors is
survivorship bias.

**The candidate pool is restricted to fold=train.** Seeds are already train-fold,
but FAISS retrieves over the whole corpus and would otherwise put eval-fold
profiles into training — destroying the split that separates "learned to retrieve"
from "memorised the book". This is a training-data discipline, never applied at
inference.

**``violated_preferences`` comes from the rule, not the teacher.** On 233 judged
rule-arm candidates the density rule agrees with the judge 71.2% against the
teacher's 54.5%, and where they disagree the judge sides with the rule 68 to 29.
The v4 teacher catches 4 of 91 violations. When the rule decides,
``align_fields_with_rule`` rebuilds ``reason`` and ``risk_flags`` too, so the three
constraint fields cannot contradict each other.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.backends import create_matcher
from src.http_matcher import DEFAULT_BASE_URL
from src.llm_matcher import PROMPT_VERSION, build_match_prompt
from src.preferences import constraint_violation_from_densities, is_rule_checkable, parse_preference_query
from src.query_expansion import (
    EXPANSION_CACHE_PATH,
    append_expansion_cache,
    build_expanded_queries,
    expansion_cache_key,
    load_expansion_cache,
)
from src.query_synthesis import align_fields_with_rule, normalize_violated_terms
from src.rank import load_profile_text_lookup, sha256_text, truncate_profile
from src.search import multi_query_semantic_search
from src.splits import filter_candidates_to_fold, load_fold_lookup
from src.vector_index import DEFAULT_ID_MAP_PATH, DEFAULT_INDEX_PATH

app = typer.Typer(add_completion=False)
console = Console(width=150)

DEFAULT_IN_TEXT_PATH = Path("data/processed/train_queries.jsonl")
DEFAULT_META_PATH = Path("data/processed/train_queries_meta_strat.jsonl")
DEFAULT_DENSITY_PATH = Path("data/processed/term_density.parquet")
DEFAULT_SPLITS_PATH = Path("data/processed/book_splits.parquet")
DEFAULT_OUT_PATH = Path("data/processed/sft_samples.jsonl")


def load_queries(path: Path, count: int, seed: int) -> list[dict]:
    """Sample ``count`` synthesized queries without replacement."""

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if count >= len(rows):
        return rows
    return random.Random(seed).sample(rows, count)


def load_density_table(path: Path) -> dict[str, dict[str, float]]:
    """Load the precomputed (novel, term) densities as a nested lookup."""

    frame = pd.read_parquet(path)
    terms = [column for column in frame.columns if column not in {"novel_id", "char_count"}]
    return {
        str(row["novel_id"]): {term: float(row[term]) for term in terms}
        for row in frame.to_dict(orient="records")
    }


def sample_candidates(candidates: list[dict], *, head: int, tail: int, key: str) -> list[tuple[int, dict]]:
    """Take every one of the top ``head``, then ``tail`` sampled from the rest.

    Returns ``(pool_rank, candidate)`` pairs: the rank is recorded per sample and
    recovering it later by searching the pool is both quadratic and wrong the moment
    two candidates compare equal.

    Deterministic in ``key`` so a rerun reproduces the same pool; the alternative,
    a global RNG, makes the dataset a function of iteration order.
    """

    chosen = list(enumerate(candidates[:head]))
    remainder = list(enumerate(candidates[head:], start=head))
    if remainder and tail > 0:
        rng = random.Random(sha256_text(key))
        chosen.extend(rng.sample(remainder, min(tail, len(remainder))))
    return chosen


def rule_terms(unwanted: list[str]) -> list[str]:
    """The subset of a query's exclusions the density rule can decide."""

    return [term for term in unwanted if is_rule_checkable(term)]


@app.command()
def main(
    in_text: Path = typer.Option(DEFAULT_IN_TEXT_PATH, help="Rule-checkable training queries."),
    meta: Path = typer.Option(DEFAULT_META_PATH, help="Semantic-exclusion training queries."),
    density: Path = typer.Option(DEFAULT_DENSITY_PATH, help="Precomputed term densities (script 16)."),
    splits: Path = typer.Option(DEFAULT_SPLITS_PATH, help="Fold assignment."),
    profiles: Path = typer.Option(Path("data/processed/novel_profiles.parquet"), help="Novel profiles."),
    index_path: Path = typer.Option(DEFAULT_INDEX_PATH, help="FAISS index."),
    id_map_path: Path = typer.Option(DEFAULT_ID_MAP_PATH, help="FAISS row -> novel_id map."),
    out: Path = typer.Option(DEFAULT_OUT_PATH, help="SFT samples."),
    n_in_text: int = typer.Option(3000, help="Rule-checkable queries to use."),
    n_meta: int = typer.Option(2000, help="Semantic-exclusion queries to use."),
    candidate_k: int = typer.Option(100, help="Retrieval pool per query, after fold filtering."),
    top_k_per_query: int = typer.Option(100, help="FAISS depth per expanded query."),
    sample_head: int = typer.Option(10, help="Top candidates always included."),
    sample_tail: int = typer.Option(10, help="Candidates sampled from the rest of the pool."),
    model: str = typer.Option("Qwen/Qwen3-32B", help="Teacher model."),
    base_url: str = typer.Option(DEFAULT_BASE_URL, help="OpenAI-compatible endpoint."),
    embedding_model: str = typer.Option("Qwen/Qwen3-Embedding-8B", help="Encoder for retrieval."),
    device: str = typer.Option("cuda:2", help="Encoder device. Keep off the GPUs serving the teacher."),
    max_workers: int = typer.Option(64, help="Concurrent teacher requests per query."),
    query_chunk: int = typer.Option(
        4,
        help="Queries scored concurrently. One query alone caps the server at 20 "
        "in-flight requests (the candidate count); 4 keeps it near the 64-way "
        "concurrency the throughput estimate assumes.",
    ),
    profile_max_chars: int = typer.Option(1200, help="Profile budget in the prompt."),
    use_expansion: bool = typer.Option(True, help="Expand queries before retrieval, as inference does."),
    max_expanded: int = typer.Option(5, help="Retrieval query variants per preference."),
    seed: int = typer.Option(20260810, help="Query sampling seed."),
    limit: int | None = typer.Option(None, help="Stop after N queries (smoke run)."),
    overwrite: bool = typer.Option(False, help="Overwrite existing output."),
) -> None:
    """Retrieve, score, rule-label, and write SFT samples."""

    if out.exists() and not overwrite:
        raise typer.BadParameter(f"Output already exists: {out}. Use --overwrite to replace it.")

    queries = [
        {**row, "arm": "in_text"} for row in load_queries(in_text, n_in_text, seed)
    ] + [
        {**row, "arm": "meta"} for row in load_queries(meta, n_meta, seed + 1)
    ]
    random.Random(seed).shuffle(queries)
    if limit is not None:
        queries = queries[:limit]

    console.print(f"Queries: {len(queries)}  (in_text {sum(q['arm'] == 'in_text' for q in queries)} / meta {sum(q['arm'] == 'meta' for q in queries)})")

    densities = load_density_table(density)
    folds = load_fold_lookup(splits)
    profile_lookup = load_profile_text_lookup(profiles)
    expansion_cache = load_expansion_cache()

    from src.app_pipeline import resolve_device
    from src.embed import load_embedding_model
    from src.vector_index import load_faiss_index, load_id_map

    resolved_device = resolve_device(device)
    console.print(f"Encoder on {resolved_device}; teacher at {base_url}")
    encoder = load_embedding_model(embedding_model, device=resolved_device)
    index = load_faiss_index(index_path)
    id_map = load_id_map(id_map_path)
    matcher = create_matcher(backend="http", model_name=model, base_url=base_url, max_workers=max_workers)

    def prewarm_expansions(rows: list[dict]) -> int:
        """Fetch every missing query expansion up front, concurrently.

        Expansion is one LLM call per query. Left inside the retrieval loop it runs
        one at a time between scoring batches and costs 3.5s of the 7.5s per query —
        more than half the run, spent at a concurrency of one. Fetched as its own
        pass it is a few minutes total, and the cache also makes the resulting
        candidate pools reproducible across runs.
        """

        from concurrent.futures import ThreadPoolExecutor

        pending = {}
        for row in rows:
            query = str(row["query"])
            key = expansion_cache_key(query, model, max_expanded - 1)
            if key not in expansion_cache:
                pending[key] = query
        if not pending:
            return 0

        def fetch(item: tuple[str, str]) -> tuple[str, str]:
            key, query = item
            try:
                return key, matcher.expand_queries(raw_query=query, max_queries=max_expanded - 1)
            except Exception:  # noqa: BLE001 - a failed expansion falls back to the raw query
                return key, ""

        console.print(f"Pre-warming {len(pending)} query expansions ...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with typer.progressbar(length=len(pending), label="Expanding") as expand_bar:
                for key, text in executor.map(fetch, pending.items()):
                    if text:
                        expansion_cache[key] = text
                        append_expansion_cache(EXPANSION_CACHE_PATH, key, text)
                    expand_bar.update(1)
        return len(pending)

    def retrieve(position: int, row: dict) -> dict | None:
        """Build one query's scoring batch. Runs on the main thread: GPU, not HTTP."""

        query = str(row["query"])
        expanded = build_expanded_queries(
            raw_query=query,
            structured_preference=parse_preference_query(query),
            llm_provider=matcher if use_expansion else None,
            use_llm_expansion=use_expansion,
            expansion_cache=expansion_cache,
            llm_model=model,
            max_expanded_queries=max_expanded,
        )
        pool = multi_query_semantic_search(
            expanded_queries=expanded,
            model=encoder,
            index=index,
            id_map=id_map,
            top_k_per_query=top_k_per_query,
            final_candidate_k=candidate_k * 2,
        )
        pool = filter_candidates_to_fold(pool, folds, "train")[:candidate_k]
        if not pool:
            return None
        picked = sample_candidates(pool, head=sample_head, tail=sample_tail, key=f"{seed}:{position}:{query}")
        return {
            "position": position,
            "row": row,
            "query": query,
            "unwanted": [str(term) for term in row.get("unwanted") or []],
            "picked": picked,
            "items": [
                (candidate, truncate_profile(profile_lookup.get(str(candidate.get("novel_id", "")), ""), profile_max_chars))
                for _, candidate in picked
            ],
        }

    written = 0
    stats: Counter[str] = Counter()
    timing: Counter[str] = Counter()
    if use_expansion:
        clock = time.perf_counter()
        fetched = prewarm_expansions(queries)
        timing['expand'] = time.perf_counter() - clock
        console.print(f"Expansions fetched: {fetched} in {timing['expand']:.0f}s")
    out.parent.mkdir(parents=True, exist_ok=True)
    handle = out.open("w", encoding="utf-8")
    try:
        # Queries are scored a chunk at a time rather than one by one. `score_many`
        # opens min(max_workers, len(items)) connections, so a single query caps the
        # server at 20 concurrent requests and then blocks on the slowest of them;
        # measured that way the run sat at 1.3 calls/s against 4.9 at 64 concurrency,
        # turning 6 hours into 21. Retrieval stays on the main thread (it is GPU work
        # and the encoder is not shared safely), scoring fans out across the chunk.
        from concurrent.futures import ThreadPoolExecutor

        with typer.progressbar(length=len(queries), label="Building") as bar:
            for start in range(0, len(queries), query_chunk):
                chunk = queries[start : start + query_chunk]
                prepared = []
                clock = time.perf_counter()
                for offset, row in enumerate(chunk):
                    batch = retrieve(start + offset, row)
                    if batch is None:
                        stats["empty_pool"] += 1
                        bar.update(1)
                        continue
                    prepared.append(batch)
                timing["retrieve"] += time.perf_counter() - clock
                if not prepared:
                    continue

                clock = time.perf_counter()
                with ThreadPoolExecutor(max_workers=len(prepared)) as executor:
                    scored = list(
                        executor.map(
                            lambda batch: matcher.score_many(batch["query"], batch["items"], max_profile_chars=profile_max_chars),
                            prepared,
                        )
                    )
                timing["score"] += time.perf_counter() - clock

                for batch, results in zip(prepared, scored, strict=True):
                    position, row, query = batch["position"], batch["row"], batch["query"]
                    unwanted = batch["unwanted"]
                    terms = rule_terms(unwanted)
                    for (rank_in_pool, candidate), (_, profile_text), result in zip(
                        batch["picked"], batch["items"], results, strict=True
                    ):
                        if result is None:
                            stats["teacher_failed"] += 1
                            continue
                        novel_id = str(candidate.get("novel_id", ""))
                        verdict = constraint_violation_from_densities(densities.get(novel_id, {}), terms) if terms else None
                        if verdict is None:
                            # No rule signal: either a semantic exclusion the rule
                            # cannot see, or a density inside the abstention band.
                            # Either way the teacher's fields stand — flagged, so
                            # training can weight or drop them without re-deriving
                            # why. Its wording is still normalized onto the query's
                            # exclusions, so the field means the same thing on both
                            # arms.
                            payload = result.to_dict()
                            payload["violated_preferences"] = normalize_violated_terms(
                                payload.get("violated_preferences", []), unwanted
                            )
                            label_source = "teacher" if not terms else "rule_abstained"
                        else:
                            payload = align_fields_with_rule(result, terms, verdict)
                            label_source = "rule"
                        stats[label_source] += 1
                        stats["violates" if verdict else "clean" if verdict is False else "no_verdict"] += 1

                        handle.write(json.dumps({
                            "query_id": f"t{position:06d}",
                            "query": query,
                            "arm": row["arm"],
                            "seed_novel_id": row.get("seed_novel_id", ""),
                            "novel_id": novel_id,
                            "title_guess": str(candidate.get("title_guess", "")),
                            "pool_rank": rank_in_pool,
                            "selection": "head" if rank_in_pool < sample_head else "tail",
                            "constraint_terms": terms,
                            "rule_verdict": verdict,
                            "label_source": label_source,
                            "confidence": payload.get("confidence", ""),
                            "prompt": build_match_prompt(query, candidate, profile_text, max_profile_chars=profile_max_chars),
                            "target": json.dumps(payload, ensure_ascii=False),
                            "prompt_version": PROMPT_VERSION,
                            "teacher_model": model,
                        }, ensure_ascii=False) + "\n")
                        written += 1
                    bar.update(1)
    finally:
        handle.close()

    summary = Table(title="SFT Sample Assembly")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Queries", str(len(queries)))
    summary.add_row("Samples written", str(written))
    summary.add_row("Label from rule", f"{stats['rule']} (violates {stats['violates']} / clean {stats['clean']})")
    summary.add_row("Rule abstained (in_text)", str(stats["rule_abstained"]))
    summary.add_row("Teacher only (meta)", str(stats["teacher"]))
    summary.add_row("Teacher request failed", str(stats["teacher_failed"]))
    summary.add_row("Empty pool after fold filter", str(stats["empty_pool"]))
    summary.add_row("Time in expansion pre-warm", f"{timing['expand']:.0f}s")
    summary.add_row("Time in retrieval", f"{timing['retrieve']:.0f}s ({timing['retrieve']/max(len(queries),1):.1f}s/query)")
    summary.add_row("Time in teacher scoring", f"{timing['score']:.0f}s ({timing['score']/max(len(queries),1):.1f}s/query)")
    summary.add_row("Output", str(out))
    console.print(summary)


if __name__ == "__main__":
    app()
