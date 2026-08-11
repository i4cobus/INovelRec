"""Build the GRPO rollout pool: one row per (query, candidate) episode.

Three filters, all decided from the precomputed density table so the whole pass is
a dictionary lookup rather than 36 GB of re-reads:

1. **fold=train.** Same discipline as SFT. Rollouts see profiles, so an eval-fold
   book here would destroy the split that separates "learned to retrieve" from
   "memorised the book". Never applied at inference.
2. **Rule-decidable pairs only.** 15.1% of pairs land in the abstention band and
   carry no verifiable signal; sampling them spends a rollout to learn nothing.
3. **Balanced 50/50 violating / clean.** Among decidable pairs only 19.3% violate,
   and an unbalanced batch degrades the reward toward a constant — the same failure
   the original presence-based rule had, where 87% of the corpus "violated" 系统.

Filter 3 is also the anti-collapse mechanism. The predicted reward hack is a policy
that claims every candidate violates; on a 50/50 pool that scores 1 on half the batch
and 0 on the other, averaging exactly to the always-silent policy, so the only way up
is to actually discriminate.

Queries are the ones SFT did **not** see. Reusing SFT's queries for rollouts would
have the policy explore prompts it was already fit on.
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
from src.llm_matcher import build_match_prompt
from src.preferences import constraint_violation_from_densities, is_rule_checkable, parse_preference_query
from src.query_expansion import (
    EXPANSION_CACHE_PATH,
    append_expansion_cache,
    build_expanded_queries,
    expansion_cache_key,
    load_expansion_cache,
)
from src.rank import load_profile_text_lookup, truncate_profile
from src.search import multi_query_semantic_search
from src.splits import filter_candidates_to_fold, load_fold_lookup
from src.vector_index import DEFAULT_ID_MAP_PATH, DEFAULT_INDEX_PATH

app = typer.Typer(add_completion=False)
console = Console(width=150)

DEFAULT_QUERIES = Path("data/processed/train_queries.jsonl")
DEFAULT_DENSITY = Path("data/processed/term_density.parquet")
DEFAULT_SPLITS = Path("data/processed/book_splits.parquet")
DEFAULT_OUT = Path("data/processed/grpo_pool.parquet")


def load_density_table(path: Path) -> dict[str, dict[str, float]]:
    frame = pd.read_parquet(path)
    terms = [column for column in frame.columns if column not in {"novel_id", "char_count"}]
    return {
        str(row["novel_id"]): {term: float(row[term]) for term in terms}
        for row in frame.to_dict(orient="records")
    }


def sft_query_texts(path: Path, count: int, seed: int) -> set[str]:
    """Reproduce exactly which queries ``17_build_sft_data.py`` sampled.

    Same file, same seed, same call — ``random.Random(seed).sample`` is deterministic,
    so the SFT split is recoverable rather than something that had to be recorded.
    Excluding by query *text* rather than index survives any later reordering.
    """

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen = rows if count >= len(rows) else random.Random(seed).sample(rows, count)
    return {str(row["query"]) for row in chosen}


def balance(pairs: list[dict], rng: random.Random, head_depth: int = 20) -> list[dict]:
    """Equalise violating/clean, preferring candidates from the head of the pool.

    The first run balanced by rule verdict alone and drew 90% of its episodes from
    below rank 10, median rank 51. Those are candidates the policy already scores
    near zero, so "answer 0" was a globally safe action there — and it transferred
    straight to the top ten, where it is wrong and where every metric is measured.
    A training distribution that does not cover the region under evaluation cannot
    be expected to improve it.

    Within each class, head candidates are taken first and the rest fill in, so the
    pool keeps the hard deep-pool cases without being dominated by them.
    """

    def ordered(items: list[dict]) -> list[dict]:
        head = [item for item in items if item["pool_rank"] < head_depth]
        tail = [item for item in items if item["pool_rank"] >= head_depth]
        rng.shuffle(head)
        rng.shuffle(tail)
        return head + tail

    violating = ordered([item for item in pairs if item["rule_verdict"]])
    clean = ordered([item for item in pairs if not item["rule_verdict"]])
    keep = min(len(violating), len(clean))
    if keep == 0:
        return []
    return violating[:keep] + clean[:keep]


@app.command()
def main(
    queries_path: Path = typer.Option(DEFAULT_QUERIES, "--queries", help="Synthesized in_text training queries."),
    density: Path = typer.Option(DEFAULT_DENSITY, help="Precomputed term densities (script 16)."),
    splits: Path = typer.Option(DEFAULT_SPLITS, help="Fold assignment."),
    profiles: Path = typer.Option(Path("data/processed/novel_profiles.parquet"), help="Novel profiles."),
    index_path: Path = typer.Option(DEFAULT_INDEX_PATH, help="FAISS index."),
    id_map_path: Path = typer.Option(DEFAULT_ID_MAP_PATH, help="FAISS row -> novel_id map."),
    out: Path = typer.Option(DEFAULT_OUT, help="GRPO rollout pool."),
    sft_queries_used: int = typer.Option(3000, help="How many queries SFT consumed; those are excluded."),
    sft_seed: int = typer.Option(20260810, help="Seed 17_build_sft_data.py used, to reproduce its sample."),
    n_queries: int = typer.Option(4000, help="Queries to draw rollout episodes from."),
    candidate_k: int = typer.Option(100, help="Retrieval pool per query, after fold filtering."),
    top_k_per_query: int = typer.Option(100, help="FAISS depth per expanded query."),
    per_query_cap: int = typer.Option(4, help="Episodes kept per query, after balancing."),
    head_depth: int = typer.Option(
        20,
        help="Ranks counted as the head of the retrieval pool. At least half of each "
        "query's episodes are drawn from it, because that is the region the "
        "evaluation metric is computed over.",
    ),
    model: str = typer.Option("Qwen/Qwen3-32B", help="Model name the expansion cache is keyed on."),
    base_url: str = typer.Option(DEFAULT_BASE_URL, help="Endpoint, used only for uncached expansions."),
    embedding_model: str = typer.Option("Qwen/Qwen3-Embedding-8B", help="Encoder for retrieval."),
    device: str = typer.Option("cuda:2", help="Encoder device."),
    max_workers: int = typer.Option(64, help="Concurrency for the expansion pre-warm."),
    profile_max_chars: int = typer.Option(1200, help="Profile budget in the prompt."),
    max_expanded: int = typer.Option(5, help="Retrieval query variants per preference."),
    seed: int = typer.Option(20260811, help="Sampling seed."),
    limit: int | None = typer.Option(None, help="Stop after N queries (smoke run)."),
    overwrite: bool = typer.Option(False, help="Overwrite existing output."),
) -> None:
    """Retrieve, rule-label, balance, and write the GRPO episode pool."""

    if out.exists() and not overwrite:
        raise typer.BadParameter(f"Output already exists: {out}. Use --overwrite to replace it.")

    rows = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    used = sft_query_texts(queries_path, sft_queries_used, sft_seed)
    available = [row for row in rows if str(row["query"]) not in used]
    console.print(f"Queries: {len(rows)} total, {len(used)} consumed by SFT, {len(available)} available")

    rng = random.Random(seed)
    if n_queries < len(available):
        available = rng.sample(available, n_queries)
    if limit is not None:
        available = available[:limit]

    densities = load_density_table(density)
    folds = load_fold_lookup(splits)
    profile_lookup = load_profile_text_lookup(profiles)
    expansion_cache = load_expansion_cache()

    from src.app_pipeline import resolve_device
    from src.embed import load_embedding_model
    from src.vector_index import load_faiss_index, load_id_map

    resolved_device = resolve_device(device)
    encoder = load_embedding_model(embedding_model, device=resolved_device)
    index = load_faiss_index(index_path)
    id_map = load_id_map(id_map_path)
    matcher = create_matcher(backend="http", model_name=model, base_url=base_url, max_workers=max_workers)

    # Expansion is one LLM call per query and, left inside the retrieval loop, runs at
    # a concurrency of one. Pinned to the same model name the evaluation uses so the
    # candidate pools stay comparable across everything in this project.
    from concurrent.futures import ThreadPoolExecutor

    pending = {}
    for row in available:
        key = expansion_cache_key(str(row["query"]), model, max_expanded - 1)
        if key not in expansion_cache:
            pending[key] = str(row["query"])
    if pending:
        console.print(f"Pre-warming {len(pending)} expansions ...")

        def fetch(item: tuple[str, str]) -> tuple[str, str]:
            key, text = item
            try:
                return key, matcher.expand_queries(raw_query=text, max_queries=max_expanded - 1)
            except Exception:  # noqa: BLE001 - a failed expansion falls back to the raw query
                return key, ""

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with typer.progressbar(length=len(pending), label="Expanding") as bar:
                for key, text in executor.map(fetch, pending.items()):
                    if text:
                        expansion_cache[key] = text
                        append_expansion_cache(EXPANSION_CACHE_PATH, key, text)
                    bar.update(1)

    episodes: list[dict] = []
    stats: Counter[str] = Counter()
    clock = time.perf_counter()
    with typer.progressbar(available, label="Building") as bar:
        for row in bar:
            query = str(row["query"])
            unwanted = [str(term) for term in row.get("unwanted") or []]
            terms = [term for term in unwanted if is_rule_checkable(term)]
            if not terms:
                stats["no_rule_checkable_term"] += 1
                continue

            expanded = build_expanded_queries(
                raw_query=query,
                structured_preference=parse_preference_query(query),
                llm_provider=matcher,
                use_llm_expansion=True,
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

            decidable = []
            for rank_in_pool, candidate in enumerate(pool):
                novel_id = str(candidate.get("novel_id", ""))
                verdict = constraint_violation_from_densities(densities.get(novel_id, {}), terms)
                if verdict is None:
                    stats["abstained"] += 1
                    continue
                profile_text = truncate_profile(profile_lookup.get(novel_id, ""), profile_max_chars)
                if not profile_text:
                    continue
                decidable.append({
                    "query": query,
                    "seed_novel_id": str(row.get("seed_novel_id", "")),
                    "novel_id": novel_id,
                    "title_guess": str(candidate.get("title_guess", "")),
                    "pool_rank": rank_in_pool,
                    "constraint_terms": terms,
                    "rule_verdict": bool(verdict),
                    "prompt": build_match_prompt(query, candidate, profile_text, max_profile_chars=profile_max_chars),
                })

            balanced = balance(decidable, rng, head_depth=head_depth)
            if not balanced:
                stats["unbalanceable_query"] += 1
                continue
            rng.shuffle(balanced)
            episodes.extend(balanced[:per_query_cap])
            stats["queries_used"] += 1

    frame = pd.DataFrame(episodes)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)

    summary = Table(title="GRPO Episode Pool")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Queries offered", str(len(available)))
    summary.add_row("Queries used", str(stats["queries_used"]))
    summary.add_row("  dropped: no rule-checkable term", str(stats["no_rule_checkable_term"]))
    summary.add_row("  dropped: cannot balance", str(stats["unbalanceable_query"]))
    summary.add_row("Candidates skipped (rule abstained)", str(stats["abstained"]))
    summary.add_row("Episodes", str(len(frame)))
    if len(frame):
        violating = int(frame["rule_verdict"].sum())
        summary.add_row("  violating / clean", f"{violating} / {len(frame) - violating}")
        summary.add_row("  distinct queries / novels", f"{frame['query'].nunique()} / {frame['novel_id'].nunique()}")
    summary.add_row("Elapsed", f"{time.perf_counter() - clock:.0f}s")
    summary.add_row("Output", str(out))
    console.print(summary)


if __name__ == "__main__":
    app()
