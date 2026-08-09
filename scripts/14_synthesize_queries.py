"""Generate training queries from train-fold novels with the teacher.

These are the queries SFT and GRPO train on. The evaluation set is hand written
and frozen; the two must never mix, so seeds are drawn only from ``fold=train``
and every generated query is checked against the evaluation set before it is kept.

Where a constraint is rule-checkable the teacher's satisfies/violates claim is
verified against the novel's own text — the text is ground truth, the teacher's
opinion is not.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.backends import create_matcher
from src.config import DEFAULT_OUTPUT_PATH
from src.evaluation import load_eval_queries
from src.http_matcher import DEFAULT_BASE_URL
from src.query_synthesis import (
    DEFAULT_QUERIES_PER_BOOK,
    SynthesisTask,
    build_synthesis_prompt,
    SynthesizedQuery,
    deduplicate,
    label_constraint_by_rule,
    looks_truncated,
    parse_synthesis_response,
    verify_constraint_claim,
)
from src.splits import load_fold_lookup

app = typer.Typer(add_completion=False)
console = Console(width=140)

DEFAULT_SPLITS_PATH = Path("data/processed/book_splits.parquet")
DEFAULT_PROFILES_PATH = Path("data/processed/novel_profiles.parquet")
DEFAULT_OUT_PATH = Path("data/processed/train_queries.jsonl")


def load_tasks(profiles_path: Path, splits_path: Path, limit: int | None, seed: int) -> list[SynthesisTask]:
    """Sample train-fold novels to generate queries from."""

    profiles = pd.read_parquet(profiles_path, columns=["novel_id", "title_guess", "profile_text"])
    folds = load_fold_lookup(splits_path)
    profiles = profiles[profiles["novel_id"].astype(str).map(lambda value: folds.get(value) == "train")]
    if limit is not None and limit < len(profiles):
        profiles = profiles.sample(n=limit, random_state=seed)
    return [
        SynthesisTask(novel_id=str(row.novel_id), title=str(row.title_guess), profile=str(row.profile_text))
        for row in profiles.itertuples()
    ]


@app.command()
def main(
    profiles: Path = typer.Option(DEFAULT_PROFILES_PATH, help="Novel profiles parquet."),
    splits: Path = typer.Option(DEFAULT_SPLITS_PATH, help="Book fold assignment; only fold=train is used."),
    inventory: Path = typer.Option(DEFAULT_OUTPUT_PATH, help="Stage 1 inventory, for verifying constraint claims."),
    eval_file: Path = typer.Option(Path("eval/eval_queries.jsonl"), help="Evaluation queries to deduplicate against."),
    out: Path = typer.Option(DEFAULT_OUT_PATH, help="Generated training queries."),
    model: str = typer.Option("Qwen/Qwen3-32B", help="Teacher model name."),
    base_url: str = typer.Option(DEFAULT_BASE_URL, help="OpenAI-compatible endpoint."),
    limit: int | None = typer.Option(None, help="Number of seed novels. Omit for the whole train fold."),
    per_book: int = typer.Option(DEFAULT_QUERIES_PER_BOOK, help="Queries requested per novel."),
    max_workers: int = typer.Option(16, help="Concurrent teacher requests."),
    max_new_tokens: int = typer.Option(
        1200,
        help="Token budget per response. At 600 a third of responses were cut off mid-JSON and "
        "silently yielded nothing; 1200 drops that to 8% and roughly doubles the query count.",
    ),
    seed: int = typer.Option(20260809, help="Sampling seed."),
    exclusion_kind: str = typer.Option(
        "in_text",
        help="Which exclusion vocabulary the teacher draws from: in_text (rule-checkable, "
        "carries a verifiable reward) or meta (爽文/圣母/小白 …, SFT material only but 42 of "
        "the 59 evaluation queries).",
    ),
    verify_constraints: bool = typer.Option(True, help="Check rule-checkable claims against the novel text."),
    overwrite: bool = typer.Option(False, help="Overwrite an existing output file."),
) -> None:
    """Synthesise training queries and write them as JSONL."""

    if exclusion_kind not in {"in_text", "meta"}:
        raise typer.BadParameter("exclusion-kind must be in_text or meta")
    if out.exists() and not overwrite:
        raise typer.BadParameter(f"Output already exists: {out}. Use --overwrite to replace it.")

    tasks = load_tasks(profiles, splits, limit, seed)
    if not tasks:
        raise typer.BadParameter("No train-fold novels found. Run 12_build_splits.py first.")
    eval_queries = load_eval_queries(eval_file)
    reserved = [query.query for query in eval_queries]
    reserved_queries = [
        SynthesizedQuery(query=query.query, shape="kw", unwanted=list(query.unwanted or []))
        for query in eval_queries
    ]
    console.print(f"Seed novels (fold=train): {len(tasks)}")
    console.print(f"Reserved evaluation queries: {len(reserved)}")

    matcher = create_matcher(backend="http", model_name=model, base_url=base_url, max_workers=max_workers, max_new_tokens=max_new_tokens)
    prompts = [(SynthesisTask(**{**task.__dict__}), build_synthesis_prompt(task, count=per_book, exclusion_kind=exclusion_kind)) for task in tasks]

    generated = []
    failed = truncated = barren = 0
    with typer.progressbar(length=len(prompts), label="Synthesising") as bar:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(matcher.generate_response, prompt): task for task, prompt in prompts}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    response = future.result()
                except Exception:  # noqa: BLE001 - one dead request must not kill the run
                    failed += 1
                    bar.update(1)
                    continue
                parsed = parse_synthesis_response(response, task)
                # A response that parses to nothing is counted, not shrugged off:
                # truncation and a genuinely unusable answer look identical in the
                # output list, and only one of them is fixed by a bigger budget.
                if not parsed:
                    barren += 1
                    if looks_truncated(response):
                        truncated += 1
                generated.extend(parsed)
                bar.update(1)

    kept, duplicates, leaked = deduplicate(generated, reserved, reserved_queries=reserved_queries)

    agreed = disagreed = abstained = semantic = 0
    if verify_constraints:
        from src.evidence import load_raw_text_lookup

        needed = {item.seed_novel_id for item in kept if item.constraint_checkable}
        texts = load_raw_text_lookup(inventory, needed)
        labelled = []
        for item in kept:
            text = texts.get(item.seed_novel_id, "")
            # Measure the teacher first, then overwrite it. The agreement rate is a
            # result worth reporting; the teacher's claim is not supervision.
            outcome = verify_constraint_claim(item, text)
            if outcome is True:
                agreed += 1
            elif outcome is False:
                disagreed += 1
            relabelled = label_constraint_by_rule(item, text)
            if relabelled is None:
                # The rule looked and landed in the undecidable band. No trustworthy
                # label, so this is not training data — distinct from a constraint
                # the rule cannot see at all, which stays for the judge.
                abstained += 1
                continue
            if not relabelled.constraint_checkable:
                semantic += 1
            labelled.append(relabelled)
        kept = labelled

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(item.to_dict(), ensure_ascii=False) for item in kept) + "\n", encoding="utf-8")

    shapes = Counter(item.shape for item in kept)
    summary = Table(title="Training Query Synthesis")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Seed novels", str(len(tasks)))
    summary.add_row("Failed requests", str(failed))
    summary.add_row("Parsed to nothing", f"{barren} (of which truncated: {truncated})")
    summary.add_row("Raw queries", str(len(generated)))
    summary.add_row("Dropped as duplicates", str(duplicates))
    summary.add_row("Dropped as eval leakage", str(leaked))
    total_claims = agreed + disagreed
    rate = f"{agreed / total_claims:.0%}" if total_claims else "n/a"
    summary.add_row("Teacher claim agreed with rule", f"{agreed} / {total_claims} ({rate})")
    summary.add_row("Relabelled by rule (text wins)", str(disagreed))
    summary.add_row("Rule abstained (dropped)", str(abstained))
    summary.add_row("Semantic exclusion (judge only)", str(semantic))
    summary.add_row("Kept", str(len(kept)))
    summary.add_row("  satisfies / violates", f"{sum(1 for i in kept if i.seed_satisfies_constraint)} / {sum(1 for i in kept if not i.seed_satisfies_constraint)}")
    summary.add_row("  rule-checkable", str(sum(1 for item in kept if item.constraint_checkable)))
    summary.add_row("  shapes", ", ".join(f"{k}:{v}" for k, v in shapes.most_common()))
    summary.add_row("Output", str(out))
    console.print(summary)


if __name__ == "__main__":
    app()
