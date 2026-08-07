"""Build a single-annotator relevance sheet with evidence embedded.

A 200-row task is only feasible if the annotator never has to open a novel. Each
row therefore carries the same independently sampled evidence the LLM judge saw,
so human and judge labels are directly comparable — that comparison is what the
Cohen's kappa in 11_agreement.py measures.

Sampling is stratified across queries and rank bands so the sheet is not
dominated by a handful of queries or by top-1 results alone.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.config import DEFAULT_OUTPUT_PATH
from src.evidence import DEFAULT_WINDOWS, DEFAULT_WINDOW_CHARS, load_raw_text_lookup, sample_judge_evidence

app = typer.Typer(add_completion=False)
console = Console(width=180)

SHEET_COLUMNS = [
    "query_id",
    "query",
    "wanted",
    "unwanted",
    "novel_id",
    "title_guess",
    "rank",
    "system_variant",
    "evidence",
    "relevance_label",
    "constraint_violation",
    "notes",
]

RANK_BANDS = ((1, 3), (4, 10))
MINUTES_PER_ROW = 2.0


def rank_band(rank: int) -> str:
    for low, high in RANK_BANDS:
        if low <= rank <= high:
            return f"{low}-{high}"
    return "other"


def stratified_sample(results: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Sample evenly across (query_id, rank band), deduplicated by (query, novel)."""

    deduped = results.drop_duplicates(subset=["query_id", "novel_id"]).copy()
    deduped["rank_band"] = deduped["rank"].astype(int).map(rank_band)
    groups = list(deduped.groupby(["query_id", "rank_band"], sort=True))
    if not groups:
        return deduped.head(0)

    per_group = max(1, sample_size // len(groups))
    picked = [group.sample(n=min(per_group, len(group)), random_state=seed) for _, group in groups]
    sampled = pd.concat(picked)

    if len(sampled) < sample_size:
        remainder = deduped.drop(sampled.index)
        if not remainder.empty:
            extra = remainder.sample(n=min(sample_size - len(sampled), len(remainder)), random_state=seed)
            sampled = pd.concat([sampled, extra])
    return sampled.head(sample_size).sort_values(["query_id", "rank"])


@app.command()
def main(
    results_csv: Path = typer.Option(Path("eval/results/eval_results.csv"), help="Output of 07_evaluate.py."),
    eval_file: Path = typer.Option(Path("eval/eval_queries.jsonl"), help="Evaluation query JSONL."),
    inventory: Path = typer.Option(DEFAULT_OUTPUT_PATH, help="Stage 1 inventory parquet, used to read raw text."),
    out: Path = typer.Option(Path("eval/manual_judgements_sheet.csv"), help="Sheet to hand to the annotator."),
    sample_size: int = typer.Option(200, help="Rows to annotate."),
    top_k: int = typer.Option(10, help="Only sample results ranked <= top-k."),
    windows: int = typer.Option(DEFAULT_WINDOWS, help="Evidence windows per novel."),
    window_chars: int = typer.Option(DEFAULT_WINDOW_CHARS, help="Characters per evidence window."),
    seed: int = typer.Option(20260807, help="Sampling seed, for a reproducible sheet."),
    overwrite: bool = typer.Option(False, help="Overwrite an existing sheet."),
) -> None:
    """Write a stratified annotation sheet with evidence inlined."""

    if out.exists() and not overwrite:
        raise typer.BadParameter(f"Output already exists: {out}. Use --overwrite to replace it.")

    from src.evaluation import load_eval_queries

    results = pd.read_csv(results_csv)
    results = results[results["rank"] <= top_k]
    if results.empty:
        raise typer.BadParameter(f"No rows with rank <= {top_k} in {results_csv}")

    sampled = stratified_sample(results, sample_size, seed)
    queries_by_id = {
        query.query_id: {"wanted": query.wanted, "unwanted": query.unwanted}
        for query in load_eval_queries(eval_file)
    }
    texts = load_raw_text_lookup(inventory, {str(value) for value in sampled["novel_id"].unique()})

    rows = []
    skipped = 0
    for row in sampled.to_dict(orient="records"):
        novel_id = str(row["novel_id"])
        evidence = sample_judge_evidence(texts.get(novel_id, ""), novel_id, windows=windows, window_chars=window_chars)
        if not evidence:
            skipped += 1
            continue
        meta = queries_by_id.get(str(row["query_id"]), {})
        rows.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "wanted": "|".join(meta.get("wanted", [])),
                "unwanted": "|".join(meta.get("unwanted", [])),
                "novel_id": novel_id,
                "title_guess": row.get("title_guess", ""),
                "rank": row["rank"],
                "system_variant": row.get("system_variant", ""),
                "evidence": evidence,
                "relevance_label": "",
                "constraint_violation": "",
                "notes": "",
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    table = Table(title="Annotation Sheet")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Rows to annotate", str(len(rows)))
    table.add_row("Skipped (no readable text)", str(skipped))
    table.add_row("Distinct queries", str(sampled["query_id"].nunique()))
    table.add_row("Estimated effort", f"~{len(rows) * MINUTES_PER_ROW / 60:.1f} h at {MINUTES_PER_ROW:.0f} min/row")
    console.print(table)
    console.print(f"Wrote {out}")
    console.print(
        "Fill [bold]relevance_label[/bold] (0 not relevant / 1 partly / 2 highly) and "
        "[bold]constraint_violation[/bold] (true/false), judging ONLY from the evidence column."
    )


if __name__ == "__main__":
    app()
