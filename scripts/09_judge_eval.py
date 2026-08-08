"""Judge evaluation results with an LLM, under a hard USD cap.

Reads `eval/results/eval_results.csv` (from 07_evaluate.py), samples judge
evidence from the RAW novel text at offsets the Stage 2 profile does not use,
and writes judged labels alongside each result row.

The judge deliberately never sees the profile the system ranked on: grading the
system's own summary would only measure profile-query agreement, not whether the
book actually matches.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.config import DEFAULT_OUTPUT_PATH
from src.evidence import DEFAULT_WINDOWS, DEFAULT_WINDOW_CHARS, load_raw_text_lookup, sample_judge_evidence
from src.http_matcher import DEFAULT_BASE_URL, HTTPChatTransport
from src.judge import (
    JUDGE_CACHE_PATH,
    BudgetExceeded,
    BudgetGuard,
    JudgeTask,
    PricePerMillion,
    judge_cache_key,
    run_judgements,
)

app = typer.Typer(add_completion=False)
console = Console(width=180)


def load_pinned_evidence(sheet_path: Path) -> dict[tuple[str, str], str]:
    """Read the evidence a human actually saw, keyed by (query_id, novel_id).

    Evidence is regenerated from the raw text on each run, and it depends on which
    chapters the *profile* sampled — so changing the profile silently changes it.
    That is what happened here: an annotation sheet was filled in, the profile was
    then revised, and the judge went on to read six different chapters than the
    annotator had. Agreement measured across two different sets of excerpts is not
    agreement at all, and three kappa runs had to be thrown away.

    Once a sheet exists it is the authority for those pairs. The judge reuses it
    verbatim rather than recomputing.
    """

    if not sheet_path.exists():
        return {}
    sheet = pd.read_csv(sheet_path)
    if not {"query_id", "novel_id", "evidence"}.issubset(sheet.columns):
        return {}
    return {
        (str(row.query_id), str(row.novel_id)): str(row.evidence)
        for row in sheet.itertuples()
        if isinstance(row.evidence, str) and row.evidence.strip()
    }


def build_tasks(
    results: pd.DataFrame,
    queries_by_id: dict[str, dict],
    texts: dict[str, str],
    windows: int,
    window_chars: int,
    pinned: dict[tuple[str, str], str] | None = None,
) -> tuple[list[JudgeTask], int, int]:
    """Build one task per unique (query_id, novel_id).

    Returns tasks, the number skipped, and how many reused pinned evidence.
    """

    pinned = pinned or {}
    tasks: list[JudgeTask] = []
    skipped = 0
    reused = 0
    for (query_id, novel_id), group in results.groupby(["query_id", "novel_id"], sort=False):
        evidence = pinned.get((str(query_id), str(novel_id)), "")
        if evidence:
            reused += 1
        else:
            text = texts.get(str(novel_id))
            if not text:
                skipped += 1
                continue
            evidence = sample_judge_evidence(text, str(novel_id), windows=windows, window_chars=window_chars)
        if not evidence:
            skipped += 1
            continue
        meta = queries_by_id.get(str(query_id), {})
        tasks.append(
            JudgeTask(
                query_id=str(query_id),
                query=str(group.iloc[0]["query"]),
                novel_id=str(novel_id),
                title=str(group.iloc[0].get("title_guess", "")),
                evidence=evidence,
                wanted=meta.get("wanted", []),
                unwanted=meta.get("unwanted", []),
            )
        )
    return tasks, skipped, reused


@app.command()
def main(
    results_csv: Path = typer.Option(Path("eval/results/eval_results.csv"), help="Output of 07_evaluate.py."),
    eval_file: Path = typer.Option(Path("eval/eval_queries.jsonl"), help="Evaluation query JSONL, for wanted/unwanted."),
    inventory: Path = typer.Option(DEFAULT_OUTPUT_PATH, help="Stage 1 inventory parquet, used to read raw text."),
    out: Path = typer.Option(Path("eval/results/eval_results_judged.csv"), help="Judged output CSV."),
    sheet: Path = typer.Option(Path("eval/manual_judgements_sheet.csv"), help="Annotation sheet; its evidence is reused verbatim so judge and human read the same text."),
    judge_model: str = typer.Option(..., help="Judge model name as the endpoint expects it."),
    base_url: str = typer.Option(DEFAULT_BASE_URL, help="OpenAI-compatible base URL. Env: INOVELREC_LLM_BASE_URL."),
    budget_usd: float = typer.Option(200.0, help="Hard spend cap. The run refuses to start if the estimate exceeds it."),
    price_input: float = typer.Option(..., help="USD per million input tokens."),
    price_output: float = typer.Option(..., help="USD per million output tokens."),
    top_k: int = typer.Option(10, help="Judge only results ranked <= top-k."),
    windows: int = typer.Option(DEFAULT_WINDOWS, help="Evidence windows sampled per novel."),
    window_chars: int = typer.Option(DEFAULT_WINDOW_CHARS, help="Characters per evidence window."),
    max_workers: int = typer.Option(8, help="Concurrent judge requests."),
    cache_path: Path = typer.Option(JUDGE_CACHE_PATH, help="Judge verdict cache."),
    dry_run: bool = typer.Option(False, help="Estimate cost and exit without calling the judge."),
) -> None:
    """Attach judge labels to evaluation results without exceeding the budget."""

    from src.evaluation import load_eval_queries

    results = pd.read_csv(results_csv)
    results = results[results["rank"] <= top_k].copy()
    if results.empty:
        raise typer.BadParameter(f"No rows with rank <= {top_k} in {results_csv}")

    queries_by_id = {
        query.query_id: {"wanted": query.wanted, "unwanted": query.unwanted}
        for query in load_eval_queries(eval_file)
    }
    pinned = load_pinned_evidence(sheet)
    novel_ids = {str(value) for value in results["novel_id"].unique()}
    texts = load_raw_text_lookup(inventory, novel_ids)
    tasks, skipped, reused = build_tasks(results, queries_by_id, texts, windows, window_chars, pinned=pinned)

    prices = PricePerMillion(input_usd=price_input, output_usd=price_output)
    budget = BudgetGuard(limit_usd=budget_usd, prices=prices)

    console.print(f"Result rows (rank <= {top_k}): {len(results)}")
    console.print(f"Unique (query, novel) pairs : {len(tasks)}  [deduplicated across variants]")
    console.print(f"Skipped (no readable text)  : {skipped}")
    console.print(f"Reusing annotator's evidence: {reused}  [判定与人工基于同一份材料]")
    console.print(f"Worst-case estimate         : ${budget.estimate_usd(len(tasks)):.2f} of ${budget_usd:.2f}")
    console.print("[dim]Cached pairs cost nothing; the real charge is usually well below this.[/dim]")

    if dry_run:
        console.print("[yellow]Dry run: no requests sent.[/yellow]")
        return

    transport = HTTPChatTransport(model=judge_model, base_url=base_url)
    try:
        verdicts, summary = run_judgements(
            tasks,
            transport,
            judge_model,
            budget,
            cache_path=cache_path,
            max_workers=max_workers,
        )
    except BudgetExceeded as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    finally:
        transport.close()

    lookup = {
        (tasks[index].query_id, tasks[index].novel_id): verdicts.get(judge_cache_key(tasks[index], judge_model))
        for index in range(len(tasks))
    }
    results["judge_relevance_label"] = [
        getattr(lookup.get((str(row.query_id), str(row.novel_id))), "relevance_label", None)
        for row in results.itertuples()
    ]
    results["judge_constraint_violation"] = [
        getattr(lookup.get((str(row.query_id), str(row.novel_id))), "constraint_violation", None)
        for row in results.itertuples()
    ]
    results["judge_confidence"] = [
        getattr(lookup.get((str(row.query_id), str(row.novel_id))), "judge_confidence", None)
        for row in results.itertuples()
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False, encoding="utf-8-sig")

    table = Table(title="Judge Run Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Requested pairs", str(summary.requested))
    table.add_row("Cache hits", str(summary.cache_hits))
    table.add_row("Newly judged", str(summary.judged))
    table.add_row("Failed requests", str(summary.failed))
    table.add_row("Skipped (over budget)", str(summary.skipped_over_budget))
    table.add_row("Input tokens", f"{summary.usage.prompt_tokens:,}")
    table.add_row("Output tokens", f"{summary.usage.completion_tokens:,}")
    table.add_row("Spent USD", f"${summary.spent_usd:.2f}")
    table.add_row("Remaining USD", f"${budget.remaining_usd():.2f}")
    console.print(table)
    console.print(f"Wrote {out}")

    if summary.stopped_early:
        console.print(
            f"[yellow]Budget cap reached: {summary.skipped_over_budget} pairs were not judged. "
            "Completed verdicts are cached, so re-running after raising --budget-usd resumes.[/yellow]"
        )


if __name__ == "__main__":
    app()
