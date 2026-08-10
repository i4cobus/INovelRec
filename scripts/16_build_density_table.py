"""Precompute term densities for every (novel, rule-checkable term) pair.

The constraint reward is defined on the *full* novel text, but GRPO evaluates it
inside the rollout loop, once per candidate per rollout. Re-reading a 3M-character
serial there is not affordable, and it is also pure waste: the density of 「系统」
in a given book never changes. One pass over the corpus turns the reward into a
dictionary lookup.

Sizing: 7,653 novels x 13 terms is ~100k floats — a table small enough to hold in
memory in every rollout worker. What it replaces is 36 GB of re-reads per epoch.

The thresholds are deliberately NOT applied here. This file stores measurements;
``preferences.violation_from_density`` stores the rule. Baking 3.0 into the table
would fork the rule into two artifacts that drift apart, and the divergence plot
compares against exactly one rule.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.config import DEFAULT_OUTPUT_PATH
from src.preferences import IN_TEXT_NEGATIVES, term_density

app = typer.Typer(add_completion=False)
console = Console(width=140)

DEFAULT_DENSITY_PATH = Path("data/processed/term_density.parquet")
TERMS = sorted(IN_TEXT_NEGATIVES)


def measure_one(row: dict[str, object]) -> dict[str, object] | None:
    """Read one novel and measure every term's density in it.

    Runs in a worker process, so it re-imports rather than closing over module
    state; the pool is spawned, not forked, to match Stage 1/2.
    """

    from src.profile import read_text_with_encoding

    if row.get("read_status") != "ok":
        return None
    try:
        text = read_text_with_encoding(
            Path(str(row["absolute_path"])),
            row.get("detected_encoding"),
            allow_lossy=int(row.get("decode_replacement_chars", 0) or 0) > 0,
        )
    except (OSError, UnicodeError, LookupError, ValueError):
        return None
    record: dict[str, object] = {"novel_id": str(row["novel_id"]), "char_count": len(text)}
    for term in TERMS:
        record[term] = term_density(text, term)
    return record


@app.command()
def main(
    inventory: Path = typer.Option(DEFAULT_OUTPUT_PATH, help="Stage 1 inventory."),
    out: Path = typer.Option(DEFAULT_DENSITY_PATH, help="Density table."),
    limit: int | None = typer.Option(None, help="Measure only the first N novels (smoke run)."),
    max_workers: int = typer.Option(32, help="Reader processes."),
    overwrite: bool = typer.Option(False, help="Overwrite an existing table."),
) -> None:
    """Measure every rule-checkable term's density in every novel."""

    if out.exists() and not overwrite:
        raise typer.BadParameter(f"Output already exists: {out}. Use --overwrite to replace it.")

    frame = pd.read_parquet(
        inventory,
        columns=["novel_id", "absolute_path", "detected_encoding", "read_status", "decode_replacement_chars"],
    )
    if limit is not None:
        frame = frame.head(limit)
    rows = frame.to_dict(orient="records")
    console.print(f"Novels: {len(rows)}  Terms: {len(TERMS)}")

    records = []
    unreadable = 0
    with typer.progressbar(length=len(rows), label="Measuring") as bar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(measure_one, rows, chunksize=8):
                if result is None:
                    unreadable += 1
                else:
                    records.append(result)
                bar.update(1)

    table_frame = pd.DataFrame.from_records(records)
    out.parent.mkdir(parents=True, exist_ok=True)
    table_frame.to_parquet(out, index=False)

    summary = Table(title="Term Density Table")
    summary.add_column("Term")
    summary.add_column("median", justify="right")
    summary.add_column("p90", justify="right")
    summary.add_column("max", justify="right")
    summary.add_column(">=3.0", justify="right")
    summary.add_column("<=1.0", justify="right")
    for term in TERMS:
        values = table_frame[term]
        summary.add_row(
            term,
            f"{values.median():.2f}",
            f"{values.quantile(0.9):.2f}",
            f"{values.max():.1f}",
            f"{(values >= 3.0).mean():.1%}",
            f"{(values <= 1.0).mean():.1%}",
        )
    console.print(summary)
    console.print(f"Rows: {len(table_frame)}  Unreadable: {unreadable}  ->  {out}")


if __name__ == "__main__":
    app()
