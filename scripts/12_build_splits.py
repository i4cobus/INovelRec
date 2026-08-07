"""Assign every unique novel to a train or eval fold.

Run this once, right after Stage 1, and before any teacher labelling. Folds are
derived from ``content_sha256``, so duplicate copies of one novel always land
together and adding books later never reshuffles the existing assignment.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.config import DEFAULT_OUTPUT_PATH, PROCESSED_DATA_DIR
from src.splits import DEFAULT_EVAL_FRACTION, SPLIT_SALT, build_splits

app = typer.Typer(add_completion=False)
console = Console()

DEFAULT_SPLITS_PATH = PROCESSED_DATA_DIR / "book_splits.parquet"


@app.command()
def main(
    inventory: Path = typer.Option(DEFAULT_OUTPUT_PATH, help="Stage 1 inventory parquet."),
    out: Path = typer.Option(DEFAULT_SPLITS_PATH, help="Split assignment parquet."),
    eval_fraction: float = typer.Option(DEFAULT_EVAL_FRACTION, help="Share of unique novels held out."),
    salt: str = typer.Option(SPLIT_SALT, help="Changing this reshuffles everything. Only do so deliberately."),
    overwrite: bool = typer.Option(False, help="Overwrite an existing split file."),
) -> None:
    """Write a stable novel_id -> fold assignment."""

    if out.exists() and not overwrite:
        raise typer.BadParameter(
            f"Splits already exist: {out}. Overwriting reassigns folds and invalidates every "
            "result measured against the old split. Use --overwrite only if that is intended."
        )

    frame = pd.read_parquet(inventory)
    splits, report = build_splits(frame, eval_fraction=eval_fraction, salt=salt)
    out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(out, index=False)

    summary = Table(title="Book Split")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Inventory rows", str(len(frame)))
    summary.add_row("Skipped (unreadable)", str(report.skipped_unreadable))
    summary.add_row("Skipped (duplicates)", str(report.skipped_duplicates))
    summary.add_row("Unique novels split", str(report.total))
    summary.add_row("train", str(report.train_novels))
    summary.add_row("eval (held out)", str(report.eval_novels))
    summary.add_row(
        "Actual eval share",
        f"{report.eval_novels / report.total:.3f}" if report.total else "n/a",
    )
    summary.add_row("Output", str(out))
    console.print(summary)
    console.print(
        "[dim]Teacher labelling and SFT must draw only from fold=train. "
        "Anything measured on fold=eval stays attributable.[/dim]"
    )


if __name__ == "__main__":
    app()
