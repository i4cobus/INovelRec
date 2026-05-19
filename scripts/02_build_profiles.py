"""CLI entry point for Stage 2 profile generation."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.config import DEFAULT_OUTPUT_PATH
from src.profile import DEFAULT_PROFILE_OUTPUT_PATH, build_profiles, write_profiles

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    inventory: Path = typer.Option(DEFAULT_OUTPUT_PATH, help="Stage 1 inventory parquet path."),
    out: Path = typer.Option(DEFAULT_PROFILE_OUTPUT_PATH, help="Profile parquet output path."),
    limit: int | None = typer.Option(None, help="Maximum number of inventory rows to process."),
    overwrite: bool = typer.Option(False, help="Overwrite existing output parquet."),
) -> None:
    """Build compact cleaned novel profiles for later embeddings."""

    if out.exists() and not overwrite:
        raise typer.BadParameter(f"Output already exists: {out}. Use --overwrite to replace it.")

    result = build_profiles(inventory_path=inventory, limit=limit)
    df = write_profiles(result, out)

    summary = Table(title="Profile Build Summary")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Inventory", str(inventory))
    summary.add_row("Output", str(out))
    summary.add_row("Profiles written", str(len(df)))
    summary.add_row("Skipped failed inventory rows", str(result.skipped_failed))
    summary.add_row("Skipped missing files", str(result.skipped_missing))
    summary.add_row("Skipped read errors", str(result.skipped_read_error))
    if not df.empty:
        summary.add_row("Avg profile chars", f"{df['profile_text'].str.len().mean():.2f}")
        summary.add_row("Avg chapter count", f"{df['estimated_chapter_count'].mean():.2f}")
    console.print(summary)


if __name__ == "__main__":
    app()

