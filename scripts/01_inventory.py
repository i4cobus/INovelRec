"""CLI entry point for Stage 1 dataset inventory."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.config import DEFAULT_OUTPUT_PATH, RAW_DATA_DIR
from src.ingest import inventory_novels, write_inventory

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    raw_dir: Path = typer.Option(RAW_DATA_DIR, help="Directory containing raw .txt files."),
    out: Path = typer.Option(DEFAULT_OUTPUT_PATH, help="Parquet output path."),
    limit: int | None = typer.Option(None, help="Maximum number of files to process."),
    overwrite: bool = typer.Option(False, help="Overwrite existing output parquet."),
    max_workers: int | None = typer.Option(None, help="Process pool size. Defaults to a capped share of available cores."),
    store_sample_text: bool = typer.Option(False, "--store-sample-text/--no-store-sample-text", help="Persist a 4.5k-char sample per novel. Nothing downstream reads it; it costs an extra full pass."),
) -> None:
    """Run the Stage 1 novel inventory pipeline."""

    if out.exists() and not overwrite:
        raise typer.BadParameter(f"Output already exists: {out}. Use --overwrite to replace it.")

    records, duplicates = inventory_novels(
        raw_dir=raw_dir,
        limit=limit,
        max_workers=max_workers,
        store_sample_text=store_sample_text,
    )
    df = write_inventory(records, out)

    ok_count = int((df["read_status"] == "ok").sum()) if not df.empty else 0
    failed_count = int((df["read_status"] == "failed").sum()) if not df.empty else 0

    summary = Table(title="Inventory Summary")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Raw directory", str(raw_dir))
    summary.add_row("Output", str(out))
    summary.add_row("Files processed", str(len(df)))
    summary.add_row("Read OK", str(ok_count))
    summary.add_row("Read failed", str(failed_count))
    summary.add_row("Exact content duplicates", str(duplicates.exact_duplicates))
    summary.add_row("Duplicate groups", str(duplicates.duplicate_groups))
    summary.add_row("Title collisions (not dropped)", str(duplicates.title_collisions))
    summary.add_row("Unique novels for Stage 2", str(ok_count - duplicates.exact_duplicates))
    summary.add_row("Total size MB", f"{df['file_size_mb'].sum():.2f}" if not df.empty else "0.00")
    summary.add_row("Avg chapter estimate", f"{df['estimated_chapter_count'].mean():.2f}" if not df.empty else "0.00")
    console.print(summary)


if __name__ == "__main__":
    app()

