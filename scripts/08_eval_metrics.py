"""Compute metrics from completed manual recommendation judgements."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.evaluation import compute_manual_metrics, load_manual_judgements

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    judgements: Path = typer.Option(Path("eval/manual_judgements.csv"), help="Completed manual judgement CSV."),
    k: int = typer.Option(10, help="Compute metrics over ranks <= k."),
) -> None:
    """Read manual judgements and print aggregate metrics by system variant."""

    df = load_manual_judgements(judgements)
    metrics = compute_manual_metrics(df, k=k)
    table = Table(title=f"Manual Evaluation Metrics @ {k}")
    for column in metrics.columns:
        table.add_column(str(column), justify="right" if column != "system_variant" else "left")
    for row in metrics.to_dict(orient="records"):
        table.add_row(
            *[
                f"{value:.3f}" if isinstance(value, float) else str(value)
                for value in row.values()
            ]
        )
    console.print(table)


if __name__ == "__main__":
    app()
