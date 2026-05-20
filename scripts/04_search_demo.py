"""CLI demo for semantic search over the Stage 3 FAISS index."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.embed import DEFAULT_EMBEDDING_MODEL, load_embedding_model
from src.search import load_faiss_index, load_id_map, semantic_search
from src.vector_index import DEFAULT_ID_MAP_PATH, DEFAULT_INDEX_PATH

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    query: str = typer.Argument(..., help="Search query."),
    index: Path = typer.Option(DEFAULT_INDEX_PATH, help="FAISS index path."),
    id_map: Path = typer.Option(DEFAULT_ID_MAP_PATH, help="Row metadata JSON path."),
    model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer model name."),
    device: str | None = typer.Option(None, help="Optional torch device, for example cuda or cpu."),
    top_k: int = typer.Option(10, help="Number of results to return."),
) -> None:
    """Run a semantic search query against the local FAISS index."""

    embedding_model = load_embedding_model(model, device=device)
    faiss_index = load_faiss_index(index)
    row_map = load_id_map(id_map)
    results = semantic_search(query, embedding_model, faiss_index, row_map, top_k=top_k)

    table = Table(title="Semantic Search Results")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Title")
    table.add_column("Novel ID")
    table.add_column("Preview")

    for result in results:
        table.add_row(
            str(result["rank"]),
            f"{result['score']:.4f}",
            str(result["title_guess"]),
            str(result["novel_id"]),
            str(result["profile_text_preview"]).replace("\n", " ")[:120],
        )
    console.print(table)


if __name__ == "__main__":
    app()
