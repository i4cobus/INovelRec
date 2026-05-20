"""CLI entry point for Stage 3 embedding generation and FAISS indexing."""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.embed import DEFAULT_EMBEDDING_MODEL, encode_texts, load_embedding_model
from src.vector_index import (
    DEFAULT_ID_MAP_PATH,
    DEFAULT_INDEX_METADATA_PATH,
    DEFAULT_INDEX_PATH,
    DEFAULT_PROFILES_PATH,
    build_faiss_index,
    ensure_can_write,
    load_profiles_for_index,
    make_id_map,
    make_index_metadata,
    save_faiss_index,
    save_id_map,
    save_index_metadata,
)

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    profiles: Path = typer.Option(DEFAULT_PROFILES_PATH, help="Novel profiles parquet path."),
    index_out: Path = typer.Option(DEFAULT_INDEX_PATH, help="Output FAISS index path."),
    id_map_out: Path = typer.Option(DEFAULT_ID_MAP_PATH, help="Output row metadata JSON path."),
    metadata_out: Path = typer.Option(DEFAULT_INDEX_METADATA_PATH, help="Output index metadata JSON path."),
    model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer model name."),
    device: str | None = typer.Option(None, help="Optional torch device, for example cuda or cpu."),
    batch_size: int = typer.Option(32, help="Embedding batch size."),
    limit: int | None = typer.Option(None, help="Maximum number of profile rows to process."),
    overwrite: bool = typer.Option(False, help="Overwrite existing output files."),
) -> None:
    """Build embeddings and a FAISS IndexFlatIP vector index."""

    started_at = time.perf_counter()
    ensure_can_write([index_out, id_map_out, metadata_out], overwrite=overwrite)

    loaded = load_profiles_for_index(profiles_path=profiles, limit=limit)
    if loaded.dataframe.empty:
        raise typer.BadParameter("No valid profiles found for indexing.")

    texts = loaded.dataframe["profile_text"].tolist()
    console.print(f"Loaded profiles: {len(loaded.dataframe)}")
    console.print(f"Skipped profiles: {loaded.skipped_rows}")

    embedding_model = load_embedding_model(model, device=device)
    embeddings = encode_texts(embedding_model, texts, batch_size=batch_size, normalize_embeddings=True)
    index = build_faiss_index(embeddings)
    id_map = make_id_map(loaded.dataframe)
    metadata = make_index_metadata(
        model_name=model,
        embedding_dim=int(embeddings.shape[1]),
        num_vectors=int(embeddings.shape[0]),
        normalize_embeddings=True,
        source_profiles=profiles,
    )

    save_faiss_index(index, index_out)
    save_id_map(id_map, id_map_out)
    save_index_metadata(metadata, metadata_out)

    elapsed = time.perf_counter() - started_at
    summary = Table(title="Index Build Summary")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Profiles loaded", str(len(loaded.dataframe)))
    summary.add_row("Profiles skipped", str(loaded.skipped_rows))
    summary.add_row("Embedding shape", str(tuple(embeddings.shape)))
    summary.add_row("Device", device or "auto")
    summary.add_row("FAISS index size", str(index.ntotal))
    summary.add_row("Index output", str(index_out))
    summary.add_row("ID map output", str(id_map_out))
    summary.add_row("Metadata output", str(metadata_out))
    summary.add_row("Runtime seconds", f"{elapsed:.2f}")
    console.print(summary)


if __name__ == "__main__":
    app()
