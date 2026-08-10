"""Convert the GRPO episode pool into verl's dataset layout.

verl expects, per row: ``prompt`` as a chat message list, ``data_source`` for
reward-function dispatch, ``reward_model.ground_truth``, and a free-form
``extra_info``. The pool is a flat table, so this is a reshape plus a train/validation
split.

Two details that would silently corrupt training if got wrong:

* **The rule verdict travels as a string.** ``"true"`` / ``"false"`` / ``"none"``,
  never a numpy bool. Parquet round-trips numpy bools inconsistently through verl's
  non-tensor batch, and ``None`` must stay distinguishable from ``False`` — one means
  "no reward signal", the other means "the rule read the whole novel and found it
  clean". Collapsing them teaches the model that ambiguity is punished.
* **The split is by query, not by row.** Episodes from one query share the query
  string and overlap in candidates, so a row-level split leaks near-duplicates into
  validation, exactly as it would have in SFT.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console(width=140)

DEFAULT_POOL = Path("data/processed/grpo_pool.parquet")
DEFAULT_OUT_DIR = Path("data/processed/verl")
DATA_SOURCE = "inovelrec_constraint"


def to_verl_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for index, row in enumerate(frame.itertuples()):
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": str(row.prompt)}],
                "ability": "constraint_rerank",
                "reward_model": {"style": "rule", "ground_truth": "true" if bool(row.rule_verdict) else "false"},
                "extra_info": {
                    "index": index,
                    "query": str(row.query),
                    "novel_id": str(row.novel_id),
                    "constraint_terms": list(row.constraint_terms),
                    "rule_verdict": "true" if bool(row.rule_verdict) else "false",
                    "pool_rank": int(row.pool_rank),
                },
            }
        )
    return rows


@app.command()
def main(
    pool: Path = typer.Option(DEFAULT_POOL, help="Episode pool from 20_build_grpo_data.py."),
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, help="Directory for train.parquet / val.parquet."),
    val_queries: int = typer.Option(200, help="Queries held out for validation."),
    seed: int = typer.Option(20260811, help="Split seed."),
    overwrite: bool = typer.Option(False, help="Overwrite existing outputs."),
) -> None:
    """Write verl-format train/validation parquet files."""

    train_path, val_path = out_dir / "train.parquet", out_dir / "val.parquet"
    if (train_path.exists() or val_path.exists()) and not overwrite:
        raise typer.BadParameter(f"Outputs exist in {out_dir}. Use --overwrite.")

    frame = pd.read_parquet(pool)
    queries = sorted(frame["query"].unique())
    rng = random.Random(seed)
    rng.shuffle(queries)
    held_out = set(queries[: min(val_queries, max(len(queries) // 10, 1))])

    validation = frame[frame["query"].isin(held_out)]
    training = frame[~frame["query"].isin(held_out)]

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(to_verl_rows(training)).to_parquet(train_path, index=False)
    pd.DataFrame(to_verl_rows(validation)).to_parquet(val_path, index=False)

    summary = Table(title="verl Dataset")
    summary.add_column("Split")
    summary.add_column("Episodes", justify="right")
    summary.add_column("Queries", justify="right")
    summary.add_column("Violating", justify="right")
    for name, part in (("train", training), ("validation", validation)):
        violating = int(part["rule_verdict"].sum())
        summary.add_row(name, str(len(part)), str(part["query"].nunique()), f"{violating} ({violating / max(len(part), 1):.0%})")
    console.print(summary)
    console.print(f"Wrote {train_path} and {val_path}")


if __name__ == "__main__":
    app()
