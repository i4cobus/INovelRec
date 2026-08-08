"""Regenerate eval/eval_queries.jsonl from the hand-edited draft table.

`eval/query_set_draft.md` is the source of truth for *new* queries: it is the file
a human reviews and edits. The JSONL is a build product, so editing it directly
loses the reasoning recorded alongside each row.

Queries already present in the JSONL are carried over untouched. That is
deliberate: their anchors were chosen before their baseline ranks were known, and
revising them afterwards would mean selecting for what the system already does
well. Frozen means frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.preferences import is_rule_checkable

app = typer.Typer(add_completion=False)
console = Console()

DRAFT_PATH = Path("eval/query_set_draft.md")
QUERIES_PATH = Path("eval/eval_queries.jsonl")
EXPECTED_COLUMNS = 8
LIST_SEPARATOR = "、"


def split_terms(cell: str) -> list[str]:
    """Split a table cell into terms, tolerating an empty or placeholder cell."""

    cleaned = cell.strip()
    if not cleaned or cleaned in {"—", "-"}:
        return []
    return [term.strip() for term in cleaned.split(LIST_SEPARATOR) if term.strip()]


def parse_draft(path: Path) -> list[dict]:
    """Read the candidate table, keeping only rows marked to retain."""

    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != EXPECTED_COLUMNS or cells[1] == "id" or set(cells[0]) <= set("-"):
            continue
        if cells[0] != "✓":
            continue
        unwanted = split_terms(cells[6])
        rows.append(
            {
                "query_id": cells[1],
                "query": cells[4],
                "wanted": split_terms(cells[5]),
                "unwanted": unwanted,
                "anchor_titles": [],
                "genre": cells[2],
                "shape": cells[3],
                # Recomputed here rather than trusted from the table: the reviewer
                # edits the constraint, not this flag.
                "constraint_checkable": bool(unwanted) and all(is_rule_checkable(term) for term in unwanted),
                "notes": f"{cells[2]} / {cells[3]}. Added from query_set_draft.md.",
            }
        )
    return rows


@app.command()
def main(
    draft: Path = typer.Option(DRAFT_PATH, help="Hand-edited candidate table."),
    out: Path = typer.Option(QUERIES_PATH, help="Evaluation query JSONL to regenerate."),
    dry_run: bool = typer.Option(False, help="Report what would change without writing."),
) -> None:
    """Merge frozen queries with the reviewed draft."""

    frozen: list[dict] = []
    if out.exists():
        frozen = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    frozen_ids = {row["query_id"] for row in frozen}

    candidates = parse_draft(draft)
    added = [row for row in candidates if row["query_id"] not in frozen_ids]
    skipped = [row for row in candidates if row["query_id"] in frozen_ids]

    merged = frozen + added
    merged.sort(key=lambda row: row["query_id"])

    checkable = [row for row in merged if row.get("constraint_checkable")]
    with_negative = [row for row in merged if row.get("unwanted")]
    anchored = [row for row in merged if row.get("anchor_titles")]

    summary = Table(title="Eval Query Set")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Frozen (carried over untouched)", str(len(frozen)))
    summary.add_row("Added from draft", str(len(added)))
    summary.add_row("Skipped (id already frozen)", str(len(skipped)))
    summary.add_row("Total queries", str(len(merged)))
    summary.add_row("With a negative constraint", str(len(with_negative)))
    summary.add_row("  of which rule-checkable", str(len(checkable)))
    summary.add_row("  of which judge-only", str(len(with_negative) - len(checkable)))
    summary.add_row("With anchors", str(len(anchored)))
    console.print(summary)

    if dry_run:
        console.print("[yellow]Dry run: nothing written.[/yellow]")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in merged) + "\n", encoding="utf-8")
    console.print(f"Wrote {out}")
    console.print(
        f"[dim]{len(merged) - len(anchored)} queries have no anchors yet. Anchors are optional — "
        "a query with no defensible archetype is better left unanchored than given a wrong one.[/dim]"
    )


if __name__ == "__main__":
    app()
