"""Calibrate the LLM judge against the human annotation sheet.

Without this step the judge is an unvalidated oracle and every downstream number
rests on it. With it, the claim becomes "judged by <model>, calibrated at
kappa=X against N human labels" — and, more importantly, the human column becomes
an independent yardstick for the constraint metric that GRPO optimises against a
keyword rule. Those two must never be the same source, or the reward-hacking
divergence plot cannot exist.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.evaluation import compute_manual_metrics, judge_human_agreement

app = typer.Typer(add_completion=False)
console = Console(width=180)

JOIN_KEYS = ["query_id", "novel_id"]


def normalize_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce annotation columns filled in by hand into usable dtypes."""

    frame = frame.copy()
    frame["relevance_label"] = pd.to_numeric(frame["relevance_label"], errors="coerce")
    frame["constraint_violation"] = (
        frame["constraint_violation"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    )
    for key in JOIN_KEYS:
        frame[key] = frame[key].astype(str)
    return frame.dropna(subset=["relevance_label"])


@app.command()
def main(
    human: Path = typer.Option(Path("eval/manual_judgements_sheet.csv"), help="Annotation sheet, filled in."),
    judged: Path = typer.Option(Path("eval/results/eval_results_judged.csv"), help="Output of 09_judge_eval.py."),
    out: Path | None = typer.Option(None, help="Optional CSV of the paired rows, for error review."),
    k: int = typer.Option(10, help="Compute manual metrics over ranks <= k."),
) -> None:
    """Report judge-vs-human agreement and human-labelled system metrics."""

    human_frame = normalize_labels(pd.read_csv(human))
    if human_frame.empty:
        raise typer.BadParameter(f"No filled-in rows in {human}. Fill relevance_label before running this.")

    judged_frame = pd.read_csv(judged)
    for key in JOIN_KEYS:
        judged_frame[key] = judged_frame[key].astype(str)
    judge_columns = [*JOIN_KEYS, "judge_relevance_label", "judge_constraint_violation"]
    missing = set(judge_columns).difference(judged_frame.columns)
    if missing:
        raise typer.BadParameter(f"{judged} is missing columns: {sorted(missing)}")

    judge_side = judged_frame[judge_columns].drop_duplicates(subset=JOIN_KEYS)
    merged = human_frame.merge(judge_side, on=JOIN_KEYS, how="inner")
    merged = merged.dropna(subset=["judge_relevance_label"])
    if merged.empty:
        raise typer.BadParameter("No rows carry both human and judge labels. Run 09_judge_eval.py over the same results.")

    merged["judge_constraint_violation"] = (
        merged["judge_constraint_violation"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    )
    agreement = judge_human_agreement(merged)

    table = Table(title="Judge vs Human Agreement")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Human rows annotated", str(len(human_frame)))
    table.add_row("Paired with judge", str(agreement["paired_items"]))
    table.add_row("Relevance weighted kappa", f"{agreement['relevance_weighted_kappa']:.3f} ({agreement['relevance_kappa_band']})")
    table.add_row("Relevance exact agreement", f"{agreement['relevance_exact_agreement']:.3f}")
    table.add_row(
        "Constraint violation kappa",
        f"{agreement['constraint_violation_kappa']:.3f} ({agreement['constraint_violation_kappa_band']})",
    )
    console.print(table)

    if agreement["relevance_weighted_kappa"] < 0.4:
        console.print(
            "[yellow]Kappa below 0.4: the judge is not yet a usable stand-in for human labels. "
            "Revise the judge prompt or widen the evidence before trusting judged sweeps.[/yellow]"
        )

    if "system_variant" in merged.columns and "rank" in merged.columns:
        console.print(compute_manual_metrics(merged, k=k).to_string(index=False))

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        disagreements = merged[merged["relevance_label"] != merged["judge_relevance_label"]]
        merged.to_csv(out, index=False, encoding="utf-8-sig")
        console.print(f"Wrote {out} ({len(disagreements)} disagreements worth reviewing)")


if __name__ == "__main__":
    app()
