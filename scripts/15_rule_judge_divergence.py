"""Measure the gap between the verifiable reward rule and the evaluation source.

This is the baseline for the project's headline figure. GRPO scores negative
constraints with ``constraint_violation_by_rule``; evaluation scores them with the
calibrated judge. The two are deliberately different sources — if they were the
same, optimising the metric would prove nothing — so the gap between them is what
reward hacking would widen.

Run this before training to fix the baseline, and again after, on the same queries.
A model that has learned the preference moves both lines together; a model that has
learned the keyword rule moves only the rule line, and the two diverge. That
divergence is the result, whichever way it comes out.

The rule abstains on the band between its two thresholds. Abstentions are reported
separately and excluded from agreement, because "no signal" is not a prediction and
scoring it as one would flatter or punish the rule arbitrarily.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.config import DEFAULT_OUTPUT_PATH
from src.preferences import constraint_violation_by_rule

app = typer.Typer(add_completion=False)
console = Console(width=120)

TRUTHY = {"true", "1", "yes", "1.0"}


def load_rule_checkable_queries(path: Path) -> dict[str, list[str]]:
    """Query id to exclusion terms, for queries the rule can decide at all."""

    queries: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("constraint_checkable") and row.get("unwanted"):
            queries[row["query_id"]] = list(row["unwanted"])
    return queries


def judge_verdicts(results_path: Path, queries: dict[str, list[str]]) -> pd.DataFrame:
    """One row per (query, novel) the judge labelled, deduplicated across variants.

    A novel retrieved by both variants gets one row: the judge's verdict depends on
    the query and the evidence, not on which system surfaced it, so counting it
    twice would weight it by how many variants happened to retrieve it.
    """

    frame = pd.read_csv(results_path)
    frame = frame[frame["query_id"].isin(queries)]
    frame = frame[frame["judge_constraint_violation"].notna()]
    frame["judge"] = frame["judge_constraint_violation"].astype(str).str.lower().isin(TRUTHY)
    return frame.drop_duplicates(subset=["query_id", "novel_id"])


@app.command()
def main(
    results: Path = typer.Option(Path("eval/results/eval_results_judged.csv"), help="Judged evaluation results."),
    eval_file: Path = typer.Option(Path("eval/eval_queries.jsonl"), help="Evaluation queries."),
    inventory: Path = typer.Option(DEFAULT_OUTPUT_PATH, help="Stage 1 inventory, for full novel text."),
    out: Path = typer.Option(Path("eval/results/rule_judge_divergence.json"), help="Where to write the report."),
    label: str = typer.Option("baseline", help="Name for this measurement, e.g. baseline / after_grpo."),
) -> None:
    """Compare rule verdicts against judge verdicts on the rule-checkable arm."""

    from src.profile import read_text_with_encoding

    queries = load_rule_checkable_queries(eval_file)
    if not queries:
        raise typer.BadParameter("No rule-checkable queries found. Run 13_sync_eval_queries.py first.")
    labelled = judge_verdicts(results, queries)
    console.print(f"Rule-checkable queries: {len(queries)} | judged (query, novel) pairs: {len(labelled)}")

    stock = pd.read_parquet(
        inventory,
        columns=["novel_id", "absolute_path", "detected_encoding", "decode_replacement_chars"],
    ).set_index("novel_id")

    records = []
    for row in labelled.itertuples():
        novel_id = str(row.novel_id)
        if novel_id not in stock.index:
            continue
        book = stock.loc[novel_id]
        try:
            text = read_text_with_encoding(
                Path(str(book["absolute_path"])),
                book["detected_encoding"],
                allow_lossy=int(book.get("decode_replacement_chars", 0) or 0) > 0,
            )
        except (OSError, UnicodeError, LookupError, ValueError):
            continue
        terms = queries[row.query_id]
        records.append(
            {
                "query_id": row.query_id,
                "term": terms[0],
                "novel_id": novel_id,
                "rule": constraint_violation_by_rule(text, terms),
                "judge": bool(row.judge),
            }
        )

    frame = pd.DataFrame(records)
    decided = frame[frame["rule"].notna()].copy()
    decided["rule"] = decided["rule"].astype(bool)

    true_positive = int((decided["rule"] & decided["judge"]).sum())
    false_positive = int((decided["rule"] & ~decided["judge"]).sum())
    false_negative = int((~decided["rule"] & decided["judge"]).sum())
    true_negative = int((~decided["rule"] & ~decided["judge"]).sum())
    judged_violations = true_positive + false_negative

    table = Table(title=f"Rule vs judge on the rule-checkable arm ({label})")
    table.add_column("")
    table.add_column("judge: violates", justify="right")
    table.add_column("judge: clean", justify="right")
    table.add_row("rule: violates", str(true_positive), str(false_positive))
    table.add_row("rule: clean", str(false_negative), str(true_negative))
    console.print(table)

    summary = {
        "label": label,
        "pairs": len(frame),
        "decided": len(decided),
        "abstained": len(frame) - len(decided),
        "agreement": (decided["rule"] == decided["judge"]).mean() if len(decided) else 0.0,
        "rule_miss_rate": false_negative / judged_violations if judged_violations else 0.0,
        "rule_false_positive": false_positive,
        "confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
        "by_term": {},
    }
    console.print(
        f"Agreement {summary['agreement']:.0%} | rule misses {false_negative}/{judged_violations} "
        f"({summary['rule_miss_rate']:.0%}) of judge-confirmed violations | abstained {summary['abstained']}"
    )

    per_term = Table(title="By exclusion term — the two failure directions are opposite")
    for column in ("term", "n", "agreement", "rule misses", "rule over-fires"):
        per_term.add_column(column, justify="left" if column == "term" else "right")
    for term, group in decided.groupby("term"):
        misses = int((~group["rule"] & group["judge"]).sum())
        over = int((group["rule"] & ~group["judge"]).sum())
        per_term.add_row(str(term), str(len(group)), f"{(group['rule'] == group['judge']).mean():.0%}", str(misses), str(over))
        summary["by_term"][str(term)] = {
            "n": len(group),
            "agreement": (group["rule"] == group["judge"]).mean(),
            "misses": misses,
            "over_fires": over,
        }
    console.print(per_term)

    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    existing[label] = summary
    out.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"Wrote {out} (key: {label})")


if __name__ == "__main__":
    app()
