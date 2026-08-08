"""Lightweight evaluation helpers for recommendation results."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EvalQuery:
    """One manually designed evaluation query."""

    query_id: str
    query: str
    wanted: list[str] = field(default_factory=list)
    unwanted: list[str] = field(default_factory=list)
    anchor_titles: list[str] = field(default_factory=list)
    notes: str = ""


def load_eval_queries(path: Path) -> list[EvalQuery]:
    """Load JSONL evaluation queries."""

    queries: list[EvalQuery] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        if not data.get("query_id") or not data.get("query"):
            raise ValueError(f"Missing query_id/query at line {line_no}")
        queries.append(
            EvalQuery(
                query_id=str(data["query_id"]),
                query=str(data["query"]),
                wanted=[str(item) for item in data.get("wanted", [])],
                unwanted=[str(item) for item in data.get("unwanted", [])],
                anchor_titles=[str(item) for item in data.get("anchor_titles", [])],
                notes=str(data.get("notes", "")),
            )
        )
    return queries


def normalize_title(title: str) -> str:
    """Normalize titles for loose anchor matching."""

    title = re.sub(r"[《》〈〉「」『』\[\]【】\s_\-]+", "", title)
    return title.lower()


def title_matches_anchor(title: str, anchor: str) -> bool:
    """Return True when a corpus title names the anchored work.

    Matching is by *prefix*, not by substring anywhere in the title. Bidirectional
    substring matching inflates the only automatic metric this project has: in
    this corpus, anchor 凡人修仙传 also matched 《小小凡人修仙传》 and anchor 遮天
    matched 《医手遮天》 — different books entirely. A system could then score an
    anchor hit without retrieving the anchored work at all.

    A prefix still tolerates the edition and volume suffixes the corpus is full
    of (《盗墓笔记》（实体封面》, 《坏蛋是怎样炼成的Ⅰ》).
    """

    normalized_title = normalize_title(title)
    normalized_anchor = normalize_title(anchor)
    if not normalized_title or not normalized_anchor:
        return False
    return normalized_title.startswith(normalized_anchor)


def first_anchor_rank(results: list[dict[str, Any]], anchors: list[str], k: int) -> int | None:
    """Return the first top-k rank containing any anchor title."""

    if not anchors:
        return None
    for item in results[:k]:
        title = str(item.get("title_guess", ""))
        rank = int(item.get("rank", item.get("final_rank", 0)) or 0)
        if any(title_matches_anchor(title, anchor) for anchor in anchors):
            return rank or results.index(item) + 1
    return None


def compute_anchor_metrics(rows: list[dict[str, Any]], queries: list[EvalQuery], ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, Any]:
    """Compute anchor Hit@K and average first-anchor rank by system variant."""

    by_query_variant: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("query_id", "")), str(row.get("system_variant", "")))
        by_query_variant.setdefault(key, []).append(row)
    for items in by_query_variant.values():
        items.sort(key=lambda row: int(row.get("rank", 0) or 0))

    query_map = {query.query_id: query for query in queries}
    variants = sorted({str(row.get("system_variant", "")) for row in rows if row.get("system_variant")})
    summary: dict[str, Any] = {
        "num_queries": len(queries),
        "num_queries_with_anchors": sum(1 for query in queries if query.anchor_titles),
        "variants": {},
    }

    for variant in variants:
        anchor_queries = [query for query in queries if query.anchor_titles]
        variant_summary: dict[str, Any] = {"queries_with_anchors": len(anchor_queries)}
        ranks: list[int] = []
        total_anchors = sum(len(query.anchor_titles) for query in anchor_queries)
        for k in ks:
            hits = 0
            anchors_found = 0
            for query in anchor_queries:
                results = by_query_variant.get((query.query_id, variant), [])
                rank = first_anchor_rank(results, query.anchor_titles, k)
                if rank is not None:
                    hits += 1
                    if k == max(ks):
                        ranks.append(rank)
                anchors_found += sum(
                    1
                    for anchor in query.anchor_titles
                    if any(title_matches_anchor(str(item.get("title_guess", "")), anchor) for item in results[:k])
                )
            # Hit@K: share of queries with at least one anchor in the top K.
            # Recall@K: share of all anchors found. These coincide only when every
            # query carries exactly one anchor, which is why they were previously
            # reporting the same number twice.
            variant_summary[f"Anchor Hit@{k}"] = hits / len(anchor_queries) if anchor_queries else 0.0
            variant_summary[f"Anchor Recall@{k}"] = (
                anchors_found / total_anchors if total_anchors else 0.0
            )
        variant_summary["average_first_anchor_rank"] = sum(ranks) / len(ranks) if ranks else None
        summary["variants"][variant] = variant_summary
    return summary


def write_eval_outputs(rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    """Write evaluation results as CSV and JSONL."""

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "eval_results.csv"
    jsonl_path = out_dir / "eval_results.jsonl"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return csv_path, jsonl_path


def load_manual_judgements(path: Path) -> pd.DataFrame:
    """Load manual judgement CSV with expected columns."""

    df = pd.read_csv(path)
    required = {
        "query_id",
        "query",
        "system_variant",
        "rank",
        "title_guess",
        "novel_id",
        "relevance_label",
        "constraint_violation",
        "notes",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing judgement columns: {sorted(missing)}")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["relevance_label"] = pd.to_numeric(df["relevance_label"], errors="coerce").fillna(0)
    df["constraint_violation"] = df["constraint_violation"].astype(str).str.lower().isin({"true", "1", "yes", "y"})
    return df


def compute_manual_metrics(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Compute manual relevance and constraint metrics by system variant."""

    if k <= 0:
        raise ValueError("k must be positive")
    topk = df[df["rank"] <= k].copy()
    rows: list[dict[str, Any]] = []
    for variant, group in topk.groupby("system_variant"):
        rows.append(
            {
                "system_variant": variant,
                "evaluated_results": int(len(group)),
                f"Precision@{k}": float((group["relevance_label"] >= 1).mean()) if len(group) else 0.0,
                f"Strong Precision@{k}": float((group["relevance_label"] == 2).mean()) if len(group) else 0.0,
                "average_relevance": float(group["relevance_label"].mean()) if len(group) else 0.0,
                "constraint_violation_rate": float(group["constraint_violation"].mean()) if len(group) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_manual_judgement_template(path: Path) -> None:
    """Write an empty manual judgement CSV template."""

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "query_id",
        "query",
        "system_variant",
        "rank",
        "title_guess",
        "novel_id",
        "relevance_label",
        "constraint_violation",
        "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)


def confusion_counts(rater_a: list[int], rater_b: list[int], categories: list[int]) -> dict[tuple[int, int], int]:
    """Count co-occurrences of two raters' labels."""

    index = {category: position for position, category in enumerate(categories)}
    counts: dict[tuple[int, int], int] = {}
    for left, right in zip(rater_a, rater_b, strict=True):
        if left not in index or right not in index:
            raise ValueError(f"Label outside declared categories {categories}: {left}, {right}")
        key = (left, right)
        counts[key] = counts.get(key, 0) + 1
    return counts


def cohen_kappa(
    rater_a: list[int],
    rater_b: list[int],
    categories: list[int] | None = None,
    weighted: bool = False,
) -> float:
    """Cohen's kappa, optionally with linear weights for ordinal labels.

    Use ``weighted=True`` for the 0/1/2 relevance scale, where confusing 0 with 1
    is a smaller error than confusing 0 with 2. Use the unweighted form for the
    binary constraint-violation flag.
    """

    if len(rater_a) != len(rater_b):
        raise ValueError("Raters must label the same number of items")
    if not rater_a:
        raise ValueError("Cannot compute kappa over an empty sample")

    levels = sorted(set(categories if categories is not None else [*rater_a, *rater_b]))
    if len(levels) < 2:
        # Both raters used a single category: agreement is total but chance-corrected
        # kappa is undefined. Report perfect agreement rather than dividing by zero.
        return 1.0 if rater_a == rater_b else 0.0

    total = len(rater_a)
    observed = confusion_counts(rater_a, rater_b, levels)
    marginal_a = {level: rater_a.count(level) / total for level in levels}
    marginal_b = {level: rater_b.count(level) / total for level in levels}
    span = len(levels) - 1

    def weight(left: int, right: int) -> float:
        if not weighted:
            return 0.0 if left == right else 1.0
        return abs(levels.index(left) - levels.index(right)) / span

    numerator = sum(weight(left, right) * count / total for (left, right), count in observed.items())
    denominator = sum(weight(left, right) * marginal_a[left] * marginal_b[right] for left in levels for right in levels)
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return 1.0 - (numerator / denominator)


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch style bands, for reporting only."""

    if kappa < 0.0:
        return "worse than chance"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"


def judge_human_agreement(merged: pd.DataFrame) -> dict[str, Any]:
    """Compare judge and human columns on the rows both have labelled.

    Expects ``relevance_label`` / ``constraint_violation`` (human) alongside
    ``judge_relevance_label`` / ``judge_constraint_violation``.
    """

    required = {"relevance_label", "constraint_violation", "judge_relevance_label", "judge_constraint_violation"}
    missing = required.difference(merged.columns)
    if missing:
        raise ValueError(f"Missing agreement columns: {sorted(missing)}")

    paired = merged.dropna(subset=list(required))
    if paired.empty:
        raise ValueError("No rows carry both human and judge labels")

    human_relevance = [int(value) for value in paired["relevance_label"]]
    judge_relevance = [int(value) for value in paired["judge_relevance_label"]]
    human_violation = [int(bool(value)) for value in paired["constraint_violation"]]
    judge_violation = [int(bool(value)) for value in paired["judge_constraint_violation"]]

    relevance_kappa = cohen_kappa(human_relevance, judge_relevance, categories=[0, 1, 2], weighted=True)
    violation_kappa = cohen_kappa(human_violation, judge_violation, categories=[0, 1])
    exact = sum(1 for left, right in zip(human_relevance, judge_relevance, strict=True) if left == right)

    return {
        "paired_items": int(len(paired)),
        "relevance_weighted_kappa": round(relevance_kappa, 4),
        "relevance_kappa_band": interpret_kappa(relevance_kappa),
        "relevance_exact_agreement": round(exact / len(paired), 4),
        "constraint_violation_kappa": round(violation_kappa, 4),
        "constraint_violation_kappa_band": interpret_kappa(violation_kappa),
    }
