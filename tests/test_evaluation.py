import json
from pathlib import Path

import pandas as pd

from src.evaluation import (
    compute_anchor_metrics,
    compute_manual_metrics,
    load_eval_queries,
    load_manual_judgements,
    title_matches_anchor,
    write_eval_outputs,
)


def test_eval_query_jsonl_loading(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q001",
                "query": "凡人流 仙侠",
                "wanted": ["凡人流"],
                "unwanted": ["系统"],
                "anchor_titles": ["凡人修仙传"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    queries = load_eval_queries(path)
    assert queries[0].query_id == "q001"
    assert queries[0].anchor_titles == ["凡人修仙传"]


def test_anchor_title_matching() -> None:
    assert title_matches_anchor("《凡人修仙传》", "凡人修仙传")
    assert title_matches_anchor("凡人修仙传 完本", "凡人修仙传")
    assert not title_matches_anchor("百炼成仙", "凡人修仙传")


def test_anchor_hit_calculation() -> None:
    queries = load_eval_queries_from_items(
        [
            {"query_id": "q001", "query": "x", "anchor_titles": ["凡人修仙传"]},
            {"query_id": "q002", "query": "y", "anchor_titles": []},
        ]
    )
    rows = [
        {"query_id": "q001", "system_variant": "baseline", "rank": 1, "title_guess": "百炼成仙"},
        {"query_id": "q001", "system_variant": "baseline", "rank": 3, "title_guess": "凡人修仙传"},
    ]
    metrics = compute_anchor_metrics(rows, queries, ks=(1, 5, 10))
    assert metrics["num_queries"] == 2
    assert metrics["num_queries_with_anchors"] == 1
    assert metrics["variants"]["baseline"]["Anchor Hit@1"] == 0.0
    assert metrics["variants"]["baseline"]["Anchor Hit@5"] == 1.0


def test_manual_metric_calculation_from_fake_csv(tmp_path: Path) -> None:
    path = tmp_path / "judgements.csv"
    pd.DataFrame(
        [
            {
                "query_id": "q001",
                "query": "q",
                "system_variant": "baseline",
                "rank": 1,
                "title_guess": "A",
                "novel_id": "a",
                "relevance_label": 2,
                "constraint_violation": False,
                "notes": "",
            },
            {
                "query_id": "q001",
                "query": "q",
                "system_variant": "baseline",
                "rank": 2,
                "title_guess": "B",
                "novel_id": "b",
                "relevance_label": 0,
                "constraint_violation": True,
                "notes": "",
            },
        ]
    ).to_csv(path, index=False)
    metrics = compute_manual_metrics(load_manual_judgements(path), k=2)
    row = metrics.iloc[0]
    assert row["Precision@2"] == 0.5
    assert row["Strong Precision@2"] == 0.5
    assert row["constraint_violation_rate"] == 0.5


def test_eval_result_output_schema(tmp_path: Path) -> None:
    rows = [{"query_id": "q001", "system_variant": "baseline", "rank": 1, "title_guess": "A"}]
    csv_path, jsonl_path = write_eval_outputs(rows, tmp_path)
    assert csv_path.exists()
    assert jsonl_path.exists()
    loaded = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    assert loaded["query_id"] == "q001"


def load_eval_queries_from_items(items: list[dict]) -> list:
    path = Path("temp_eval_queries_for_test.jsonl")
    try:
        path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in items), encoding="utf-8")
        return load_eval_queries(path)
    finally:
        if path.exists():
            path.unlink()
