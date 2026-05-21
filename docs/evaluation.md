# Evaluation

This project does not have official relevance labels, genre labels, or user behavior logs. Evaluation is therefore designed as a lightweight framework that can compare retrieval and reranking variants without overclaiming objective quality.

## Why Evaluation Is Needed

The recommender combines semantic retrieval, query expansion, local LLM reranking, and grounded explanation. Each layer can help or hurt result quality. Evaluation makes those tradeoffs visible:

- FAISS-only retrieval shows the raw semantic baseline.
- Full LLM reranking shows whether local Qwen3 scoring improves preference fit.
- Manual judgement captures relevance and constraint violations.

## Evaluation Dataset Format

Evaluation queries live in `eval/eval_queries.jsonl`. Each line is a JSON object:

```json
{
  "query_id": "q001",
  "query": "凡人流 仙侠 慢热 理性主角 不系统",
  "wanted": ["凡人流", "仙侠", "慢热", "理性主角"],
  "unwanted": ["系统"],
  "anchor_titles": ["凡人修仙传"],
  "notes": "Tests whether the system can retrieve classic slow-burn xianxia novels."
}
```

`anchor_titles` are optional. They are useful for recall checks when a known title may exist in the corpus, but the framework does not assume every anchor is present.

## Baseline vs Full System

| Variant | Meaning |
|---|---|
| `baseline_faiss` | Raw query -> Qwen3 embedding -> FAISS top-k retrieval |
| `full_llm_rerank` | Query expansion -> multi-query FAISS retrieval -> local Qwen3 LLM reranking |

Run baseline-only evaluation:

```bash
uv run python scripts/07_evaluate.py --mode baseline --top-k 10 --candidate-k 100
```

Run both variants:

```bash
uv run python scripts/07_evaluate.py --mode both --top-k 10 --candidate-k 100 --llm-candidate-k 10
```

Outputs:

- `eval/results/eval_results.csv`
- `eval/results/eval_results.jsonl`

## Automatic Anchor Metrics

Anchor metrics check whether optional known titles appear in top-k results.

| Metric | Meaning |
|---|---|
| Anchor Hit@K | Whether at least one anchor title appears in the top K |
| Anchor Recall@K | Fraction of anchored queries with an anchor in the top K |
| Average First Anchor Rank | Average rank of the first matched anchor when found |

Anchor matching uses normalized substring matching, so `《凡人修仙传》` and `凡人修仙传` match.

## Manual Relevance Judgement

Use `eval/manual_judgement_template.csv` as the judgement format. Copy it to `eval/manual_judgements.csv`, fill in the result rows, then run:

```bash
uv run python scripts/08_eval_metrics.py --judgements eval/manual_judgements.csv --k 10
```

Manual fields:

- `relevance_label`: `0` not relevant, `1` partially relevant, `2` highly relevant
- `constraint_violation`: `true` if the result violates explicit negative preferences
Manual metrics:

| Metric | Meaning |
|---|---|
| Precision@K | Fraction of top K results judged relevant or partially relevant |
| Strong Precision@K | Fraction of top K results judged highly relevant |
| Average Relevance | Mean `relevance_label` score |
| Constraint Violation Rate | Fraction of results violating explicit negative preferences |

## Limitations

- The corpus has no official labels, so manual judgement is required for relevance.
- Anchors are optional and may not exist in the private corpus.
- Results are based on compact sampled profiles, not full human reading of entire novels.
- Local LLM reranking is slower than FAISS-only retrieval.
- The framework compares system variants but does not claim production-grade benchmark results.
