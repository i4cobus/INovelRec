# Evaluation artifacts

Every number quoted in the top-level `README.md` is computed from the files in
this directory. Six arms were evaluated against an identical retrieval stage, so
`baseline_faiss` rows are expected to agree across all of them — that agreement
is the check that the comparison isolates the reranker.

| file | what it is |
|---|---|
| `eval_results_judged.csv` | top-level run: system output joined with the LLM judge's labels |
| `eval_results_judged_clean.csv` | same, after 313 unparsed judge verdicts were dropped rather than recorded as label 0 |
| `<arm>/eval_results_judged.csv` | one arm each: `instruct_4b`, `sft_student`, `grpo_4b` (v1), `grpo_v2`, `grpo_v3`, `grpo_v4` |
| `<arm>/eval_run_config.json` | the exact retrieval and prompt settings that produced that arm — the fields that must match across arms for the comparison to hold |
| `judge_calibration.json` | judge vs. 200 human labels: exact, adjacent, weighted kappa, Gwet AC1 |
| `rule_judge_divergence.json` | the reward rule vs. the judge, overall and per excluded term |
| `arm_precheck.json` | per-query pool composition, used to decide which queries carry a rule-checkable exclusion |
| `anchor_ranks_baseline.json` | corpus rank of each anchor title under retrieval alone |

Three kinds of file are deliberately **not** here:

- The pre-judge `eval_results.csv` / `.jsonl`. The judged CSV is a strict superset
  of the same rows, so keeping both only duplicates them.
- `agreement_*.csv` and the annotation sheets. They carry an `evidence` column of
  up to ~5,000 characters of raw novel text, and the corpus is private.
- Model checkpoints. See the top-level README for how to reproduce them.
