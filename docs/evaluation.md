# Evaluation

The corpus has no relevance labels, genre labels, or behaviour logs. Everything below exists to
compare *system variants* against each other under identical retrieval — not to claim benchmark
numbers.

## Three constraints that hold the protocol together

Breaking any one silently invalidates every number the project reports.

**1. The judge never reads the text the system ranked on.** `src/evidence.py` and `src/profile.py`
share `profile_chapter_indices` / `substantive_chapter_indices` / `window_fractions`; the profile
uses them to *pick* chapters and the evidence sampler uses them to *avoid* those chapters — the
complement in chapter mode, the midpoints between windows in the character-window fallback. Tests
assert zero overlap in both modes.

Grading the judge on the same text the system was ranked on would measure profile↔query agreement
rather than book↔query relevance, and would inherit every sampling mistake the profile made.

One deliberate exception: the author's 内容简介 appears on both sides. Without it an annotator
cannot determine genre at all — measured, not assumed. A noisy judge destroys the whole downstream
chain; a bounded and documented overlap does not.

**2. The reward rule and the evaluation source must differ.** GRPO scores negative constraints with
a keyword-density rule; evaluation scores them with a judge plus human labels. Same source and
optimising the metric would prove nothing.

**3. Human calibration.** 200 rows, single annotator. See below.

## Evaluation dataset

`eval/eval_queries.jsonl`, **frozen**: 59 queries — 17 whose exclusion is a narrative word the
density rule can count, 42 whose exclusion is a reader-applied meta-label it cannot. 31 queries
carry 55 anchor titles.

```json
{
  "query_id": "q001",
  "query": "凡人流 仙侠 慢热 理性主角 不系统",
  "wanted": ["凡人流", "仙侠", "慢热", "理性主角"],
  "unwanted": ["系统"],
  "anchor_titles": ["凡人修仙传"],
  "constraint_checkable": true,
  "notes": "Tests whether the system can retrieve classic slow-burn xianxia novels."
}
```

`constraint_checkable` is **derived, not frozen**. `13_sync_eval_queries.py` carries frozen queries
over untouched except this flag, which it recomputes from the current rule: query text, constraints
and anchors are the experimental design, but a derived flag left stale misreports which arm a query
is in.

## Variants

Any number of scoring arms can be compared, provided they share retrieval. The project's headline
comparison is five:

| variant | scoring model |
|---|---|
| `baseline_faiss` | none — raw query → embedding → FAISS top-k |
| `teacher_32b` | Qwen3-32B |
| `instruct_4b` | off-the-shelf Qwen3-4B-Instruct |
| `sft_student` | Qwen3-4B-Base after SFT |
| `grpo_v2` … `grpo_v4` | the SFT student after each GRPO round |

**`baseline_faiss` must be row-for-row identical across arms.** It is the check that says the only
variable is the scoring model. When it diverged once, that was how a cache poisoned with
parse-failure rows got caught.

```bash
uv run python scripts/07_evaluate.py --mode both --top-k 10 --llm-candidate-k 20
```

Outputs `eval/results/eval_results.csv` / `.jsonl`, plus `eval/results/eval_run_config.json`. Only
the judge-labelled join is committed — see [`../eval/results/README.md`](../eval/results/README.md).

Runs are reproducible to the row: the same configuration twice yields 1180/1180 identical lines.
That is not free — query expansion goes through vLLM, where `temperature=0` is *not* sufficient
because continuous batching changes batch composition and flips near-ties, and one different token
in an expansion swaps the entire candidate pool. Expansions are therefore cached under a key that
includes the prompt version. **A result file that cannot state how it was produced cannot be
compared to another one.**

## Anchor metrics

`compute_anchor_metrics` in `src/evaluation.py`.

| metric | definition | denominator |
|---|---|---|
| `Anchor Hit@K` | share of anchored queries with **at least one** anchor in the top K | 31 anchored queries |
| `Anchor Recall@K` | share of **all anchors** found in the top K | 55 anchors |
| average first anchor rank | mean rank of the first matched anchor, where one was found | queries with a hit |

Hit@K and Recall@K coincide only when every query carries exactly one anchor, which is why they were
once reporting the same number twice.

**Matching is by prefix, not substring.** `凡人修仙传` must not also match 《小小凡人修仙传》.
A prefix still tolerates the edition and volume suffixes the corpus is full of.

Pass `anchor_folds=` to break Recall@K down by fold. 17 of 22 original anchors sit in the train
fold, so after training the train/eval gap measures memorisation rather than retrieval.

**Anchors are a smoke test, never the headline.** They are a small, hand-picked, biased proxy for
recall: with 55 of them, Hit@10 moves 0.09 for two books. Three disciplines follow — the set is
frozen (tuning it after seeing baseline ranks means selecting for what the system already does
well), Hit@50/100 is always reported alongside so there is resolution at this n, and the headline
metric is always the judge plus human labels.

There is also a leakage risk specific to anchors: the 32B teacher recognises 《凡人修仙传》 and
《琅琊榜》 from pretraining and may assert facts absent from the sampled text, which would turn
Anchor Hit@K into a memorisation check that happens to inflate the metric in use.

## Relevance and constraint metrics

`compute_manual_metrics` in `src/evaluation.py`, over rows with `rank <= k`, grouped by
`system_variant`.

| metric | definition |
|---|---|
| `Precision@K` | share of judged rows with `relevance_label >= 1` |
| `Strong Precision@K` | share of judged rows with `relevance_label == 2` |
| `average_relevance` | mean `relevance_label` |
| `constraint_violation_rate` | share of judged rows with `constraint_violation` true |

**All four are micro-averages over candidate rows, not averages of per-query averages.** With 59
queries at top-10 that is up to 590 rows per arm. Paired sign tests are computed separately, on
per-query means — which is why their win/loss counts are counts of queries, not of rows.

`constraint_violation_rate` is only meaningful over queries that carry a negative constraint, and
is worth splitting by arm: the 17 rule-checkable queries and the 42 meta-label ones have very
different amounts of available headroom, and averaging them together hides that.

Labels:

- `relevance_label`: `0` not relevant, `1` partially relevant, `2` highly relevant
- `constraint_violation`: `true` if the result violates an explicit negative preference

## Human judgement

```bash
cp eval/manual_judgement_template.csv eval/manual_judgements.csv   # fill it in
uv run python scripts/08_eval_metrics.py --judgements eval/manual_judgements.csv --k 10
```

Annotation runs through `src/annotate_app.py` (Streamlit). Two decisions in it matter:

- **Rank and `system_variant` are not displayed.** Knowing that the system ranked something first
  biases an annotator toward agreeing, and the entire value of these labels is independence. A test
  scans the source and asserts neither field is rendered.
- **Every save appends to `eval/annotations.jsonl` and flushes.** 200 rows is not one sitting; a
  crash, a closed tab or a restart loses nothing and reopening jumps to the next unlabelled row.

## LLM judge

```bash
uv run python scripts/09_judge_eval.py --judge-model <name> \
    --price-input 3 --price-output 15 --budget-usd 200 --dry-run
```

**Always `--dry-run` first** — it prints the worst-case spend. `BudgetGuard` pre-flights the run and
refuses to start if the projection exceeds the cap, accumulates the endpoint-reported `usage` during
the run, and stops at the ceiling, flushing completed verdicts so a re-run resumes.

Verdicts cache on `(query_id, novel_id, evidence hash, model, prompt version)`. Re-sampling evidence
invalidates the entry, because a verdict is only valid for text the judge actually read. **A failed
request is absent from the results rather than recorded as label 0** — never conflate "no answer"
with "not relevant". Ignoring that once produced 313 fabricated zero rows that simultaneously
understated relevance and overstated violations.

**Judge evidence must come from the annotation sheet when one exists.** Evidence derives from the
complement of the profile's sampled chapters, so revising the profile silently changes it. A sheet
annotated under one profile and later judged under another had 40 of 40 sampled rows differ; three
agreement runs were lost this way. `09_judge_eval.py` now reuses the sheet's evidence verbatim.

## Judge–human agreement

```bash
uv run python scripts/11_agreement.py
```

| measure | value |
|---|---|
| exact agreement | 0.560 |
| chance agreement `pe` | 0.456 |
| Cohen's κ (linear-weighted, ordinal 0/1/2) | 0.191–0.263 |
| **adjacent agreement (≤1 level)** | **0.925** |
| **Gwet AC1** | **0.396** |
| severe disagreement (2 levels) | 15/200 |
| constraint violation agreement | 0.795 (κ 0.417, unweighted) |

**Never report κ alone for this task.** Both raters place ~60% of items in the top level, which
pushes chance agreement to 0.456 and eats most of the observed agreement. That is the Kappa Paradox,
not an unreliable judge: the 15 severe disagreements were adjudicated by hand and the judge was
right 10 to 5, with no directional bias. Report adjacent agreement and Gwet AC1 alongside it;
`eval/results/judge_calibration.json` holds the full set.

`src/evaluation.py` provides `cohen_kappa` (linear-weighted for the ordinal scale, unweighted for
the binary flag) and `judge_human_agreement`. `11_agreement.py` warns when weighted κ < 0.4.

Judge model choice is empirical. A cheaper model could not resolve three levels at all — 46.5% zeros
against a human 12.5%, and 3.0% ones against a human 28.5% — and two prompt revisions moved κ the
wrong way (0.244 → 0.224). That is a capability limit, not a wording problem.

## Rule–judge divergence

```bash
uv run python scripts/15_rule_judge_divergence.py
```

Compares the GRPO reward's density rule against judge labels. This is the instrument that separates
"learned the preference" from "learned the keyword rule", so it is run before and after training
with `--label`.

Baseline over the 233 candidates the rule adjudicated (45 declined): 71.2% agreement, 32 of 91 real
violations missed, 35 false positives. Per-term the failures are near-unidirectional — 异能 only
under-fires (44% agreement), 系统 only over-fires (16 false positives, from 火控系统 and similar).
The errors are a fixed per-term bias rather than noise, and a model optimised against the rule
learns the bias.

## Limitations

- **Statistical power is the binding constraint.** 17 rule-checkable queries; a paired sign test
  needs a near-sweep to clear 0.05.
- The corpus has no official labels, so every relevance number traces back to a judge or a human.
- Anchors are optional, hand-picked, and biased toward famous titles.
- Judgements are made on sampled evidence, not on complete novels.
- These metrics compare system variants. They are not benchmark claims.
