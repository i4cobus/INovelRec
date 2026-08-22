# Preference Alignment for a Chinese Web Novel Reranker (SFT + GRPO)

Post-training a 4B reranker to obey **negative preferences** — "仙侠 慢热 **不要系统**" — over a
private corpus of 7,653 Chinese web novels (~36 GB of plain text).

A retrieval system supplies the task and the evaluation harness. The work is the post-training:
distilling a domain reranker from a 32B teacher, then optimising constraint compliance with GRPO
against a rule-verifiable reward, and measuring what that reward does and does not buy.

## Result

Every arm below ranks the *same* retrieved candidates — same index, same cached query expansions,
same `candidate_k` — so the only variable is the scoring model. `baseline_faiss` is identical
across all five arms, which is the check that says so.

| arm | P@10 | strong P@10 | mean relevance | **violation rate** | rule-checkable arm |
|---|---|---|---|---|---|
| teacher 32B | **0.956** | **0.679** | **1.635** | 0.251 | 0.314 |
| off-the-shelf 4B-Instruct | 0.932 | 0.616 | 1.548 | 0.271 | 0.294 |
| 4B after SFT | 0.952 | 0.621 | 1.573 | 0.276 | 0.359 |
| **4B after GRPO** | 0.954 | 0.642 | 1.596 | **0.241** | **0.271** |

Denominators, since none of these are benchmark numbers: 59 hand-written evaluation queries, top-10
each. All four columns are **micro-averages over judged candidate rows** — up to 590 per arm — not
averages of per-query averages. `P@10` is the share of rows a judge scored 1 or 2 on a 0/1/2 scale,
`strong P@10` the share scored 2, `mean relevance` the mean label. `violation rate` covers only rows
from queries that carry a negative constraint, and the last column narrows that to the 17 queries
whose exclusion is a narrative word the density rule can actually count. The paired sign tests
quoted below run on per-query means instead, which is why their win/loss counts are query counts.

Labels come from an LLM judge calibrated against 200 human annotations. The 4B model ends up
**better at obeying negative constraints than the 32B teacher it was distilled from**, at roughly
an eighth of the inference cost,<sup>†</sup> with relevance unchanged from SFT.

<sup>†</sup> "an eighth" is the weight-memory ratio (8 GB vs 64 GB in bf16). Measured throughput on
one A100 is 30.5 gen/s for the 4B against 16 scoring calls/s for the teacher at TP=2 — about 4x per
GPU, ~7–8x once the teacher's second card is counted.

Honest bound: on the 17 rule-checkable queries a paired sign test gives p=0.092 against the SFT
model. The direction is consistent across all four aggregate metrics and retrieval is provably
identical, but "confirmed" is the wrong word for n=17.

Every judged row behind this table is in [`eval/results/`](eval/results/).

## What the project is actually about

Real products never have a single verifiable reward. Here the reward mixes a **hard, checkable**
signal (does this novel contain the excluded trope?) with a **soft, model-judged** one (is it
relevant?). The question worth answering is where a policy goes when both are in the same
objective.

It goes straight for the checkable half — and that half only covers part of what you wanted:

| GRPO round | reward-rule agreement | judge-measured violation rate |
|---|---|---|
| v1 | 0.547 → 0.705 | 0.359 → 0.353 (**no movement**) |
| **v2** | 0.553 → 0.704 | 0.359 → **0.271** |
| v3 | 0.559 → **0.768** | 0.359 → 0.282 |
| v4 | 0.533 → 0.717 | 0.359 → 0.276 |

Across four rounds the training metric rose 0.15–0.21 every time while the metric that matters
moved at most 0.09. The gap is measurable rather than mysterious: the density rule *sees* only 48%
of real violations (63% after widening its vocabulary), which caps a perfect rule-follower at a
0.141 violation rate. Reaching 0.271 captured roughly 40% of the space that was actually available.

Seeing this at all required an evaluation whose labels never touch the reward — the discipline that
makes the two columns above independent.

Round 4 is also why the vocabulary stopped widening. Recall and precision trade against each other:
the wider list raised the ceiling by 0.04 but pushed the semantic arm from 0.229 to 0.254, because
six clean candidates now fired the rule and were demoted. v2 shipped; v3 and v4 both kept climbing
on the training metric while the system metric stalled or regressed.

## Two policy collapses

Both were caught by monitoring curves rather than by a drop in the final metric — in both cases the
mean reward was *rising* while the policy degenerated.

**"Claim every candidate violates."** Predicted before training and observed at step 25 of round 1:
the claim rate on held-out data went 0.10 → 0.80. The rollout pool is balanced 50/50 between
violating and clean candidates precisely so this shortcut earns the same as saying nothing, and it
fell back to 0.45 within 25 steps without intervention.

**"Score everything 0."** Not predicted. The score reward paid for *lowering* the score on
violations and paid nothing on clean candidates, so answering 0 everywhere was its trivial optimum
— and since that field carries weight 0.50 in the ranking formula, the reranker's relevance
contribution vanished (paired sign test against SFT, p=0.012). Replacing it with a margin against a
contrasting candidate of the same query makes any constant score exactly average instead of
optimal.

## The reward

A negative constraint is decided by **density, never presence**. A 3M-character novel that says
消化系统 once is not a 系统流 story; scoring presence marked 87% of the corpus as violating and
turned the reward into a constant. Two thresholds instead of one, so the ambiguous middle can be
declined rather than guessed:

```python
density >= 3.0  → violates      # occurrences per 100k characters
density <= 1.0  → clean
otherwise       → None, meaning no reward signal — never "no violation"
```

Thresholds are calibrated against human labels: violating novels sit at a median density of 3.45,
clean ones at 0.03, with overlapping tails. A single threshold tops out at F1 0.65; the two-sided
version declines the 6% in between. Merging *abstain* into *clean* would teach the model that
ambiguity is safe.

Each exclusion is a *set* of surface forms, because a novel saturated with 觉醒者 and 能力者 may
mention 异能 at density 0.25. Members were screened by corpus-wide firing rate and co-occurrence
with the base term, **never by agreement with the judge** — selecting the reward's vocabulary to
match the evaluation would make any later gain self-fulfilling.

Exclusions readers attach from *outside* the book (宠文, 种马) are held in a separate set and never
enter the verifiable reward: 《娇女》 is a 宠文 and the word never appears in it, so a keyword
rule's recall there is zero by construction, not merely low.

The rule's errors are not noise — each term fails in one direction. 异能 only under-fires (44%
agreement; those novels say 觉醒 and 能力者 instead), while 系统 only over-fires (military fiction
is full of 火控系统). A model optimised against the rule learns those biases, which is exactly what
the two columns above are instrumented to detect.

The whole thing precomputes to a 1.2 MB lookup table over 7,656 novels — one dict lookup per
rollout. That property matters more than the price: a judge cannot be precomputed, because its
input contains the query.

## Pipeline

```
data/raw/*.txt                      7,653 novels, up to 3M characters each
  → 01  ingest, dedupe by content hash
  → 12  train/eval folds from content hash, not path
  → 02  profile: 黄金三章 + distributed sampling, capped at 8000 chars
  → 03  Qwen3-Embedding-8B → FAISS IndexFlatIP (4096-d, L2-normalised)
  → 04  retrieval + rank            ← the task
  → 16  term-density table          ← the verifiable reward, precomputed to 1.2 MB
  → 14  synthesise training queries from train-fold books
  → 17  SFT data from a 32B teacher, candidates restricted to fold=train
  → 18  SFT Qwen3-4B-Base           4x A100, 1 epoch in 2.2h
  → 20  GRPO episode pool, balanced 50/50 and weighted to the head of retrieval
  → 22  GRPO via verl               reward = src/grpo_reward.py, G=8
  → 07  evaluate  → 09 LLM judge  → 11 human agreement
```

## Training data

Queries are synthesised **backwards, from books**: the teacher reads one train-fold profile and
writes the reader need that would point at it, so every query ships with a seed positive and needs
no annotation. 24,792 survive deduplication and leak checks.

The design decision worth stating: **labels come from the rule, not from the teacher's claim.** The
first version had the teacher also declare whether the book violated the exclusion it had just
picked, with the rule used only to discard contradictions. On a 20-book smoke test the violating
half collapsed from an expected ~35% to 3% — the half that carries the entire constraint signal.
The teacher reads an 8000-character profile and calls 系统 on a single mention; the rule reads all
3M characters. When ground truth is free to compute, having a model guess it and then deleting data
over the guess is self-inflicted.

That leaves a number worth reporting: the teacher's constraint claims agree with the full text only
**70%** of the time. Since the reranker also only ever sees a profile, that 70% is a meaningful
upper bound on the teacher, not just a data-cleaning artifact.

Three leak barriers, and the third is the one that is easy to miss: seed books are restricted to the
train fold, synthesised queries are deduplicated against the evaluation set, **and the candidate
pool is filtered to the train fold too** — FAISS retrieves corpus-wide, so without that filter the
student would see eval-fold profiles and the fold split would stop separating "learned to retrieve"
from "memorised the corpus". No filtering is applied at inference; it is a discipline on the
training data, not a limit on the model.

## Evaluation protocol

Three constraints. Breaking any one silently invalidates every number above.

**1. The judge never reads the text the system ranked on.** `src/evidence.py` and `src/profile.py`
share the chapter-selection functions — one picks, the other takes the complement (or, in the
character-window fallback, the midpoints between). Tests assert zero overlap in both modes. Grading
the system's own summary would measure profile↔query agreement and inherit every sampling mistake
the profile made.

**2. The reward rule and the evaluation source are different.** GRPO scores constraints with the
density rule; evaluation scores them with a judge plus human labels. If they were the same source,
optimising the metric would prove nothing and the divergence table above could not exist.

**3. The judge is calibrated against humans.** 200 rows, single annotator:

| | value |
|---|---|
| exact agreement | 0.560 |
| chance agreement `pe` | 0.456 |
| Cohen's κ (linear-weighted) | 0.191–0.263 |
| **adjacent agreement (≤1 level)** | **0.925** |
| **Gwet AC1** | **0.396** |
| severe disagreement (2 levels) | 15/200 |

**Never read κ alone here.** Both raters put ~60% of items in the top level, which pushes chance
agreement to 0.456 and eats most of the observed agreement — the Kappa Paradox, not an unreliable
judge. The 15 severe disagreements were adjudicated by hand and the judge was right 10 to 5, with
no directional bias. Full numbers in
[`eval/results/judge_calibration.json`](eval/results/judge_calibration.json).

Model choice was empirical, not rhetorical: a cheaper judge could not resolve three levels at all
(46.5% zeros against a human 12.5%, and only 3.0% ones against a human 28.5%), and two prompt
revisions moved κ the wrong way. Total judge spend across the project was $11.54 against a $200 cap
enforced in code — the guard pre-flights a run, refuses to start if the projection exceeds the cap,
accumulates endpoint-reported usage, and flushes completed verdicts when it stops.

**The evaluation set is frozen**: 59 queries, 17 with a rule-checkable exclusion and 42 without, 31
carrying 55 anchor titles. Anchors match by **prefix, not substring** — 凡人修仙传 must not also
match 小小凡人修仙传 — and they are only ever a smoke test. With 55 anchors, Hit@10 moves 0.09 for
two books, and the teacher recognises the famous ones from pretraining, which would turn the metric
into a memorisation check. Recall is therefore reported split by fold, so the train/eval gap
measures memorisation directly.

Runs are reproducible to the row: the same configuration twice produces 1180/1180 identical lines,
and `07_evaluate.py` writes its configuration into `eval/results/eval_run_config.json`. A result
file that cannot say how it was produced cannot be compared to another one.

## The retrieval system

The task and the harness, not the contribution — but two of its decisions carry the post-training.

**Profiles are embedded, not novels.** A book averages over a million characters. Each profile is
the author's own 内容简介 (extractable in 86.6% of the corpus) plus excerpts from 10 chapters: the
**first four**, then six spread across the rest, stopping short of the finale. Capped at 8000
characters, 5,815 on average.

The front-loading is 黄金三章 — a Chinese web novel states its genre, protagonist and 金手指 in its
opening chapters, because that is where readers decide whether to continue. Sampling evenly spends
nine slots on the thinnest part of the book and one on the richest. Truncation is sentence-aligned;
chapters are coherent narrative units and cutting mid-sentence throws away the style signal.

Chapter detection fails in two measured ways, so there is a fallback: 章回体 novels open with a
table of contents whose headings have empty bodies, and 140 novels yield no headings at all and
collapse into a single "chapter". Chapters are filtered to those carrying real text, and books with
fewer than three fall back to evenly distributed character windows.

**This design is frozen.** It went through three data-driven revisions and one domain-driven one;
once the baseline numbers were known, further changes risk being reverse-engineered from the metric.

Its limits are structural rather than fixable by better sampling: 慢热 is a first derivative over
two million characters, and 不系统 is a universal negative. Neither is observable in any 8000
characters — that is what the reranker is for, and what an offline tagging pass would be for.

**Ranking, then explanation.** Stage 4 ranks:

```text
final_score = 0.40 * normalised_semantic + 0.50 * llm_match + 0.10 * confidence − risk_penalty
```

Candidates the LLM never scored stay on the *same* scale: their `llm_match` and `confidence` are
imputed from the mean of the scored ones — treated as average, not as bad. The older two-formula
behaviour survives as `fallback_policy="legacy_semantic"` purely as an A/B control, with paired
tests showing it lets a candidate the LLM rated 0.1 outrank a strictly better unscored one.

Do not trust the nominal weights: `score_component_contributions()` measures what each term actually
moves. On this corpus it is `llm_match` 75.3%, semantic 24.7%, confidence 0.0% — and the semantic
share was 0.04 until normalisation was made unconditional, because a pool of cosine similarities
already sits inside [0, 1] and the old guard therefore never fired.

Stage 5 explains and **may not reorder**. It sees only Stage 4's output fields plus profile
evidence, and its prompt forbids inventing plot, popularity, author, ratings, or completion status.
Invalid JSON falls back to a deterministic rendering of Stage 4's fields.

Two id spaces, easy to conflate: `novel_id` is a sha1 of the relative path and is stable across
runs; the FAISS row index is positional and only meaningful through `novel_id_map.json`. Folds are
assigned from `content_sha256` instead, so duplicate copies of a book land on the same side and
adding books never reshuffles the existing ones.

## Repository layout

| path | what |
|---|---|
| `src/` | library. Models are injected through `Protocol`s, so tests run on CPU with stubs and no downloads |
| `scripts/` | one `typer` CLI per stage, numbered in execution order |
| `eval/results/` | the judged rows and aggregate JSONs behind every number above |
| `tests/` | `uv run pytest` |
| `docs/` | [architecture](docs/architecture.md) · [evaluation](docs/evaluation.md) |

`src/preferences.py` holds the single definition of constraint violation, shared by the GRPO reward
and by training-data synthesis. Two copies would drift, and the divergence table compares *that*
rule against human labels.

## Running it

```bash
uv sync
uv run pytest                                     # CPU only, no model downloads
```

Retrieval and ranking, once `data/raw/*.txt` exists:

```bash
uv run python scripts/01_inventory.py --overwrite
uv run python scripts/12_build_splits.py
uv run python scripts/02_build_profiles.py --overwrite
uv run python scripts/03_build_index.py --overwrite --multi-gpu
uv run python scripts/05_recommend_demo.py "凡人流 仙侠 慢热 理性主角 不系统" \
    --candidate-k 200 --llm-candidate-k 10 --top-k 10 --device cuda:1
uv run streamlit run src/streamlit_app.py
```

Evaluation, then the training path:

```bash
uv run python scripts/07_evaluate.py --mode both --top-k 10 --llm-candidate-k 20
uv run python scripts/09_judge_eval.py --judge-model <name> --budget-usd 200 --dry-run
uv run python scripts/16_build_density_table.py
uv run python scripts/14_synthesize_queries.py --stratify
uv run python scripts/17_build_sft_data.py
uv run python scripts/18_sft_train.py
bash scripts/22_grpo_train.sh
```

Two things that are easy to get wrong on a shared box. Pass an explicit `--device cuda:N` to any
stage that loads the embedder if a vLLM server is holding GPU 0 — `auto` resolves to `cuda:0` and
OOMs against the resident weights. And the reranker's API key comes from `INOVELREC_LLM_API_KEY`,
never a CLI flag, because arguments are world-readable in `ps`.

`vllm` is deliberately **not** a dependency of this project: it carries its own torch pin. It runs
in a separate environment and is reached over HTTP, which is also what lets the same client talk to
a hosted gateway for the judge.

## What's here and what isn't

Committed: all code and tests, the frozen evaluation queries, and the judge-labelled rows for every
arm.

Not committed: the corpus, the profiles, the FAISS index, model checkpoints, and the annotation
sheets — the sheets inline several thousand characters of novel text per row. See
[`eval/results/README.md`](eval/results/README.md) for what is kept and why the rest is held back.

## Limitations

- **Statistical power is the binding constraint.** 17 rule-checkable queries; a paired sign test
  needs a near-sweep to clear 0.05. Direction and magnitude are consistent, significance is not
  established.
- **SFT bought format, not ranking.** Against off-the-shelf 4B-Instruct it shows no measurable gain
  (relevance p=0.880, constraints p=1.000) — and that test had power, since the same method
  separates the student from the teacher at p=0.006. What it did buy is a 99.9% valid-JSON rate,
  which is what GRPO needed to start from.
- **Relevance is capped by the teacher.** It has no verifiable reward, so there is no mechanism by
  which the student exceeds it. Only the checkable dimension went past the 32B model.
- **The rule is biased, per term, by design.** 71.2% agreement with the judge, 35% of real
  violations missed. That gap is the instrument, not a defect — a perfect rule would make the
  experiment meaningless.
- **Meta-label exclusions (宠文, 种马, 爽文) have no verifiable reward at all**, so 42 of the 59
  evaluation queries can only be improved indirectly.
- **No online system.** No A/B, no latency SLO, no serving path. The throughput figures here are
  offline measurements, and the per-request budget is derived from them rather than load-tested.
- **GRPO ran 50 steps per round.** This is a methodology run on limited GPU time; the four-round
  comparison carries more weight than any single round's step count.
- **Anchors are a smoke test.** 55 of them, hand-picked, biased toward famous titles the teacher may
  recognise from pretraining.
