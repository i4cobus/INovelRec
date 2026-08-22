# Architecture

Each stage writes an artifact the next one consumes. **The artifact boundaries are the contract:
a later stage never re-derives an earlier one.**

## End-to-end flow

```text
data/raw/*.txt
  → 01  ingest        encoding detection, content-hash dedupe, novel_id = sha1(relative path)
  → novels.parquet
  → 12  splits        train/eval folds from content_sha256
  → 02  profile       boilerplate strip, chapter split, 10-chapter sampling
  → novel_profiles.parquet        (≤8000 chars per novel)
  → 03  embed         Qwen3-Embedding-8B → L2-normalised → FAISS IndexFlatIP (4096-d)
  → data/index/
  → 04  rank          expansion → multi-query retrieval → candidate selection → LLM scoring → fusion
  → 05  explain       grounded explanation, may not reorder
  → 06  Streamlit / CLI
```

The post-training path branches off the same artifacts:

```text
novels.parquet        → 16  term-density table    → the GRPO verifiable reward (1.2 MB lookup)
novel_profiles.parquet
  + data/index/       → 14  synthesise training queries from train-fold books
                      → 17  SFT data from a 32B teacher
                      → 18  SFT Qwen3-4B-Base
                      → 20  GRPO episode pool, balanced 50/50
                      → 21  export for verl  → 22  GRPO
novel_profiles.parquet → src/evidence.py → 09  LLM judge → 11  human agreement
```

## Stage 2: profiles

Profiles are embedded, not novels — a book averages over a million characters, and embedding one
whole would blend every arc and register into a single oversized input.

A profile is the author's own 内容简介 plus excerpts from 10 chapters: the first four, then six
distributed across the rest, stopping short of the finale. Truncation is sentence-aligned. The
front-loading follows 黄金三章; sampling evenly would spend nine slots on the thinnest part of the
book and one on the richest.

Chapter detection fails in two measured ways, so chapters are first filtered to those carrying real
body text, and books with fewer than three substantive chapters fall back to evenly distributed
character windows.

`profile_chapter_indices` / `substantive_chapter_indices` / `window_fractions` are **shared with
`src/evidence.py`**: the profile uses them to pick chapters, the judge's evidence sampler uses them
to avoid those chapters. That is what keeps evaluation from grading the system on its own sampling.

Any change here invalidates the FAISS index, the judge's evidence, and any annotation sheet built
under the old profile. The design is frozen for that reason as much as any other.

## Stage 3: vector retrieval

`Qwen3-Embedding-8B`, 4096 dimensions, through `encode_query` / `encode_document` so the model gets
its instruction-aware prompts — query text and novel profiles are not the same kind of input, and
encoding both identically runs the model outside its intended mode. Role selection falls back to
plain `encode` so test stubs keep working.

The index is `IndexFlatIP` over L2-normalised vectors, so **inner product is cosine**. Never add
unnormalised vectors. Exact search is deliberate: 7,653 vectors scan in microseconds, and an
approximate index would trade recall for an unmeasurable speedup. `faiss-cpu` is the dependency for
the same reason, and stays correct up to chunk-level scale.

Outputs:

- `data/index/faiss.index`
- `data/index/novel_id_map.json`
- `data/index/index_metadata.json`

**Two id spaces.** `novel_id` is a sha1 of the relative path, stable across runs. The FAISS row
index is positional and only meaningful through `novel_id_map.json` (whose JSON string keys must be
converted back to `int` on load). Folds are assigned from `content_sha256` instead of `novel_id`, so
duplicate copies of a book land on the same side of the split and adding books never reshuffles the
existing ones.

## Stage 4: ranking

Stage 4 does not rebuild embeddings, regenerate profiles, or re-clean text.

```text
query
  → structured preference parsing
  → LLM / domain query expansion          (cached; key includes the prompt version)
  → multi-query FAISS retrieval
  → candidate merge
  → candidate selection for the LLM        (three quota views)
  → LLM scoring                            (cached; key includes the prompt version)
  → fusion
```

**Domain hints are retrieval-only.** `DOMAIN_HINTS` widens recall — `凡人流 → 普通资质, 草根修仙,
谨慎, 炼气, 筑基, 宗门` — and never contributes to the final score. Otherwise a hand-written prior
would be scoring results directly.

**Candidate selection fills quotas from three views** so a strong recall candidate buried in the
merged order still reaches the LLM: `retrieval_score_top`, `semantic_score_top`,
`best_faiss_rank_top`. Each selected row carries its `llm_selection_reasons` through to the output
and the CLI table. `--debug-target-title` forces a specific title in for diagnosis.

Final score:

```text
final_score =
  0.40 * normalized_semantic_score
+ 0.50 * llm_match_score
+ 0.10 * confidence_score
- risk_penalty
```

**Every candidate is scored on one scale.** Candidates the LLM never saw reuse this same formula
with `llm_match_score` and `confidence_score` imputed from the mean of the scored ones — treated as
average, not as bad. The older two-formula behaviour survives as
`fallback_policy="legacy_semantic"` purely as an A/B control; paired tests show it lets a candidate
the LLM rated 0.1 outrank a strictly better unscored one. Setting `llm_candidate_k` to
`LLM_CANDIDATE_K_ALL` scores everything and removes the unscored path entirely.

**`normalize_semantic_scores` min-maxes unconditionally.** It used to pass scores through whenever
they already fell in [0, 1], which — for a pool of cosine similarities clustered in 0.64–0.74 — was
almost always, so a nominal `semantic_weight=0.40` contributed about 0.04 of real spread. Call
`score_component_contributions(ranked)` to see what each component actually moves; do not trust the
nominal weights. On this corpus: `llm_match` 75.3%, semantic 24.7%, confidence 0.0%.

**Failure is distinguished from a low score.** `rank.py` uses the matcher's `score_many` when it
offers one, else per-candidate calls. A `None` in a batch slot means *the request died*, not *the
model rated it low*; those rows become `rule_fallback` and are deliberately **not** cached, because
caching a failure would freeze it in.

### Backends

`src/backends.py` is the single construction point. `create_matcher(backend=...)` returns either
matcher and imports `transformers` or `httpx` lazily, so choosing one never pays the other's import
cost. Scripts `05`/`06`/`07` and the Streamlit sidebar all route through it.

- `transformers` — local Hugging Face inference, one device pinned by `resolve_single_device`.
  Never `device_map="auto"`: it pipeline-shards a 4B model across every visible GPU and lands
  slower.
- `http` — `OpenAICompatibleMatcher` speaks `/chat/completions`, so one client covers a locally
  served vLLM *and* a hosted gateway. It satisfies `CandidateMatcher`, `LLMExpansionProvider` and
  `ExplanationGenerator` at once, and its `ChatTransport` seam lets tests run against a stub.

There is deliberately **no `--api-key` flag** — CLI arguments are world-readable in `ps`, so the key
comes from `INOVELREC_LLM_API_KEY`. `is_local_url` decides whether to bypass the ambient HTTP proxy;
a proxy answering a request for `127.0.0.1` by closing the connection surfaces as
`Server disconnected without sending a response`, which names nothing about proxies.

Two parsing rules apply to every LLM response: reasoning is disabled by default
(`chat_template_kwargs: {enable_thinking: false}`, because a `<think>` block will eat the whole
token budget and every verdict then parses as a failure), and JSON is extracted by **brace-depth
matching** after stripping reasoning traces — a greedy `{.*}` spans from a brace inside the trace to
the closing brace of the answer.

## Stage 5: explanation

**Stage 4 ranks. Stage 5 explains. `explain_recommendations` may not reorder.**

It is fed only Stage 4 output fields plus profile evidence: query, final rank, title, scores,
matched and violated preferences, risk flags, Stage 4's reason. The prompt explicitly forbids
inventing plot details, popularity, author facts, ratings, or completion status. Invalid JSON falls
back to `fallback_explanation`, a deterministic rendering of the same Stage 4 fields, so the report
pipeline cannot fail open.

## Stage 6: Streamlit

`src/app_pipeline.py` is the single shared Stage 4→5 driver used by both the app and
`06_explain_demo.py`. Pipeline changes go there, not into the Streamlit layer, which holds only
caching and rendering:

- `st.cache_resource` for the embedding model, the LLM and the FAISS index
- `st.cache_data` for the id map and profile lookup

The app does not rebuild the index, regenerate embeddings, or re-clean profiles.

## Testability

Collaborators are injected through `Protocol`s and never constructed inside logic: `SupportsEncode`
(`embed.py`), `CandidateMatcher` (`rank.py`), `ExplanationGenerator` (`explain.py`),
`LLMExpansionProvider` (`query_expansion.py`).

Every test runs on CPU with stub objects and **no model downloads**. Heavy imports
(`transformers`, `sentence_transformers`) are deferred inside functions so importing `src.*` stays
cheap — which is also what lets the verl training environment call `src/grpo_reward.py` with nothing
but `PYTHONPATH`, without installing this project's dependencies.
