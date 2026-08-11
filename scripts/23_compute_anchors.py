"""Attach the pairwise reward's partner anchor to every GRPO episode.

The reward scores *separation*: a rule-violating candidate should land below a clean
candidate of the same query, and vice versa. That needs a reference point per
episode, and it has to be fixed before training — an anchor that moved with the
policy could be satisfied by shifting both sides together, which is the constant
solution wearing a disguise.

The anchor is the **reference policy's** (the SFT checkpoint's) score on the opposite
class of the same query, decoded greedily so it is a stable number rather than a
sample. That is also the model KL is measured against, so the reward and the KL term
agree about what "unchanged behaviour" means.

Runs under the vLLM environment, not this project's:
  PYTHONPATH=/data/huangyanyu/INovelRec /data/huangyanyu/.venv-verl/bin/python \
      scripts/23_compute_anchors.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grpo_reward import parse_verdict  # noqa: E402

DEFAULT_POOL = "data/processed/grpo_pool.parquet"
DEFAULT_OUT = "data/processed/grpo_pool_anchored.parquet"
DEFAULT_MODEL = "data/checkpoints/sft-qwen3-4b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default=DEFAULT_POOL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Reference policy; must be the KL reference.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    pool = pd.read_parquet(args.pool)
    print(f"episodes: {len(pool)}  queries: {pool['query'].nunique()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in pool["prompt"]
    ]

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=4096)
    # Greedy: the anchor is a fixed reference value, not a sample.
    outputs = llm.generate(
        prompts,
        SamplingParams(n=1, temperature=0.0, max_tokens=args.max_tokens,
                       stop_token_ids=[tokenizer.convert_tokens_to_ids("<|im_end|>")]),
    )

    scores: list[float | None] = []
    for out in outputs:
        verdict = parse_verdict(out.outputs[0].text)
        scores.append(float(verdict["llm_match_score"]) if verdict else None)
    pool["ref_score"] = scores
    parsed = pool["ref_score"].notna().sum()
    print(f"reference scores parsed: {parsed}/{len(pool)} ({parsed / len(pool):.1%})")

    # The anchor is the mean reference score of the *opposite* class within the same
    # query. A mean rather than one partner: a single partner makes the reward hostage
    # to one noisy generation, while the class mean is what "the other side of this
    # query looks like" actually means.
    anchors: list[float | None] = []
    by_query = {
        query: (
            group.loc[group["rule_verdict"], "ref_score"].mean(),
            group.loc[~group["rule_verdict"], "ref_score"].mean(),
        )
        for query, group in pool.groupby("query")
    }
    for row in pool.itertuples():
        violating_mean, clean_mean = by_query[row.query]
        partner = clean_mean if row.rule_verdict else violating_mean
        anchors.append(None if pd.isna(partner) else float(partner))
    pool["partner_anchor"] = anchors

    have = pool["partner_anchor"].notna().sum()
    print(f"episodes with an anchor: {have}/{len(pool)} ({have / len(pool):.1%})")
    described = pool["partner_anchor"].describe()
    print(f"anchor: mean {described['mean']:.4f}  min {described['min']:.4f}  max {described['max']:.4f}")
    print("\nreference score by class (the separation the reward will train):")
    for verdict, group in pool.groupby("rule_verdict"):
        label = "violating" if verdict else "clean"
        print(f"  {label:<10} n={len(group):<6} mean ref_score {group['ref_score'].mean():.4f}")
    print("\npool_rank distribution (must cover the head, where the metric lives):")
    print(f"  from rank <10: {(pool['pool_rank'] < 10).mean():.1%}   <20: {(pool['pool_rank'] < 20).mean():.1%}   median {pool['pool_rank'].median():.0f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
