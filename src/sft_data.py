"""Turn assembled SFT samples into tokenized, loss-masked training examples.

Kept out of the training script so it can be tested on CPU with a stub tokenizer,
like every other injected collaborator in this project. Nothing here loads a model.

The one thing this module exists to get right is that **the text the student is
trained on has to be byte-identical to the text it will be served**. Stage 4 talks
to the reranker over ``/chat/completions`` with
``chat_template_kwargs={"enable_thinking": False}``, and Qwen3's template renders
that as a *pre-filled empty think block*::

    <|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n

Train on the bare ``<|im_start|>assistant\\n`` instead and the student meets an
unfamiliar prefix on the very first token it has to generate at inference. That
failure is silent: the model still emits JSON, just worse, and nothing in the
metrics says why.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# Chat turns end on <|im_end|>, not on the base model's <|endoftext|>. A student
# trained to stop on the wrong token runs past its own answer.
CHAT_TURN_END = "<|im_end|>"


class SupportsChatTemplate(Protocol):
    """The slice of a tokenizer this module needs."""

    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: Any) -> str: ...

    def __call__(self, text: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class SFTExample:
    """One tokenized sample with the prompt masked out of the loss."""

    input_ids: list[int]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.input_ids)


def load_samples(path: Path) -> list[dict[str, Any]]:
    """Read assembled samples from ``17_build_sft_data.py``."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_prompt(tokenizer: SupportsChatTemplate, prompt: str) -> str:
    """Render the served prompt prefix, thinking disabled.

    ``enable_thinking=False`` is passed explicitly rather than relying on the
    template default: the default renders a *live* think block, and Qwen3 then
    spends the whole token budget reasoning before any JSON appears.
    """

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_example(
    tokenizer: SupportsChatTemplate,
    prompt: str,
    target: str,
    *,
    max_length: int,
    ignore_index: int = -100,
) -> SFTExample | None:
    """Tokenize one (prompt, target) pair, masking the prompt out of the loss.

    Returns ``None`` when the pair does not fit: truncating the *target* would
    train the model to emit JSON that stops mid-object, which is worse than not
    training on the sample at all. Truncating the prompt instead would teach it to
    score a profile it cannot see.
    """

    prefix = render_prompt(tokenizer, prompt)
    completion = target + CHAT_TURN_END

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if len(prefix_ids) + len(completion_ids) > max_length:
        return None

    return SFTExample(
        input_ids=list(prefix_ids) + list(completion_ids),
        labels=[ignore_index] * len(prefix_ids) + list(completion_ids),
    )


def split_by_query(samples: list[dict[str, Any]], *, eval_fraction: float, seed: int) -> tuple[list, list]:
    """Hold out whole queries, never individual samples.

    Twenty samples share a query string and overlap heavily in candidates, so a
    per-sample split puts near-duplicates of every validation item into training
    and reports a validation loss that means nothing.
    """

    import random

    query_ids = sorted({str(sample.get("query_id", "")) for sample in samples})
    rng = random.Random(seed)
    rng.shuffle(query_ids)
    held_out = set(query_ids[: max(1, int(len(query_ids) * eval_fraction))])
    train = [sample for sample in samples if str(sample.get("query_id", "")) not in held_out]
    validation = [sample for sample in samples if str(sample.get("query_id", "")) in held_out]
    return train, validation


def build_dataset(
    tokenizer: SupportsChatTemplate,
    samples: Iterable[dict[str, Any]],
    *,
    max_length: int,
) -> tuple[list[SFTExample], int]:
    """Encode samples, returning the examples and how many were dropped as too long."""

    examples: list[SFTExample] = []
    dropped = 0
    for sample in samples:
        encoded = encode_example(
            tokenizer,
            str(sample.get("prompt", "")),
            str(sample.get("target", "")),
            max_length=max_length,
        )
        if encoded is None:
            dropped += 1
            continue
        examples.append(encoded)
    return examples, dropped


def collate(batch: list[SFTExample], *, pad_token_id: int, ignore_index: int = -100) -> dict[str, Any]:
    """Right-pad a batch and build the attention mask."""

    import torch

    width = max(len(item) for item in batch)
    input_ids, labels, attention = [], [], []
    for item in batch:
        padding = width - len(item)
        input_ids.append(item.input_ids + [pad_token_id] * padding)
        labels.append(item.labels + [ignore_index] * padding)
        attention.append([1] * len(item) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }
