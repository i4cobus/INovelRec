"""Tests for SFT example construction. CPU only, stub tokenizer, no downloads."""

from __future__ import annotations

from typing import Any

from src.sft_data import CHAT_TURN_END, build_dataset, collate, encode_example, render_prompt, split_by_query


class StubTokenizer:
    """Character-level stand-in that renders Qwen3's thinking-disabled template."""

    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: Any) -> str:
        content = conversation[0]["content"]
        tail = "<think>\n\n</think>\n\n" if kwargs.get("enable_thinking") is False else ""
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n{tail}"

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) for character in text]}


def sample(query_id: str, prompt: str = "p", target: str = '{"llm_match_score":0.5}') -> dict[str, Any]:
    return {"query_id": query_id, "prompt": prompt, "target": target}


def test_training_text_carries_the_prefilled_think_block_inference_sends() -> None:
    """The served prefix and the trained prefix must be the same string.

    Stage 4 calls the endpoint with ``chat_template_kwargs={"enable_thinking": False}``,
    which Qwen3 renders as an *empty but present* think block. Training on the bare
    ``<|im_start|>assistant\\n`` leaves the student facing an unfamiliar prefix on the
    first token it must generate — and it fails silently, still emitting JSON, just
    worse, with nothing in the metrics naming the cause.
    """

    rendered = render_prompt(StubTokenizer(), "PROMPT")

    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "PROMPT" in rendered


def test_loss_is_masked_to_the_target_only() -> None:
    example = encode_example(StubTokenizer(), "p", "T", max_length=4096)

    assert example is not None
    prefix_length = len(render_prompt(StubTokenizer(), "p"))
    assert example.labels[:prefix_length] == [-100] * prefix_length
    # Everything after the prefix is supervised, and the turn ends on <|im_end|>.
    assert example.labels[prefix_length:] == example.input_ids[prefix_length:]
    assert example.input_ids[prefix_length:] == [ord(c) for c in "T" + CHAT_TURN_END]


def test_a_pair_that_does_not_fit_is_dropped_rather_than_truncated() -> None:
    """Truncation would teach the model to stop mid-JSON."""

    assert encode_example(StubTokenizer(), "p" * 5000, "T", max_length=64) is None
    examples, dropped = build_dataset(StubTokenizer(), [sample("q1", prompt="p" * 5000)], max_length=64)
    assert (examples, dropped) == ([], 1)


def test_validation_holds_out_whole_queries() -> None:
    """Twenty samples share a query and overlap in candidates.

    Splitting per sample puts near-duplicates of every validation item into
    training, and the resulting validation loss measures nothing.
    """

    samples = [sample(f"q{index // 20:03d}") for index in range(200)]
    train, validation = split_by_query(samples, eval_fraction=0.2, seed=7)

    train_queries = {row["query_id"] for row in train}
    validation_queries = {row["query_id"] for row in validation}
    assert not (train_queries & validation_queries)
    assert len(train) + len(validation) == len(samples)
    assert validation_queries


def test_collate_pads_right_and_masks_padding_out_of_the_loss() -> None:
    batch = [
        encode_example(StubTokenizer(), "p", "T", max_length=4096),
        encode_example(StubTokenizer(), "p", "TTTT", max_length=4096),
    ]
    collated = collate([item for item in batch if item is not None], pad_token_id=0)

    assert collated["input_ids"].shape == collated["labels"].shape == collated["attention_mask"].shape
    lengths = collated["attention_mask"].sum(dim=1).tolist()
    assert lengths[0] < lengths[1]
    # Padding contributes neither attention nor loss.
    padded_row = collated["labels"][0].tolist()
    assert padded_row[-1] == -100
    assert collated["attention_mask"][0].tolist()[-1] == 0
