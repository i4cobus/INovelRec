"""SFT Qwen3-4B-Base on teacher-labelled reranker verdicts.

Written against ``transformers.Trainer`` directly rather than a training framework.
LLaMA-Factory and TRL both carry their own torch pins, and this host is already
constrained to the cu128 wheel (driver 570.124.06 caps at CUDA 12.8, which caps
torch at 2.11.0); the same reasoning that keeps vLLM out of this project's
dependencies applies here. The task is also genuinely simple — single turn, fixed
output schema, loss on the completion — so a framework would mostly be buying
config surface.

Data construction lives in ``src/sft_data.py`` so it is testable on CPU with a stub
tokenizer. In particular, the training text reproduces the *served* prompt exactly,
including Qwen3's pre-filled empty ``<think>`` block; see that module.

Multi-GPU via torchrun + FSDP. Full-parameter bf16 on 4B needs roughly 64 GB of
weights plus optimizer state, which fits one A100-80GB but leaves no room to grow;
sharding across four cards is both faster and less fragile.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.sft_data import build_dataset, collate, load_samples, split_by_query

app = typer.Typer(add_completion=False)
console = Console(width=140)

DEFAULT_SAMPLES = Path("data/processed/sft_samples.jsonl")
DEFAULT_OUTPUT = Path("data/checkpoints/sft-qwen3-4b")
BASE_MODEL = "Qwen/Qwen3-4B-Base"


@app.command()
def main(
    samples_path: Path = typer.Option(DEFAULT_SAMPLES, "--samples", help="Assembled SFT samples (script 17)."),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT, "--output-dir", help="Checkpoint directory."),
    base_model: str = typer.Option(BASE_MODEL, help="Starting checkpoint."),
    max_length: int = typer.Option(2048, help="Token budget per sample; pairs that exceed it are dropped, not cut."),
    epochs: float = typer.Option(1.0, help="Passes over the data."),
    learning_rate: float = typer.Option(1e-5, help="Peak LR for full-parameter SFT."),
    per_device_batch: int = typer.Option(4, help="Sequences per device per step."),
    grad_accum: int = typer.Option(8, help="Accumulation steps; global batch = devices x batch x accum."),
    warmup_ratio: float = typer.Option(0.03, help="Linear warmup fraction."),
    eval_fraction: float = typer.Option(0.02, help="Fraction of *queries* held out for validation."),
    save_steps: int = typer.Option(200, help="Checkpoint interval."),
    save_total_limit: int = typer.Option(2, help="Checkpoints kept. Full-parameter states are ~56 GB each."),
    logging_steps: int = typer.Option(10, help="Log interval."),
    seed: int = typer.Option(20260810, help="Split and training seed."),
    limit: int | None = typer.Option(None, help="Use only the first N samples (smoke run)."),
    max_steps: int = typer.Option(-1, help="Stop after N optimizer steps; -1 runs the full schedule."),
    dry_run: bool = typer.Option(False, help="Build and report the dataset, then exit without training."),
) -> None:
    """Fine-tune the base model on assembled reranker verdicts."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    samples = load_samples(samples_path)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise typer.BadParameter(f"No samples in {samples_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    train_rows, eval_rows = split_by_query(samples, eval_fraction=eval_fraction, seed=seed)
    train_set, train_dropped = build_dataset(tokenizer, train_rows, max_length=max_length)
    eval_set, eval_dropped = build_dataset(tokenizer, eval_rows, max_length=max_length)

    lengths = sorted(len(item) for item in train_set)
    report = Table(title="SFT Dataset")
    report.add_column("Metric")
    report.add_column("Value", justify="right")
    report.add_row("Samples", str(len(samples)))
    report.add_row("Train / validation", f"{len(train_set)} / {len(eval_set)}")
    report.add_row("Held-out queries", str(len({str(row.get('query_id', '')) for row in eval_rows})))
    report.add_row("Dropped as too long", f"{train_dropped} / {eval_dropped}")
    if lengths:
        report.add_row("Tokens: median / p95 / max", f"{lengths[len(lengths) // 2]} / {lengths[int(0.95 * len(lengths))]} / {lengths[-1]}")
        report.add_row("Supervised tokens", f"{sum(sum(1 for label in item.labels if label != -100) for item in train_set):,}")
    console.print(report)

    if not train_set:
        raise typer.BadParameter("Every training pair exceeded max_length; nothing to train on.")
    if dry_run:
        console.print("[yellow]--dry-run: dataset built, exiting before model load.[/yellow]")
        return

    # <|im_end|> ends an assistant turn. The base model's configured eos is
    # <|endoftext|>, which a chat-served student would never emit — leave it as the
    # pad token and make the turn terminator the generation stop.
    turn_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    model.config.pad_token_id = pad_id
    model.generation_config.eos_token_id = turn_end_id
    model.generation_config.pad_token_id = pad_id
    model.config.use_cache = False  # incompatible with gradient checkpointing

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        max_grad_norm=1.0,
        logging_steps=logging_steps,
        eval_strategy="steps" if eval_set else "no",
        eval_steps=save_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_only_model=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to=[],
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_set,
        eval_dataset=eval_set or None,
        data_collator=lambda batch: collate(batch, pad_token_id=pad_id),
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sft_run_config.json").write_text(
        json.dumps(
            {
                "base_model": base_model,
                "samples_path": str(samples_path),
                "samples": len(samples),
                "train": len(train_set),
                "validation": len(eval_set),
                "max_length": max_length,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "global_batch": per_device_batch * grad_accum * max(torch.cuda.device_count(), 1),
                "seed": seed,
                "train_loss": result.training_loss,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]Saved to {output_dir}[/green]  train_loss={result.training_loss:.4f}")


if __name__ == "__main__":
    app()
