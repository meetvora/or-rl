from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path
from typing import Any

from .data import NormalizedExample, load_train_examples, setup_logging, split_train_validation
from .modeling import (
    load_text_causal_lm,
    load_text_tokenizer,
    prepare_lora_model_if_needed,
    str2bool,
    validate_quantization_args,
)
from .prompts import GENERATION_INSTRUCTION

logger = logging.getLogger(__name__)


def enable_kv_cache_and_disable_checkpointing(model) -> None:
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()


def build_sft_prompt(tokenizer, example: NormalizedExample) -> str:
    user = f"{example.prompt.strip()}\n\n{GENERATION_INSTRUCTION}"
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user


def build_sft_completion(example: NormalizedExample, eos_token: str | None) -> str:
    code = (example.reference_code or "# No reference code available").strip()
    completion = f"<CODE>\n{code}\n</CODE>"
    if eos_token:
        completion += eos_token
    return completion


def tokenize_completion_only_example(
    tokenizer,
    example: NormalizedExample,
    max_seq_length: int,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(
        build_sft_prompt(tokenizer, example),
        add_special_tokens=False,
    )["input_ids"]
    completion_ids = tokenizer(
        build_sft_completion(example, tokenizer.eos_token),
        add_special_tokens=False,
    )["input_ids"]
    if len(completion_ids) >= max_seq_length:
        completion_ids = completion_ids[:max_seq_length]
        prompt_ids = []
    else:
        prompt_budget = max_seq_length - len(completion_ids)
        prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = prompt_ids + completion_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids,
        "loss_token_count": [len(completion_ids)],
        "prompt_token_count": [len(prompt_ids)],
    }


def build_completion_only_dataset(tokenizer, examples: list[NormalizedExample], max_seq_length: int):
    from datasets import Dataset

    return Dataset.from_list(
        [tokenize_completion_only_example(tokenizer, ex, max_seq_length) for ex in examples]
    )


class CompletionOnlyDataCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        pad_token_id = self.tokenizer.pad_token_id
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * pad_length)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_length)
            batch["labels"].append(feature["labels"] + [-100] * pad_length)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional SFT warm-start for OR-Tools format.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--train_path", default="data/train/complex_or_variations.jsonl")
    parser.add_argument("--output_dir", default="outputs/sft_adapter")
    parser.add_argument("--eval_split_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_examples", type=int, default=32)
    parser.add_argument("--use_lora", type=str2bool, default=True)
    parser.add_argument("--use_qlora", type=str2bool, default=False)
    parser.add_argument("--load_in_4bit", type=str2bool, default=False)
    parser.add_argument("--load_in_8bit", type=str2bool, default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    if args.use_qlora:
        args.load_in_4bit = True
    setup_logging(args.log_level)
    validate_quantization_args(args.load_in_4bit, args.load_in_8bit)

    try:
        import torch
        from transformers import Trainer, TrainingArguments
        if args.use_lora:
            from peft import LoraConfig, get_peft_model
    except Exception as exc:
        raise RuntimeError(
            "SFT requires torch, transformers, datasets, trl, peft, and bitsandbytes for k-bit LoRA."
        ) from exc

    examples = load_train_examples(args.train_path, args.max_train_examples)
    train_examples, val_examples = split_train_validation(examples, args.eval_split_ratio, args.seed)

    tokenizer = load_text_tokenizer(args.model_name_or_path)
    train_ds = build_completion_only_dataset(tokenizer, train_examples, args.max_seq_length)
    val_ds = (
        build_completion_only_dataset(tokenizer, val_examples, args.max_seq_length)
        if val_examples
        else None
    )
    train_loss_tokens = sum(row["loss_token_count"][0] for row in train_ds)
    train_prompt_tokens = sum(row["prompt_token_count"][0] for row in train_ds)
    logger.info(
        "completion_only_sft train_examples=%d prompt_tokens=%d loss_tokens=%d",
        len(train_ds),
        train_prompt_tokens,
        train_loss_tokens,
    )
    model = load_text_causal_lm(
        args.model_name_or_path,
        args.load_in_4bit,
        args.load_in_8bit,
        trainable_adapter=True,
    )
    enable_kv_cache_and_disable_checkpointing(model)

    peft_config = None
    if args.use_lora:
        model = prepare_lora_model_if_needed(model, args.use_lora, args.load_in_4bit, args.load_in_8bit)
        enable_kv_cache_and_disable_checkpointing(model)
        if getattr(model, "peft_config", None) is not None:
            from .modeling import enable_trainable_adapter_parameters

            enable_trainable_adapter_parameters(model)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        if getattr(model, "peft_config", None) is None:
            model = get_peft_model(model, peft_config)
            peft_config = None

    training_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "optim": args.optim,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "bf16": torch.cuda.is_available(),
        "gradient_checkpointing": False,
        "remove_unused_columns": False,
        "report_to": "none",
        "seed": args.seed,
    }
    eval_strategy_arg = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_kwargs[eval_strategy_arg] = "epoch" if val_ds is not None else "no"
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=CompletionOnlyDataCollator(tokenizer),
    )

    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Saved SFT adapter/model to %s", args.output_dir)


if __name__ == "__main__":
    main()
