from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from time import perf_counter

from .data import load_train_examples, setup_logging
from .modeling import (
    enable_trainable_adapter_parameters,
    load_text_causal_lm,
    load_text_tokenizer,
    prepare_lora_model_if_needed,
    str2bool,
    validate_quantization_args,
)
from .prompts import GENERATION_INSTRUCTION
from .rewards import add_reward_args, config_from_args, make_trl_reward_func

logger = logging.getLogger(__name__)


def format_grpo_prompt(tokenizer, problem: str) -> str:
    user = f"{problem.strip()}\n\n{GENERATION_INSTRUCTION}"
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user


def log_phase(name: str, started_at: float) -> float:
    now = perf_counter()
    logger.info("startup_phase=%s elapsed_seconds=%.2f", name, now - started_at)
    return now


def log_trainable_parameters(model) -> None:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    percent = 100 * trainable / total if total else 0
    logger.info(
        "trainable_parameters=%d total_parameters=%d trainable_percent=%.4f",
        trainable,
        total,
        percent,
    )
    if trainable == 0:
        raise RuntimeError("No trainable parameters found after trainer/model setup.")


def enable_kv_cache_and_disable_checkpointing(model) -> None:
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()


def build_logging_grpo_trainer(base_cls):
    class LoggingGRPOTrainer(base_cls):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            started_at = perf_counter()
            logger.info(
                "train_compute_loss_start global_step=%s microbatch_size=%d num_generations=%s max_completion_length=%s",
                self.state.global_step,
                len(inputs),
                getattr(self, "num_generations", "unknown"),
                getattr(self, "max_completion_length", "unknown"),
            )
            loss = super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
            logger.info(
                "train_compute_loss_end global_step=%s elapsed_seconds=%.2f",
                self.state.global_step,
                perf_counter() - started_at,
            )
            return loss

    return LoggingGRPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="RLVR fine-tuning with TRL GRPO.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--train_path", default="data/train/complex_or_variations.jsonl")
    parser.add_argument("--output_dir", default="outputs/rlvr_adapter")
    parser.add_argument("--max_train_examples", type=int, default=32)
    parser.add_argument(
        "--resume_step",
        type=int,
        default=0,
        help="Start training from this normalized training-example index.",
    )
    parser.add_argument("--save_steps", type=int, default=5)
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=0,
        help="Maximum checkpoints to keep. Use 0 to keep all checkpoints.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default="",
        help="Optional Trainer checkpoint path, for example outputs/rlvr_adapter/checkpoint-25.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_lora", type=str2bool, default=True)
    parser.add_argument("--load_in_4bit", type=str2bool, default=False)
    parser.add_argument("--load_in_8bit", type=str2bool, default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--log_level", default="INFO")
    add_reward_args(parser)
    args = parser.parse_args()
    save_total_limit = args.save_total_limit if args.save_total_limit > 0 else None
    resume_from_checkpoint = args.resume_from_checkpoint or None
    setup_logging(args.log_level)
    validate_quantization_args(args.load_in_4bit, args.load_in_8bit)
    phase_start = perf_counter()

    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        GRPOTrainer = build_logging_grpo_trainer(GRPOTrainer)
        if args.use_lora:
            from peft import LoraConfig
    except Exception as exc:
        raise RuntimeError(
            "RLVR training requires a TRL version with GRPOTrainer plus datasets and peft. "
            "Install the pinned requirements; if your TRL release lacks GRPO, upgrade TRL rather "
            "than silently switching algorithms."
        ) from exc
    phase_start = log_phase("imports", phase_start)

    all_examples = load_train_examples(args.train_path)
    random.Random(args.seed).shuffle(all_examples)
    if args.resume_step < 0:
        raise ValueError("--resume_step must be non-negative.")
    if args.resume_step >= len(all_examples):
        raise ValueError(
            f"--resume_step={args.resume_step} leaves no training examples "
            f"from {len(all_examples)} loaded examples."
        )
    examples = all_examples[args.resume_step :]
    if args.max_train_examples is not None:
        examples = examples[: args.max_train_examples]
    logger.info(
        "training_data_window start_index=%d examples=%d total_loaded=%d max_train_examples=%s",
        args.resume_step,
        len(examples),
        len(all_examples),
        args.max_train_examples,
    )
    logger.info("training_data_window_first_ids=%s", [ex.id for ex in examples[:5]])
    phase_start = log_phase("load_train_examples", phase_start)
    tokenizer = load_text_tokenizer(args.model_name_or_path)
    phase_start = log_phase("load_tokenizer", phase_start)
    rows = []
    for ex in examples:
        rows.append(
            {
                "prompt": format_grpo_prompt(tokenizer, ex.prompt),
                "true_answer": ex.true_answer,
                "reference_code": ex.reference_code,
            }
        )
    dataset = Dataset.from_list(rows)
    phase_start = log_phase("build_dataset", phase_start)
    reward_func = make_trl_reward_func(config_from_args(args))
    model = load_text_causal_lm(
        args.model_name_or_path,
        args.load_in_4bit,
        args.load_in_8bit,
        trainable_adapter=True,
    )
    phase_start = log_phase("load_model", phase_start)
    model = prepare_lora_model_if_needed(model, args.use_lora, args.load_in_4bit, args.load_in_8bit)
    enable_kv_cache_and_disable_checkpointing(model)
    phase_start = log_phase("prepare_model", phase_start)
    if getattr(model, "peft_config", None) is not None:
        reenabled = enable_trainable_adapter_parameters(model)
        logger.info("adapter_trainable_parameters_reenabled=%d", reenabled)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    peft_config = None
    if args.use_lora and not hasattr(model, "peft_config"):
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
    phase_start = log_phase("build_peft_config", phase_start)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        optim=args.optim,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=save_total_limit,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        num_generations=args.num_generations,
        gradient_checkpointing=False,
        use_cache=True,
        report_to="none",
        seed=args.seed,
        temperature=0.7
    )
    phase_start = log_phase("build_training_args", phase_start)

    kwargs = {
        "model": model,
        "reward_funcs": reward_func,
        "args": training_args,
        "train_dataset": dataset,
        "peft_config": peft_config,
        "processing_class": tokenizer,
    }
    try:
        trainer = GRPOTrainer(**kwargs)
    except TypeError as exc:
        try:
            kwargs.pop("processing_class", None)
            trainer = GRPOTrainer(**kwargs)
        except TypeError:
            raise RuntimeError(
                "The installed TRL GRPOTrainer API is incompatible with this script. "
                "Pin or upgrade TRL and rerun; no fallback model or trainer was used."
            ) from exc
    if getattr(trainer, "generation_config", None) is not None:
        trainer.generation_config.use_cache = True
        logger.info("trainer_generation_config_use_cache=%s", trainer.generation_config.use_cache)
    phase_start = log_phase("build_trainer", phase_start)
    log_trainable_parameters(trainer.model)
    logger.info("resume_from_checkpoint=%s", resume_from_checkpoint or "none")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    logger.info("Saved RLVR adapter/model to %s", args.output_dir)


if __name__ == "__main__":
    main()
