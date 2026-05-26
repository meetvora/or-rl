from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from .code_exec import (
    answer_matches,
    detect_ortools_usage,
    execute_code,
    extract_numeric_answer,
    extract_python_code,
)
from .data import load_eval_examples, setup_logging
from .modeling import load_text_causal_lm, load_text_tokenizer, validate_quantization_args
from .prompts import GENERATION_INSTRUCTION

logger = logging.getLogger(__name__)


def load_model(model_name_or_path: str, load_in_4bit: bool = False, load_in_8bit: bool = False):
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise RuntimeError("Install transformers and torch to run model evaluation.") from exc
    tokenizer = load_text_tokenizer(model_name_or_path)
    model = load_text_causal_lm(model_name_or_path, load_in_4bit, load_in_8bit)
    model.eval()
    return tokenizer, model


def format_prompt(tokenizer, problem: str) -> str:
    user = f"{problem.strip()}\n\n{GENERATION_INSTRUCTION}"
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user

def generate_responses_batch(tokenizer, model, prompts: list[str], max_new_tokens: int) -> list[str]:
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    responses = []
    prompt_length = inputs["input_ids"].shape[-1]
    for i, output in enumerate(outputs):
        generated = output[prompt_length:]
        responses.append(tokenizer.decode(generated, skip_special_tokens=True))
    return responses

def generate_response(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_response(
    example,
    response: str,
    code_timeout_seconds: int,
    answer_tolerance: float,
) -> dict:
    code = extract_python_code(response)
    uses_ortools = detect_ortools_usage(code)
    exec_result = execute_code(code, code_timeout_seconds) if code else None
    text_answer = extract_numeric_answer(response)
    stdout_answer = extract_numeric_answer(exec_result.stdout if exec_result else None)
    extracted = text_answer if text_answer is not None else stdout_answer
    code_exec_answer = None
    if exec_result is not None and exec_result.success:
        code_exec_answer = exec_result.stdout
    result = {
        "id": example.id,
        "prompt": example.prompt,
        "response": response,
        "extracted_code": code,
        "uses_ortools": uses_ortools,
        "execution_success": bool(exec_result and exec_result.success),
        "execution_timeout": bool(exec_result and exec_result.timeout),
        "stdout": exec_result.stdout if exec_result else "",
        "stderr": exec_result.stderr if exec_result else "",
        "return_code": exec_result.return_code if exec_result else None,
        "extracted_answer": extracted,
        "code_exec_answer": code_exec_answer,
        "true_answer": example.true_answer,
        "answer_correct": answer_matches(extracted, example.true_answer, answer_tolerance),
        "error": exec_result.error if exec_result else ("no code extracted" if code is None else None),
    }
    result["failure_type"] = classify_failure(result)
    return result


def classify_failure(result: dict) -> Optional[str]:
    if result.get("answer_correct"):
        return None
    error = (result.get("error") or "").lower()
    stderr = (result.get("stderr") or "").lower()
    extracted_code = result.get("extracted_code")
    if not extracted_code:
        return "no_code_extracted"
    if "invalid python syntax" in error or "syntaxerror" in stderr:
        return "syntax_error"
    if result.get("execution_timeout"):
        return "execution_timeout"
    if result.get("return_code") not in (0, None):
        return "runtime_error"
    if result.get("execution_success") and result.get("extracted_answer") is None:
        return "answer_parse_failure"
    if result.get("execution_success"):
        return "wrong_answer"
    if error:
        return "execution_validation_failure"
    if result.get("extracted_answer") is None:
        return "answer_parse_failure"
    return "unknown_failure"


def compute_metrics(results: list[dict]) -> dict:
    total = len(results) or 1
    failure_counts = Counter(r.get("failure_type") or "correct" for r in results)
    return {
        "num_examples": len(results),
        "answer_accuracy": sum(bool(r["answer_correct"]) for r in results) / total,
        "code_execution_ratio": sum(bool(r["execution_success"]) for r in results) / total,
        "ortools_usage_ratio": sum(bool(r["uses_ortools"]) for r in results) / total,
        "executable_ortools_ratio": sum(
            bool(r["uses_ortools"] and r["execution_success"]) for r in results
        )
        / total,
        "timeout_ratio": sum(bool(r["execution_timeout"]) for r in results) / total,
        "parse_failure_ratio": sum(r["extracted_answer"] is None for r in results) / total,
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "failure_type_ratios": {
            key: value / total for key, value in sorted(failure_counts.items())
        },
    }


def metrics_path_for(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_suffix(path.suffix + ".metrics.json")


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_generation_log(rows: list[dict], path: str | Path | None) -> None:
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def chunks(xs, batch_size):
    for i in range(0, len(xs), batch_size):
        yield xs[i : i + batch_size]


def run_evaluation(args: argparse.Namespace) -> tuple[list[dict], dict]:
    examples = load_eval_examples(args.eval_dir, args.max_eval_examples)
    if args.partial_run:
        examples = examples[: args.batch_size]
        logger.info("Partial run enabled; evaluating first batch only (%d examples)", len(examples))
    tokenizer = model = None

    if not args.evaluate_cached_responses:
        tokenizer, model = load_model(args.model_name_or_path, args.load_in_4bit, args.load_in_8bit)

    results = []

    if args.evaluate_cached_responses:
        responses = [ex.metadata.get("baseline_response") or "" for ex in examples]
        append_generation_log(
            [
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "event": "cached_response",
                    "index": index,
                    "id": ex.id,
                    "response": response,
                }
                for index, (ex, response) in enumerate(zip(examples, responses), start=1)
            ],
            args.generation_log_path,
        )
    else:
        responses = []
        total_batches = (len(examples) + args.batch_size - 1) // args.batch_size
        for batch_index, batch in enumerate(chunks(examples, args.batch_size), start=1):
            start = (batch_index - 1) * args.batch_size + 1
            end = start + len(batch) - 1
            logger.info(
                "Generating responses for eval examples %d-%d/%d (batch %d/%d)",
                start,
                end,
                len(examples),
                batch_index,
                total_batches,
            )
            prompts = [format_prompt(tokenizer, ex.prompt) for ex in batch]
            batch_responses = generate_responses_batch(tokenizer, model, prompts, args.max_new_tokens)
            responses.extend(batch_responses)
            append_generation_log(
                [
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "event": "generated_response",
                        "batch_index": batch_index,
                        "batch_total": total_batches,
                        "index": start + offset,
                        "id": example.id,
                        "prompt": example.prompt,
                        "response": response,
                    }
                    for offset, (example, response) in enumerate(zip(batch, batch_responses))
                ],
                args.generation_log_path,
            )
            logger.info("Finished generation batch %d/%d", batch_index, total_batches)

    for index, (example, response) in enumerate(zip(examples, responses), start=1):
        try:
            result = evaluate_response(
                example,
                response,
                args.code_timeout_seconds,
                args.answer_tolerance,
            )
        except Exception as exc:
            logger.exception("Evaluation failed for %s", example.id)
            result = {
                "id": example.id,
                "prompt": example.prompt,
                "response": response,
                "extracted_code": None,
                "uses_ortools": False,
                "execution_success": False,
                "execution_timeout": False,
                "stdout": "",
                "stderr": "",
                "return_code": None,
                "extracted_answer": None,
                "true_answer": example.true_answer,
                "answer_correct": False,
                "error": str(exc),
            }
            result["failure_type"] = classify_failure(result)

        results.append(result)

        if index % 10 == 0:
            logger.info("Evaluated %d/%d examples", index, len(examples))

    metrics = compute_metrics(results)
    write_jsonl(results, args.output_path)
    metrics_path = metrics_path_for(args.output_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Wrote %s and %s", args.output_path, metrics_path)
    return results, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OR-Tools code-generation model.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--eval_dir", default="data/eval/complex_or_eval.jsonl")
    parser.add_argument("--output_path", default="outputs/baseline_eval.jsonl")
    parser.add_argument("--code_timeout_seconds", type=int, default=120)
    parser.add_argument("--answer_tolerance", type=float, default=1e-6)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--max_eval_examples", type=int)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--evaluate_cached_responses", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--generation_log_path")
    parser.add_argument("--partial_run", action="store_true")
    args = parser.parse_args()
    setup_logging(args.log_level)
    validate_quantization_args(args.load_in_4bit, args.load_in_8bit)
    run_evaluation(args)


if __name__ == "__main__":
    main()
