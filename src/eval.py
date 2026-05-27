from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
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

FAILURE_PRIORITY = [
    "syntax_error",
    "execution_timeout",
    "runtime_error",
    "no_code_extracted",
    "answer_parse_failure",
    "wrong_answer",
    "execution_validation_failure",
    "unknown_failure",
]


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

def generate_response_candidates_batch(
    tokenizer,
    model,
    prompts: list[str],
    max_new_tokens: int,
    num_samples_per_prompt: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[list[str]]:
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
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["num_return_sequences"] = num_samples_per_prompt
        outputs = model.generate(**inputs, **gen_kwargs)

    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    responses: list[list[str]] = [[] for _ in prompts]
    if do_sample and num_samples_per_prompt > 1:
        for prompt_index, prompt_length in enumerate(prompt_lengths):
            base = prompt_index * num_samples_per_prompt
            for sample_index in range(num_samples_per_prompt):
                output = outputs[base + sample_index]
                generated = output[prompt_length:]
                responses[prompt_index].append(tokenizer.decode(generated, skip_special_tokens=True))
    else:
        for prompt_index, prompt_length in enumerate(prompt_lengths):
            output = outputs[prompt_index]
            generated = output[prompt_length:]
            responses[prompt_index].append(tokenizer.decode(generated, skip_special_tokens=True))
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
    example_id: str,
    prompt: str,
    true_answer: float,
    response: str,
    code_timeout_seconds: int,
    answer_tolerance: float,
    sample_index: int = 0,
    prompt_id: Optional[str] = None,
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
        "id": example_id,
        "prompt_id": prompt_id or example_id,
        "sample_index": sample_index,
        "prompt": prompt,
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
        "true_answer": true_answer,
        "answer_correct": answer_matches(extracted, true_answer, answer_tolerance),
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


def load_response_rows(path: str | Path, max_examples: int | None = None) -> list[dict]:
    rows: list[dict] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if max_examples is not None and len(rows) >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping bad JSON row in %s line %d", path, line_no)
                continue
            response = raw.get("response") or raw.get("baseline_response") or raw.get("completion")
            prompt = raw.get("prompt") or raw.get("problem_statement") or raw.get("question")
            prompt_id = raw.get("prompt_id") or raw.get("id") or raw.get("unique_id")
            sample_index = raw.get("sample_index", 0)
            true_answer = raw.get("true_answer")
            if true_answer is None and "answer_json" in raw:
                true_answer = raw.get("answer_json")
            if true_answer is None:
                true_answer = raw.get("answer")
            true_answer = true_answer if isinstance(true_answer, (int, float)) else extract_numeric_answer(str(true_answer) if true_answer is not None else None)
            if not prompt or response is None or true_answer is None:
                logger.warning("Skipping incomplete response row in %s line %d", path, line_no)
                continue
            rows.append(
                {
                    "id": str(raw.get("id") or raw.get("unique_id") or f"{path.stem}:{line_no}"),
                    "prompt_id": str(prompt_id) if prompt_id is not None else str(raw.get("id") or raw.get("unique_id") or f"{path.stem}:{line_no}"),
                    "sample_index": int(sample_index) if str(sample_index).isdigit() else sample_index,
                    "prompt": str(prompt),
                    "response": str(response),
                    "true_answer": true_answer,
                    "raw": raw,
                }
            )
    return rows


def summarize_prompt_failure_type(sample_results: list[dict]) -> str:
    failures = [r.get("failure_type") or "unknown_failure" for r in sample_results]
    for failure_type in FAILURE_PRIORITY:
        if failure_type in failures:
            return failure_type
    return "unknown_failure"


def compute_metrics(results: list[dict], pass_k: int = 5) -> dict:
    total_samples = len(results) or 1
    sample_failure_counts = Counter(r.get("failure_type") or "correct" for r in results)
    prompt_groups: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        prompt_groups[str(row.get("prompt_id") or row.get("id"))].append(row)
    prompt_failure_counts = Counter()
    prompt_passes = 0
    for _, rows in prompt_groups.items():
        ordered = sorted(rows, key=lambda r: int(r.get("sample_index", 0)))
        topk = ordered[:pass_k]
        if any(r.get("answer_correct") for r in topk):
            prompt_passes += 1
            prompt_failure_counts["passed"] += 1
        else:
            prompt_failure_counts[summarize_prompt_failure_type(topk)] += 1
    total_prompts = len(prompt_groups) or 1
    return {
        "num_samples": len(results),
        "num_prompts": len(prompt_groups),
        "pass_k": pass_k,
        "pass_at_k": prompt_passes / total_prompts,
        "answer_accuracy": sum(bool(r["answer_correct"]) for r in results) / total_samples,
        "code_execution_ratio": sum(bool(r["execution_success"]) for r in results) / total_samples,
        "ortools_usage_ratio": sum(bool(r["uses_ortools"]) for r in results) / total_samples,
        "executable_ortools_ratio": sum(
            bool(r["uses_ortools"] and r["execution_success"]) for r in results
        )
        / total_samples,
        "timeout_ratio": sum(bool(r["execution_timeout"]) for r in results) / total_samples,
        "parse_failure_ratio": sum(r["extracted_answer"] is None for r in results) / total_samples,
        "sample_failure_type_counts": dict(sorted(sample_failure_counts.items())),
        "sample_failure_type_ratios": {
            key: value / total_samples for key, value in sorted(sample_failure_counts.items())
        },
        "prompt_failure_type_counts": dict(sorted(prompt_failure_counts.items())),
        "prompt_failure_type_ratios": {
            key: value / total_prompts for key, value in sorted(prompt_failure_counts.items())
        },
        "failure_type_counts": dict(sorted(sample_failure_counts.items())),
        "failure_type_ratios": {
            key: value / total_samples for key, value in sorted(sample_failure_counts.items())
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
    if args.precompute_responses:
        examples = load_eval_examples(args.eval_dir, args.max_eval_examples)
        if args.partial_run:
            examples = examples[: args.batch_size]
            logger.info("Partial run enabled; generating first batch only (%d examples)", len(examples))
        tokenizer, model = load_model(args.model_name_or_path, args.load_in_4bit, args.load_in_8bit)
        total_batches = (len(examples) + args.batch_size - 1) // args.batch_size
        response_rows: list[dict] = []
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
            batch_candidates = generate_response_candidates_batch(
                tokenizer,
                model,
                prompts,
                args.max_new_tokens,
                num_samples_per_prompt=args.num_samples_per_prompt,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
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
                        "responses": responses,
                    }
                    for offset, (example, responses) in enumerate(zip(batch, batch_candidates))
                ],
                args.generation_log_path,
            )
            for example, candidate_list in zip(batch, batch_candidates):
                for sample_index, response in enumerate(candidate_list):
                    response_rows.append(
                        {
                            "id": example.id,
                            "prompt_id": example.id,
                            "sample_index": sample_index,
                            "prompt": example.prompt,
                            "true_answer": example.true_answer,
                            "response": response,
                        }
                    )
            logger.info("Finished generation batch %d/%d", batch_index, total_batches)

        write_jsonl(response_rows, args.responses_path)
        logger.info("Wrote precomputed responses to %s", args.responses_path)
        return response_rows, {"num_examples": len(response_rows)}

    if args.responses_path and Path(args.responses_path).exists():
        response_rows = load_response_rows(args.responses_path, args.max_eval_examples)
        if args.partial_run:
            max_rows = args.batch_size * max(1, args.num_samples_per_prompt)
            response_rows = response_rows[:max_rows]
            logger.info("Partial run enabled; scoring first prompt batch only (%d response rows)", len(response_rows))
        results = []
        for index, row in enumerate(response_rows, start=1):
            try:
                result = evaluate_response(
                    row["id"],
                    row["prompt"],
                    float(row["true_answer"]),
                    row["response"],
                    args.code_timeout_seconds,
                    args.answer_tolerance,
                    sample_index=int(row.get("sample_index", 0)),
                    prompt_id=str(row.get("prompt_id") or row["id"]),
                )
            except Exception as exc:
                logger.exception("Evaluation failed for %s", row["id"])
                result = {
                    "id": row["id"],
                    "prompt_id": str(row.get("prompt_id") or row["id"]),
                    "sample_index": int(row.get("sample_index", 0)),
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "extracted_code": None,
                    "uses_ortools": False,
                    "execution_success": False,
                    "execution_timeout": False,
                    "stdout": "",
                    "stderr": "",
                    "return_code": None,
                    "extracted_answer": None,
                    "true_answer": row["true_answer"],
                    "answer_correct": False,
                    "error": str(exc),
                }
                result["failure_type"] = classify_failure(result)

            results.append(result)

            if index % 10 == 0:
                logger.info("Evaluated %d/%d examples", index, len(response_rows))

        metrics = compute_metrics(results, pass_k=args.pass_k)
        write_jsonl(results, args.output_path)
        metrics_path = metrics_path_for(args.output_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info("Wrote %s and %s", args.output_path, metrics_path)
        return results, metrics

    examples = load_eval_examples(args.eval_dir, args.max_eval_examples)
    if args.partial_run:
        examples = examples[: args.batch_size]
        logger.info("Partial run enabled; evaluating first batch only (%d examples)", len(examples))
    tokenizer, model = load_model(args.model_name_or_path, args.load_in_4bit, args.load_in_8bit)
    results = []
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
        batch_candidates = generate_response_candidates_batch(
            tokenizer,
            model,
            prompts,
            args.max_new_tokens,
            num_samples_per_prompt=(args.num_samples_per_prompt if args.do_sample else 1),
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
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
                    "responses": responses,
                }
                for offset, (example, responses) in enumerate(zip(batch, batch_candidates))
            ],
            args.generation_log_path,
        )
        for example, candidate_list in zip(batch, batch_candidates):
            for sample_index, response in enumerate(candidate_list):
                try:
                    result = evaluate_response(
                        example.id,
                        example.prompt,
                        example.true_answer,
                        response,
                        args.code_timeout_seconds,
                        args.answer_tolerance,
                        sample_index=sample_index,
                        prompt_id=example.id,
                    )
                except Exception as exc:
                    logger.exception("Evaluation failed for %s", example.id)
                    result = {
                        "id": example.id,
                        "prompt_id": example.id,
                        "sample_index": sample_index,
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
        logger.info("Finished generation batch %d/%d", batch_index, total_batches)

    for index, _ in enumerate(results, start=1):
        if index % 10 == 0:
            logger.info("Evaluated %d/%d examples", index, len(results))

    metrics = compute_metrics(results, pass_k=args.pass_k)
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
    parser.add_argument("--responses_path", default="outputs/eval_responses.jsonl")
    parser.add_argument("--precompute_responses", action="store_true")
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
