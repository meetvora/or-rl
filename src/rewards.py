from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .code_exec import (
    answer_matches,
    detect_ortools_usage,
    execute_code,
    extract_numeric_answer,
    validate_script_features,
)

logger = logging.getLogger(__name__)

CODE_BLOCK_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.I | re.S)


def extract_tagged_code_blocks(text: str) -> list[str]:
    return [match.group(1).strip() for match in CODE_BLOCK_RE.finditer(text or "") if match.group(1).strip()]


def text_outside_code_blocks(text: str) -> str:
    return CODE_BLOCK_RE.sub("", text or "").strip()


def strip_prompt_prefix(text: str, prompt: Optional[str]) -> str:
    if not prompt:
        return text
    if text.startswith(prompt):
        return text[len(prompt) :].lstrip()
    prompt_tail = prompt[-1000:]
    if prompt_tail and text.startswith(prompt_tail):
        return text[len(prompt_tail) :].lstrip()
    tag_positions = [pos for tag in ("<THINK>", "<think>", "<CODE>", "<code>", "<ANSWER>", "<answer>") if (pos := text.find(tag)) > 0]
    if tag_positions:
        return text[min(tag_positions) :].lstrip()
    return text


@dataclass
class RewardConfig:
    answer_weight: float = 1.0
    exec_weight: float = 0.3
    ortools_weight: float = 0.2
    format_weight: float = 0.1
    script_validation_weight: float = 0.1
    syntax_weight: float = 1.0
    syntax_error_penalty: float = -2.0
    execution_timeout_penalty: float = -1.5
    execution_error_penalty: float = -1.0
    answer_tolerance: float = 1e-6
    code_timeout_seconds: int = 30
    generation_log_path: Optional[str] = None
    generation_preview_chars: int = 500
    enable_reference_code_similarity_reward: bool = False


def compute_reward(
    response: str,
    true_answer: float,
    config: RewardConfig,
    reference_code: Optional[str] = None,
) -> tuple[float, dict[str, Any]]:
    try:
        code_blocks = extract_tagged_code_blocks(response)
        outside_code = text_outside_code_blocks(response)
        has_prose_outside_code = bool(outside_code)
        code = code_blocks[0] if len(code_blocks) == 1 else None
        uses_ortools = detect_ortools_usage(code)
        checks = validate_script_features(code)
        syntax_reward = 1.0 if code and checks["valid_ast"] else config.syntax_error_penalty
        should_execute = bool(
            code
            and checks["valid_ast"]
            and checks["has_executable_statement"]
        )
        exec_result = execute_code(code, config.code_timeout_seconds) if should_execute else None
        stdout_answer = extract_numeric_answer(exec_result.stdout if exec_result else None)
        predicted = stdout_answer if exec_result and exec_result.success else None

        answer_reward = 1.0 if answer_matches(predicted, true_answer, config.answer_tolerance) else 0.0
        if not code_blocks:
            exec_reward = -1.0
        elif len(code_blocks) != 1:
            exec_reward = -0.7
        elif exec_result is None:
            exec_reward = -0.5
        elif exec_result.timeout:
            exec_reward = config.execution_timeout_penalty
        elif exec_result.success:
            exec_reward = 1.0
        else:
            exec_reward = config.execution_error_penalty
        ortools_reward = 1.0 if uses_ortools else 0.0
        format_reward = 1.0 if len(code_blocks) == 1 and not has_prose_outside_code else 0.0

        script_reward = sum(1.0 for passed in checks.values() if passed) / len(checks)

        reward = (
            config.answer_weight * answer_reward
            + config.exec_weight * exec_reward
            + config.ortools_weight * ortools_reward
            + config.format_weight * format_reward
            + config.script_validation_weight * script_reward
            + config.syntax_weight * syntax_reward
        )
        info = {
            "predicted_answer": predicted,
            "answer_reward": answer_reward,
            "exec_reward": exec_reward,
            "ortools_reward": ortools_reward,
            "format_reward": format_reward,
            "script_validation_reward": script_reward,
            "syntax_reward": syntax_reward,
            "valid_python_syntax": bool(code and checks["valid_ast"]),
            "code_block_count": len(code_blocks),
            "has_prose_outside_code": has_prose_outside_code,
            "execution_success": bool(exec_result and exec_result.success),
            "execution_timeout": bool(exec_result and exec_result.timeout),
            "execution_error": exec_result.error if exec_result else None,
            "execution_skipped": bool(code and not should_execute),
            "uses_ortools": uses_ortools,
        }
        return float(reward), info
    except Exception as exc:
        logger.exception("Reward computation failed")
        return 0.0, {"error": str(exc)}


def make_trl_reward_func(config: RewardConfig):
    def reward_func(completions, true_answer=None, reference_code=None, **kwargs):
        answers = true_answer or kwargs.get("true_answers") or []
        refs = reference_code or kwargs.get("reference_codes") or [None] * len(completions)
        prompts = kwargs.get("prompts") or [None] * len(completions)
        rewards = []
        generation_rows = []
        for i, completion in enumerate(completions):
            text = completion
            if isinstance(completion, list) and completion:
                text = completion[0].get("content", "")
            elif isinstance(completion, dict):
                text = completion.get("content", "")
            try:
                target = float(answers[i])
            except Exception:
                target = 0.0
            raw_text = str(text)
            prompt = prompts[i] if i < len(prompts) else None
            text = strip_prompt_prefix(raw_text, str(prompt) if prompt is not None else None)
            reward, info = compute_reward(text, target, config, refs[i] if i < len(refs) else None)
            rewards.append(reward)
            generation_rows.append(
                {
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "index": i,
                    "prompt": prompt,
                    "true_answer": target,
                    "completion": text,
                    "raw_completion": raw_text,
                    "stripped_prompt_prefix": text != raw_text,
                    "reward": reward,
                    "reward_info": info,
                }
            )
            preview = text.replace("\n", "\\n")
            if len(preview) > config.generation_preview_chars:
                preview = preview[: config.generation_preview_chars] + "..."
            logger.info(
                "generation_preview index=%d reward=%.4f chars=%d text=%s",
                i,
                reward,
                len(text),
                preview,
            )
        if config.generation_log_path:
            path = Path(config.generation_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                for row in generation_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info("logged_generations count=%d path=%s", len(generation_rows), path)
        return rewards

    return reward_func


def add_reward_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reward_answer_weight", type=float, default=1.0)
    parser.add_argument("--reward_exec_weight", type=float, default=0.3)
    parser.add_argument("--reward_ortools_weight", type=float, default=0.2)
    parser.add_argument("--reward_format_weight", type=float, default=0.1)
    parser.add_argument("--reward_script_validation_weight", type=float, default=0.1)
    parser.add_argument("--reward_syntax_weight", type=float, default=1.0)
    parser.add_argument("--syntax_error_penalty", type=float, default=-2.0)
    parser.add_argument("--execution_timeout_penalty", type=float, default=-1.5)
    parser.add_argument("--execution_error_penalty", type=float, default=-1.0)
    parser.add_argument("--answer_tolerance", type=float, default=1e-6)
    parser.add_argument("--code_timeout_seconds", type=int, default=30)
    parser.add_argument("--generation_log_path")
    parser.add_argument("--generation_preview_chars", type=int, default=500)
    parser.add_argument("--enable_reference_code_similarity_reward", default="false")


def config_from_args(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        answer_weight=args.reward_answer_weight,
        exec_weight=args.reward_exec_weight,
        ortools_weight=args.reward_ortools_weight,
        format_weight=args.reward_format_weight,
        script_validation_weight=args.reward_script_validation_weight,
        syntax_weight=args.reward_syntax_weight,
        syntax_error_penalty=args.syntax_error_penalty,
        execution_timeout_penalty=args.execution_timeout_penalty,
        execution_error_penalty=args.execution_error_penalty,
        answer_tolerance=args.answer_tolerance,
        code_timeout_seconds=args.code_timeout_seconds,
        generation_log_path=args.generation_log_path,
        generation_preview_chars=args.generation_preview_chars,
        enable_reference_code_similarity_reward=str(args.enable_reference_code_similarity_reward).lower()
        == "true",
    )
