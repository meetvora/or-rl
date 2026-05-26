from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

from .prompts import GENERATION_INSTRUCTION

@dataclass
class NormalizedExample:
    id: str
    prompt: str
    reference_code: Optional[str]
    true_answer: float
    metadata: dict[str, Any]

    def asdict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "reference_code": self.reference_code,
            "true_answer": self.true_answer,
            "metadata": self.metadata,
        }


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _first_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            try:
                return _first_numeric(json.loads(text))
            except Exception:
                return None
    if isinstance(value, list):
        for item in value:
            parsed = _first_numeric(item)
            if parsed is not None:
                return parsed
    if isinstance(value, dict):
        for key in ("answer", "objective", "value", "result", "output"):
            if key in value:
                parsed = _first_numeric(value[key])
                if parsed is not None:
                    return parsed
    return None


def _message_content(record: dict[str, Any]) -> str:
    try:
        return record.get("choices", [{}])[0].get("message", {}).get("content") or ""
    except Exception:
        return ""


def _message_reasoning(record: dict[str, Any]) -> str:
    try:
        return record.get("choices", [{}])[0].get("message", {}).get("reasoning") or ""
    except Exception:
        return ""


def load_train_examples(
    path: str | Path = "data/train/complex_or_variations.jsonl",
    max_examples: Optional[int] = None,
) -> list[NormalizedExample]:
    path = Path(path)
    examples: list[NormalizedExample] = []
    skipped: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if max_examples is not None and len(examples) >= max_examples:
                break
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skipped["bad_json"] = skipped.get("bad_json", 0) + 1
                continue
            prompt = raw.get("problem_statement") or raw.get("prompt") or raw.get("question")
            reference_code = raw.get("or_tools_script") or raw.get("reference_code")
            true_answer = _first_numeric(raw.get("answer_json", raw.get("answer")))
            if not prompt:
                skipped["missing_prompt"] = skipped.get("missing_prompt", 0) + 1
                continue
            if true_answer is None:
                skipped["missing_true_answer"] = skipped.get("missing_true_answer", 0) + 1
                continue
            example_id = str(raw.get("id") or f"{path.stem}:{line_no}")
            metadata = {"source": str(path), "line_no": line_no, "raw": raw}
            examples.append(
                NormalizedExample(
                    id=example_id,
                    prompt=str(prompt),
                    reference_code=str(reference_code) if reference_code else None,
                    true_answer=true_answer,
                    metadata=metadata,
                )
            )
    if skipped:
        logger.warning("Skipped train examples from %s: %s", path, skipped)
    logger.info("Loaded %d train examples from %s", len(examples), path)
    return examples


def iter_eval_files(eval_dir: str | Path) -> list[Path]:
    path = Path(eval_dir)
    if path.is_file():
        return [path]
    if not path.exists():
        fallback = Path("responses")
        if fallback.is_dir():
            logger.warning("%s not found; falling back to %s", path, fallback)
            path = fallback
        else:
            raise FileNotFoundError(f"eval path not found: {eval_dir}")
    return sorted(p for p in path.rglob("*") if p.suffix in {".json", ".jsonl"})


def _load_eval_jsonl(path: Path, max_remaining: Optional[int]) -> list[NormalizedExample]:
    examples: list[NormalizedExample] = []
    skipped: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if max_remaining is not None and len(examples) >= max_remaining:
                break
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skipped["bad_json"] = skipped.get("bad_json", 0) + 1
                continue
            prompt = raw.get("question") or raw.get("prompt") or raw.get("problem_statement")
            true_answer = _first_numeric(raw.get("answer") or raw.get("true_answer"))
            if not prompt or true_answer is None:
                skipped["missing_required"] = skipped.get("missing_required", 0) + 1
                continue
            example_id = str(raw.get("unique_id") or raw.get("id") or f"{path.stem}:{line_no}")
            examples.append(
                NormalizedExample(
                    id=example_id,
                    prompt=str(prompt),
                    reference_code=None,
                    true_answer=true_answer,
                    metadata={"source": str(path), "line_no": line_no, "raw": raw},
                )
            )
    if skipped:
        logger.warning("Skipped eval JSONL rows from %s: %s", path, skipped)
    return examples


def _load_eval_response(path: Path) -> Optional[NormalizedExample]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Skipping %s: %s", path, exc)
        return None
    metadata = raw.get("metadata") or {}
    prompt = (
        metadata.get("original_prompt")
        or raw.get("prompt")
        or raw.get("question")
    )
    true_answer = _first_numeric(metadata.get("true_answer", raw.get("answer")))
    if not prompt or true_answer is None:
        logger.warning("Skipping %s: missing prompt or true answer", path)
        return None
    example_id = str(metadata.get("unique_id") or raw.get("id") or path.stem)
    return NormalizedExample(
        id=example_id,
        prompt=str(prompt),
        reference_code=None,
        true_answer=true_answer,
        metadata={
            "source": str(path),
            "raw": raw,
            "baseline_response": _message_content(raw),
            "baseline_reasoning": _message_reasoning(raw),
            "baseline_extracted_answer": metadata.get("extracted_answer"),
        },
    )


def load_eval_examples(
    eval_dir: str | Path = "data/eval/responses",
    max_examples: Optional[int] = None,
) -> list[NormalizedExample]:
    examples: list[NormalizedExample] = []
    for path in iter_eval_files(eval_dir):
        remaining = None if max_examples is None else max_examples - len(examples)
        if remaining is not None and remaining <= 0:
            break
        if path.suffix == ".jsonl":
            examples.extend(_load_eval_jsonl(path, remaining))
        else:
            ex = _load_eval_response(path)
            if ex is not None:
                examples.append(ex)
    logger.info("Loaded %d eval examples from %s", len(examples), eval_dir)
    return examples


def split_train_validation(
    examples: list[NormalizedExample],
    eval_split_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[NormalizedExample], list[NormalizedExample]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    val_size = int(round(len(shuffled) * eval_split_ratio))
    if len(shuffled) > 1 and eval_split_ratio > 0:
        val_size = max(1, min(val_size, len(shuffled) - 1))
    return shuffled[val_size:], shuffled[:val_size]


def build_sft_text(example: NormalizedExample) -> str:
    code = example.reference_code or "# No reference code available"
    return (
        f"{example.prompt.strip()}\n\n{GENERATION_INSTRUCTION}\n\n"
        "<CODE>\n"
        f"{code.strip()}\n"
        "</CODE>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect normalized OR-RL datasets.")
    parser.add_argument("--train_path", default="data/train/complex_or_variations.jsonl")
    parser.add_argument("--eval_dir", default="data/eval/responses")
    parser.add_argument("--eval_split_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_examples", type=int)
    parser.add_argument("--max_eval_examples", type=int)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    train = load_train_examples(args.train_path, args.max_train_examples)
    train_split, val_split = split_train_validation(train, args.eval_split_ratio, args.seed)
    eval_examples = load_eval_examples(args.eval_dir, args.max_eval_examples)
    print(
        json.dumps(
            {
                "train_total": len(train),
                "train_split": len(train_split),
                "validation_split": len(val_split),
                "eval_total": len(eval_examples),
                "sample_train": train[0].asdict() if train else None,
                "sample_eval": eval_examples[0].asdict() if eval_examples else None,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
