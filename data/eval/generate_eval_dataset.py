from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_jsonl(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    return pd.read_json(path, lines=True)


def sample_eval_rows(
    df: pd.DataFrame,
    sample_fraction: float,
    seed: int,
    group_columns: list[str],
) -> pd.DataFrame:
    if sample_fraction <= 0 or sample_fraction >= 1:
        return df
    missing = [column for column in group_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Cannot sample by missing column(s): {missing}")
    return df.groupby(group_columns, group_keys=False).apply(
        lambda group: group.sample(
            n=max(1, int(len(group) * sample_fraction)),
            random_state=seed,
        )
    )


def build_row(raw: dict[str, Any], row_index: int) -> tuple[dict[str, Any] | None, str | None]:
    example_id = raw.get("unique_id") or raw.get("id") or f"canonical:{row_index}"
    problem_statement = raw.get("question") or raw.get("problem_statement") or raw.get("prompt")
    true_answer = raw.get("answer") if "answer" in raw else raw.get("true_answer")

    if not problem_statement:
        return None, "missing_problem_statement"
    if true_answer is None:
        return None, "missing_true_answer"

    return (
        {
            "id": str(example_id),
            "problem_statement": str(problem_statement),
            "answer_json": [true_answer],
        },
        None,
    )


def generate_eval_dataset(
    input_path: str | Path,
    output_path: str | Path,
    sample_fraction: float = 0.2,
    seed: int = 42,
    group_columns: list[str] | None = None,
) -> dict[str, Any]:
    group_columns = group_columns or ["problem_class", "size"]
    df = load_jsonl(input_path)
    sampled = sample_eval_rows(df, sample_fraction, seed, group_columns)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for row_index, raw in enumerate(sampled.to_dict(orient="records")):
        row, skip_reason = build_row(raw, row_index)
        if row is None:
            reason = skip_reason or "unknown"
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        rows.append(row)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_rows": len(df),
        "rows": len(rows),
        "sample_fraction": sample_fraction,
        "seed": seed,
        "group_columns": group_columns,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate train-schema eval JSONL directly from canonical OPTEngine JSONL."
    )
    parser.add_argument("--input_path", default="data/eval/canonical.jsonl")
    parser.add_argument("--output_path", default="data/eval/complex_or_eval.jsonl")
    parser.add_argument("--sample_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group_columns",
        nargs="+",
        default=["problem_class", "size"],
        help="Columns used for stratified sampling. Set --sample_fraction 1 to disable sampling.",
    )
    args = parser.parse_args()
    summary = generate_eval_dataset(
        args.input_path,
        args.output_path,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
        group_columns=args.group_columns,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
