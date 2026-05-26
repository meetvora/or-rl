from __future__ import annotations

import argparse
import subprocess
import sys


def run_module(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module] + args
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFT, RLVR, or SFT then RLVR.")
    parser.add_argument("--stage", choices=["sft", "rlvr", "sft_then_rlvr"], required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--train_path", default="data/train/complex_or_variations.jsonl")
    parser.add_argument("--sft_output_dir", default="outputs/sft_adapter")
    parser.add_argument("--rlvr_output_dir", default="outputs/rlvr_adapter")
    parser.add_argument("--max_train_examples", default="32")
    parser.add_argument("--code_timeout_seconds", default="120")
    parser.add_argument("--answer_tolerance", default="1e-6")
    parser.add_argument("--load_in_4bit", default="false")
    parser.add_argument("--load_in_8bit", default="false")
    known, _ = parser.parse_known_args()

    if known.stage in {"sft", "sft_then_rlvr"}:
        run_module(
            "src.train_sft",
            [
                "--model_name_or_path",
                known.model_name_or_path,
                "--train_path",
                known.train_path,
                "--output_dir",
                known.sft_output_dir,
                "--max_train_examples",
                known.max_train_examples,
                "--load_in_4bit",
                known.load_in_4bit,
                "--load_in_8bit",
                known.load_in_8bit,
            ],
        )
    if known.stage in {"rlvr", "sft_then_rlvr"}:
        rlvr_model = known.sft_output_dir if known.stage == "sft_then_rlvr" else known.model_name_or_path
        run_module(
            "src.train_rlvr",
            [
                "--model_name_or_path",
                rlvr_model,
                "--train_path",
                known.train_path,
                "--output_dir",
                known.rlvr_output_dir,
                "--max_train_examples",
                known.max_train_examples,
                "--code_timeout_seconds",
                known.code_timeout_seconds,
                "--answer_tolerance",
                known.answer_tolerance,
                "--load_in_4bit",
                known.load_in_4bit,
                "--load_in_8bit",
                known.load_in_8bit,
            ],
        )


if __name__ == "__main__":
    main()
