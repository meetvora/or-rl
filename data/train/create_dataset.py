import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


STATEMENT_RE = re.compile(r"<problem_statement>\n?(.*?)\n?</problem_statement>", re.S)
SCRIPT_RE = re.compile(r"<or_tools_script><!\[CDATA\[\n?(.*?)\n?\]\]></or_tools_script>", re.S)
FILENAME_RE = re.compile(r"model_output_(\d+)\.xml$")
DEFAULT_COLAB_ARROW_OUT = "complex_or_dataset.arrow"
DEFAULT_COLAB_PARQUET_OUT = "complex_or_dataset.parquet"
REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
CODE_START = "<CODE>"
CODE_END = "</CODE>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an Arrow dataset from generated ComplexOR variations."
    )
    parser.add_argument(
        "--base-dir",
        default="data/train/ComplexOR",
        help="Directory containing per-problem variation folders.",
    )
    parser.add_argument(
        "--arrow-out",
        default="data/train/complex_or_variations.arrow",
        help="Output Arrow IPC/Feather file.",
    )
    parser.add_argument(
        "--parquet-out",
        default="data/train/complex_or_variations.parquet",
        help="Output Parquet file.",
    )
    parser.add_argument(
        "--jsonl-out",
        default="data/train/complex_or_variations.jsonl",
        help="Optional JSONL output for easy inspection.",
    )
    parser.add_argument(
        "--colab-arrow-out",
        default=DEFAULT_COLAB_ARROW_OUT,
        help=(
            "Compatibility Feather file for Qwen_Finetuning.ipynb. "
            "Use an empty string to disable."
        ),
    )
    parser.add_argument(
        "--colab-parquet-out",
        default=DEFAULT_COLAB_PARQUET_OUT,
        help=(
            "Compatibility Parquet file matching --colab-arrow-out. "
            "Use an empty string to disable."
        ),
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Do not execute embedded OR-Tools scripts to compute answer_json.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Seconds allowed for each embedded OR-Tools script.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of variation files to process.",
    )
    return parser.parse_args()


def parse_variation_xml(path):
    text = path.read_text(encoding="utf-8")
    statement_match = STATEMENT_RE.search(text)
    script_match = SCRIPT_RE.search(text)

    if not statement_match:
        raise ValueError("missing problem_statement block")
    if not script_match:
        raise ValueError("missing or_tools_script CDATA block")

    return statement_match.group(1).strip(), script_match.group(1).strip()


def execute_script(script, timeout):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".py", delete=False
    ) as script_file:
        script_path = Path(script_file.name)
        script_file.write(script)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "timeout", None, "", f"timed out after {timeout} seconds"
    finally:
        script_path.unlink(missing_ok=True)

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode:
        return "runtime_error", None, stdout, stderr

    try:
        answer = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return "bad_json", None, stdout, str(exc)

    return "ok", answer, stdout, stderr


def first_numeric(value):
    if value is None or isinstance(value, bool):
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
                return first_numeric(json.loads(text))
            except Exception:
                return None
    if isinstance(value, list):
        for item in value:
            parsed = first_numeric(item)
            if parsed is not None:
                return parsed
    if isinstance(value, dict):
        for key in ("answer", "objective", "value", "result", "output"):
            if key in value:
                parsed = first_numeric(value[key])
                if parsed is not None:
                    return parsed
    return None


def format_answer(value):
    if value is None:
        return None
    value = float(value)
    return str(int(value)) if value.is_integer() else format(value, ".12g")


def make_training_prompt(problem_statement):
    return (
        f"{problem_statement.strip()}\n\n"
        "Solve the operations research problem. Return exactly one response with: "
        f"a concise reasoning section between {REASONING_START} and {REASONING_END}, "
        f"Python OR-Tools code between {CODE_START} and {CODE_END}, "
        f"and only the final numeric answer between {SOLUTION_START} and {SOLUTION_END}."
    )


def make_target_completion(or_tools_script, answer):
    if answer is None:
        return None
    return (
        f"{REASONING_START}Set up the optimization model, solve it with OR-Tools, "
        f"and report the objective value.{REASONING_END}\n"
        f"{CODE_START}\n{or_tools_script.strip()}\n{CODE_END}\n"
        f"{SOLUTION_START}{answer}{SOLUTION_END}"
    )


def iter_variation_files(base_dir):
    for problem_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        variations_dir = problem_dir / "variations"
        if not variations_dir.is_dir():
            continue
        for xml_path in sorted(variations_dir.glob("model_output_*.xml")):
            match = FILENAME_RE.match(xml_path.name)
            if match:
                yield problem_dir.name, int(match.group(1)), xml_path


def build_rows(args):
    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"base directory not found: {base_dir}")

    rows = []
    for problem_type, variation_index, xml_path in iter_variation_files(base_dir):
        if args.limit is not None and len(rows) >= args.limit:
            break

        problem_statement, or_tools_script = parse_variation_xml(xml_path)
        execution_status = "not_run"
        answer_json = None
        raw_output = ""
        execution_error = ""

        if not args.no_execute:
            execution_status, answer_json, raw_output, execution_error = execute_script(
                or_tools_script, args.timeout
            )

        answer_json_text = json.dumps(answer_json) if answer_json is not None else None
        correct_answer = first_numeric(answer_json)
        answer = format_answer(correct_answer)
        rows.append(
            {
                "id": f"{problem_type}:{variation_index:03d}",
                "problem_type": problem_type,
                "variation_index": variation_index,
                "source_path": str(xml_path),
                "problem_statement": problem_statement,
                "prompt": problem_statement,
                "training_prompt": make_training_prompt(problem_statement),
                "or_tools_script": or_tools_script,
                "reference_code": or_tools_script,
                "answer_json": answer_json_text,
                "output": answer_json_text,
                "answer": answer,
                "correct_answer": correct_answer,
                "target_completion": make_target_completion(or_tools_script, answer),
                "raw_output": raw_output,
                "execution_status": execution_status,
                "execution_error": execution_error,
            }
        )
        if len(rows) % 50 == 0:
            print(f"processed={len(rows)}", flush=True)

    return rows


def write_jsonl(rows, path):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_feather(table, path):
    if not path:
        return
    import pyarrow.feather as feather

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(table, path)


def write_parquet(table, path):
    if not path:
        return
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def main():
    args = parse_args()
    import pyarrow as pa

    rows = build_rows(args)
    table = pa.Table.from_pylist(rows)

    write_feather(table, args.arrow_out)
    write_parquet(table, args.parquet_out)
    write_feather(table, args.colab_arrow_out)
    write_parquet(table, args.colab_parquet_out)
    write_jsonl(rows, args.jsonl_out)

    status_counts = {}
    for row in rows:
        status_counts[row["execution_status"]] = status_counts.get(row["execution_status"], 0) + 1

    print(f"rows={len(rows)}")
    print(f"arrow={args.arrow_out}")
    print(f"parquet={args.parquet_out}")
    if args.colab_arrow_out:
        print(f"colab_arrow={args.colab_arrow_out}")
    if args.colab_parquet_out:
        print(f"colab_parquet={args.colab_parquet_out}")
    if args.jsonl_out:
        print(f"jsonl={args.jsonl_out}")
    print(f"execution_status={status_counts}")


if __name__ == "__main__":
    main()
