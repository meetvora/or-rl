import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_FILES = ("description.txt", "sample.json")
SCRIPT_RE = re.compile(r"<or_tools_script><!\[CDATA\[\n?(.*?)\n?\]\]></or_tools_script>", re.S)
TRAINING_EXAMPLE_RE = re.compile(
    r"<training_example\s+index=\"(\d+)\">\n?(.*?)\n?</training_example>",
    re.S,
)
USAGE_LIMIT_MARKER = "You've hit your usage limit"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate model_output.xml files for ComplexOR problem directories "
            "by running the local Codex CLI."
        )
    )
    parser.add_argument("--base-dir", default="ComplexOR", help="Problem directory root.")
    parser.add_argument(
        "--prompt",
        default="prompt_template.md",
        help="System instruction file for Codex.",
    )
    parser.add_argument(
        "--output-name",
        default="model_output.xml",
        help=(
            "Output filename written inside each problem directory when "
            "--variations is 1."
        ),
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=50,
        help=(
            "Number of diverse variations to generate per problem. Use 1 for "
            "the legacy single model_output.xml behavior."
        ),
    )
    parser.add_argument(
        "--variations-dir",
        default="variations",
        help="Subdirectory for generated variation XML files.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="One-based variation index to start from.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional list of problem directory names to generate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs that already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show what would be generated without running Codex.",
    )
    parser.add_argument(
        "--codex-command",
        default="codex",
        help="Codex CLI executable to run.",
    )
    parser.add_argument(
        "--codex-model",
        help="Optional Codex model name passed through to `codex exec --model`.",
    )
    parser.add_argument(
        "--codex-sandbox",
        default="read-only",
        choices=("read-only", "workspace-write", "danger-full-access"),
        help="Sandbox mode passed through to `codex exec`.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Maximum seconds to wait for each `codex exec` run.",
    )
    parser.add_argument(
        "--verify-timeout",
        type=int,
        default=30,
        help="Maximum seconds to wait when executing a generated OR-Tools script.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Additional Codex attempts when generated XML fails verification.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of variations to request from Codex in one call.",
    )
    parser.add_argument(
        "--no-verify-generated",
        action="store_true",
        help="Do not execute generated OR-Tools scripts before saving them.",
    )
    return parser.parse_args()


def read_text(path):
    return path.read_text(encoding="utf-8")


def problem_dirs(base_dir, only=None):
    only_set = set(only or [])
    for path in sorted(base_dir.iterdir()):
        if path.is_dir() and (not only_set or path.name in only_set):
            yield path


def missing_required_files(problem_dir):
    return [name for name in REQUIRED_FILES if not (problem_dir / name).is_file()]


def model_spec_path(problem_dir):
    gt_model_path = problem_dir / "gt_model.txt"
    if gt_model_path.is_file():
        return gt_model_path

    candidates = sorted(
        path
        for path in problem_dir.glob("*.py")
        if path.name != "code_example.py" and not path.name.startswith("__")
    )
    if candidates:
        return candidates[0]
    return gt_model_path


def build_user_payload(problem_dir):
    desc = read_text(problem_dir / "description.txt")
    model_path = model_spec_path(problem_dir)
    gt = read_text(model_path)
    sample = read_text(problem_dir / "sample.json")
    return (
        f"DESCRIPTION:\n{desc}\n\n"
        f"MODEL_SPEC ({model_path.name}):\n{gt}\n\n"
        f"SAMPLE:\n{sample}"
    )


def build_codex_prompt(system_prompt, user_payload, variation_indices=None, total_variations=1):
    variation_indices = variation_indices or []
    if not variation_indices:
        variation_context = "VARIATION_MODE: false\n"
    elif len(variation_indices) == 1:
        variation_index = variation_indices[0]
        variation_context = (
            "VARIATION_MODE: true\n"
            "BATCH_MODE: false\n"
            f"VARIATION_INDEX: {variation_index}\n"
            f"TOTAL_VARIATIONS_REQUESTED: {total_variations}\n"
            "DIVERSITY_REQUIREMENT: Generate a different setting, entity names, "
            "wording, structure, tone, specificity level, units where appropriate, "
            "dimensions where feasible, and numeric data from other variations for "
            "this same base problem. Some variations should read like a natural "
            "chatbot request rather than a textbook problem statement.\n"
        )
    else:
        indices = ", ".join(str(index) for index in variation_indices)
        variation_context = (
            "VARIATION_MODE: true\n"
            "BATCH_MODE: true\n"
            f"VARIATION_INDICES: {indices}\n"
            f"TOTAL_VARIATIONS_REQUESTED: {total_variations}\n"
            "DIVERSITY_REQUIREMENT: Generate a different setting, entity names, "
            "wording, structure, tone, specificity level, units where appropriate, "
            "dimensions where feasible, and numeric data from other variations for "
            "this same base problem. Some variations should read like a natural "
            "chatbot request rather than a textbook problem statement.\n"
            "BATCH_OUTPUT_REQUIREMENT: Return exactly one <training_example "
            "index=\"N\">...</training_example> block for each requested index N. "
            "Inside each training_example block, put exactly one complete "
            "<problem_statement>...</problem_statement> and exactly one complete "
            "<or_tools_script><![CDATA[...]]></or_tools_script>. Do not put text "
            "outside the training_example blocks.\n"
        )

    return (
        "Follow these system instructions exactly and return only the requested XML. "
        "Do not edit files, run shell commands, or add commentary.\n\n"
        f"{system_prompt}\n\n"
        f"{variation_context}\n"
        "Input data:\n"
        f"{user_payload}"
    )


def output_paths(problem_dir, args):
    if args.variations == 1:
        yield None, problem_dir / args.output_name
        return

    variation_dir = problem_dir / args.variations_dir
    end_index = args.start_index + args.variations
    for variation_index in range(args.start_index, end_index):
        filename = f"model_output_{variation_index:03d}.xml"
        yield variation_index, variation_dir / filename


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run_codex(args, prompt_text):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".xml", delete=False
    ) as output_file:
        output_path = Path(output_file.name)

    cmd = [
        args.codex_command,
        "-C",
        str(Path.cwd()),
        "--sandbox",
        args.codex_sandbox,
        "--ask-for-approval",
        "never",
    ]
    if args.codex_model:
        cmd.extend(["--model", args.codex_model])

    cmd.extend(
        [
            "exec",
            "--skip-git-repo-check",
            "--output-last-message",
            str(output_path),
            "-",
        ]
    )

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(prompt_text, timeout=args.timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                stdout, stderr = proc.communicate()
            raise RuntimeError(f"codex exec timed out after {args.timeout} seconds") from exc
        result = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        output_text = read_text(output_path).strip()
    finally:
        output_path.unlink(missing_ok=True)

    if result.returncode:
        details = (result.stderr or result.stdout).strip()
        if USAGE_LIMIT_MARKER in details:
            raise RuntimeError("codex exec failed: usage limit reached")
        if len(details) > 4000:
            details = details[:4000] + "\n... [truncated]"
        raise RuntimeError(f"codex exec failed with exit code {result.returncode}: {details}")

    if not output_text:
        output_text = result.stdout.strip()
    if not output_text:
        raise RuntimeError("codex exec completed without producing output")

    return output_text


def verify_generated_xml(xml_text, args):
    match = SCRIPT_RE.search(xml_text)
    if not match:
        raise RuntimeError("generated XML is missing an or_tools_script CDATA block")

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".py", delete=False
    ) as script_file:
        script_path = Path(script_file.name)
        script_file.write(match.group(1))

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            text=True,
            capture_output=True,
            check=False,
            timeout=args.verify_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"generated OR-Tools script timed out after {args.verify_timeout} seconds"
        ) from exc
    finally:
        script_path.unlink(missing_ok=True)

    if result.returncode:
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"generated OR-Tools script failed: {details}")

    try:
        json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"generated OR-Tools script did not print valid JSON: {result.stdout.strip()}"
        ) from exc


def split_generated_output(output_text, variation_indices):
    if not variation_indices:
        return {None: output_text.strip()}

    if len(variation_indices) == 1 and "<training_example" not in output_text:
        return {variation_indices[0]: output_text.strip()}

    found = {}
    for match in TRAINING_EXAMPLE_RE.finditer(output_text):
        index = int(match.group(1))
        found[index] = match.group(2).strip()

    missing = sorted(set(variation_indices) - set(found))
    extra = sorted(set(found) - set(variation_indices))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing variation indices: {missing}")
        if extra:
            details.append(f"unexpected variation indices: {extra}")
        raise RuntimeError("; ".join(details))

    return found


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    prompt_path = Path(args.prompt)

    if not base_dir.is_dir():
        print(f"Base directory not found: {base_dir}", file=sys.stderr)
        return 2
    if not prompt_path.is_file():
        print(f"Prompt file not found: {prompt_path}", file=sys.stderr)
        return 2

    system_prompt = read_text(prompt_path)
    selected_dirs = list(problem_dirs(base_dir, args.only))

    if args.only:
        found = {path.name for path in selected_dirs}
        missing = sorted(set(args.only) - found)
        if missing:
            print(f"Requested problem directories not found: {', '.join(missing)}")

    if not selected_dirs:
        print("No problem directories found.")
        return 1

    failures = 0
    generated = 0
    skipped = 0

    if args.variations < 1:
        print("--variations must be at least 1", file=sys.stderr)
        return 2
    if args.start_index < 1:
        print("--start-index must be at least 1", file=sys.stderr)
        return 2
    if args.batch_size < 1:
        print("--batch-size must be at least 1", file=sys.stderr)
        return 2

    for problem_dir in selected_dirs:
        missing_files = missing_required_files(problem_dir)

        if missing_files:
            failures += 1
            print(
                f"Failed on {problem_dir.name}: missing {', '.join(missing_files)}",
                file=sys.stderr,
            )
            continue

        pending = []
        for variation_index, output_path in output_paths(problem_dir, args):
            if output_path.exists() and not args.overwrite:
                skipped += 1
                print(f"Skipping {output_path}: already exists")
                continue

            if args.dry_run:
                print(f"Would generate {output_path} with {args.codex_command} exec")
                continue

            pending.append((variation_index, output_path))

        if args.dry_run or not pending:
            continue

        user_payload = build_user_payload(problem_dir)
        effective_batch_size = 1 if args.variations == 1 else args.batch_size

        for batch in chunks(pending, effective_batch_size):
            variation_indices = [index for index, _ in batch if index is not None]
            if not variation_indices:
                variation_indices = []
            output_by_index = {}

            try:
                prompt_text = build_codex_prompt(
                    system_prompt,
                    user_payload,
                    variation_indices=variation_indices,
                    total_variations=args.variations,
                )
                for attempt in range(args.retries + 1):
                    first_path = batch[0][1]
                    last_path = batch[-1][1]
                    batch_label = (
                        str(first_path)
                        if first_path == last_path
                        else f"{first_path} .. {last_path}"
                    )
                    print(
                        f"Generating {batch_label}"
                        f"{'' if attempt == 0 else f' (retry {attempt})'}...",
                        flush=True,
                    )
                    candidate = run_codex(args, prompt_text)
                    if args.no_verify_generated:
                        output_by_index = split_generated_output(candidate, variation_indices)
                        break
                    try:
                        output_by_index = split_generated_output(candidate, variation_indices)
                        for xml_text in output_by_index.values():
                            verify_generated_xml(xml_text, args)
                        break
                    except Exception as exc:
                        if attempt >= args.retries:
                            raise
                        print(f"Verification failed for {batch_label}: {exc}", file=sys.stderr)

                for variation_index, output_path in batch:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    key = variation_index if variation_index is not None else None
                    output_path.write_text(output_by_index[key] + "\n", encoding="utf-8")
                    generated += 1
                    print(f"Generated {output_path}", flush=True)
            except Exception as exc:
                failures += len(batch)
                first_path = batch[0][1]
                last_path = batch[-1][1]
                batch_label = (
                    str(first_path) if first_path == last_path else f"{first_path} .. {last_path}"
                )
                print(f"Failed on {batch_label}: {exc}", file=sys.stderr)

    print(
        f"Done. generated={generated}, skipped={skipped}, failures={failures}, "
        f"dry_run={args.dry_run}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
