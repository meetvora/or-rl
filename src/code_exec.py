from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*[:=]\s*(%s)" % NUMBER_RE.pattern, re.I)
ANSWER_TAG_RE = re.compile(r"<answer>\s*(%s)\s*</answer>" % NUMBER_RE.pattern, re.I)
PY_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.I | re.S)
CODE_TAG_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.I | re.S)
ORTOOLS_TOKENS = (
    "ortools",
    "pywraplp",
    "cp_model",
    "CpModel",
    "CpSolver",
    "Solver.CreateSolver",
    "linear_solver",
    "sat.python.cp_model",
)


def _is_placeholder_code(code: str) -> bool:
    stripped = code.strip().lower()
    return stripped in {"# code", "code", "python code", "# reference or-tools code"}


@dataclass
class ExecutionResult:
    success: bool
    timeout: bool
    stdout: str
    stderr: str
    return_code: Optional[int]
    error: Optional[str] = None


def extract_python_code(text: str) -> Optional[str]:
    if not text:
        return None
    blocks = [match.group(1).strip() for match in PY_BLOCK_RE.finditer(text)]
    blocks.extend(match.group(1).strip() for match in CODE_TAG_RE.finditer(text))
    blocks = [b for b in blocks if b and not _is_placeholder_code(b)]
    pythonish = [b for b in blocks if "import " in b or "from " in b or "print(" in b]
    if pythonish:
        return max(pythonish, key=len)
    if blocks:
        return max(blocks, key=len)
    if "from ortools" in text or "import ortools" in text:
        return text.strip()
    return None


def detect_ortools_usage(code: Optional[str]) -> bool:
    if not code:
        return False
    if any(token in code for token in ORTOOLS_TOKENS):
        return True
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.startswith("ortools") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("ortools"):
                return True
    return False


def extract_numeric_answer(*texts: Optional[str]) -> Optional[float]:
    for text in texts:
        if not text:
            continue
        for regex in (FINAL_ANSWER_RE, ANSWER_TAG_RE):
            match = regex.search(text)
            if match:
                return float(match.group(1))
        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, list) and parsed:
                return float(parsed[0])
            if isinstance(parsed, (int, float)):
                return float(parsed)
        except Exception:
            pass
        numbers = NUMBER_RE.findall(text)
        if numbers:
            return float(numbers[-1])
    return None


def validate_script_features(code: Optional[str]) -> dict[str, bool]:
    checks = {
        "valid_ast": False,
        "has_executable_statement": False,
        "uses_ortools": detect_ortools_usage(code),
        "defines_variables": False,
        "defines_constraints": False,
        "defines_objective_or_solver": False,
        "calls_solver": False,
        "prints_numeric_candidate": False,
    }
    if not code:
        return checks
    try:
        tree = ast.parse(code)
        checks["valid_ast"] = True
    except SyntaxError:
        return checks
    checks["has_executable_statement"] = any(
        not isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass))
        for node in tree.body
    )
    checks["defines_variables"] = any(
        token in code for token in ("NewIntVar", "NewBoolVar", "NumVar", "IntVar", "BoolVar")
    )
    checks["defines_constraints"] = any(token in code for token in ("model.Add", ".Add(", "AddConstraint"))
    checks["defines_objective_or_solver"] = any(
        token in code for token in ("Minimize", "Maximize", "Objective", "CpSolver", "CreateSolver")
    )
    checks["calls_solver"] = any(token in code for token in (".Solve(", "solver.Solve("))
    checks["prints_numeric_candidate"] = "print(" in code or "Final answer" in code
    return checks


def execute_code(code: Optional[str], timeout_seconds: int = 120) -> ExecutionResult:
    if not code:
        return ExecutionResult(False, False, "", "", None, "no code to execute")
    checks = validate_script_features(code)
    if not checks["valid_ast"]:
        return ExecutionResult(False, False, "", "", None, "invalid python syntax")
    if not checks["has_executable_statement"]:
        return ExecutionResult(False, False, "", "", None, "no executable statements")
    with tempfile.TemporaryDirectory(prefix="or_rl_exec_") as tmp:
        tmp_path = Path(tmp)
        script_path = tmp_path / "solution.py"
        sitecustomize = tmp_path / "sitecustomize.py"
        script_path.write_text(code, encoding="utf-8")
        sitecustomize.write_text(
            "import socket\n"
            "def _blocked(*args, **kwargs):\n"
            "    raise RuntimeError('network access disabled during evaluation')\n"
            "socket.socket = _blocked\n"
            "socket.create_connection = _blocked\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONNOUSERSITE"] = "1"
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmp,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                success=False,
                timeout=True,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                return_code=None,
                error=f"timed out after {timeout_seconds} seconds",
            )
        except Exception as exc:
            return ExecutionResult(False, False, "", "", None, str(exc))
    if proc.returncode != 0:
        return ExecutionResult(
            success=False,
            timeout=False,
            stdout=proc.stdout,
            stderr=proc.stderr,
            return_code=proc.returncode,
            error="nonzero return code",
        )
    if checks["uses_ortools"]:
        if not checks["defines_constraints"]:
            return ExecutionResult(
                False,
                False,
                proc.stdout,
                proc.stderr,
                proc.returncode,
                "ortools script completed without constraints",
            )
        if not checks["defines_objective_or_solver"]:
            return ExecutionResult(
                False,
                False,
                proc.stdout,
                proc.stderr,
                proc.returncode,
                "ortools script completed without objective or solver setup",
            )
        if not checks["calls_solver"]:
            return ExecutionResult(
                False,
                False,
                proc.stdout,
                proc.stderr,
                proc.returncode,
                "ortools script completed without calling the solver",
            )
    if not checks["prints_numeric_candidate"] and not proc.stdout.strip():
        return ExecutionResult(
            False,
            False,
            proc.stdout,
            proc.stderr,
            proc.returncode,
            "script completed without producing an answer candidate",
        )
    return ExecutionResult(
        success=True,
        timeout=False,
        stdout=proc.stdout,
        stderr=proc.stderr,
        return_code=proc.returncode,
        error=None,
    )


def answer_matches(predicted: Optional[float], true_answer: float, tolerance: float = 1e-6) -> bool:
    if predicted is None:
        return False
    absolute = abs(predicted - true_answer)
    relative = absolute / max(1.0, abs(true_answer))
    return absolute <= tolerance or relative <= tolerance
