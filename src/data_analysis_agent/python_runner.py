"""Constrained prototype runner for Executor-generated local Python scripts.

This is defense in depth for a local prototype, not a production security sandbox.
It combines a conservative AST policy, an isolated working directory, a minimal
environment, a timeout, and bounded captured output. It is not an OS-level jail.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue

ALLOWED_THIRD_PARTY = {"pandas", "numpy", "scipy"}
BANNED_IMPORTS = {
    "requests",
    "urllib",
    "socket",
    "subprocess",
    "http",
    "ftplib",
    "telnetlib",
}
BANNED_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "breakpoint",
    "system",
    "popen",
    "spawnl",
    "spawnlp",
    "remove",
    "unlink",
    "rmdir",
    "removedirs",
    "rmtree",
    "getenv",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "rename",
    "replace",
    "truncate",
}
WRITE_METHODS = {
    "write_text",
    "write_bytes",
    "to_csv",
    "to_json",
    "to_parquet",
    "save",
    "savefig",
}
READ_METHODS = {"read_csv", "read_table", "read_text", "read_bytes", "loadtxt"}


class PythonExecutionResult(BaseModel):
    """Captured factual result and artifact paths for one script version."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    version: int
    exit_code: int | None
    stdout: str
    stderr: str
    result: dict[str, JsonValue]
    error: str | None = None
    duration_seconds: float
    timed_out: bool = False
    script_path: str
    artifact_paths: list[str] = Field(default_factory=list)


class PythonPolicyError(ValueError):
    """Raised when generated source violates the prototype AST policy."""


def _literal_path(
    node: ast.AST | None, constants: dict[str, str] | None = None
) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and constants is not None:
        return constants.get(node.id)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Path"
        and node.args
    ):
        return _literal_path(node.args[0], constants)
    return None


def _within(path: Path, roots: list[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def validate_generated_code(
    code: str, *, run_directory: Path, allowed_files: list[Path]
) -> None:
    """Reject obvious network, process, environment, deletion, and path access."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Syntax errors are mechanical failures eligible for the single repair.
        return
    run_root = run_directory.resolve()
    allowed = {path.resolve() for path in allowed_files}
    standard_library = set(getattr(sys, "stdlib_module_names", ()))
    constants: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
            node.value, ast.AST
        ):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = _literal_path(node.value, constants)
            if value is not None:
                for target in targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = value

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                root = name.split(".", 1)[0]
                if root in BANNED_IMPORTS:
                    raise PythonPolicyError(f"Prohibited import: {root}")
                if root not in standard_library and root not in ALLOWED_THIRD_PARTY:
                    raise PythonPolicyError(f"Library is not allowed: {root}")
        if isinstance(node, ast.Attribute) and node.attr == "environ":
            raise PythonPolicyError("Environment-variable access is prohibited")
        if not isinstance(node, ast.Call):
            continue
        call_name = None
        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            call_name = node.func.attr
        if call_name in BANNED_CALLS:
            raise PythonPolicyError(f"Prohibited call: {call_name}")

        path_argument = node.args[0] if node.args else None
        if call_name == "open":
            is_path_method = isinstance(node.func, ast.Attribute)
            supplied = (
                _literal_path(node.func.value, constants)
                if is_path_method
                else _literal_path(path_argument, constants)
            )
            mode_index = 0 if is_path_method else 1
            mode = (
                _literal_path(node.args[mode_index], constants)
                if len(node.args) > mode_index
                else "r"
            )
            for keyword in node.keywords:
                if keyword.arg == "mode":
                    mode = _literal_path(keyword.value, constants)
            if supplied is not None:
                resolved = (run_root / supplied).resolve()
                if any(flag in (mode or "r") for flag in "wax+"):
                    if not _within(resolved, [run_root]):
                        raise PythonPolicyError("Writing outside the run directory")
                elif resolved not in allowed:
                    raise PythonPolicyError("Reading a file that was not staged")
            else:
                raise PythonPolicyError("Dynamic file paths are prohibited")
        elif call_name in WRITE_METHODS:
            supplied = (
                _literal_path(node.func.value, constants)
                if isinstance(node.func, ast.Attribute)
                and call_name in {"write_text", "write_bytes"}
                else _literal_path(path_argument, constants)
            )
            if supplied is not None:
                resolved = (run_root / supplied).resolve()
                if not _within(resolved, [run_root]):
                    raise PythonPolicyError("Writing outside the run directory")
            else:
                raise PythonPolicyError("Dynamic file paths are prohibited")
        elif call_name in READ_METHODS:
            supplied = (
                _literal_path(node.func.value, constants)
                if isinstance(node.func, ast.Attribute)
                and call_name in {"read_text", "read_bytes"}
                else _literal_path(path_argument, constants)
            )
            if supplied is not None:
                resolved = Path(supplied).resolve()
                if resolved not in allowed:
                    raise PythonPolicyError("Reading a file that was not staged")
            else:
                raise PythonPolicyError("Dynamic file paths are prohibited")


def _parse_json_result(stdout: str, goal_directory: Path) -> dict[str, JsonValue]:
    result_path = goal_directory / "generated_outputs" / "result.json"
    if result_path.is_file():
        raw = result_path.read_text(encoding="utf-8")
    else:
        lines = [line for line in stdout.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Generated script produced no JSON result")
        raw = lines[-1]
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Generated result must be one JSON object")
    # Pydantic's JsonValue validation is applied by PythonExecutionResult.
    return parsed


def _bounded_text(value: str | bytes | None, limit: int) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return (value or "")[-limit:]


class LocalPythonRunner:
    """Save and run generated code in a per-goal artifact directory."""

    def __init__(self, *, timeout_seconds: float = 30, output_limit: int = 20_000):
        self.timeout_seconds = timeout_seconds
        self.output_limit = output_limit

    def run(
        self,
        *,
        code: str,
        goal_directory: Path,
        allowed_files: list[Path],
        version: int,
    ) -> PythonExecutionResult:
        goal_directory.mkdir(parents=True, exist_ok=True)
        (goal_directory / "generated_outputs").mkdir(exist_ok=True)
        script_path = goal_directory / f"generated_code_v{version}.py"
        script_path.write_text(code, encoding="utf-8")
        started = time.perf_counter()
        stdout = ""
        stderr = ""
        exit_code: int | None = None
        timed_out = False
        error_message: str | None = None
        parsed_result: dict[str, JsonValue] = {}

        try:
            validate_generated_code(
                code,
                run_directory=goal_directory,
                allowed_files=allowed_files,
            )
            environment = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONHASHSEED": "0",
                "HOME": str(goal_directory),
                "LANG": "C.UTF-8",
            }
            completed = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=goal_directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            exit_code = completed.returncode
            stdout = completed.stdout[-self.output_limit :]
            stderr = completed.stderr[-self.output_limit :]
            if exit_code != 0:
                error_message = f"Generated Python exited with code {exit_code}"
            else:
                try:
                    parsed_result = _parse_json_result(stdout, goal_directory)
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    ValueError,
                ) as error:
                    error_message = f"Invalid JSON output: {error}"
        except PythonPolicyError as error:
            error_message = f"PythonPolicyError: {error}"
        except subprocess.TimeoutExpired as error:
            timed_out = True
            stdout = _bounded_text(error.stdout, self.output_limit)
            stderr = _bounded_text(error.stderr, self.output_limit)
            error_message = (
                f"Execution timed out after {self.timeout_seconds:g} seconds"
            )

        duration = time.perf_counter() - started
        stdout_path = goal_directory / f"stdout_v{version}.txt"
        stderr_path = goal_directory / f"stderr_v{version}.txt"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        (goal_directory / "stdout.txt").write_text(stdout, encoding="utf-8")
        (goal_directory / "stderr.txt").write_text(stderr, encoding="utf-8")
        artifact_paths = [str(script_path), str(stdout_path), str(stderr_path)]
        for output_path in sorted((goal_directory / "generated_outputs").glob("*")):
            if output_path.is_file():
                artifact_paths.append(str(output_path))
        result = PythonExecutionResult(
            success=error_message is None,
            version=version,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            result=parsed_result,
            error=error_message,
            duration_seconds=duration,
            timed_out=timed_out,
            script_path=str(script_path),
            artifact_paths=artifact_paths,
        )
        execution_path = goal_directory / f"execution_result_v{version}.json"
        execution_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        result.artifact_paths.append(str(execution_path))
        (goal_directory / "execution_result.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        return result


def write_execution_metadata(
    *,
    goal_directory: Path,
    run_id: str,
    goal_id: str,
    executions: list[PythonExecutionResult],
    strategy_reason: str,
) -> Path:
    """Record reproducibility metadata without registering code as a capability."""
    successful = next((item for item in reversed(executions) if item.success), None)
    metadata = {
        "run_id": run_id,
        "goal_id": goal_id,
        "script_paths": [item.script_path for item in executions],
        "creation_timestamp": datetime.now(UTC).isoformat(),
        "execution_succeeded": successful is not None,
        "repair_required": len(executions) > 1,
        "output_paths": successful.artifact_paths if successful else [],
        "execution_duration_seconds": sum(item.duration_seconds for item in executions),
        "concise_strategy_reason": strategy_reason,
    }
    last_version = executions[-1].version
    path = goal_directory / f"artifact_metadata_v{last_version}.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (goal_directory / "artifact_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return path
