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
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeAlias

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
    "getenv",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
}
FILE_MUTATION_CALLS = {
    "remove",
    "unlink",
    "rmdir",
    "removedirs",
    "rmtree",
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
PATH_DISCOVERY_CALLS = {"glob", "rglob", "iterdir", "walk", "scandir", "listdir"}

StaticPathValue: TypeAlias = str | dict[str, str]
ExecutionFailureCategory: TypeAlias = Literal[
    "policy_error", "syntax_error", "runtime_error", "timeout", "result_contract_error"
]


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
    policy_validated: bool = False
    parsed_result: bool = False
    failure_category: ExecutionFailureCategory | None = None


class PythonPolicyError(ValueError):
    """Raised when generated source violates the prototype AST policy."""


def _static_path_value(
    node: ast.AST | None, bindings: dict[str, StaticPathValue]
) -> StaticPathValue | None:
    """Resolve only source-level constants; never execute generated expressions."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Path"
        and node.args
    ):
        value = _static_path_value(node.args[0], bindings)
        return value if isinstance(value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _static_path_value(node.left, bindings)
        right = _static_path_value(node.right, bindings)
        if isinstance(left, str) and isinstance(right, str):
            return str(Path(left) / right)
        return None
    if isinstance(node, ast.Dict):
        resolved: dict[str, str] = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                return None
            resolved_value = _static_path_value(value, bindings)
            if not isinstance(resolved_value, str):
                return None
            resolved[key.value] = resolved_value
        return resolved
    if isinstance(node, ast.Subscript):
        mapping = _static_path_value(node.value, bindings)
        key = node.slice
        if (
            isinstance(mapping, dict)
            and isinstance(key, ast.Constant)
            and isinstance(key.value, str)
        ):
            return mapping.get(key.value)
    return None


def _static_path(
    node: ast.AST | None, bindings: dict[str, StaticPathValue]
) -> str | None:
    value = _static_path_value(node, bindings)
    return value if isinstance(value, str) else None


def _scope_for(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> ast.Module | ast.FunctionDef | ast.AsyncFunctionDef:
    """Return the lexical module/function scope containing an expression."""
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
    raise PythonPolicyError("Unable to determine generated-code scope")


def _scope_bindings(
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    before_line: int,
    parents: dict[ast.AST, ast.AST],
) -> dict[str, StaticPathValue]:
    """Resolve only unconditional earlier assignments in the lexical scope."""
    bindings: dict[str, StaticPathValue] = {}
    if not isinstance(scope, ast.Module):
        outer = _scope_for(scope, parents)
        bindings.update(
            _scope_bindings(
                outer,
                before_line=scope.lineno,
                parents=parents,
            )
        )
        for argument in (
            *scope.args.posonlyargs,
            *scope.args.args,
            *scope.args.kwonlyargs,
        ):
            bindings.pop(argument.arg, None)
        if scope.args.vararg:
            bindings.pop(scope.args.vararg.arg, None)
        if scope.args.kwarg:
            bindings.pop(scope.args.kwarg.arg, None)
    for statement in scope.body:
        if statement.lineno >= before_line:
            break
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        value = _static_path_value(statement.value, bindings)
        if value is not None:
            bindings[targets[0].id] = value
        else:
            bindings.pop(targets[0].id, None)
    return bindings


def _path_for_call(
    node: ast.AST | None,
    *,
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
) -> str | None:
    scope = _scope_for(call, parents)
    return _static_path(
        node,
        _scope_bindings(scope, before_line=call.lineno, parents=parents),
    )


def _within(path: Path, roots: list[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def validate_generated_code(
    code: str,
    *,
    run_directory: Path,
    allowed_files: list[Path],
    working_directory: Path | None = None,
) -> None:
    """Reject obvious network, process, environment, deletion, and path access."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Syntax errors are mechanical failures eligible for the single repair.
        return
    run_root = run_directory.resolve()
    working_root = (working_directory or run_directory).resolve()
    allowed = {path.resolve() for path in allowed_files}
    standard_library = set(getattr(sys, "stdlib_module_names", ()))
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "environ":
            raise PythonPolicyError("Environment-variable access is prohibited")
        if not isinstance(node, ast.Call):
            continue
        call_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else None
        )
        if call_name in BANNED_CALLS:
            raise PythonPolicyError(f"Prohibited call: {call_name}")
        if call_name in PATH_DISCOVERY_CALLS:
            raise PythonPolicyError(f"Path discovery is prohibited: {call_name}")
    path_variables: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
            node.value, ast.AST
        ):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "Path"
                ):
                    path_variables.add(target.id)

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
        if call_name in PATH_DISCOVERY_CALLS:
            raise PythonPolicyError(f"Path discovery is prohibited: {call_name}")
        if call_name in FILE_MUTATION_CALLS:
            receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
            filesystem_receiver = (
                not isinstance(node.func, ast.Attribute)
                or (
                    isinstance(receiver, ast.Name)
                    and receiver.id in {"os", "shutil", *path_variables}
                )
                or (
                    isinstance(receiver, ast.Call)
                    and isinstance(receiver.func, ast.Name)
                    and receiver.func.id == "Path"
                )
            )
            if filesystem_receiver:
                raise PythonPolicyError(f"Prohibited file operation: {call_name}")

        path_argument = node.args[0] if node.args else None
        if call_name == "open":
            is_path_method = isinstance(node.func, ast.Attribute)
            supplied = (
                _path_for_call(node.func.value, call=node, parents=parents)
                if is_path_method
                else _path_for_call(path_argument, call=node, parents=parents)
            )
            mode_index = 0 if is_path_method else 1
            mode = (
                _path_for_call(node.args[mode_index], call=node, parents=parents)
                if len(node.args) > mode_index
                else "r"
            )
            for keyword in node.keywords:
                if keyword.arg == "mode":
                    mode = _path_for_call(keyword.value, call=node, parents=parents)
            if supplied is not None:
                resolved = (working_root / supplied).resolve()
                if any(flag in (mode or "r") for flag in "wax+"):
                    if not _within(resolved, [run_root]):
                        raise PythonPolicyError("Writing outside the run directory")
                elif resolved not in allowed:
                    raise PythonPolicyError("Reading a file that was not staged")
            else:
                raise PythonPolicyError("Dynamic file paths are prohibited")
        elif call_name in WRITE_METHODS:
            supplied = (
                _path_for_call(node.func.value, call=node, parents=parents)
                if isinstance(node.func, ast.Attribute)
                and call_name in {"write_text", "write_bytes"}
                else _path_for_call(path_argument, call=node, parents=parents)
            )
            if supplied is not None:
                resolved = (working_root / supplied).resolve()
                if not _within(resolved, [run_root]):
                    raise PythonPolicyError("Writing outside the run directory")
            else:
                raise PythonPolicyError("Dynamic file paths are prohibited")
        elif call_name in READ_METHODS:
            supplied = (
                _path_for_call(node.func.value, call=node, parents=parents)
                if isinstance(node.func, ast.Attribute)
                and call_name in {"read_text", "read_bytes"}
                else _path_for_call(path_argument, call=node, parents=parents)
            )
            if supplied is not None:
                resolved = (working_root / supplied).resolve()
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

    def __init__(
        self,
        *,
        timeout_seconds: float = 30,
        output_limit: int = 20_000,
        progress_callback: Callable[[str], None] | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.output_limit = output_limit
        self.progress_callback = progress_callback

    def run(
        self,
        *,
        code: str,
        goal_directory: Path,
        allowed_files: list[Path],
        version: int,
        working_directory: Path | None = None,
    ) -> PythonExecutionResult:
        goal_directory.mkdir(parents=True, exist_ok=True)
        execution_directory = (working_directory or goal_directory).resolve()
        script_path = goal_directory / f"generated_code_v{version}.py"
        script_path.write_text(code, encoding="utf-8")
        started = time.perf_counter()
        if self.progress_callback:
            self.progress_callback("code execution — starting...")
        stdout = ""
        stderr = ""
        exit_code: int | None = None
        timed_out = False
        error_message: str | None = None
        parsed_result: dict[str, JsonValue] = {}
        policy_validated = False
        parsed_result_available = False
        failure_category: ExecutionFailureCategory | None = None

        try:
            validate_generated_code(
                code,
                run_directory=goal_directory,
                allowed_files=allowed_files,
                working_directory=execution_directory,
            )
            policy_validated = True
            environment = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONHASHSEED": "0",
                "HOME": str(goal_directory),
                "LANG": "C.UTF-8",
            }
            completed = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=execution_directory,
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
                failure_category = (
                    "syntax_error" if "SyntaxError" in stderr else "runtime_error"
                )
            else:
                try:
                    parsed_result = _parse_json_result(stdout, goal_directory)
                    parsed_result_available = True
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    ValueError,
                ) as error:
                    error_message = f"Invalid JSON output: {error}"
                    failure_category = "result_contract_error"
        except PythonPolicyError as error:
            error_message = f"PythonPolicyError: {error}"
            failure_category = "policy_error"
        except subprocess.TimeoutExpired as error:
            timed_out = True
            stdout = _bounded_text(error.stdout, self.output_limit)
            stderr = _bounded_text(error.stderr, self.output_limit)
            error_message = (
                f"Execution timed out after {self.timeout_seconds:g} seconds"
            )
            failure_category = "timeout"

        duration = time.perf_counter() - started
        if self.progress_callback:
            self.progress_callback(f"code execution — completed in {duration:.1f}s")
        stdout_path = goal_directory / f"stdout_v{version}.txt"
        stderr_path = goal_directory / f"stderr_v{version}.txt"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        (goal_directory / "stdout.txt").write_text(stdout, encoding="utf-8")
        (goal_directory / "stderr.txt").write_text(stderr, encoding="utf-8")
        artifact_paths = [str(script_path), str(stdout_path), str(stderr_path)]
        generated_outputs = goal_directory / "generated_outputs"
        for output_path in sorted(generated_outputs.glob("*")):
            if output_path.is_file():
                artifact_paths.append(str(output_path))
        try:
            generated_outputs.rmdir()
        except OSError:
            # Preserve the directory whenever a script actually saved anything.
            pass
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
            policy_validated=policy_validated,
            parsed_result=parsed_result_available,
            failure_category=failure_category,
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
