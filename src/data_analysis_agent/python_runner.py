"""Constrained prototype runner for Executor-generated local Python scripts.

This is defense in depth for a local prototype, not a production security sandbox.
It combines a conservative AST policy, an isolated working directory, a minimal
environment, a timeout, and bounded captured output. It is not an OS-level jail.
"""

from __future__ import annotations

import ast
import io
import json
import os
import subprocess
import sys
import time
import tokenize
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from data_analysis_agent.schemas import ExecutionFailureCategory

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
ResultMode: TypeAlias = Literal["agent_variable", "legacy_stdout"]
RESULT_VARIABLE = "__agent_result__"
RESULT_CONTRACT_MARKER = "AGENT_RESULT_CONTRACT_ERROR:"


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
    deterministic_result_recovery_attempted: bool = False


class PythonPolicyError(ValueError):
    """Raised when generated source violates the prototype AST policy."""


class PythonCodeContractError(ValueError):
    """Raised before execution when generated source violates its strict contract."""


def validate_generated_python_contract(code: str) -> None:
    """Require executable, comment-free module source with an explicit result.

    This check deliberately happens before sandbox policy validation and before a
    generated file is written.  Textual mentions of the result variable are not
    sufficient: the assignment must be an unconditional statement in the module
    body so the trusted runner can reliably retrieve it after ``runpy.run_path``.
    """
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                raise PythonCodeContractError(
                    "generated source contains prohibited comments"
                )
    except tokenize.TokenError as error:
        raise PythonCodeContractError(
            f"generated source does not tokenize: {error}"
        ) from error
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise PythonCodeContractError(
            f"generated source does not parse: {error.msg}"
        ) from error
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == RESULT_VARIABLE
                for target in statement.targets
            ):
                return
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == RESULT_VARIABLE
            and statement.value is not None
        ):
            return
    raise PythonCodeContractError(
        f"missing executable module-level assignment to {RESULT_VARIABLE}"
    )


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


def _parse_legacy_stdout_result(
    stdout: str, goal_directory: Path
) -> dict[str, JsonValue]:
    """Compatibility-only parser for the explicitly legacy one-shot approach."""
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


def _trusted_runner_source(*, script_path: Path, result_path: Path) -> str:
    """Build a runner-owned epilogue; generated source is never interpolated."""
    return f'''import json
import runpy
import sys
from datetime import date, datetime, time
from pathlib import Path

RESULT_MARKER = {RESULT_CONTRACT_MARKER!r}


def fail(message):
    sys.stderr.write(RESULT_MARKER + " " + message + "\\n")
    raise SystemExit(86)


def json_default(value):
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if type(value).__module__.split(".", 1)[0] == "numpy" and hasattr(value, "item"):
        scalar = value.item()
        if scalar is None or isinstance(scalar, (str, int, float, bool)):
            return scalar
    raise TypeError(f"unsupported result value type: {{type(value).__name__}}")


namespace = runpy.run_path({str(script_path)!r}, run_name="__main__")
if {RESULT_VARIABLE!r} not in namespace:
    fail("result variable missing: {RESULT_VARIABLE}")
result = namespace[{RESULT_VARIABLE!r}]
if not isinstance(result, dict):
    fail("result is not an object")
try:
    serialized = json.dumps(
        result,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=json_default,
    )
except TypeError:
    fail("result is not JSON-serializable")
except ValueError:
    fail("result contains NaN or Infinity")
try:
    Path({str(result_path)!r}).write_text(serialized, encoding="utf-8")
except (OSError, UnicodeError):
    fail("trusted serialization failed")
'''


def _parse_trusted_result(goal_directory: Path) -> dict[str, JsonValue]:
    """Parse only the file written by the trusted runner epilogue."""
    result_path = goal_directory / "generated_outputs" / "result.json"
    if not result_path.is_file():
        raise ValueError("trusted result file missing")
    try:
        parsed = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("trusted serialization failed") from error
    if not isinstance(parsed, dict):
        raise ValueError("trusted result is not an object")
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
        result_mode: ResultMode = "agent_variable",
    ) -> PythonExecutionResult:
        goal_directory.mkdir(parents=True, exist_ok=True)
        execution_directory = (
            (working_directory or goal_directory).resolve()
            if result_mode == "legacy_stdout"
            else goal_directory.resolve()
        )
        script_path = goal_directory / f"generated_code_v{version}.py"
        generated_outputs = goal_directory / "generated_outputs"
        result_path = generated_outputs / "result.json"
        runner_path: Path | None = None
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
        deterministic_result_recovery_attempted = False

        try:
            if result_mode == "agent_variable":
                validate_generated_python_contract(code)
            validate_generated_code(
                code,
                run_directory=goal_directory,
                allowed_files=allowed_files,
                working_directory=execution_directory,
            )
            policy_validated = True
            script_path.write_text(code, encoding="utf-8")
            try:
                result_path.unlink()
            except FileNotFoundError:
                pass
            if result_mode == "agent_variable":
                generated_outputs.mkdir(parents=True, exist_ok=True)
                runner_path = goal_directory / f"runner_entry_v{version}.py"
                runner_path.write_text(
                    _trusted_runner_source(
                        script_path=script_path.resolve(),
                        result_path=result_path.resolve(),
                    ),
                    encoding="utf-8",
                )
            deterministic_result_recovery_attempted = result_mode == "agent_variable"
            environment = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONHASHSEED": "0",
                "HOME": str(goal_directory),
                "LANG": "C.UTF-8",
            }
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str((runner_path or script_path).resolve()),
                ],
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
                contract_error = next(
                    (
                        line.split(RESULT_CONTRACT_MARKER, 1)[1].strip()
                        for line in stderr.splitlines()
                        if line.startswith(RESULT_CONTRACT_MARKER)
                    ),
                    None,
                )
                if contract_error is not None:
                    error_message = f"ResultContractError: {contract_error}"
                    failure_category = "result_contract_error"
                else:
                    error_message = f"Generated Python exited with code {exit_code}"
                    failure_category = (
                        "syntax_error" if "SyntaxError" in stderr else "runtime_error"
                    )
            else:
                try:
                    parsed_result = (
                        _parse_trusted_result(goal_directory)
                        if result_mode == "agent_variable"
                        else _parse_legacy_stdout_result(stdout, goal_directory)
                    )
                    parsed_result_available = True
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    ValueError,
                ) as error:
                    error_message = f"ResultContractError: {error}"
                    failure_category = "result_contract_error"
        except PythonPolicyError as error:
            error_message = f"PythonPolicyError: {error}"
            failure_category = "policy_error"
        except PythonCodeContractError as error:
            error_message = f"PythonCodeContractError: {error}"
            failure_category = "generation_contract_error"
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
        artifact_paths = [str(stdout_path), str(stderr_path)]
        if script_path.is_file():
            artifact_paths.insert(0, str(script_path))
        if runner_path is not None and runner_path.is_file():
            artifact_paths.append(str(runner_path))
        if generated_outputs.is_dir():
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
            deterministic_result_recovery_attempted=(
                deterministic_result_recovery_attempted
            ),
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
        "script_paths": [item.script_path for item in executions if item.script_path],
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
