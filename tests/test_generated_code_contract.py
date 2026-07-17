"""Regression tests for the line-oriented generated-Python contract."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from data_analysis_agent.nodes import _normalize_generated_comments
from data_analysis_agent.python_runner import LocalPythonRunner
from data_analysis_agent.schemas import PythonGeneration


def _contract(lines: list[str]) -> PythonGeneration:
    return PythonGeneration(kind="python", code_lines=lines, summary="Test source.")


def test_code_lines_reconstruct_exact_physical_lines() -> None:
    contract = _contract(["value = 1", "__agent_result__ = {'value': value}"])

    assert contract.source() == "value = 1\n__agent_result__ = {'value': value}\n"


@pytest.mark.parametrize("line", ["value = 1\n", "value = 1\r"])
def test_code_lines_reject_embedded_line_breaks(line: str) -> None:
    with pytest.raises(ValidationError, match="physical line"):
        _contract([line])


def test_comment_swallowed_result_is_rejected_before_script_write(
    tmp_path: Path,
) -> None:
    result = LocalPythonRunner().run(
        code=(
            "import pandas as pd  # all following text is a comment "
            "__agent_result__ = {}\n"
        ),
        goal_directory=tmp_path / "goal",
        allowed_files=[],
        version=1,
    )

    assert result.failure_category == "generation_contract_error"
    assert "prohibited comments" in (result.error or "")
    assert not (tmp_path / "goal/generated_code_v1.py").exists()


def test_hash_inside_string_is_not_a_comment(tmp_path: Path) -> None:
    result = LocalPythonRunner().run(
        code="label = '# not a comment'\n__agent_result__ = {'label': label}\n",
        goal_directory=tmp_path / "goal",
        allowed_files=[],
        version=1,
    )

    assert result.success
    assert result.result == {"label": "# not a comment"}


def test_model_comment_normalization_preserves_executable_source(
    tmp_path: Path,
) -> None:
    normalized = _normalize_generated_comments(
        "value = 2  # ordinary model comment\n__agent_result__ = {'value': value}\n"
    )

    result = LocalPythonRunner().run(
        code=normalized,
        goal_directory=tmp_path / "goal",
        allowed_files=[],
        version=1,
    )

    assert "ordinary model comment" not in normalized
    assert result.success
    assert result.result == {"value": 2}


@pytest.mark.parametrize(
    "source",
    [
        "text = '__agent_result__ = {}'\n",
        "def build():\n    __agent_result__ = {}\n",
        "class Build:\n    __agent_result__ = {}\n",
        "if True:\n    __agent_result__ = {}\n",
    ],
)
def test_only_module_body_assignment_is_accepted(tmp_path: Path, source: str) -> None:
    result = LocalPythonRunner().run(
        code=source,
        goal_directory=tmp_path / "goal",
        allowed_files=[],
        version=1,
    )

    assert result.failure_category == "generation_contract_error"
    assert "module-level assignment" in (result.error or "")


def test_annotated_module_assignment_is_accepted(tmp_path: Path) -> None:
    result = LocalPythonRunner().run(
        code="__agent_result__: dict = {'value': 1}\n",
        goal_directory=tmp_path / "goal",
        allowed_files=[],
        version=1,
    )

    assert result.success
    assert result.result == {"value": 1}


def test_dataframe_result_remains_a_typed_runner_failure(tmp_path: Path) -> None:
    result = LocalPythonRunner().run(
        code=(
            "import pandas as pd\n"
            "frame = pd.DataFrame({'value': [1]})\n"
            "__agent_result__ = {'frame': frame}\n"
        ),
        goal_directory=tmp_path / "goal",
        allowed_files=[],
        version=1,
    )

    assert result.failure_category == "result_contract_error"
    assert "not JSON-serializable" in (result.error or "")


def test_summary_never_enters_reconstructed_source() -> None:
    contract = PythonGeneration(
        kind="python",
        code_lines=["__agent_result__ = {'value': 1}"],
        summary="raise RuntimeError('metadata only')",
    )

    assert "metadata only" not in contract.source()
    assert json.loads(contract.model_dump_json())["summary"] == contract.summary
