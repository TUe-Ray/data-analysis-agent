"""Strict structured-output boundaries for generated and repaired Python."""

import json
from pathlib import Path

import pytest

from data_analysis_agent.graph import build_graph
from data_analysis_agent.models import ScriptedRoleModel


def _plan() -> str:
    return json.dumps(
        {
            "scientific_objective": "Return one deterministic value.",
            "goals": [
                {
                    "goal_id": "G1",
                    "objective": "Return one deterministic value.",
                    "required_outputs": ["value"],
                    "constraints": [],
                    "success_criteria": ["The value is returned."],
                    "depends_on": [],
                }
            ],
        }
    )


def _strategy() -> str:
    return json.dumps(
        {
            "strategy": "generated_python",
            "capability_name": None,
            "arguments": {},
            "concise_reason": "Generated Python is sufficient.",
        }
    )


def _generation(code: str, summary: str = "Generate the value.") -> str:
    return json.dumps(
        {"kind": "python", "code": code, "summary": summary},
        ensure_ascii=False,
    )


def _repair(code: str, summary: str = "Repair the result contract.") -> str:
    return json.dumps(
        {
            "kind": "python_repair",
            "code": code,
            "summary": summary,
            "addressed_failure_category": "result_contract_error",
        },
        ensure_ascii=False,
    )


def _state(tmp_path: Path, **overrides: object) -> dict[str, object]:
    return {
        "question": "Return one deterministic value.",
        "file_paths": [],
        "staged_file_paths": [],
        "input_context": "No input files.",
        "run_directory": str(tmp_path / "run"),
        "replan_count": 0,
        "max_replans": 1,
        "max_code_repair_attempts": 4,
        "max_code_repair_no_progress_attempts": 3,
        "trace": [],
        **overrides,
    }


def test_valid_generation_executes_only_code_and_preserves_summary(
    tmp_path: Path,
) -> None:
    code = "print('debug')\n__agent_result__ = {'value': 2}\n"
    summary = "Summary text is metadata; raise RuntimeError('never execute it')."
    model = ScriptedRoleModel(
        {
            "planner": [_plan()],
            "executor": [_strategy(), _generation(code, summary)],
            "verifier": [
                '{"decision":"PASS","feedback":"The value is present."}'
            ],
        }
    )

    result = build_graph(model).invoke(_state(tmp_path))

    goal = tmp_path / "run/goals/G1"
    assert result["status"] == "completed"
    assert (goal / "generated_code_v1.py").read_text(encoding="utf-8") == code
    metadata = json.loads(
        (goal / "python_generation_v1.json").read_text(encoding="utf-8")
    )
    assert metadata["summary"] == summary
    assert summary not in (goal / "generated_code_v1.py").read_text(encoding="utf-8")
    assert model.calls[2].structured_schema_name == "python_generation"


@pytest.mark.parametrize(
    "invalid_response",
    [
        _generation("__agent_result__ = {'value': 1}\n") + "\nextra prose",
        json.dumps({"kind": "python", "code": "   ", "summary": "empty"}),
        "```json\n{}\n```",
    ],
)
def test_invalid_generation_is_never_written_or_executed(
    tmp_path: Path, invalid_response: str
) -> None:
    model = ScriptedRoleModel(
        {"planner": [_plan()], "executor": [_strategy(), invalid_response]}
    )

    result = build_graph(model).invoke(
        _state(tmp_path, max_code_repair_no_progress_attempts=1)
    )

    goal = tmp_path / "run/goals/G1"
    assert result["status"] == "mechanical_execution_failed"
    assert result["execution_failure_category"] == "generation_contract_error"
    assert list(goal.glob("generated_code_v*.py")) == []
    assert (goal / "python_generation_invalid_v1.txt").is_file()
    assert result["generated_script_count"] == 0


def test_python_repair_uses_its_own_contract_and_metadata(tmp_path: Path) -> None:
    repaired_code = "__agent_result__ = {'value': 3}\n"
    summary = "Add the required fixed result variable."
    model = ScriptedRoleModel(
        {
            "planner": [_plan()],
            "executor": [
                _strategy(),
                _generation("print('missing result')\n"),
                _repair(repaired_code, summary),
            ],
            "verifier": [
                '{"decision":"PASS","feedback":"The repaired value is present."}'
            ],
        }
    )

    result = build_graph(model).invoke(_state(tmp_path))

    goal = tmp_path / "run/goals/G1"
    assert result["status"] == "completed"
    assert result["code_repair_count"] == 1
    assert (goal / "generated_code_v2.py").read_text(encoding="utf-8") == repaired_code
    metadata = json.loads(
        (goal / "python_repair_v2.json").read_text(encoding="utf-8")
    )
    assert metadata["summary"] == summary
    assert summary not in (goal / "generated_code_v2.py").read_text(encoding="utf-8")
    assert model.calls[3].structured_schema_name == "python_repair"


def test_invalid_repair_response_never_becomes_a_second_script(tmp_path: Path) -> None:
    invalid_repair = (
        _repair("open('should_not_exist', 'w').write('bad')\n") + " trailing prose"
    )
    model = ScriptedRoleModel(
        {
            "planner": [_plan()],
            "executor": [
                _strategy(),
                _generation("print('missing result')\n"),
                invalid_repair,
            ],
        }
    )

    result = build_graph(model).invoke(
        _state(tmp_path, max_code_repair_attempts=1)
    )

    goal = tmp_path / "run/goals/G1"
    assert result["status"] == "mechanical_execution_failed"
    assert (goal / "generated_code_v1.py").is_file()
    assert not (goal / "generated_code_v2.py").exists()
    assert not (goal / "should_not_exist").exists()
    assert (goal / "python_repair_invalid_v2.txt").is_file()
