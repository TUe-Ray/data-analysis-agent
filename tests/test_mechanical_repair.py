"""Regression coverage for the mechanical generated-code repair loop."""

import json
from pathlib import Path

import pytest

from data_analysis_agent.graph import build_graph
from data_analysis_agent.models import ScriptedRoleModel
from data_analysis_agent.python_runner import LocalPythonRunner


def _plan(*goal_ids: str) -> str:
    return json.dumps(
        {
            "scientific_objective": "Return one deterministic value per goal.",
            "goals": [
                {
                    "goal_id": goal_id,
                    "objective": f"Return a value for {goal_id}.",
                    "required_outputs": ["value"],
                    "constraints": ["Use the fixed method."],
                    "success_criteria": ["A value is returned."],
                    "depends_on": list(goal_ids[:index]),
                }
                for index, goal_id in enumerate(goal_ids)
            ],
        }
    )


def _strategy() -> str:
    return json.dumps(
        {
            "strategy": "generated_python",
            "capability_name": None,
            "arguments": {},
            "concise_reason": "A short local script is sufficient.",
        }
    )


def _state(tmp_path: Path, **overrides: object) -> dict[str, object]:
    return {
        "question": "Return a deterministic value.",
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


GOOD_CODE = "import json\nprint(json.dumps({'value': 2}))\n"
PASS = '{"decision":"PASS","feedback":"The required value is present."}'


@pytest.mark.parametrize(
    ("bad_code", "category"),
    [
        ("def broken(:\n", "syntax_error"),
        ("raise RuntimeError('broken')\n", "runtime_error"),
        ("import socket\n", "policy_error"),
        ("print('not json')\n", "result_contract_error"),
    ],
)
def test_failed_generated_code_repairs_without_calling_verifier(
    tmp_path: Path, bad_code: str, category: str
) -> None:
    model = ScriptedRoleModel(
        {
            "planner": [_plan("G1")],
            "executor": [_strategy(), bad_code, GOOD_CODE],
            "verifier": [PASS],
        }
    )

    result = build_graph(model).invoke(_state(tmp_path))

    assert result["status"] == "completed"
    assert result["replan_count"] == 0
    assert result["code_repair_count"] == 1
    assert result["code_execution_history"][0]["failure_category"] == category
    assert [call.role for call in model.calls].count("verifier") == 1
    assert result["trace"].index("mechanical_repair:attempt_1") < result["trace"].index(
        "verifier:PASS"
    )


def test_timeout_repairs_without_calling_verifier_on_failed_attempt(
    tmp_path: Path,
) -> None:
    model = ScriptedRoleModel(
        {
            "planner": [_plan("G1")],
            "executor": [_strategy(), "import time\ntime.sleep(1)\n", GOOD_CODE],
            "verifier": [PASS],
        }
    )

    result = build_graph(model, runner=LocalPythonRunner(timeout_seconds=0.2)).invoke(
        _state(tmp_path)
    )

    assert result["status"] == "completed"
    assert result["code_execution_history"][0]["failure_category"] == "timeout"
    assert result["replan_count"] == 0


def test_mechanical_exhaustion_has_its_own_status_and_no_scientific_replan(
    tmp_path: Path,
) -> None:
    model = ScriptedRoleModel(
        {
            "planner": [_plan("G1")],
            "executor": [
                _strategy(),
                "raise RuntimeError('one')\n",
                "raise RuntimeError('two')\n",
                "raise RuntimeError('three')\n",
            ],
        }
    )

    result = build_graph(model).invoke(
        _state(
            tmp_path,
            max_code_repair_attempts=2,
            max_code_repair_no_progress_attempts=50,
        )
    )

    assert result["status"] == "mechanical_execution_failed"
    assert result["replan_count"] == 0
    assert result["code_repair_attempts_for_current_goal"] == 2
    assert "code_repair_exhausted" in result["final_answer"]
    assert "stopped_after_max_replans" not in result["final_answer"]
    assert [call.role for call in model.calls].count("verifier") == 0
    assert [call.role for call in model.calls].count("planner") == 1


def test_identical_repairs_stop_early_for_no_progress(tmp_path: Path) -> None:
    bad_code = "raise RuntimeError('same')\n"
    model = ScriptedRoleModel(
        {
            "planner": [_plan("G1")],
            "executor": [_strategy(), bad_code, bad_code, bad_code],
        }
    )

    result = build_graph(model).invoke(
        _state(
            tmp_path,
            max_code_repair_attempts=50,
            max_code_repair_no_progress_attempts=2,
        )
    )

    assert result["status"] == "mechanical_execution_failed"
    assert result["code_repair_no_progress"] is True
    assert result["code_repair_attempts_for_current_goal"] == 2
    assert "mechanical_repair_no_progress" in result["final_answer"]
    assert result["replan_count"] == 0


def test_scientific_replan_is_the_only_path_that_calls_planner_again(
    tmp_path: Path,
) -> None:
    model = ScriptedRoleModel(
        {
            "planner": [_plan("G1"), _plan("G1")],
            "executor": [_strategy(), GOOD_CODE, _strategy(), GOOD_CODE],
            "verifier": [
                '{"decision":"REPLAN","feedback":"Methodological issue."}',
                PASS,
            ],
        }
    )

    result = build_graph(model).invoke(_state(tmp_path))

    assert result["status"] == "completed"
    assert result["replan_count"] == 1
    assert result["code_repair_count"] == 0
    assert result["failure_category"] is None
    assert [call.role for call in model.calls].count("planner") == 2


def test_repaired_g1_does_not_consume_budget_before_g2(tmp_path: Path) -> None:
    model = ScriptedRoleModel(
        {
            "planner": [_plan("G1", "G2")],
            "executor": [
                _strategy(),
                "raise RuntimeError('G1')\n",
                GOOD_CODE,
                _strategy(),
                GOOD_CODE,
            ],
            "verifier": [PASS, PASS],
        }
    )

    result = build_graph(model).invoke(_state(tmp_path))

    assert result["status"] == "completed"
    assert result["replan_count"] == 0
    assert result["code_repair_count"] == 1
    g2_attempt = next(
        record
        for record in result["code_execution_history"]
        if record["goal_id"] == "G2"
    )
    assert g2_attempt["scientific_replan_count"] == 0
    assert result["code_repair_attempts_for_current_goal"] == 0


def test_g2_mechanical_failures_do_not_call_planner_or_max_replan_finalizer(
    tmp_path: Path,
) -> None:
    model = ScriptedRoleModel(
        {
            "planner": [_plan("G1", "G2")],
            "executor": [
                _strategy(),
                GOOD_CODE,
                _strategy(),
                "raise RuntimeError('one')\n",
                "raise RuntimeError('two')\n",
                "raise RuntimeError('three')\n",
            ],
            "verifier": [PASS],
        }
    )

    result = build_graph(model).invoke(
        _state(
            tmp_path,
            max_code_repair_attempts=2,
            max_code_repair_no_progress_attempts=50,
        )
    )

    assert result["status"] == "mechanical_execution_failed"
    assert result["replan_count"] == 0
    assert [call.role for call in model.calls].count("planner") == 1
    assert [call.role for call in model.calls].count("verifier") == 1
    assert "stopped_after_max_replans" not in result["final_answer"]
