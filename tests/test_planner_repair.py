"""Deterministic coverage for bounded structural Planner-output repair."""

import json
from pathlib import Path

import pytest

from data_analysis_agent.graph import build_graph
from data_analysis_agent.models import ScriptedRoleModel


def _plan(*goal_ids: str) -> str:
    return json.dumps(
        {
            "scientific_objective": "Return one value for each planned goal.",
            "goals": [
                {
                    "goal_id": goal_id,
                    "objective": f"Compute {goal_id}.",
                    "required_outputs": ["value"],
                    "constraints": [],
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
            "concise_reason": "A local script returns the required value.",
        }
    )


GOOD_CODE = "__agent_result__ = {'value': 2}\n"
PASS = '{"decision":"PASS","feedback":"The required value is present."}'


def _generation(code: str = GOOD_CODE) -> str:
    return json.dumps({"kind": "python", "code": code, "summary": "Generate."})


def _state(tmp_path: Path, **overrides: object) -> dict[str, object]:
    return {
        "question": "Return a deterministic value.",
        "file_paths": [],
        "staged_file_paths": [],
        "input_context": "No input files.",
        "run_directory": str(tmp_path / "run"),
        "replan_count": 0,
        "max_replans": 1,
        "max_planner_repairs": 2,
        "trace": [],
        **overrides,
    }


def _duplicate_goals() -> str:
    payload = json.loads(_plan("G1"))
    payload["goals"].append({**payload["goals"][0]})
    return json.dumps(payload)


def _dependency_plan(dependency: str, *, forward: bool = False) -> str:
    first, second = json.loads(_plan("clean_visits", "pre_start_exclusions"))["goals"]
    if forward:
        first["goal_id"] = "pre_start_exclusions"
        first["depends_on"] = ["clean_visits"]
        second["goal_id"] = "clean_visits"
        second["depends_on"] = []
        goals = [first, second]
    else:
        second["depends_on"] = [dependency]
        goals = [first, second]
    return json.dumps({"scientific_objective": "Return values.", "goals": goals})


@pytest.mark.parametrize(
    "invalid_plan",
    [
        "{not valid JSON",
        "not valid JSON",
        _duplicate_goals(),
        _dependency_plan("missing_goal"),
        _dependency_plan("clean_visits", forward=True),
        _dependency_plan("Compute clean_visits."),
    ],
    ids=[
        "invalid_json_object",
        "invalid_json_text",
        "duplicate_ids",
        "missing_dependency",
        "forward_dependency",
        "objective_not_id",
    ],
)
def test_invalid_planner_output_repairs_before_execution(
    tmp_path: Path, invalid_plan: str
) -> None:
    model = ScriptedRoleModel(
        {
            "planner": [invalid_plan, _plan("G1")],
            "executor": [_strategy(), _generation()],
            "verifier": [PASS],
        }
    )

    result = build_graph(model).invoke(_state(tmp_path))

    assert result["status"] == "completed"
    assert result["planner_repair_count"] == 1
    assert result["replan_count"] == 0
    assert result.get("code_repair_count", 0) == 0
    assert [call.role for call in model.calls][:2] == ["planner", "planner"]
    assert result["planner_validation_history"][0]["valid"] is False
    assert result["planner_validation_history"][1]["valid"] is True
    assert (
        Path(result["planner_response_history"][0]["raw_response_path"]).read_text(
            encoding="utf-8"
        )
        == invalid_plan
    )
    assert Path(result["planner_validation_history"][0]["validation_path"]).is_file()


def test_planner_repair_exhaustion_has_its_own_finalizer(tmp_path: Path) -> None:
    invalid = _dependency_plan("missing_goal")
    model = ScriptedRoleModel({"planner": [invalid, invalid]})

    result = build_graph(model).invoke(_state(tmp_path, max_planner_repairs=1))

    assert result["status"] == "planner_output_failed"
    assert result["replan_count"] == 0
    assert result.get("code_repair_count", 0) == 0
    assert "executor_invoked=false" in result["final_answer"]
    assert "verifier_invoked=false" in result["final_answer"]
    assert "stopped_after_max_replans" not in result["final_answer"]
    assert [call.role for call in model.calls] == ["planner", "planner"]


def test_scientific_replan_invalid_replacement_increments_once(tmp_path: Path) -> None:
    invalid_replacement = _dependency_plan("clean_visits", forward=True)
    model = ScriptedRoleModel(
        {
            "planner": [_plan("G1"), invalid_replacement, _plan("G1")],
            "executor": [_strategy(), _generation(), _strategy(), _generation()],
            "verifier": [
                '{"decision":"REPLAN","feedback":"Scientific correction required."}',
                PASS,
            ],
        }
    )

    result = build_graph(model).invoke(_state(tmp_path))

    assert result["status"] == "completed"
    assert result["replan_count"] == 1
    assert result["planner_repair_count"] == 1
    assert result["planner_response_history"][1]["mode"] == "scientific_replan"
    assert [call.role for call in model.calls].count("planner") == 3


def test_valid_planner_response_keeps_original_single_call_path(tmp_path: Path) -> None:
    model = ScriptedRoleModel(
        {
            "planner": [_plan("G1")],
            "executor": [_strategy(), _generation()],
            "verifier": [PASS],
        }
    )

    result = build_graph(model).invoke(_state(tmp_path))

    assert result["status"] == "completed"
    assert result["planner_repair_count"] == 0
    assert [call.role for call in model.calls].count("planner") == 1
