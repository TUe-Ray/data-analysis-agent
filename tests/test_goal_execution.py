import json
from pathlib import Path
from unittest.mock import patch

import pytest

from data_analysis_agent.demo import run_demo
from data_analysis_agent.final_output import DeterministicFinalOutputProvider
from data_analysis_agent.graph import build_graph
from data_analysis_agent.models import ScriptedRoleModel, build_scripted_model
from data_analysis_agent.nodes import ExecutorOutputError
from data_analysis_agent.prompts import (
    EXECUTOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)


def run_scenario(scenario: str, tmp_path: Path):
    log_path = tmp_path / scenario / "workflow.log"
    log_path.parent.mkdir(parents=True)
    with (
        patch("data_analysis_agent.demo.load_settings") as load_settings,
        patch("data_analysis_agent.demo.create_nebius_client") as create_client,
    ):
        result = run_demo(mode="offline", scenario=scenario, log_path=log_path)
    load_settings.assert_not_called()
    create_client.assert_not_called()
    return result


def test_trusted_tool_scenario_executes_goals_sequentially(tmp_path: Path) -> None:
    result = run_scenario("trusted-tools-success", tmp_path)

    assert result["status"] == "completed"
    assert [item["goal_id"] for item in result["completed_goal_results"]] == [
        "understand_input",
        "compute_statistics",
    ]
    assert result["trusted_tool_calls"] == 2
    assert result["generated_script_count"] == 0
    assert result["validated_final_answer"]["key_results"] == pytest.approx(
        {
            "mean": 13,
            "sample_standard_error": 1.2909944487358056,
            "n_observations": 4,
        }
    )
    assert result["iteration_history"][0]["route"].endswith("Select Current Goal")


def test_generated_python_success_is_an_artifact_not_a_tool(tmp_path: Path) -> None:
    result = run_scenario("generated-python-success", tmp_path)

    assert result["status"] == "completed"
    assert result["trusted_tool_calls"] == 0
    assert result["generated_script_count"] == 1
    assert result["validated_final_answer"]["key_results"] == {
        "mean_absolute_successive_difference": 2.0
    }
    goal_dir = Path(result["run_directory"]) / "goals/compute_successive_difference"
    assert (goal_dir / "generated_code_v1.py").is_file()
    assert (goal_dir / "artifact_metadata.json").is_file()
    assert len(result["capability_catalog"]) == 3


def test_generated_python_repair_preserves_both_scripts(tmp_path: Path) -> None:
    result = run_scenario("generated-python-repair", tmp_path)
    goal_dir = Path(result["run_directory"]) / "goals/compute_successive_difference"

    assert result["status"] == "completed"
    assert result["generated_script_count"] == 2
    assert result["code_repair_count"] == 1
    assert (goal_dir / "generated_code_v1.py").is_file()
    assert (goal_dir / "generated_code_v2.py").is_file()


def test_generated_python_failure_terminates_without_approved_results(
    tmp_path: Path,
) -> None:
    result = run_scenario("generated-python-failure", tmp_path)
    goal_dir = Path(result["run_directory"]) / "goals/compute_successive_difference"

    assert result["status"] == "mechanical_execution_failed"
    assert result["completed_goal_results"] == []
    assert result["validated_final_answer"]["key_results"] == {}
    assert result["replan_count"] == 0
    assert "mechanical_repair_no_progress" in result["final_answer"]
    assert len(list(goal_dir.glob("generated_code_v*.py"))) >= 2


def test_python_policy_failure_skips_verifier_and_global_replan(tmp_path: Path) -> None:
    plan = json.dumps(
        {
            "scientific_objective": "Read one staged file.",
            "goals": [
                {
                    "goal_id": "read_input",
                    "objective": "Read one staged file.",
                    "required_outputs": ["rows"],
                    "constraints": [],
                    "success_criteria": ["A row count is returned."],
                    "depends_on": [],
                }
            ],
        }
    )
    strategy = json.dumps(
        {
            "strategy": "generated_python",
            "capability_name": None,
            "arguments": {},
            "concise_reason": "Read the staged input.",
        }
    )
    blocked_code = (
        "import pandas as pd\n"
        "def read(path):\n"
        "    return pd.read_csv(path)\n"
        "read('inputs/patients.csv')\n"
        "__agent_result__ = {'rows': 0}\n"
    )
    generation = json.dumps(
        {"kind": "python", "code": blocked_code, "summary": "Read the input."}
    )
    model = ScriptedRoleModel(
        {"planner": [plan], "executor": [strategy, generation]}
    )

    result = build_graph(model).invoke(
        {
            "question": "Read one staged file.",
            "file_paths": ["patients.csv"],
            "staged_file_paths": [str((tmp_path / "inputs/patients.csv").resolve())],
            "staged_file_display_paths": ["inputs/patients.csv"],
            "execution_working_directory": str(tmp_path),
            "input_context": "Public input only.",
            "run_directory": str(tmp_path / "run"),
            "replan_count": 0,
            "max_replans": 1,
            "max_code_repair_attempts": 1,
            "max_code_repair_no_progress_attempts": 1,
            "trace": [],
        }
    )

    assert result["status"] == "mechanical_execution_failed"
    assert result["replan_count"] == 0
    assert result["policy_failure_reason"].startswith("PythonPolicyError:")
    assert result["execution_failure_category"] == "policy_error"
    assert [call.role for call in model.calls].count("verifier") == 0


def test_structured_verifier_context_excludes_raw_input_and_role_history(
    tmp_path: Path,
) -> None:
    data_path = (
        Path(__file__).resolve().parents[1]
        / "examples/data/measurements_with_missing.csv"
    )
    model = build_scripted_model("trusted-tools-success")
    build_graph(model, DeterministicFinalOutputProvider()).invoke(
        {
            "question": "Compute mean, standard error, and count.",
            "file_paths": [data_path.name],
            "staged_file_paths": [str(data_path)],
            "input_context": data_path.read_text(encoding="utf-8"),
            "run_directory": str(tmp_path / "run"),
            "replan_count": 0,
            "max_replans": 1,
            "trace": [],
        }
    )
    verifier_calls = [call for call in model.calls if call.role == "verifier"]
    executor_calls = [call for call in model.calls if call.role == "executor"]

    assert len(verifier_calls) == 2
    for call in verifier_calls:
        context = call.messages[1]["content"]
        assert "Current IntermediateGoal:" in context
        assert "Relevant prior GoalResults:" in context
        assert "Available capability catalog:" not in context
        assert "s1,10" not in context
        assert PLANNER_SYSTEM_PROMPT not in context
        assert EXECUTOR_SYSTEM_PROMPT not in context
    assert executor_calls[0].messages[0]["content"] == EXECUTOR_SYSTEM_PROMPT
    assert "Available capability catalog:" in executor_calls[0].messages[1]["content"]


def _strategy_test_model(strategy: dict[str, object]) -> ScriptedRoleModel:
    plan = json.dumps(
        {
            "scientific_objective": "Compute one generated result.",
            "goals": [
                {
                    "goal_id": "compute",
                    "objective": "Compute one generated result.",
                    "required_outputs": ["value"],
                    "constraints": [],
                    "success_criteria": ["A value is returned."],
                    "depends_on": [],
                }
            ],
        }
    )
    return ScriptedRoleModel(
        {
            "planner": [plan],
            "executor": [
                json.dumps(strategy),
                json.dumps(
                    {
                        "kind": "python",
                        "code": "__agent_result__ = {'value': 2.0}\n",
                        "summary": "Return one value.",
                    }
                ),
            ],
            "verifier": ['{"decision":"PASS","feedback":"Value returned."}'],
        }
    )


@pytest.mark.parametrize(
    ("capability_name", "arguments", "warning_expected"),
    [
        (None, {}, False),
        ("profile_table", {}, True),
        (None, {"file_path": "inputs/data.csv"}, True),
        ("inspect_file", {"file_path": "inputs/data.csv"}, True),
    ],
)
def test_generated_python_strategy_fields_normalize_and_continue(
    tmp_path: Path,
    capability_name: str | None,
    arguments: dict[str, str],
    warning_expected: bool,
) -> None:
    model = _strategy_test_model(
        {
            "strategy": "generated_python",
            "capability_name": capability_name,
            "arguments": arguments,
            "concise_reason": "Generated code is required.",
        }
    )

    result = build_graph(model).invoke(
        {
            "question": "Compute one value.",
            "file_paths": [],
            "staged_file_paths": [],
            "input_context": "No input files.",
            "run_directory": str(tmp_path / "run"),
            "replan_count": 0,
            "max_replans": 1,
            "trace": [],
        }
    )

    assert result["status"] == "completed"
    assert result["current_strategy"]["capability_name"] is None
    assert result["current_strategy"]["arguments"] == {}
    assert bool(result.get("executor_warnings")) is warning_expected
    assert [call.role for call in model.calls].count("executor") == 2


@pytest.mark.parametrize(
    ("capability_name", "message"),
    [(None, "requires capability_name"), ("unknown_tool", "Unknown trusted")],
)
def test_invalid_trusted_tool_strategy_still_fails(
    tmp_path: Path, capability_name: str | None, message: str
) -> None:
    model = _strategy_test_model(
        {
            "strategy": "trusted_tool",
            "capability_name": capability_name,
            "arguments": {},
            "concise_reason": "Use a trusted tool.",
        }
    )

    with pytest.raises(ExecutorOutputError, match=message):
        build_graph(model).invoke(
            {
                "question": "Compute one value.",
                "file_paths": [],
                "staged_file_paths": [],
                "input_context": "No input files.",
                "run_directory": str(tmp_path / "run"),
                "replan_count": 0,
                "max_replans": 1,
                "trace": [],
            }
        )
