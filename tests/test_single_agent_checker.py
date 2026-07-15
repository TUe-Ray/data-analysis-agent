"""Controlled tests for the single-agent plus final-checker ablation."""

import json
from pathlib import Path

from data_analysis_agent.benchmark_approaches import run_single_agent_checker
from data_analysis_agent.benchmark_types import BenchmarkConfig, PublicTaskView
from data_analysis_agent.models import ScriptedRoleModel
from data_analysis_agent.python_runner import LocalPythonRunner


def _public(tmp_path: Path) -> PublicTaskView:
    data = tmp_path / "inputs/data.csv"
    data.parent.mkdir(parents=True)
    data.write_text("value\n1\n", encoding="utf-8")
    return PublicTaskView(
        task_id="ablation_fixture",
        prompt="Return one integer value.",
        data_files=["inputs/data.csv"],
        data_contents={"inputs/data.csv": "value\n1\n"},
        answer_schema={
            "type": "object",
            "required": ["value"],
            "properties": {"value": {"type": "integer"}},
            "additionalProperties": False,
        },
    )


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        model="scripted",
        task_ids=["ablation_fixture"],
        approaches=["single_agent_checker"],
    )


def _generation(value: int) -> str:
    return json.dumps(
        {
            "kind": "python",
            "code_lines": [f"__agent_result__ = {{'value': {value}}}"],
            "summary": "Produce the candidate.",
        }
    )


def test_final_checker_runs_once_without_planner_or_per_goal_verifier(
    tmp_path: Path,
) -> None:
    model = ScriptedRoleModel(
        {
            "single_agent": [_generation(1)],
            "final_checker": [
                '{"decision":"PASS","repair_scope":"none","feedback":"Complete."}'
            ],
        }
    )

    outcome = run_single_agent_checker(
        public=_public(tmp_path),
        model=model,
        run_directory=tmp_path,
        config=_config(),
    )

    assert outcome.status == "completed"
    assert outcome.candidate == {"value": 1}
    assert outcome.verifier_decisions == ["PASS"]
    assert outcome.global_checker_repair_count == 0
    assert [call.role for call in model.calls] == ["single_agent", "final_checker"]


def test_final_checker_repair_regenerates_and_executes_python(tmp_path: Path) -> None:
    model = ScriptedRoleModel(
        {
            "single_agent": [
                _generation(1),
                json.dumps(
                    {
                        "kind": "python_repair",
                        "code_lines": ["__agent_result__ = {'value': 2}"],
                        "summary": "Correct the executed analysis.",
                        "addressed_failure_category": "result_contract_error",
                    }
                ),
            ],
            "final_checker": [
                '{"decision":"REPAIR","repair_scope":"rerun_analysis","feedback":'
                '"Replace value with the checked value."}'
            ],
        }
    )

    outcome = run_single_agent_checker(
        public=_public(tmp_path),
        model=model,
        run_directory=tmp_path,
        config=_config(),
    )

    assert outcome.status == "completed"
    assert outcome.candidate == {"value": 2}
    assert outcome.global_checker_repair_count == 1
    assert [call.role for call in model.calls].count("final_checker") == 1
    assert outcome.global_replan_count == 0


def test_invalid_checker_repair_fails_without_a_second_repair(tmp_path: Path) -> None:
    model = ScriptedRoleModel(
        {
            "single_agent": [
                _generation(1),
                json.dumps(
                    {
                        "kind": "python_repair",
                        "code_lines": [
                            "__agent_result__ = {'value': 'not an integer'}"
                        ],
                        "summary": "Bad repair.",
                        "addressed_failure_category": "result_contract_error",
                    }
                ),
            ],
            "final_checker": [
                '{"decision":"REPAIR","repair_scope":"format_only",'
                '"feedback":"Repair it."}'
            ],
        }
    )

    outcome = run_single_agent_checker(
        public=_public(tmp_path),
        model=model,
        run_directory=tmp_path,
        config=_config(),
    )

    assert outcome.status == "invalid_json"
    assert outcome.global_checker_repair_count == 1
    assert [call.role for call in model.calls] == [
        "single_agent",
        "final_checker",
        "single_agent",
    ]


def test_single_agent_uses_resolved_staged_file_path_from_its_execution_cwd(
    tmp_path: Path,
) -> None:
    staged = (tmp_path / "inputs/data.csv").resolve()
    model = ScriptedRoleModel(
        {
            "single_agent": [
                json.dumps(
                    {
                        "kind": "python",
                        "code_lines": [
                            "import pandas as pd",
                            f"data = pd.read_csv({str(staged)!r})",
                            "__agent_result__ = {'value': int(data['value'].sum())}",
                        ],
                        "summary": "Read the staged data.",
                    }
                )
            ],
            "final_checker": [
                '{"decision":"PASS","repair_scope":"none","feedback":"Complete."}'
            ],
        }
    )

    outcome = run_single_agent_checker(
        public=_public(tmp_path), model=model, run_directory=tmp_path, config=_config()
    )

    assert outcome.status == "completed"
    assert outcome.candidate == {"value": 1}
    prompt = model.calls[0].messages[1]["content"]
    assert str(staged) in prompt
    assert "Relative input paths are invalid" in prompt
    rejected = LocalPythonRunner().run(
        code=(
            "import pandas as pd\n"
            "data = pd.read_csv('inputs/data.csv')\n"
            "__agent_result__ = {'value': int(data['value'].sum())}\n"
        ),
        goal_directory=tmp_path / "relative_rejected",
        allowed_files=[staged],
        version=1,
    )
    assert not rejected.success
    assert rejected.failure_category == "policy_error"


def test_format_only_checker_repair_still_uses_second_execution(tmp_path: Path) -> None:
    model = ScriptedRoleModel(
        {
            "single_agent": [
                _generation(1),
                json.dumps(
                    {
                        "kind": "python_repair",
                        "code_lines": ["__agent_result__ = {'value': 1}"],
                        "summary": "Re-emit the valid wrapper.",
                        "addressed_failure_category": "result_contract_error",
                    }
                ),
            ],
            "final_checker": [
                '{"decision":"REPAIR","repair_scope":"format_only",'
                '"feedback":"Normalize representation."}'
            ],
        }
    )

    outcome = run_single_agent_checker(
        public=_public(tmp_path), model=model, run_directory=tmp_path, config=_config()
    )

    assert outcome.status == "completed"
    assert outcome.candidate == {"value": 1}
    assert len(list((tmp_path / "single_agent_run").glob("generated_code_v*.py"))) == 2
    assert [call.role for call in model.calls].count("final_checker") == 1
