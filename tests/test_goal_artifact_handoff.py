"""Verifier-gated cross-goal artifact handoff coverage."""

import json
from pathlib import Path

from data_analysis_agent.graph import build_graph
from data_analysis_agent.models import ScriptedRoleModel
from data_analysis_agent.nodes import (
    _artifacts_available_to_current_goal,
    planner_validator_node,
    select_current_goal_node,
)


def _plan() -> str:
    return json.dumps(
        {
            "scientific_objective": "Prepare a small normalized table then consume it.",
            "goals": [
                {
                    "goal_id": "G1",
                    "objective": "Create the normalized patient table.",
                    "required_outputs": ["normalized table"],
                    "constraints": [],
                    "success_criteria": ["The table is written."],
                    "depends_on": [],
                },
                {
                    "goal_id": "G2",
                    "objective": "Read the verified normalized table.",
                    "required_outputs": ["normalized row count"],
                    "constraints": [],
                    "success_criteria": ["The artifact is read."],
                    "depends_on": ["G1"],
                },
            ],
        }
    )


def _strategy() -> str:
    return json.dumps(
        {
            "strategy": "generated_python",
            "capability_name": None,
            "arguments": {},
            "concise_reason": "The goal needs a small local transformation.",
        }
    )


def _generation(code: str) -> str:
    return json.dumps(
        {"kind": "python", "code_lines": code.splitlines(), "summary": "Run goal."}
    )


def _state(tmp_path: Path) -> dict[str, object]:
    return {
        "question": "Create and then consume a normalized patient table.",
        "file_paths": [],
        "staged_file_paths": [],
        "input_context": "No original staged files are needed for this fixture.",
        "run_directory": str(tmp_path / "run"),
        "replan_count": 0,
        "max_replans": 1,
        "max_code_repair_attempts": 2,
        "max_code_repair_no_progress_attempts": 3,
        "trace": [],
    }


def test_verified_artifact_is_registered_and_available_only_to_dependency(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "run/goals/G1/patients_normalized.csv"
    g1 = "\n".join(
        [
            "import csv",
            (
                "with open('patients_normalized.csv', 'w', encoding='utf-8', "
                "newline='') as handle:"
            ),
            "    writer = csv.writer(handle)",
            "    writer.writerow(['patient_id', 'arm'])",
            "    writer.writerow(['P001', 'A'])",
            "__agent_result__ = {'rows': 1, 'artifacts': [{"
            "'relative_name': 'patients_normalized.csv', "
            "'description': 'Normalized patient arm table', "
            "'media_type': 'text/csv'}]}",
        ]
    )
    g2 = "\n".join(
        [
            "import pandas as pd",
            f"frame = pd.read_csv({str(artifact_path)!r})",
            "__agent_result__ = {'normalized_rows': int(len(frame))}",
        ]
    )
    model = ScriptedRoleModel(
        {
            "planner": [_plan()],
            "executor": [_strategy(), _generation(g1), _strategy(), _generation(g2)],
            "verifier": [
                '{"decision":"PASS","feedback":"G1 artifact is verified."}',
                '{"decision":"PASS","feedback":"G2 used the verified artifact."}',
            ],
        }
    )

    result = build_graph(model).invoke(_state(tmp_path))

    approved = result["approved_goal_artifacts"]
    assert result["status"] == "completed"
    assert len(approved) == 1
    assert approved[0]["producer_goal_id"] == "G1"
    assert approved[0]["path"] == str(artifact_path)
    assert Path(approved[0]["path"]).is_file()
    assert (tmp_path / "run/approved_goal_artifacts.json").is_file()
    g2_generation_call = model.calls[4]
    assert str(artifact_path) in g2_generation_call.messages[1]["content"]
    assert result["completed_goal_results"][-1]["result"] == {"normalized_rows": 1}


def test_unverified_artifact_is_never_approved(tmp_path: Path) -> None:
    g1 = "\n".join(
        [
            "from pathlib import Path",
            "Path('patients_normalized.csv').write_text("
            "'patient_id\\nP001\\n', encoding='utf-8')",
            "__agent_result__ = {'artifacts': [{"
            "'relative_name': 'patients_normalized.csv', "
            "'description': 'Unverified table'}]}",
        ]
    )
    model = ScriptedRoleModel(
        {
            "planner": [_plan()],
            "executor": [_strategy(), _generation(g1)],
            "verifier": ['{"decision":"REPLAN","feedback":"Do not approve G1."}'],
        }
    )

    result = build_graph(model).invoke({**_state(tmp_path), "max_replans": 0})

    assert result["completed_goal_results"] == []
    assert result.get("approved_goal_artifacts", []) == []
    assert (tmp_path / "run/goals/G1/patients_normalized.csv").is_file()


def test_diagnostic_file_cannot_be_declared_as_an_artifact(tmp_path: Path) -> None:
    code = "\n".join(
        [
            "from pathlib import Path",
            "Path('workflow.log').write_text('private diagnostic', encoding='utf-8')",
            "__agent_result__ = {'artifacts': [{"
            "'relative_name': 'workflow.log', "
            "'description': 'Bad declaration'}]}",
        ]
    )
    model = ScriptedRoleModel(
        {"planner": [_plan()], "executor": [_strategy(), _generation(code)]}
    )

    result = build_graph(model).invoke(
        {
            **_state(tmp_path),
            "max_code_repair_attempts": 0,
            "max_code_repair_no_progress_attempts": 1,
        }
    )

    assert result["execution_failure_category"] == "result_contract_error"
    assert result.get("approved_goal_artifacts", []) == []


def test_replan_rejects_mutating_completed_artifact_producer(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    original_plan = json.loads(_plan())
    revised_plan = {
        **original_plan,
        "goals": [
            {
                **original_plan["goals"][0],
                "objective": "A materially changed replacement for G1.",
            }
        ],
    }
    state = {
        "planner_raw_response": json.dumps(revised_plan),
        "planner_raw_response_path": str(tmp_path / "run/planner_response_v1.json"),
        "planner_response_history": [{"version": 1}],
        "run_directory": str(tmp_path / "run"),
        "planner_mode": "scientific_replan",
        "high_level_plan": original_plan,
        "completed_goal_results": [{"goal_id": "G1", "success": True}],
        "approved_goal_artifacts": [
            {
                "artifact_id": "G1:patients_normalized.csv:0123456789ab",
                "producer_goal_id": "G1",
                "path": str(tmp_path / "run/goals/G1/patients_normalized.csv"),
                "relative_name": "patients_normalized.csv",
                "media_type": "text/csv",
                "description": "Original output",
                "size_bytes": 1,
                "sha256": "0" * 64,
            }
        ],
        "planner_repair_count": 0,
        "replan_count": 1,
        "trace": [],
    }

    updates = planner_validator_node(state)

    assert "completed goal ID G1 was reused" in updates["planner_validation_error"]


def test_full_scientific_replan_preserves_eight_goals_and_dependency_artifacts(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    goals = [
        {
            "goal_id": f"G{index}",
            "objective": f"Original goal {index}.",
            "required_outputs": [f"output {index}"],
            "constraints": [],
            "success_criteria": [f"criterion {index}"],
            "depends_on": [] if index == 1 else [f"G{index - 1}"],
        }
        for index in range(1, 12)
    ]
    original = {"scientific_objective": "Eleven-step analysis.", "goals": goals}
    artifact_paths = []
    for producer in ("G7", "G8"):
        path = run_directory / "goals" / producer / f"{producer}.csv"
        path.parent.mkdir(parents=True)
        path.write_text("value\n1\n", encoding="utf-8")
        artifact_paths.append(
            {
                "artifact_id": f"{producer}:{producer}.csv:" + "0" * 12,
                "producer_goal_id": producer,
                "path": str(path),
                "relative_name": path.name,
                "media_type": "text/csv",
                "description": f"verified records from {producer}",
                "size_bytes": path.stat().st_size,
                "sha256": "0" * 64,
            }
        )
    invalid_goals = [dict(goal) for goal in goals[:3]]
    invalid_goals[0]["objective"] = "Unrelated replacement work."
    state = {
        "planner_raw_response": json.dumps(
            {"scientific_objective": "bad residual", "goals": invalid_goals}
        ),
        "planner_raw_response_path": str(run_directory / "planner_response_v1.json"),
        "planner_response_history": [{"version": 1}],
        "run_directory": str(run_directory),
        "planner_mode": "scientific_replan",
        "high_level_plan": original,
        "completed_goal_results": [
            {"goal_id": f"G{index}", "success": True} for index in range(1, 9)
        ],
        "approved_goal_artifacts": artifact_paths,
        "planner_repair_count": 0,
        "replan_count": 1,
        "trace": [],
    }

    rejected = planner_validator_node(state)
    assert "completed goal ID G1 was reused" in rejected["planner_validation_error"]

    revised_goals = [dict(goal) for goal in goals]
    revised_goals[8] = {
        **revised_goals[8],
        "objective": "Revised post-start exclusion using verified selections.",
        "depends_on": ["G7", "G8"],
    }
    full_revised = {
        "scientific_objective": "Eleven-step analysis.",
        "goals": revised_goals,
    }
    state.update(
        {
            "planner_raw_response": json.dumps(full_revised),
            "planner_response_history": [{"version": 2}],
            "planner_raw_response_path": str(
                run_directory / "planner_response_v2.json"
            ),
        }
    )
    updates = planner_validator_node(state)
    selected = select_current_goal_node({**state, **updates})
    available = _artifacts_available_to_current_goal({**state, **updates, **selected})

    assert len(updates["completed_goal_results"]) == 8
    assert len(updates["approved_goal_artifacts"]) == 2
    assert updates["current_goal_index"] == 8
    assert updates["remaining_goal_count"] == 3
    assert selected["current_goal"]["goal_id"] == "G9"
    assert {item.producer_goal_id for item in available} == {"G7", "G8"}
