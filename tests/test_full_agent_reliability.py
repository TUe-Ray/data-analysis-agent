"""Focused reliability coverage for the schema-aware full-agent workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_analysis_agent.benchmark_approaches import run_agent
from data_analysis_agent.benchmark_tasks import load_benchmark_task, stage_public_task
from data_analysis_agent.benchmark_types import BenchmarkConfig
from data_analysis_agent.final_output import (
    DeterministicFinalOutputProvider,
    FinalGenerationRequest,
)
from data_analysis_agent.graph import build_graph, route_after_execution
from data_analysis_agent.models import (
    ModelCallLimitError,
    RecordingRoleModel,
    ScriptedRoleModel,
)
from data_analysis_agent.nodes import (
    PlannerOutputError,
    _goal_result_limit_error,
    _validate_plan,
    make_verifier_node,
    output_validator_node,
    planner_validator_node,
    select_current_goal_node,
)
from data_analysis_agent.public_schema import (
    required_schema_paths,
    validate_against_public_schema,
)

SIMPLE_SCHEMA = {
    "type": "object",
    "required": ["status", "answer"],
    "properties": {
        "status": {"enum": ["completed"]},
        "answer": {"type": "string"},
    },
    "additionalProperties": False,
}


def _goal(goal_id: str, *, depends_on: list[str] | None = None) -> dict[str, object]:
    return {
        "goal_id": goal_id,
        "objective": f"Produce {goal_id}.",
        "required_outputs": ["value"],
        "constraints": [],
        "success_criteria": ["The value is present."],
        "depends_on": depends_on or [],
    }


def _plan(*goal_ids: str) -> str:
    return json.dumps(
        {
            "scientific_objective": "Produce bounded results.",
            "goals": [
                _goal(goal_id, depends_on=list(goal_ids[:index]))
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
            "concise_reason": "A deterministic script is required.",
        }
    )


def _generation(code: str) -> str:
    return json.dumps({"kind": "python", "code": code, "summary": "Execute."})


def _state(tmp_path: Path, **overrides: object) -> dict[str, object]:
    return {
        "question": "Produce a value.",
        "file_paths": [],
        "staged_file_paths": [],
        "input_context": '{"files": []}',
        "input_profile": {"files": []},
        "run_directory": str(tmp_path / "run"),
        "replan_count": 0,
        "max_replans": 1,
        "max_code_repair_attempts": 2,
        "max_code_repair_no_progress_attempts": 3,
        "max_failure_family_attempts": 5,
        "trace": [],
        **overrides,
    }


def test_plan_requires_final_schema_coverage_and_goal_limit() -> None:
    missing = json.loads(_plan("assemble"))
    missing["final_output_goal_id"] = "assemble"
    try:
        _validate_plan(json.dumps(missing), answer_schema=SIMPLE_SCHEMA)
    except PlannerOutputError as error:
        assert "required public output path" in str(error)
    else:
        raise AssertionError("missing public output coverage was accepted")

    fragmented = json.loads(_plan("g1", "g2", "g3"))
    try:
        _validate_plan(json.dumps(fragmented), max_plan_goals=2)
    except PlannerOutputError as error:
        assert "maximum is 2" in str(error)
    else:
        raise AssertionError("fragmented plan was accepted")


def test_final_assembly_accepts_transitive_dependency_coverage() -> None:
    payload = json.loads(_plan("load", "compute", "assemble"))
    payload["goals"][1]["depends_on"] = ["load"]
    payload["goals"][2]["depends_on"] = ["compute"]
    payload["goals"][2]["required_outputs"] = ["status", "answer"]
    payload["final_output_goal_id"] = "assemble"

    plan, structured = _validate_plan(json.dumps(payload), answer_schema=SIMPLE_SCHEMA)

    assert structured
    assert plan.final_output_goal_id == "assemble"


def test_final_assembly_does_not_merge_intermediate_fields() -> None:
    provider = DeterministicFinalOutputProvider()
    raw = provider.generate(
        FinalGenerationRequest(
            question="q",
            approved_execution_result="unused",
            verifier_decision="PASS",
            verifier_feedback="ok",
            iteration_history=[],
            answer_schema=SIMPLE_SCHEMA,
            final_output_goal_id="assemble",
            completed_goal_results=[
                {
                    "goal_id": "intermediate",
                    "success": True,
                    "result": {"must_not_leak": 1},
                },
                {
                    "goal_id": "assemble",
                    "success": True,
                    "result": {"status": "completed", "answer": "ok"},
                },
            ],
        )
    )

    assert json.loads(raw) == {"status": "completed", "answer": "ok"}


def test_public_schema_gate_rejects_missing_and_unexpected_fields() -> None:
    for candidate in (
        {"status": "completed"},
        {"status": "completed", "answer": "ok", "unexpected": True},
    ):
        updates = output_validator_node(
            {
                "raw_final_output": json.dumps(candidate),
                "answer_schema": SIMPLE_SCHEMA,
                "output_repair_count": 0,
                "max_output_repairs": 0,
                "output_validation_history": [],
                "trace": [],
            }
        )
        assert updates["output_validation_status"] == "INVALID"
        assert updates["validated_final_answer"] is None


def test_valid_final_assembly_candidate_passes_exact_schema() -> None:
    candidate = {"status": "completed", "answer": "ok"}
    updates = output_validator_node(
        {
            "raw_final_output": json.dumps(candidate),
            "answer_schema": SIMPLE_SCHEMA,
            "output_repair_count": 0,
            "max_output_repairs": 0,
            "output_validation_history": [],
            "trace": [],
        }
    )

    assert updates["output_validation_status"] == "VALID"
    assert updates["validated_final_answer"] == candidate


def test_large_goal_result_is_rejected_with_artifact_guidance(tmp_path: Path) -> None:
    error = _goal_result_limit_error(
        _state(tmp_path, max_goal_result_list_length=2),
        {"rows": [{"value": 1}, {"value": 2}, {"value": 3}]},
    )

    assert error is not None
    assert "declared artifacts" in error


def test_invalid_executor_strategy_receives_one_structural_repair(
    tmp_path: Path,
) -> None:
    model = ScriptedRoleModel(
        {
            "planner": [_plan("g1")],
            "executor": [
                "{}",
                _strategy(),
                _generation("__agent_result__ = {'value': 1}\n"),
            ],
            "verifier": ['{"decision":"PASS","feedback":"ok"}'],
        }
    )

    result = build_graph(model).invoke(_state(tmp_path))

    assert result["status"] == "completed"
    assert result["executor_strategy_repair_count"] == 1
    assert [call.structured_schema_name for call in model.calls].count(
        "executor_strategy_repair"
    ) == 1


def test_missing_upstream_contract_escalates_without_python_repairs(
    tmp_path: Path,
) -> None:
    bad = "raise KeyError('upstream_value')\n__agent_result__ = {}\n"
    contract_plan = json.loads(_plan("g1"))
    contract_plan["goals"][0]["required_outputs"] = ["upstream_value"]
    serialized_plan = json.dumps(contract_plan)
    model = ScriptedRoleModel(
        {
            "planner": [serialized_plan, serialized_plan],
            "executor": [
                _strategy(),
                _generation(bad),
                _strategy(),
                _generation("__agent_result__ = {'value': 1}\n"),
            ],
            "verifier": ['{"decision":"PASS","feedback":"ok"}'],
        }
    )

    result = build_graph(model).invoke(_state(tmp_path))

    assert result["status"] == "completed"
    assert result["replan_count"] == 1
    assert result.get("code_repair_count", 0) == 0
    assert result["contract_escalated_goal_ids"] == ["g1"]


def test_trusted_tool_failure_never_routes_to_verifier() -> None:
    state = {
        "current_strategy": {"strategy": "trusted_tool"},
        "current_goal_result": {"success": False, "error": "operational"},
        "contract_escalation_required": False,
    }

    assert route_after_execution(state) == "mechanical_failure"


def test_global_model_call_ceiling_is_enforced() -> None:
    scripted = ScriptedRoleModel({"planner": ["first", "second"]})
    bounded = RecordingRoleModel(scripted, max_calls=1)

    assert bounded.generate(role="planner", messages=[]) == "first"
    with pytest.raises(ModelCallLimitError, match="ceiling reached"):
        bounded.generate(role="planner", messages=[])


def test_rollback_removes_only_target_descendants_and_their_artifacts(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    original_goals = [
        _goal("root"),
        _goal("unrelated"),
        _goal("descendant", depends_on=["root"]),
    ]
    revised_goals = [dict(goal) for goal in original_goals]
    revised_goals[0] = {**revised_goals[0], "objective": "Recompute root."}
    revised_goals[2] = {**revised_goals[2], "objective": "Recompute descendant."}
    artifacts = []
    for goal_id in ("root", "unrelated", "descendant"):
        path = run_directory / f"{goal_id}.csv"
        path.write_text("value\n1\n", encoding="utf-8")
        artifacts.append(
            {
                "artifact_id": f"{goal_id}:artifact:" + "0" * 12,
                "producer_goal_id": goal_id,
                "path": str(path),
                "relative_name": path.name,
                "description": goal_id,
                "size_bytes": path.stat().st_size,
                "sha256": "0" * 64,
            }
        )
    revised = {
        "scientific_objective": "revised",
        "goals": revised_goals,
        "invalidate_from_goal_id": "root",
    }
    state = {
        "planner_raw_response": json.dumps(revised),
        "planner_raw_response_path": str(run_directory / "planner.json"),
        "planner_response_history": [{"version": 1}],
        "run_directory": str(run_directory),
        "planner_mode": "scientific_replan",
        "high_level_plan": {
            "scientific_objective": "original",
            "goals": original_goals,
        },
        "completed_goal_results": [
            {"goal_id": goal_id, "success": True}
            for goal_id in ("root", "unrelated", "descendant")
        ],
        "approved_goal_artifacts": artifacts,
        "rollback_count": 0,
        "max_goal_rollbacks": 1,
        "max_rollback_goals": 3,
        "trace": [],
    }

    updates = planner_validator_node(state)

    assert updates["invalidated_goal_ids"] == ["root", "descendant"]
    assert [item["goal_id"] for item in updates["completed_goal_results"]] == [
        "unrelated"
    ]
    assert [
        item["producer_goal_id"] for item in updates["approved_goal_artifacts"]
    ] == ["unrelated"]


def test_goal_selection_skips_completed_ids_after_replan(tmp_path: Path) -> None:
    plan = json.loads(_plan("g1", "g2"))
    selected = select_current_goal_node(
        _state(
            tmp_path,
            high_level_plan=plan,
            current_goal_index=0,
            completed_goal_results=[{"goal_id": "g1", "success": True}],
        )
    )

    assert selected["current_goal"]["goal_id"] == "g2"
    assert selected["current_goal_index"] == 1


def test_verifier_replaces_duplicate_completed_goal_result(tmp_path: Path) -> None:
    goal = _goal("g1")
    model = ScriptedRoleModel({"verifier": ['{"decision":"PASS","feedback":"ok"}']})
    updates = make_verifier_node(model)(
        _state(
            tmp_path,
            structured_plan=True,
            plan=json.dumps({"scientific_objective": "x", "goals": [goal]}),
            high_level_plan={"scientific_objective": "x", "goals": [goal]},
            current_goal=goal,
            current_goal_index=0,
            current_strategy={"strategy": "generated_python"},
            current_goal_result={
                "goal_id": "g1",
                "success": True,
                "strategy": "generated_python",
                "capability_name": None,
                "result": {"value": 2},
                "warnings": [],
                "error": None,
                "artifact_paths": [],
            },
            completed_goal_results=[
                {
                    "goal_id": "g1",
                    "success": True,
                    "strategy": "generated_python",
                    "capability_name": None,
                    "result": {"value": 1},
                    "warnings": [],
                    "error": None,
                    "artifact_paths": [],
                }
            ],
            execution_result='{"success": true}',
            pending_goal_artifacts=[],
        )
    )

    assert len(updates["completed_goal_results"]) == 1
    assert updates["completed_goal_results"][0]["result"] == {"value": 2}


def _minimal_schema_value(schema: object) -> object:
    if not isinstance(schema, dict):
        return None
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    kind = schema.get("type")
    if kind == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        return {
            name: _minimal_schema_value(properties.get(name, {})) for name in required
        }
    if kind == "array":
        return []
    if kind == "string":
        return ""
    if kind == "integer":
        return 0
    if kind == "number":
        return 0.0
    return None


def test_longitudinal_public_files_complete_through_real_subprocess(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    loaded = load_benchmark_task(
        project_root / "benchmark_tasks", "longitudinal_treatment_response"
    )
    attempt = tmp_path / "attempt"
    public = stage_public_task(loaded.public, attempt)
    candidate = _minimal_schema_value(public.answer_schema)
    assert isinstance(candidate, dict)
    paths = [(attempt / path).resolve() for path in public.data_files]
    code_lines = ["import csv"]
    for index, path in enumerate(paths):
        code_lines.extend(
            [
                f"with open({str(path)!r}, encoding='utf-8') as handle_{index}:",
                f"    rows_{index} = list(csv.DictReader(handle_{index}))",
            ]
        )
    code_lines.append(f"__agent_result__ = {candidate!r}")
    goal_id = "assemble_public_answer"
    plan = json.dumps(
        {
            "scientific_objective": "Read staged inputs and assemble the answer.",
            "goals": [
                {
                    "goal_id": goal_id,
                    "objective": "Read all staged CSVs and assemble the full answer.",
                    "required_outputs": required_schema_paths(public.answer_schema),
                    "constraints": [],
                    "success_criteria": ["The exact public schema validates."],
                    "depends_on": [],
                }
            ],
            "final_output_goal_id": goal_id,
        }
    )
    model = ScriptedRoleModel(
        {
            "planner": [plan],
            "executor": [
                _strategy(),
                json.dumps(
                    {
                        "kind": "python",
                        "code_lines": code_lines,
                        "summary": "Read the real public CSV files.",
                    }
                ),
            ],
            "verifier": ['{"decision":"PASS","feedback":"schema complete"}'],
        }
    )

    outcome = run_agent(
        public=public,
        model=model,
        run_directory=attempt,
        config=BenchmarkConfig(
            model="offline",
            task_ids=[public.task_id],
            approaches=["agent"],
            max_replans=1,
        ),
    )

    assert outcome.status == "completed"
    assert outcome.candidate is not None
    validate_against_public_schema(outcome.candidate, public.answer_schema)
    assert outcome.execution_exit_code == 0
    planner_prompt = model.calls[0].messages[-1]["content"]
    assert "Exact public answer schema" in planner_prompt
    assert public.data_contents[public.data_files[0]] not in planner_prompt
    generated = attempt / "agent_run/goals/assemble_public_answer/generated_code_v1.py"
    assert generated.is_file()
